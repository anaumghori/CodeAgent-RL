import math
import subprocess
import threading
from collections import deque
from multiprocessing.connection import Connection

from modal.exception import RemoteError

from src.checkpointing.weight_sync import init_inference_weight_transfer
from src.config.config import PipelineConfig
from src.data.datasets import load_eval_subsets, load_training_streams
from src.data.prompt_queue import PromptQueue, SOURCE_CODECONTESTS
from src.environments.pool import EnvironmentPool, build_environment
from src.environments.swe_env import METRICS as SWE_ENVIRONMENT_METRICS
from src.helpers import format_exception, gpu_memory_snapshot, log_event, log_json
from src.inference.rollout import RolloutGenerator, RolloutGroup
from src.inference.vllm_server import VLLMServer
from src.training.rewards import (
    compute_auxiliary_rewards,
    compute_correctness_reward,
    compute_effort_x,
    compute_group_advantages,
    compute_length_penalty,
)
from src.training.evaluator import Evaluator
from src.training.launcher import TrainerLauncher
from src.training.model_loader import load_tokenizer_only
from src.training.rollout_buffer import RolloutBuffer
from src.training.wandb_logger import WandbLogger

IPC_WAIT_POLL_SEC = 1.0

class Orchestrator:
    """
    Top-level pipeline driver running in the orchestrator parent process.
    Owns the vLLM inference engine (on the GPU exposed via
    `CUDA_VISIBLE_DEVICES`), the prompt queue, the rollout generator and
    environment workers, the in-memory rollout buffer, the evaluator, and
    the W&B logger. The DeepSpeed-backed Trainer lives in two child
    subprocesses spawned by `TrainerLauncher`; this process communicates
    with rank 0 over a Unix-domain socket IPC channel.
    """

    def __init__(self, cfg: PipelineConfig, resume_from: str | None = None) -> None:
        self.cfg = cfg
        self.resume_from = resume_from
        self.eval_subsets = load_eval_subsets(cfg)
        self.streams = load_training_streams(cfg, self.eval_subsets)
        self.prompt_queue = PromptQueue(cfg, self.streams)
        self.env_pool = EnvironmentPool(cfg)
        self._env_build_failures: int = 0
        self.rollout_buffer = RolloutBuffer(
            max_groups=cfg.infra.rollout_buffer_max_groups,
            max_staleness=cfg.grpo.max_policy_staleness,
        )
        self.logger = WandbLogger(cfg)
        self.server = VLLMServer(cfg)
        self.tokenizer = None
        self.evaluator: Evaluator | None = None
        self.rollout_generator: RolloutGenerator | None = None
        self.launcher: TrainerLauncher | None = None
        self.ipc: Connection | None = None
        self._stop = threading.Event()
        self._ipc_send_lock = threading.Lock()
        self._step_event = threading.Event()
        self._step_payload: tuple | None = None
        self._save_event = threading.Event()
        self._save_payload: tuple | None = None
        self._load_event = threading.Event()
        self._load_payload: tuple | None = None
        self._inference_error: BaseException | None = None
        self._dispatcher_error: BaseException | None = None
        self._prompt_backlog: deque = deque()
        self.training_step: int = 0
        self.policy_version: int = 0
        self._client_state: dict = {
            "training_step": 0,
            "policy_version": 0,
            "prompt_index": 0,
            "curriculum_stage": 1,
            "wandb_run_id": None,
        }


    def _prewarm_env_count(self, prompt) -> int:
        """Return how many environments to prepare ahead of time for `prompt`."""
        if prompt.source == SOURCE_CODECONTESTS:
            return 1
        return min(2, self.cfg.grpo.group_size)


    def _schedule_env_prewarm(self, prompt) -> None:
        """Submit background environment setup for an upcoming prompt."""
        self.env_pool.submit(prompt, count=self._prewarm_env_count(prompt))


    def _ensure_prompt_backlog(self, count: int) -> None:
        """
        Keep at least `count` prompts prefetched so environment setup can overlap
        with training and inference.
        """
        while len(self._prompt_backlog) < count:
            prompt = self.prompt_queue.next_prompt(self.training_step)
            self._prompt_backlog.append(prompt)
            self._schedule_env_prewarm(prompt)


    def _next_prompt(self):
        """Return the next prompt, prefetched when possible, and replenish the backlog."""
        self._ensure_prompt_backlog(1)
        prompt = self._prompt_backlog.popleft()
        self._ensure_prompt_backlog(2)
        return prompt


    def _compute_rewards_for_group(self, group: RolloutGroup) -> None:
        """Run correctness, auxiliary, and length-penalty rewards over the group."""
        rewards: list[float] = []
        successes = 0
        for rollout in group.rollouts:
            corr = compute_correctness_reward(rollout.test_result, rollout.source, self.cfg)
            aux = compute_auxiliary_rewards(rollout.metadata, self.cfg)
            n_think = sum(s.n_think_tokens for s in rollout.segments)
            n_tool_call = sum(s.n_tool_call_tokens for s in rollout.segments)
            n_tool_out = sum(s.n_tool_output_tokens for s in rollout.segments)
            n_final = sum(s.n_final_tokens for s in rollout.segments)
            x = compute_effort_x(
                think_tokens=n_think,
                tool_call_tokens=n_tool_call,
                tool_output_tokens=n_tool_out,
                final_tokens=n_final,
                n_tool_calls=rollout.n_tool_calls,
                n_turns=rollout.n_turns,
            )
            penalty = compute_length_penalty(x, self.cfg)
            total = corr + aux["total_weighted"] - penalty
            rollout.reward = total
            rewards.append(total)
            success = corr > 0.0
            successes += 1 if success else 0
            self.prompt_queue.update_task_outcome(rollout.prompt_id, success=success)
        advantages = compute_group_advantages(rewards)
        for rollout, adv in zip(group.rollouts, advantages):
            rollout.advantage = adv
        reward_mean = sum(rewards) / max(1, len(rewards))
        reward_variance = sum((reward - reward_mean) ** 2 for reward in rewards) / max(1, len(rewards))
        reward_std = math.sqrt(reward_variance)
        success_rate = successes / max(1, len(group.rollouts))
        log_json(
            "orchestrator",
            "group_reward_summary",
            {
                "prompt_id": group.prompt.prompt_id,
                "source": group.prompt.source,
                "success_rate": success_rate,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_min": min(rewards) if rewards else 0.0,
                "reward_max": max(rewards) if rewards else 0.0,
            },
        )
        if success_rate < self.cfg.curriculum.hard_task_success_threshold:
            self.prompt_queue.maybe_requeue_prompt(group.prompt, self.training_step)


    def _inference_loop(self) -> None:
        """Background thread: continuously generate rollout groups and push them to the buffer."""
        gen = self.rollout_generator
        assert gen is not None
        try:
            while not self._stop.is_set():
                prompt = self._next_prompt()
                self._client_state["prompt_index"] = self.prompt_queue.global_index

                def env_factory(p=prompt):
                    wait_timeout = 10.0 if p.source != SOURCE_CODECONTESTS else 1.0
                    env = self.env_pool.acquire_for_prompt(p, timeout=wait_timeout)
                    if env is not None:
                        return env
                    self.env_pool.claim_inline_build(p)
                    success = False
                    try:
                        env = build_environment(p, self.cfg)
                        env.setup()
                        setattr(env, "_composer_prepared", True)
                        success = True
                        return env
                    finally:
                        self.env_pool.release_inline_build(p, success=success)

                # SWE rollouts can fail during environment setup (image pull,
                # build, or git clone errors). Skip the prompt and continue
                # rather than tearing down the inference loop, since these
                # errors are per-instance and not pipeline-wide.
                try:
                    group = gen.generate_group(prompt, env_factory, self.policy_version)
                except (subprocess.CalledProcessError, RuntimeError, OSError, RemoteError) as exc:
                    self._env_build_failures += 1
                    log_event(
                        "orchestrator",
                        f"Environment setup failed for prompt {prompt.prompt_id} "
                        f"(source={prompt.source}): {exc}; skipping prompt.",
                    )
                    continue
                self._compute_rewards_for_group(group)
                self.rollout_buffer.push(group)
        except BaseException as exc:
            # Capture the error so the main thread surfaces it instead of
            # silently hanging on an empty rollout buffer.
            self._inference_error = exc
            self._stop.set()
            self.rollout_buffer.unblock_consumers()
            log_event("orchestrator", f"Inference loop failed:\n{format_exception(exc)}")
            raise


    def _ipc_send(self, payload: tuple) -> None:
        with self._ipc_send_lock:
            self.ipc.send(payload)


    def _ipc_dispatcher(self) -> None:
        """
        Background thread: read every message from rank 0 and either service
        a vLLM pause/update/resume request or signal a completion event.
        """
        try:
            while not self._stop.is_set():
                try:
                    msg = self.ipc.recv()
                except (EOFError, OSError):
                    return
                kind = msg[0]
                if kind == "vllm_pause_and_update":
                    _, names, dtype_names, shapes = msg
                    self.server.pause_for_weight_sync()
                    self.server.begin_weight_update(names, dtype_names, shapes)
                    self._ipc_send(("ack",))
                elif kind == "vllm_join_and_resume":
                    self.server.end_weight_update()
                    self.server.resume_after_weight_sync()
                    self._ipc_send(("ack",))
                elif kind == "step_done":
                    _, step, policy_version, metrics = msg
                    self.training_step = step
                    self.policy_version = policy_version
                    self._step_payload = (step, policy_version, metrics)
                    self._step_event.set()
                elif kind == "save_checkpoint_done":
                    self._save_payload = msg[1:]
                    self._save_event.set()
                elif kind == "load_checkpoint_done":
                    self._load_payload = msg[1:]
                    self._load_event.set()
                else:
                    raise RuntimeError(f"Orchestrator received unknown IPC message: {msg!r}")
        except BaseException as exc:
            self._dispatcher_error = exc
            self._stop.set()
            # Wake any thread blocked on a completion event so it can observe the error.
            self._step_event.set()
            self._save_event.set()
            self._load_event.set()
            log_event("orchestrator", f"IPC dispatcher failed:\n{format_exception(exc)}")
            raise


    def _check_health(self) -> None:
        """Surface any latent error from the inference thread, dispatcher, or trainer subprocesses."""
        if self._inference_error is not None:
            raise RuntimeError("Inference thread failed") from self._inference_error
        if self._dispatcher_error is not None:
            raise RuntimeError("IPC dispatcher failed") from self._dispatcher_error
        if self.launcher is not None:
            crashed = self.launcher.crashed_ranks()
            if crashed:
                raise RuntimeError(f"Trainer subprocess(es) exited unexpectedly: ranks={crashed}")


    def _wait_for(self, event: threading.Event, payload_attr: str) -> tuple:
        """
        Block until `event` fires, polling for trainer / inference health while
        waiting so a crash in any background actor surfaces instead of hanging.
        """
        while not event.wait(timeout=IPC_WAIT_POLL_SEC):
            self._check_health()
        event.clear()
        # The dispatcher signals every event on shutdown to unblock waiters; if
        # health is bad here, surface the error rather than returning a stale payload.
        self._check_health()
        payload = getattr(self, payload_attr)
        setattr(self, payload_attr, None)
        return payload


    def _run_training_step(self, groups: list[RolloutGroup]) -> dict:
        """Send a `step` command to rank 0 and wait for completion."""
        self._ipc_send(("step", groups))
        step, policy_version, metrics = self._wait_for(self._step_event, "_step_payload")
        return metrics


    def _save_checkpoint(self, checkpoint_number: int, client_state: dict) -> None:
        """Send a checkpoint save command to rank 0 and wait for completion."""
        self._ipc_send(("save_checkpoint", checkpoint_number, client_state))
        self._wait_for(self._save_event, "_save_payload")


    def _load_checkpoint(self, checkpoint_tag: str) -> dict:
        """Send a checkpoint load command to rank 0 and wait for the restored client state."""
        self._ipc_send(("load_checkpoint", checkpoint_tag))
        payload = self._wait_for(self._load_event, "_load_payload")
        return payload[0] if payload else {}


    def _maybe_evaluate_and_checkpoint(self, latest_metrics: dict) -> None:
        """Run evaluation + push a recovery checkpoint at the configured cadence."""
        step = self.training_step
        if step == 0 or step % self.cfg.eval.eval_interval != 0:
            return
        checkpoint_number = step // self.cfg.checkpoint.recovery_checkpoint_interval
        log_event(
            "orchestrator",
            f"Starting checkpoint/evaluation cycle at step={step}, checkpoint_number={checkpoint_number}.",
        )
        self._client_state.update({
            "training_step": step,
            "policy_version": self.policy_version,
            "prompt_index": self.prompt_queue.global_index,
            "wandb_run_id": self.logger.run.id if self.logger.run else None,
        })
        self._save_checkpoint(checkpoint_number, self._client_state)
        self.evaluator.run(training_step=step)


    def _environment_metrics(self) -> dict:
        """Snapshot per-step Modal environment metrics for the W&B log payload."""
        out = SWE_ENVIRONMENT_METRICS.snapshot_and_reset()
        out["env/build_failures_cumulative"] = self._env_build_failures
        return out


    def setup(self) -> None:
        """Materialise every component (vLLM, trainer subprocesses, evaluator) before training begins."""
        log_event("orchestrator", "Starting setup().")
        log_event("orchestrator", "Loading tokenizer.")
        self.tokenizer = load_tokenizer_only(self.cfg)
        server_error: list[BaseException] = []

        def _start_server() -> None:
            try:
                self.server.start()
            except BaseException as exc:
                server_error.append(exc)

        log_event("orchestrator", "Starting vLLM server and trainer subprocesses in parallel.")
        server_thread = threading.Thread(target=_start_server, daemon=True)
        server_thread.start()
        self.launcher = TrainerLauncher(self.cfg)
        self.ipc = self.launcher.start()
        server_thread.join()
        if server_error:
            raise RuntimeError("vLLM server startup failed") from server_error[0]
        log_event("orchestrator", "vLLM server and trainer subprocess launch completed.")

        # Handshake with rank 0: it tells us once DeepSpeed engine init is
        # complete; we then concurrently run NCCL inference init while rank 0
        # runs trainer_init, both blocking on the rendezvous.
        first = self.ipc.recv()
        if first != ("ds_init_done",):
            raise RuntimeError(f"Expected ds_init_done from trainer rank 0, got {first!r}")
        log_event("orchestrator", "Trainer DeepSpeed initialization complete; starting NCCL handshake.")

        nccl_inference_thread = threading.Thread(
            target=lambda: init_inference_weight_transfer(self.cfg, self.server.llm),
            daemon=True,
        )
        nccl_inference_thread.start()
        self.ipc.send(("init_nccl",))
        nccl_done = self.ipc.recv()
        if nccl_done != ("nccl_done",):
            raise RuntimeError(f"Expected nccl_done from trainer rank 0, got {nccl_done!r}")
        nccl_inference_thread.join()
        log_event("orchestrator", "NCCL weight transfer channel is ready.")

        self.rollout_generator = RolloutGenerator(self.cfg, self.server, self.tokenizer)
        self.evaluator = Evaluator(self.cfg, self.server, self.tokenizer,
                                   self.eval_subsets, self.logger)

        # Spin up the IPC dispatcher before any subsequent commands so vLLM
        # scheduler sleep / wake requests during weight sync are serviced
        # asynchronously.
        threading.Thread(target=self._ipc_dispatcher, daemon=True).start()

        run_id = None
        if self.resume_from:
            log_event("orchestrator", f"Resuming from checkpoint {self.resume_from}.")
            self._client_state = self._load_checkpoint(self.resume_from)
            self.training_step = self._client_state.get("training_step", 0)
            self.policy_version = self._client_state.get("policy_version", 0)
            self.prompt_queue.restore_position(self._client_state.get("prompt_index", 0))
            run_id = self._client_state.get("wandb_run_id")

        self.logger.init(run_id=run_id)
        self._ensure_prompt_backlog(2)
        log_event("orchestrator", "Setup complete.")


    def run(self) -> None:
        """Drive the training loop: feed groups to the trainer until the total step budget is reached."""
        log_event("orchestrator", "Starting training loop.")
        inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        inference_thread.start()

        target_groups = max(1, self.cfg.effective_batch_size // self.cfg.grpo.group_size)
        try:
            while self.training_step < self.cfg.training.total_steps:
                self._check_health()
                groups = self.rollout_buffer.pull(
                    count=target_groups,
                    current_version=self.policy_version,
                    timeout=IPC_WAIT_POLL_SEC,
                )
                if not groups:
                    continue
                log_event(
                    "orchestrator",
                    f"Dispatching {len(groups)} rollout groups to trainer at step={self.training_step}, "
                    f"policy_version={self.policy_version}.",
                )
                metrics = self._run_training_step(groups)
                # `resource/...`) by the trainer; do not re-prefix.
                metrics["train/buffer_depth"] = self.rollout_buffer.depth()
                metrics["train/staleness_drop_rate"] = self.rollout_buffer.pop_staleness_drops()
                metrics["train/buffer_underruns"] = self.rollout_buffer.pop_underruns()
                metrics["train/policy_version"] = self.policy_version
                # cuda:0 in the parent == global inference GPU.
                inference_mem = gpu_memory_snapshot(0)
                metrics["resource/gpu_inference_memory_allocated_gb"] = inference_mem["allocated_gb"]
                metrics["resource/gpu_inference_memory_reserved_gb"] = inference_mem["reserved_gb"]
                metrics["resource/gpu_inference_max_memory_allocated_gb"] = inference_mem["max_allocated_gb"]
                metrics["resource/gpu_inference_max_memory_reserved_gb"] = inference_mem["max_reserved_gb"]
                metrics.update(self._environment_metrics())
                if (self.training_step % self.cfg.logging.log_interval) == 0:
                    self.logger.log(metrics, step=self.training_step)
                self._maybe_evaluate_and_checkpoint(metrics)
                log_event(
                    "orchestrator",
                    f"Completed step {self.training_step}; policy_version={self.policy_version}; "
                    f"loss={metrics.get('train/loss', 0.0):.4f}; "
                    f"mean_reward={metrics.get('train/mean_reward', 0.0):.4f}; "
                    f"step_time_sec={metrics.get('timing/step_time_sec', 0.0):.2f}; "
                    f"buffer_depth={metrics.get('train/buffer_depth', 0.0):.0f}.",
                )
        except BaseException as exc:
            log_event("orchestrator", f"Run loop failed:\n{format_exception(exc)}")
            raise
        finally:
            log_event("orchestrator", "Beginning orchestrator shutdown.")
            self._stop.set()
            try:
                self._ipc_send(("shutdown",))
            except (BrokenPipeError, OSError):
                pass
            self.rollout_buffer.unblock_consumers()
            self.env_pool.shutdown()
            self.server.shutdown()
            self.logger.finish()
            if self.launcher is not None:
                self.launcher.shutdown()
