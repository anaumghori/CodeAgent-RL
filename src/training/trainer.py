import math
import os
import time
import deepspeed
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

from src.config.config import PipelineConfig
from src.config.deepspeed_config import build_deepspeed_config
from src.data.sequence_packing import TrainingSequence, pack_sequences, PackedMicrobatch
from src.helpers import gpu_memory_gb, timed
from src.inference.rollout import RolloutGroup
from src.training.loss import compute_grpo_loss


REQUIRED_DIST_ENV = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK")


def _cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    """Cosine schedule from 1.0 to 0.0 with linear warmup over `warmup` steps."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class Trainer:
    """
    Multi-rank DeepSpeed-backed GRPO trainer. One instance lives in each
    trainer subprocess (rank 0 and rank 1). Rank 0 receives rollout groups
    over IPC from the orchestrator parent, packs sequences across both
    ranks, broadcasts microbatches via `dist.broadcast_object_list`, and
    drives weight synchronisation. Rank 1 participates only in collective
    operations (broadcast receive, ZeRO-3 gathers, all-reduce of metrics).
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        model,
        tokenizer,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer

        for name in REQUIRED_DIST_ENV:
            if name not in os.environ:
                raise RuntimeError(
                    f"Trainer requires the torch.distributed environment variable {name} "
                    f"to be set by the launcher. Got env keys: {sorted(k for k in os.environ if k.startswith(('RANK','WORLD','MASTER','LOCAL')))}"
                )

        self.weight_transfer_metadata = self._capture_weight_transfer_metadata(model)

        ds_config = build_deepspeed_config(cfg)

        def _build_scheduler(base_optimizer):
            """DeepSpeed invokes this callable with the underlying torch optimizer
            (`engine.basic_optimizer`), keeping `LambdaLR`'s `isinstance` check
            satisfied. DeepSpeed then steps the returned scheduler automatically
            from inside `engine.step()` at gradient-accumulation boundaries."""
            return LambdaLR(
                base_optimizer,
                lr_lambda=lambda s: _cosine_with_warmup(
                    s, cfg.training.warmup_steps, cfg.training.total_steps,
                ),
            )

        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, config=ds_config, lr_scheduler=_build_scheduler,
        )
        self.engine = engine
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.step: int = 0
        self.policy_version: int = 0
        self._max_seq_length: int = cfg.sequence.max_training_seq_length


    def _capture_weight_transfer_metadata(
        self,
        model,
    ) -> tuple[list[str], list[str], list[tuple[int, ...]]]:
        """
        Capture the full parameter metadata before DeepSpeed ZeRO-3 wraps the
        model. vLLM expects the metadata passed to `update_weights` to match
        the exact full tensors later broadcast over NCCL.

        :param model: The pre-DeepSpeed Hugging Face model.
        :returns: Weight names, dtypes, and full shapes in parameter order.
        """
        named_parameters = list(model.named_parameters())
        names = [name for name, _ in named_parameters]
        dtype_names = [str(param.dtype).replace("torch.", "") for _, param in named_parameters]
        shapes = [tuple(param.shape) for _, param in named_parameters]
        return names, dtype_names, shapes


    def update_max_sequence_length(self) -> None:
        """Apply the curriculum-driven sequence-length extension at the configured step."""
        if self.step >= self.cfg.curriculum.seq_length_extension_step:
            self._max_seq_length = self.cfg.curriculum.extended_max_training_seq_length


    def _groups_to_sequences(self, groups: list[RolloutGroup]) -> tuple[list[TrainingSequence], dict]:
        """Flatten rollout groups into training sequences and gather aggregate stats."""
        sequences: list[TrainingSequence] = []
        rewards: list[float] = []
        advantages: list[float] = []
        turns: list[int] = []
        token_counts: list[int] = []
        correctness_hits = 0
        for group in groups:
            for rollout in group.rollouts:
                rewards.append(rollout.reward)
                advantages.append(rollout.advantage)
                turns.append(rollout.n_turns)
                tokens = sum(len(s.token_ids) for s in rollout.segments)
                token_counts.append(tokens)
                if rollout.test_result.get("fail_to_pass_passed", 0) > 0 \
                        or rollout.test_result.get("tests_passed", 0) > 0:
                    correctness_hits += 1
                for segment in rollout.segments:
                    sequences.append(TrainingSequence(
                        input_ids=segment.token_ids,
                        loss_mask=segment.loss_mask,
                        old_logprobs=segment.logprobs,
                        advantage=rollout.advantage,
                        policy_version=rollout.policy_version,
                        rollout_id=rollout.rollout_id,
                    ))
        n = max(1, len(rewards))
        stats = {
            "mean_reward": sum(rewards) / n,
            "mean_advantage": sum(advantages) / n,
            "mean_rollout_length": sum(token_counts) / n,
            "mean_turns": sum(turns) / n,
            "correctness_rate": correctness_hits / n,
        }
        return sequences, stats


    def _broadcast_microbatches(
        self,
        per_rank: list[list[PackedMicrobatch]] | None,
    ) -> list[PackedMicrobatch]:
        """
        Rank 0 broadcasts per-rank packed microbatches to every rank using
        raw `dist.broadcast` for the tensor payloads plus a tiny
        `broadcast_object_list` for per-microbatch shapes.

        Each rank reconstructs only the microbatches that target itself.
        """
        device = next(self.engine.module.parameters()).device

        if self.rank == 0:
            shape_meta = [
                [int(mb.input_ids.shape[1]) for mb in rank_list]
                for rank_list in per_rank
            ]
        else:
            shape_meta = None
        meta_payload = [shape_meta]
        dist.broadcast_object_list(meta_payload, src=0)
        shape_meta = meta_payload[0]

        local_microbatches: list[PackedMicrobatch] = []
        for r, rank_meta in enumerate(shape_meta):
            for i, T in enumerate(rank_meta):
                if self.rank == 0:
                    src_mb = per_rank[r][i]
                    input_ids = src_mb.input_ids.to(device)
                    attention_mask = src_mb.attention_mask.to(device)
                    loss_mask = src_mb.loss_mask.to(device)
                    old_logprobs = src_mb.old_logprobs.to(device)
                    advantages = src_mb.advantages.to(device)
                else:
                    input_ids = torch.empty((1, T), dtype=torch.long, device=device)
                    attention_mask = torch.empty((1, T), dtype=torch.long, device=device)
                    loss_mask = torch.empty((1, T - 1), dtype=torch.float32, device=device)
                    old_logprobs = torch.empty((1, T - 1), dtype=torch.float32, device=device)
                    advantages = torch.empty((1,), dtype=torch.float32, device=device)
                for tensor in (input_ids, attention_mask, loss_mask, old_logprobs, advantages):
                    dist.broadcast(tensor, src=0)
                if r == self.rank:
                    local_microbatches.append(PackedMicrobatch(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask,
                        old_logprobs=old_logprobs,
                        advantages=advantages,
                        rank=r,
                    ))
        return local_microbatches


    def _forward_backward(
        self,
        microbatches: list[PackedMicrobatch],
        timing: dict,
    ) -> dict:
        """Run forward + backward + optimizer step over the local microbatch list."""
        self.engine.train()
        agg = {"pg_loss": 0.0, "kl_loss": 0.0, "clip_fraction": 0.0,
               "mean_ratio": 0.0, "mean_kl": 0.0, "loss": 0.0}
        device = next(self.engine.module.parameters()).device

        f_total = 0.0
        b_total = 0.0
        opt_total = 0.0
        for i, mb in enumerate(microbatches):
            is_last = i == len(microbatches) - 1
            self.engine.set_gradient_accumulation_boundary(is_last)
            input_ids = mb.input_ids
            attention = mb.attention_mask
            old_lp = mb.old_logprobs
            advantages = mb.advantages
            mask = mb.loss_mask
            if input_ids.device != device:
                input_ids = input_ids.to(device)
                attention = attention.to(device)
                old_lp = old_lp.to(device)
                advantages = advantages.to(device)
                mask = mask.to(device)

            t0 = time.perf_counter()
            outputs = self.engine(input_ids=input_ids, attention_mask=attention)
            f_total += time.perf_counter() - t0

            loss, diag = compute_grpo_loss(
                logits=outputs.logits,
                input_ids=input_ids,
                old_logprobs=old_lp,
                advantages=advantages,
                loss_mask=mask,
                kl_coeff=self.cfg.grpo.kl_coefficient,
                clip_epsilon=self.cfg.grpo.ppo_clip_epsilon,
            )
            t0 = time.perf_counter()
            self.engine.backward(loss)
            b_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            self.engine.step()
            opt_total += time.perf_counter() - t0

            agg["loss"] += float(loss.detach().item())
            for k in ("pg_loss", "kl_loss", "clip_fraction", "mean_ratio", "mean_kl"):
                agg[k] += diag[k]
        n = max(1, len(microbatches))
        for k in agg:
            agg[k] /= n
        timing["forward_time_sec"] = f_total
        timing["backward_time_sec"] = b_total
        timing["optimizer_time_sec"] = opt_total
        # DeepSpeed steps the LR scheduler internally inside engine.step() at
        # gradient-accumulation boundaries; no manual scheduler.step() here.
        return agg


    def _all_reduce_metrics(self, metrics: dict) -> dict:
        """Average scalar metrics across all DP ranks."""
        if self.world_size <= 1:
            return metrics
        keys = sorted(metrics.keys())
        tensor = torch.tensor([metrics[k] for k in keys], dtype=torch.float64,
                              device=next(self.engine.module.parameters()).device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return {k: float(tensor[i].item()) for i, k in enumerate(keys)}


    def execute_step(self, groups: list[RolloutGroup] | None) -> dict:
        """
        Execute one optimizer step. Rank 0 receives the rollout groups and
        produces packed microbatches; rank > 0 ignores the `groups` argument
        and obtains its microbatches over `dist.broadcast`.

        Returns a metrics dictionary on every rank with keys already
        namespaced as `train/...`, `timing/...`, and `resource/...`.
        Scalar metrics are averaged across ranks; the rank-local resource
        metric is added by rank 0 only after the all-reduce.
        """
        self.update_max_sequence_length()
        timing: dict = {}
        stats: dict = {}

        with timed(timing, "step_time_sec"):
            if self.rank == 0:
                if not groups:
                    raise RuntimeError("Rank 0 received empty groups list for execute_step.")
                sequences, stats = self._groups_to_sequences(groups)
                packed = pack_sequences(
                    sequences,
                    num_ranks=self.world_size,
                    max_seq_length=self._max_seq_length,
                )
            else:
                packed = None
            local_microbatches = self._broadcast_microbatches(packed)
            stats_payload = [stats] if self.rank == 0 else [None]
            dist.broadcast_object_list(stats_payload, src=0)
            stats = stats_payload[0]
            loss_metrics = self._forward_backward(local_microbatches, timing)

        self.step += 1

        # Averaged-across-ranks scalar metrics under their final namespaces.
        averaged: dict = {}
        for k, v in loss_metrics.items():
            averaged[f"train/{k}"] = v
        for k, v in stats.items():
            averaged[f"train/{k}"] = v
        for k, v in timing.items():
            averaged[f"timing/{k}"] = v
        averaged = self._all_reduce_metrics(averaged)
        averaged["train/learning_rate"] = self.scheduler.get_last_lr()[0]

        if self.rank == 0:
            a0, r0 = gpu_memory_gb(0)
            averaged["resource/gpu_train_memory_allocated_gb"] = a0
            averaged["resource/gpu_train_memory_reserved_gb"] = r0
        return averaged
