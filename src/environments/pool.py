import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from src.config.config import PipelineConfig
from src.data.prompt_queue import Prompt, SOURCE_CODECONTESTS
from src.environments.base import Environment
from src.environments.codecontests_env import CodeContestsEnvironment
from src.environments.swe_env import SWEEnvironment
from src.helpers import format_exception, log_event


def build_environment(prompt: Prompt, cfg: PipelineConfig) -> Environment:
    """Construct the source-appropriate environment for `prompt`."""
    if prompt.source == SOURCE_CODECONTESTS:
        return CodeContestsEnvironment(prompt.payload, test_timeout=cfg.reward.test_timeout_seconds)
    return SWEEnvironment(prompt.payload, source=prompt.source, cfg=cfg,
                          test_timeout=cfg.reward.test_timeout_seconds)


class EnvironmentPool:
    """
    Rolling pre-warming pool of pre-initialised environments. The inference
    process pushes upcoming prompts into `submit()` and pulls ready
    environments via `acquire()`. Setup runs concurrently in a worker
    thread pool so the inference loop never blocks on environment build time.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._executor = ThreadPoolExecutor(max_workers=cfg.infra.num_environment_workers)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._ready_by_prompt: dict[str, deque[Environment]] = defaultdict(deque)
        self._inflight_prewarms: dict[str, int] = defaultdict(int)
        self._inline_claimed: set[str] = set()


    def _build_and_register(self, prompt: Prompt) -> None:
        """Worker callback: build the environment and place it on the ready queue."""
        try:
            env = build_environment(prompt, self.cfg)
            env.setup()
            setattr(env, "_composer_prepared", True)
            discard = False
            with self._cond:
                if prompt.prompt_id in self._inline_claimed:
                    discard = True
                else:
                    self._ready_by_prompt[prompt.prompt_id].append(env)
                self._cond.notify_all()
            if discard:
                env.teardown()
        except BaseException as exc:
            log_event(
                "env_pool",
                f"Prewarming failed for prompt {prompt.prompt_id}:\n{format_exception(exc)}",
            )
        finally:
            with self._cond:
                self._inflight_prewarms[prompt.prompt_id] = max(
                    0,
                    self._inflight_prewarms[prompt.prompt_id] - 1,
                )
                if self._inflight_prewarms[prompt.prompt_id] == 0 and prompt.prompt_id in self._inline_claimed:
                    self._inline_claimed.discard(prompt.prompt_id)
                self._cond.notify_all()


    def submit(self, prompt: Prompt, count: int = 1) -> None:
        """Schedule pre-warming for the given prompt."""
        with self._cond:
            ready = len(self._ready_by_prompt[prompt.prompt_id])
            inflight = self._inflight_prewarms[prompt.prompt_id]
            target = max(0, count - ready - inflight)
            self._inflight_prewarms[prompt.prompt_id] += target
        for _ in range(target):
            self._executor.submit(self._build_and_register, prompt)


    def acquire_for_prompt(self, prompt: Prompt, timeout: float | None = None) -> Environment | None:
        """
        Return a pre-warmed environment for `prompt` when available.

        :param prompt: Prompt whose environment is requested.
        :param timeout: Optional maximum wait time in seconds.
        :returns: Matching environment or None if none becomes available.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                cached = self._ready_by_prompt.get(prompt.prompt_id)
                if cached:
                    env = cached.popleft()
                    return env
                inflight = self._inflight_prewarms[prompt.prompt_id]
                if inflight <= 0:
                    return None
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if deadline is not None and remaining <= 0.0:
                    return None
                self._cond.wait(timeout=remaining)


    def claim_inline_build(self, prompt: Prompt) -> None:
        """
        Mark a prompt as being taken over by the inline rollout path after a
        prewarm miss/timeout so late prewarm completions can be discarded.
        """
        with self._cond:
            self._inline_claimed.add(prompt.prompt_id)
            self._cond.notify_all()


    def release_inline_build(self, prompt: Prompt, success: bool) -> None:
        """Clear the inline-build claim for a prompt once inline setup completes."""
        with self._cond:
            if not success or self._inflight_prewarms[prompt.prompt_id] == 0:
                self._inline_claimed.discard(prompt.prompt_id)
            self._cond.notify_all()


    def shutdown(self) -> None:
        """Tear down the worker pool and any unreleased environments."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._cond:
            for envs in self._ready_by_prompt.values():
                while envs:
                    envs.popleft().teardown()
            self._ready_by_prompt.clear()
            self._inline_claimed.clear()
            self._inflight_prewarms.clear()
            self._cond.notify_all()
