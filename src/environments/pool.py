import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from src.config.config import PipelineConfig
from src.data.prompt_queue import Prompt, SOURCE_CODECONTESTS
from src.environments.base import Environment
from src.environments.codecontests_env import CodeContestsEnvironment
from src.environments.swe_env import SWEEnvironment


def build_environment(prompt: Prompt, cfg: PipelineConfig) -> Environment:
    """Construct the source-appropriate environment for `prompt`."""
    if prompt.source == SOURCE_CODECONTESTS:
        return CodeContestsEnvironment(prompt.payload, test_timeout=cfg.reward.test_timeout_seconds)
    return SWEEnvironment(prompt.payload, source=prompt.source,
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
        self._ready: queue.Queue[tuple[Prompt, Environment]] = queue.Queue(
            maxsize=cfg.infra.env_prewarming_pool_size
        )
        self._executor = ThreadPoolExecutor(max_workers=cfg.infra.num_environment_workers)
        self._lock = threading.Lock()


    def _build_and_register(self, prompt: Prompt) -> None:
        """Worker callback: build the environment and place it on the ready queue."""
        env = build_environment(prompt, self.cfg)
        env.setup()
        self._ready.put((prompt, env))


    def submit(self, prompt: Prompt) -> None:
        """Schedule pre-warming for the given prompt."""
        self._executor.submit(self._build_and_register, prompt)


    def acquire(self, timeout: float | None = None) -> tuple[Prompt, Environment]:
        """Block until a pre-warmed environment is available."""
        return self._ready.get(timeout=timeout)


    def shutdown(self) -> None:
        """Tear down the worker pool and any unreleased environments."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        while not self._ready.empty():
            try:
                _, env = self._ready.get_nowait()
                env.teardown()
            except queue.Empty:
                break
