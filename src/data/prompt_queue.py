from dataclasses import dataclass
from collections import deque
import random
import threading

from src.config.config import PipelineConfig
from src.data.datasets import TrainingStreams, codecontests_difficulty, swe_difficulty


SOURCE_SWE_V2 = "swe_v2"
SOURCE_SWE_PRS = "swe_prs"
SOURCE_CODECONTESTS = "codecontests"


@dataclass
class Prompt:
    """
    A single training task fed to the inference server. The `payload` field
    holds the original dataset row so reward computation has full access to
    `corner_cases`, `FAIL_TO_PASS`, etc.
    """
    prompt_id: str
    source: str
    task_text: str
    payload: dict
    difficulty: float = 0.0
    upsample_weight: float = 1.0


def _format_swe_prompt(ex: dict) -> str:
    statement = ex.get("problem_statement") or ex.get("pr_description", "")
    repo = ex.get("repo", "")
    base = ex.get("base_commit", "")
    return (
        f"Repository: {repo}\nBase commit: {base}\n\n"
        f"Issue / PR description:\n{statement}\n\n"
        "Diagnose the problem, locate the relevant files, make a minimal patch "
        "that satisfies the failing tests without breaking existing tests, and run the "
        "test suite to verify the fix."
    )


def _format_codecontests_prompt(ex: dict) -> str:
    name = ex.get("name", "")
    desc = ex.get("description", "")
    return (
        f"Problem: {name}\n\n{desc}\n\n"
        "Read input from stdin and write the answer to stdout. "
        "Implement, test against the provided sample cases, and submit your final solution."
    )


class PromptQueue:
    """
    Continuously interleaves prompts from the streaming training datasets
    according to per-stage mix ratios, applies single-epoch enforcement,
    excludes held-out evaluation instances, and supports reward-based
    upsampling of difficult tasks.

    Thread-safe pull/push interface is used by the inference process.
    """

    def __init__(self, cfg: PipelineConfig, streams: TrainingStreams) -> None:
        self.cfg = cfg
        self.streams = streams
        self._iters = {
            SOURCE_SWE_V2: iter(streams.swe_v2),
            SOURCE_SWE_PRS: iter(streams.swe_prs),
            SOURCE_CODECONTESTS: iter(streams.codecontests),
        }
        self._lock = threading.Lock()
        self.global_index: int = 0
        # Difficulty-bucket success rates: prompt_id -> (n_total, n_success)
        self._task_stats: dict[str, tuple[int, int]] = {}
        self._replay_queue: deque[Prompt] = deque()
        self._rng = random.Random(0)


    def restore_position(self, global_index: int) -> None:
        """
        Skip the first `global_index` prompts in each source stream after a
        crash recovery. Streaming datasets do not support random access, so
        this is a one-time linear skip per source.
        """
        with self._lock:
            self.global_index = global_index
        # Skip an equivalent number from each iter (approximate but conservative)
        for src, it in self._iters.items():
            for _ in range(global_index):
                try:
                    next(it)
                except StopIteration:
                    break


    def update_task_outcome(self, prompt_id: str, success: bool) -> None:
        """Record whether a rollout for `prompt_id` was successful."""
        with self._lock:
            n, s = self._task_stats.get(prompt_id, (0, 0))
            self._task_stats[prompt_id] = (n + 1, s + (1 if success else 0))


    def maybe_requeue_prompt(self, prompt: Prompt, step: int) -> None:
        """
        Requeue a hard prompt for one additional pass during the bounded
        streaming window used in later curriculum stages.
        """
        factor = self._upsample_factor(prompt, step)
        if factor <= 1.0:
            return
        with self._lock:
            copies = max(1, int(round(factor)) - 1)
            for _ in range(copies):
                self._replay_queue.append(prompt)


    def _current_mix(self, step: int) -> dict[str, float]:
        """Return source -> sampling weight for the current curriculum stage."""
        if step < self.cfg.curriculum.stage1_end_step:
            mix = self.cfg.curriculum.stage1_mix
            return {
                SOURCE_CODECONTESTS: mix.get("codecontests", 0.0),
                SOURCE_SWE_V2: mix.get("swe_v2", 0.0),
                SOURCE_SWE_PRS: mix.get("swe_prs", 0.0),
            }
        return {
            SOURCE_SWE_V2: self.cfg.data.mix_swe_v2_weight,
            SOURCE_SWE_PRS: self.cfg.data.mix_swe_prs_weight,
            SOURCE_CODECONTESTS: self.cfg.data.mix_codecontests_weight,
        }


    def _next_from_source(self, source: str) -> dict | None:
        """Pull the next raw row from a source iterator, returning None on exhaustion."""
        try:
            return next(self._iters[source])
        except StopIteration:
            return None


    def _build_prompt(self, source: str, ex: dict) -> Prompt | None:
        """Convert a raw dataset row into a `Prompt`, returning None to skip."""
        if source in (SOURCE_SWE_V2, SOURCE_SWE_PRS):
            instance_id = ex.get("instance_id", "")
            if not instance_id:
                return None
            if source == SOURCE_SWE_V2 and instance_id in self.streams.held_out_swe_ids:
                return None
            return Prompt(
                prompt_id=f"{source}:{instance_id}",
                source=source,
                task_text=_format_swe_prompt(ex),
                payload=ex,
                difficulty=swe_difficulty(ex),
            )
        if source == SOURCE_CODECONTESTS:
            name = ex.get("name", "")
            return Prompt(
                prompt_id=f"{source}:{name}",
                source=source,
                task_text=_format_codecontests_prompt(ex),
                payload=ex,
                difficulty=codecontests_difficulty(ex),
            )
        return None


    def _stage1_filter(self, prompt: Prompt) -> bool:
        """During stage 1 keep only easy problems."""
        return prompt.difficulty <= 0.4


    def _upsample_factor(self, prompt: Prompt, step: int) -> float:
        """Return the sampling weight given historical success rate."""
        if step < self.cfg.curriculum.stage1_end_step:
            return 1.0
        n, s = self._task_stats.get(prompt.prompt_id, (0, 0))
        if n == 0:
            return 1.0
        success_rate = s / n
        if success_rate < self.cfg.curriculum.hard_task_success_threshold:
            return self.cfg.data.hard_task_upsample_factor
        return 1.0


    def next_prompt(self, step: int) -> Prompt:
        """
        Draw the next prompt under the current curriculum mix. Blocks (loops) if
        a chosen source returns an unusable row, until a valid prompt is found.

        :param step: Current optimizer step (drives curriculum and upsampling).
        :returns: Validated `Prompt`.
        """
        while True:
            with self._lock:
                if self._replay_queue:
                    prompt = self._replay_queue.popleft()
                    self.global_index += 1
                    return prompt
            mix = self._current_mix(step)
            sources = list(mix.keys())
            weights = [mix[s] for s in sources]
            source = self._rng.choices(sources, weights=weights, k=1)[0]
            ex = self._next_from_source(source)
            if ex is None:
                # Fallback: try other sources if one is exhausted
                continue
            prompt = self._build_prompt(source, ex)
            if prompt is None:
                continue
            if step < self.cfg.curriculum.stage1_end_step and not self._stage1_filter(prompt):
                continue
            prompt.upsample_weight = self._upsample_factor(prompt, step)
            with self._lock:
                self.global_index += 1
            return prompt


    def next_batch(self, step: int, count: int) -> list[Prompt]:
        """Convenience: pull `count` prompts in sequence."""
        return [self.next_prompt(step) for _ in range(count)]
