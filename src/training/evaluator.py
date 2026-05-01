import time

from src.config.config import PipelineConfig
from src.data.datasets import EvalSubsets
from src.data.prompt_queue import (
    Prompt,
    SOURCE_CODECONTESTS,
    SOURCE_SWE_V2,
    _format_codecontests_prompt,
    _format_swe_prompt,
)
from src.environments.codecontests_env import CodeContestsEnvironment
from src.environments.swe_env import SWEEnvironment
from src.helpers import timed
from src.inference.rollout import RolloutGenerator
from src.inference.vllm_server import VLLMServer
from src.training.rewards import (
    compute_codecontests_correctness,
    compute_swe_correctness,
)
from src.training.wandb_logger import WandbLogger


class Evaluator:
    """
    Run intermediate evaluation against fixed held-out subsets and log aggregate metrics 
    to W&B. Evaluation uses greedy decoding (`eval_temperature`) for reproducibility.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        server: VLLMServer,
        tokenizer,
        eval_subsets: EvalSubsets,
        logger: WandbLogger,
    ) -> None:
        self.cfg = cfg
        self.server = server
        self.tokenizer = tokenizer
        self.subsets = eval_subsets
        self.logger = logger
        self.generator = RolloutGenerator(cfg, server, tokenizer)


    def _eval_codecontests(self) -> dict:
        """Run greedy rollouts on the held-out CodeContests-O subset."""
        passed_full = 0
        avg_score_acc = 0.0
        total = len(self.subsets.codecontests_valid)
        turns_acc = 0
        tokens_acc = 0
        for ex in self.subsets.codecontests_valid:
            prompt = Prompt(
                prompt_id=f"eval-cc:{ex.get('name','')}",
                source=SOURCE_CODECONTESTS,
                task_text=_format_codecontests_prompt(ex),
                payload=ex,
            )
            env = CodeContestsEnvironment(ex, test_timeout=self.cfg.reward.test_timeout_seconds)
            env.setup()
            try:
                rollout = self.generator._run_single_rollout(prompt, env, policy_version=-1)
                result = env.run_tests()
            finally:
                env.teardown()
            score = compute_codecontests_correctness(result["tests_passed"], result["tests_total"])
            avg_score_acc += score
            if score >= 1.0:
                passed_full += 1
            turns_acc += rollout.n_turns
            tokens_acc += sum(len(s.token_ids) for s in rollout.segments)
        n = max(1, total)
        return {
            "eval/codecontests_pass_rate": passed_full / n,
            "eval/codecontests_avg_score": avg_score_acc / n,
            "eval/mean_eval_turns_codecontests": turns_acc / n,
            "eval/mean_eval_tokens_codecontests": tokens_acc / n,
        }


    def _eval_swe(self) -> dict:
        """Run greedy rollouts on the held-out SWE-rebench-V2 subset."""
        resolved = 0
        partial = 0
        turns_acc = 0
        tokens_acc = 0
        total = len(self.subsets.swe_held_out)
        for ex in self.subsets.swe_held_out:
            prompt = Prompt(
                prompt_id=f"eval-swe:{ex.get('instance_id','')}",
                source=SOURCE_SWE_V2,
                task_text=_format_swe_prompt(ex),
                payload=ex,
            )
            env = SWEEnvironment(ex, source=SOURCE_SWE_V2,
                                 test_timeout=self.cfg.reward.test_timeout_seconds)
            env.setup()
            try:
                rollout = self.generator._run_single_rollout(prompt, env, policy_version=-1)
                result = env.run_tests()
            finally:
                env.teardown()
            reward = compute_swe_correctness(
                result["fail_to_pass_passed"], result["fail_to_pass_total"],
                result["pass_to_pass_passed"], result["pass_to_pass_total"],
            )
            if reward >= 1.0:
                resolved += 1
            elif reward > 0.0:
                partial += 1
            turns_acc += rollout.n_turns
            tokens_acc += sum(len(s.token_ids) for s in rollout.segments)
        n = max(1, total)
        return {
            "eval/swe_resolve_rate": resolved / n,
            "eval/swe_partial_rate": partial / n,
            "eval/mean_eval_turns_swe": turns_acc / n,
            "eval/mean_eval_tokens_swe": tokens_acc / n,
        }


    def run(self, training_step: int) -> dict:
        """Execute both evaluation subsets and log all results at `training_step`."""
        # Force greedy decoding for reproducibility.
        original_temp = self.cfg.sampling.temperature
        self.cfg.sampling.temperature = self.cfg.sampling.eval_temperature
        timing: dict = {}
        try:
            with timed(timing, "eval_total_time_sec"):
                with timed(timing, "eval_codecontests_time_sec"):
                    cc = self._eval_codecontests()
                with timed(timing, "eval_swe_time_sec"):
                    swe = self._eval_swe()
        finally:
            self.cfg.sampling.temperature = original_temp

        merged = {**cc, **swe}
        merged["eval/mean_eval_turns"] = (
            cc["eval/mean_eval_turns_codecontests"] + swe["eval/mean_eval_turns_swe"]
        ) / 2
        merged["eval/mean_eval_tokens"] = (
            cc["eval/mean_eval_tokens_codecontests"] + swe["eval/mean_eval_tokens_swe"]
        ) / 2
        for k, v in timing.items():
            merged[f"timing/{k}"] = v
        self.logger.log(merged, step=training_step)
        return merged
