import json
import re
import uuid
from dataclasses import dataclass, field

from src.config.config import PipelineConfig
from src.data.prompt_queue import Prompt, SOURCE_CODECONTESTS
from src.environments.base import Environment
from src.environments.tools import TOOL_DEFINITIONS, normalize_tool_arguments
from src.helpers import log_event, log_json
from src.inference.vllm_server import VLLMServer


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINKING_SYSTEM_PROMPT = (
    "Before every visible response, reason privately inside <think>...</think> "
    "and keep that reasoning out of the visible answer except through the final result."
)
SUMMARY_PROMPT = (
    "Produce a STRUCTURED SUMMARY of work so far. Cover: current understanding "
    "of the task, files examined and modified, tests run and outcomes, hypotheses "
    "explored, and remaining work. Keep it concise but information-dense."
)


@dataclass
class RolloutSegment:
    """
    One contiguous training segment of a rollout.

    `token_ids` is the entire token sequence (prompt + responses + tool
    outputs) for the segment. `loss_mask[i]` is 1 iff token `i` was generated
    by the model. `logprobs[i]` is the per-token log-probability recorded by
    vLLM for model-generated positions (zeros elsewhere; trainer never reads
    those positions because the loss mask is zero).
    """
    token_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    n_think_tokens: int = 0
    n_tool_call_tokens: int = 0
    n_tool_output_tokens: int = 0
    n_final_tokens: int = 0


@dataclass
class Rollout:
    """A complete multi-turn rollout (potentially with multiple segments)."""
    rollout_id: str
    prompt_id: str
    source: str
    segments: list[RolloutSegment] = field(default_factory=list)
    n_turns: int = 0
    n_tool_calls: int = 0
    final_response: str = ""
    test_result: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    reward: float = 0.0
    advantage: float = 0.0
    policy_version: int = 0


@dataclass
class RolloutGroup:
    """A group of `group_size` rollouts sampled from the same prompt."""
    prompt: Prompt
    rollouts: list[Rollout]
    policy_version: int


class RolloutGenerator:
    """
    Multi-turn agent rollout loop. For each prompt the generator samples
    `group_size` rollouts; each rollout iteratively prompts the model, parses
    Hermes `<tool_call>` blocks, dispatches them through the environment and
    appends `<tool_response>` messages until the model emits a final answer
    with no tool calls or hits the segment / turn limits.

    Self-summarization is triggered when the visible-token count exceeds the
    soft threshold; on trigger the current segment is closed, the model
    produces a structured summary, and a new segment begins with the system
    prompt + task + summary + the most recent turns.
    """

    def __init__(self, cfg: PipelineConfig, server: VLLMServer, tokenizer) -> None:
        self.cfg = cfg
        self.server = server
        self.tokenizer = tokenizer
        self.max_turns = 24


    def _sampling_params(self, prompt: Prompt):
        from vllm import SamplingParams
        max_tokens = self.cfg.sequence.max_generation_tokens
        if prompt.source == SOURCE_CODECONTESTS:
            max_tokens = self.cfg.sequence.codecontests_max_generation_tokens
        return SamplingParams(
            temperature=self.cfg.sampling.temperature,
            top_p=self.cfg.sampling.top_p,
            top_k=self.cfg.sampling.top_k,
            max_tokens=max_tokens,
            logprobs=1,
        )


    def _build_initial_messages(self, prompt: Prompt) -> list[dict]:
        """ChatML messages with explicit Hermes thinking instructions."""
        system_prompt = (
            f"{THINKING_SYSTEM_PROMPT}\n\n"
            "You are an expert software engineer. Use the provided tools to inspect code, "
            "run commands, write fixes, and verify your work with the test suite."
        )
        if prompt.source == SOURCE_CODECONTESTS:
            system_prompt = (
                f"{THINKING_SYSTEM_PROMPT}\n\n"
                "You are solving a competitive programming task inside a strict tool-using harness. "
                "You must use tools to complete the task. The required order is: "
                "(1) use `write_file` to create `solution.py`, "
                "(2) use `run_command` to execute and verify `solution.py`, "
                "(3) only then provide a brief final response. "
                "Do not treat a prose answer or a markdown code block as a valid solution submission. "
                "If `solution.py` has not been written and verified through tools, the task is incomplete."
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.task_text},
        ]


    def _render(self, messages: list[dict]) -> tuple[str, list[int]]:
        """Apply the chat template and return (text, token_ids) including the generation prompt."""
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tools=TOOL_DEFINITIONS, tokenize=False,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text, ids


    def _generate(self, prompt: Prompt, messages: list[dict]) -> tuple[str, list[int], list[float]]:
        """Run one generation call; return (text, generated_token_ids, per-token logprobs)."""
        sp = self._sampling_params(prompt)
        outputs = self.server.llm.chat(messages=messages, sampling_params=sp,
                                       tools=TOOL_DEFINITIONS)
        out0 = outputs[0].outputs[0]
        text = out0.text
        gen_ids = list(out0.token_ids)
        logprobs: list[float] = []
        if out0.logprobs:
            for entry in out0.logprobs:
                # `entry` maps token_id -> Logprob(logprob, ...). Pull the
                # log-prob for the actually sampled token (gen_ids[i]).
                if not entry:
                    logprobs.append(0.0)
                    continue
                tid = gen_ids[len(logprobs)]
                lp = entry.get(tid)
                logprobs.append(float(lp.logprob) if lp is not None else 0.0)
        return text, gen_ids, logprobs


    def _parse_tool_calls(self, text: str) -> list[dict]:
        """Extract `<tool_call>{...}</tool_call>` JSON payloads from generated text."""
        calls: list[dict] = []
        for match in TOOL_CALL_RE.finditer(text):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "name" in payload:
                try:
                    payload["arguments"] = normalize_tool_arguments(payload.get("arguments", {}))
                except (TypeError, json.JSONDecodeError):
                    continue
                calls.append(payload)
        return calls


    def _strip_thinking(self, text: str) -> tuple[str, str]:
        """Split `<think>...</think>` from the visible response. Returns (think, visible)."""
        think = ""
        if THINK_OPEN in text and THINK_CLOSE in text:
            start = text.find(THINK_OPEN) + len(THINK_OPEN)
            end = text.find(THINK_CLOSE)
            think = text[start:end]
            visible = text[end + len(THINK_CLOSE):]
            return think, visible
        return think, text


    def _summarize(self, prompt: Prompt, messages: list[dict]) -> tuple[str, list[int], list[float]]:
        msgs = list(messages) + [{
            "role": "user",
            "content": SUMMARY_PROMPT,
        }]
        return self._generate(prompt, msgs)


    def _segment_visible_tokens(self, segment: RolloutSegment) -> int:
        return segment.n_tool_call_tokens + segment.n_tool_output_tokens + segment.n_final_tokens


    def _should_trigger_summary(self, segment: RolloutSegment) -> bool:
        """
        Decide whether the agent should compress its current segment via
        self-summarization.

        Soft trigger: visible tokens (tool calls + tool outputs + final messages)
        exceed `summary_soft_trigger_tokens`.
        Hard trigger: total segment length exceeds `summary_hard_trigger_tokens`.
        """
        visible = self._segment_visible_tokens(segment)
        if visible >= self.cfg.sequence.summary_soft_trigger_tokens:
            return True
        if len(segment.token_ids) >= self.cfg.sequence.summary_hard_trigger_tokens:
            return True
        return False


    def _run_single_rollout(self, prompt: Prompt, env: Environment, policy_version: int) -> Rollout:
        """Generate a single rollout (possibly multiple segments)."""
        rollout = Rollout(
            rollout_id=str(uuid.uuid4()),
            prompt_id=prompt.prompt_id,
            source=prompt.source,
            policy_version=policy_version,
        )
        log_event(
            "rollout",
            f"Starting rollout {rollout.rollout_id} for prompt {prompt.prompt_id} "
            f"(source={prompt.source}, policy_version={policy_version}).",
        )

        messages = self._build_initial_messages(prompt)
        _, prompt_ids = self._render(messages)
        segment = RolloutSegment(
            token_ids=list(prompt_ids),
            loss_mask=[0] * len(prompt_ids),
            logprobs=[],
        )

        for turn in range(self.max_turns):
            text, gen_ids, logprobs = self._generate(prompt, messages)
            think, visible = self._strip_thinking(text)
            tool_calls = self._parse_tool_calls(visible)

            # Account tokens by region for the length-penalty effort variable.
            think_tok_count = len(self.tokenizer.encode(think, add_special_tokens=False)) if think else 0
            tool_call_tok_count = sum(
                len(self.tokenizer.encode(json.dumps(tc), add_special_tokens=False))
                for tc in tool_calls
            )
            final_tok_count = max(0, len(gen_ids) - think_tok_count - tool_call_tok_count)
            segment.n_think_tokens += think_tok_count
            segment.n_tool_call_tokens += tool_call_tok_count
            segment.n_final_tokens += 0 if tool_calls else final_tok_count

            segment.token_ids.extend(gen_ids)
            segment.loss_mask.extend([1] * len(gen_ids))
            segment.logprobs.extend(logprobs)

            assistant_msg = {"role": "assistant", "content": visible}
            messages.append(assistant_msg)
            rollout.n_turns += 1

            if not tool_calls:
                rollout.final_response = visible.strip()
                if hasattr(env, "record_final_response"):
                    env.record_final_response(rollout.final_response)
                break

            for tc in tool_calls:
                rollout.n_tool_calls += 1
                result = env.dispatch_tool(tc["name"], tc.get("arguments", {}))
                if not result.success:
                    preview = " ".join((result.output or "").split())[:240]
                    log_event(
                        "rollout",
                        f"Tool {tc['name']} failed during rollout {rollout.rollout_id} "
                        f"(turn={turn + 1}): {preview}",
                    )
                tool_msg = {"role": "tool", "name": tc["name"], "content": result.output}
                messages.append(tool_msg)
                # Re-render to get the tokenized representation of the tool response only.
                tool_text = self.tokenizer.apply_chat_template(
                    [tool_msg], add_generation_prompt=False, tokenize=False,
                )
                tool_ids = self.tokenizer.encode(tool_text, add_special_tokens=False)
                segment.token_ids.extend(tool_ids)
                segment.loss_mask.extend([0] * len(tool_ids))
                segment.logprobs.extend([0.0] * len(tool_ids))
                segment.n_tool_output_tokens += len(tool_ids)

            if self._should_trigger_summary(segment):
                # Close current segment, generate summary, start a new segment.
                log_event(
                    "rollout",
                    f"Triggering self-summary for rollout {rollout.rollout_id} "
                    f"after turn {turn + 1} (segment={len(rollout.segments) + 1}).",
                )
                summary, summary_ids, summary_logprobs = self._summarize(prompt, messages)
                segment.token_ids.extend(summary_ids)
                segment.loss_mask.extend([1] * len(summary_ids))
                segment.logprobs.extend(summary_logprobs)
                rollout.segments.append(segment)

                tail = messages[-2 * self.cfg.sequence.summary_context_turns:]
                messages = self._build_initial_messages(prompt) + [
                    {"role": "assistant", "content": f"Prior work summary:\n{summary}"},
                    *tail,
                ]
                _, prompt_ids = self._render(messages)
                segment = RolloutSegment(
                    token_ids=list(prompt_ids),
                    loss_mask=[0] * len(prompt_ids),
                    logprobs=[],
                )

        rollout.segments.append(segment)
        log_json(
            "rollout",
            "rollout_complete",
            {
                "rollout_id": rollout.rollout_id,
                "prompt_id": rollout.prompt_id,
                "source": rollout.source,
                "turns": rollout.n_turns,
                "tool_calls": rollout.n_tool_calls,
                "segments": len(rollout.segments),
                "total_tokens": sum(len(s.token_ids) for s in rollout.segments),
            },
        )
        return rollout


    def generate_group(self, prompt: Prompt, env_factory, policy_version: int) -> RolloutGroup:
        """
        Generate `group_size` rollouts for `prompt`. `env_factory` returns a
        fresh `Environment` for each rollout (one rollout = one environment
        instance, since environments hold mutable state).
        """
        rollouts: list[Rollout] = []
        for idx in range(self.cfg.grpo.group_size):
            env = env_factory()
            try:
                already_prepared = bool(getattr(env, "_composer_prepared", False))
                if not already_prepared:
                    env.setup()
                    setattr(env, "_composer_prepared", True)
                rollout = self._run_single_rollout(prompt, env, policy_version)
                rollout.test_result = env.run_tests()
                rollout.metadata = env.collect_metadata()
            finally:
                env.teardown()
            rollouts.append(rollout)
        return RolloutGroup(prompt=prompt, rollouts=rollouts, policy_version=policy_version)
