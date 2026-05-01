import ast
import re

from src.config.config import PipelineConfig


_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX)\b")


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """
    Dr. GRPO group-relative advantage: subtract the group mean from each
    rollout's reward. No standard-deviation normalization (this is the key
    Dr. GRPO modification — see GRPO_plan.md §3.2).

    :param rewards: Scalar rewards for the G rollouts in a single group.
    :returns: List of advantages, same length as `rewards`.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def compute_swe_correctness(
    fail_to_pass_passed: int,
    fail_to_pass_total: int,
    pass_to_pass_passed: int,
    pass_to_pass_total: int,
) -> float:
    """
    Composer 2 / SWE-rebench-V2 correctness reward.

    +1.0 if all FAIL_TO_PASS targets pass and no PASS_TO_PASS regressions occur.
    -0.2 if any PASS_TO_PASS test regresses.
    0.5 * (newly_passing / target) if partial progress without regressions.
    0.0 otherwise.
    """
    if pass_to_pass_total > 0 and pass_to_pass_passed < pass_to_pass_total:
        return -0.2
    if fail_to_pass_total == 0:
        return 0.0
    if fail_to_pass_passed == fail_to_pass_total:
        return 1.0
    if fail_to_pass_passed > 0:
        return 0.5 * (fail_to_pass_passed / fail_to_pass_total)
    return 0.0


def compute_codecontests_correctness(tests_passed: int, tests_total: int) -> float:
    """Algorithmic-task correctness reward: simple pass-rate over `corner_cases`."""
    if tests_total == 0:
        return 0.0
    return tests_passed / tests_total


def compute_correctness_reward(test_result: dict, source: str, cfg: PipelineConfig) -> float:
    """
    Dispatch to the source-appropriate correctness reward.

    `test_result` schema:
      For SWE: {"fail_to_pass_passed": int, "fail_to_pass_total": int,
                "pass_to_pass_passed": int, "pass_to_pass_total": int}
      For CodeContests: {"tests_passed": int, "tests_total": int}

    :returns: Weighted correctness reward.
    """
    if source.startswith("swe"):
        raw = compute_swe_correctness(
            test_result.get("fail_to_pass_passed", 0),
            test_result.get("fail_to_pass_total", 0),
            test_result.get("pass_to_pass_passed", 0),
            test_result.get("pass_to_pass_total", 0),
        )
    else:
        raw = compute_codecontests_correctness(
            test_result.get("tests_passed", 0),
            test_result.get("tests_total", 0),
        )
    return cfg.reward.correctness_weight * raw


def compute_effort_x(
    think_tokens: int,
    tool_call_tokens: int,
    tool_output_tokens: int,
    final_tokens: int,
    n_tool_calls: int,
    n_turns: int,
) -> float:
    """
    Compute the scalar effort variable `x` consumed by the length penalty.

    x = T_think/512 + T_tool_call/1024 + T_tool_output/4096
        + T_final/512 + 0.25 * N_tool_calls + 0.5 * N_turns
    """
    return (
        think_tokens / 512.0
        + tool_call_tokens / 1024.0
        + tool_output_tokens / 4096.0
        + final_tokens / 512.0
        + 0.25 * n_tool_calls
        + 0.5 * n_turns
    )


def compute_length_penalty(x: float, cfg: PipelineConfig) -> float:
    """
    Concave-down, monotonically increasing length penalty C_{k,q}(x):

        C(x) = ((1 + k * x) ** (1 - q) - 1) / (k * (1 - q))

    The penalty is scaled by `lambda` and returned as a *positive* magnitude
    that should be subtracted from the total reward.

    :param x: Effort variable.
    :param cfg: Pipeline config (uses `length_penalty_k`, `_q`, `_lambda`).
    :returns: Non-negative penalty magnitude.
    """
    k = cfg.reward.length_penalty_k
    q = cfg.reward.length_penalty_q
    lam = cfg.reward.length_penalty_lambda
    base = (1.0 + k * x) ** (1.0 - q)
    penalty = (base - 1.0) / (k * (1.0 - q))
    return lam * penalty


def _extract_python_blocks(text: str) -> list[str]:
    """Extract fenced ```python ... ``` blocks from a model response."""
    blocks: list[str] = []
    pattern = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text or ""):
        blocks.append(match.group(1))
    return blocks


def syntax_validity_reward(written_files: dict[str, str]) -> float:
    """
    +1 if every Python file written by the agent parses; 0 otherwise.
    Non-Python files are ignored (treated as valid).
    """
    if not written_files:
        return 0.0
    for path, content in written_files.items():
        if not path.endswith(".py"):
            continue
        try:
            ast.parse(content)
        except SyntaxError:
            return 0.0
    return 1.0


def no_todo_reward(written_files: dict[str, str]) -> float:
    """-1 if the agent wrote any TODO/FIXME/XXX markers; 0 otherwise."""
    for content in written_files.values():
        if _TODO_RE.search(content or ""):
            return -1.0
    return 0.0


def minimal_diff_reward(modified_files: set[str], task_relevant_files: set[str]) -> float:
    """
    Penalize modifications to files that are not part of the task's relevant set.
    Returns -frac_unrelated where frac_unrelated is the share of touched files
    that are unrelated to the task. 0 means perfectly minimal.
    """
    if not modified_files:
        return 0.0
    unrelated = modified_files - task_relevant_files
    return -len(unrelated) / max(1, len(modified_files))


def tool_hygiene_reward(tool_call_history: list[dict]) -> float:
    """
    Penalize duplicate tool calls (same tool, same arguments) appearing more
    than once in the conversation. Returns -frac_redundant.
    """
    if not tool_call_history:
        return 0.0
    seen: set[tuple] = set()
    redundant = 0
    for call in tool_call_history:
        key = (call.get("name"), repr(call.get("arguments")))
        if key in seen:
            redundant += 1
        else:
            seen.add(key)
    return -redundant / len(tool_call_history)


def test_discipline_reward(ran_tests_before_finish: bool) -> float:
    """+1 if the agent executed the test suite before finishing; 0 otherwise."""
    return 1.0 if ran_tests_before_finish else 0.0


def compute_auxiliary_rewards(rollout_meta: dict, cfg: PipelineConfig) -> dict[str, float]:
    """
    Compute every auxiliary reward and return both the raw values and the
    weighted sum. `rollout_meta` schema:
        {
          "written_files": {path: content, ...},
          "modified_files": set[str],
          "task_relevant_files": set[str],
          "tool_call_history": [{"name": ..., "arguments": ...}, ...],
          "ran_tests_before_finish": bool,
        }

    :returns: dict with per-component values and `"total_weighted"` key.
    """
    written = rollout_meta.get("written_files", {})
    syntax = syntax_validity_reward(written)
    no_todo = no_todo_reward(written)
    minimal = minimal_diff_reward(
        rollout_meta.get("modified_files", set()),
        rollout_meta.get("task_relevant_files", set()),
    )
    hygiene = tool_hygiene_reward(rollout_meta.get("tool_call_history", []))
    discipline = test_discipline_reward(rollout_meta.get("ran_tests_before_finish", False))

    total = (
        cfg.reward.aux_syntax_weight * syntax
        + cfg.reward.aux_no_todo_weight * no_todo
        + cfg.reward.aux_minimal_diff_weight * minimal
        + cfg.reward.aux_tool_hygiene_weight * hygiene
        + cfg.reward.aux_test_discipline_weight * discipline
    )
    return {
        "syntax": syntax,
        "no_todo": no_todo,
        "minimal_diff": minimal,
        "tool_hygiene": hygiene,
        "test_discipline": discipline,
        "total_weighted": total,
    }
