from dataclasses import dataclass
from itertools import islice
from typing import Any
import random
from datasets import load_dataset
from huggingface_hub import login as hf_login

from src.config.config import PipelineConfig
from src.helpers import log_event


@dataclass
class TrainingStreams:
    swe_v2: Any
    codecontests: Any
    held_out_swe_ids: set[str]  # set of held-out instance IDs that must be skipped during training.


@dataclass
class EvalSubsets:
    codecontests_valid: list[dict]
    swe_held_out: list[dict]


def _hf_authenticate(cfg: PipelineConfig) -> None:
    if cfg.credentials.hf_token:
        log_event("datasets", "Authenticating with Hugging Face Hub using HF_TOKEN.")
        hf_login(token=cfg.credentials.hf_token, add_to_git_credential=False)
    else:
        log_event("datasets", "HF_TOKEN not set; continuing without explicit Hugging Face login.")


def load_eval_subsets(cfg: PipelineConfig) -> EvalSubsets:
    """
    Load the two intermediate-evaluation subsets:
    - 50 problems from CodeContests-O `valid`.
    - 50 instances reserved from SWE-rebench-V2 `train`, stratified by language.

    :param cfg: Pipeline config with dataset IDs and subset sizes.
    :returns: `EvalSubsets` dataclass with two fixed lists.
    """
    _hf_authenticate(cfg)
    log_event("datasets", "Loading evaluation subsets from streaming datasets.")
    rng = random.Random(cfg.data.eval_seed)

    cc_valid_stream = load_dataset(
        cfg.data.codecontests_o_id, split="valid", streaming=True,
    )
    cc_valid_stream = cc_valid_stream.shuffle(
        seed=cfg.data.eval_seed,
        buffer_size=max(cfg.eval.eval_codecontests_count * 2, 64),
    )
    cc_subset = list(islice(cc_valid_stream, cfg.eval.eval_codecontests_count))

    # Stream the SWE train split and stratify by language for the eval reserve.
    swe_stream = load_dataset(cfg.data.swe_rebench_v2_id, split="train", streaming=True)
    by_lang: dict[str, list[dict]] = {}
    cap_per_lang = max(2, cfg.eval.eval_swe_count // 6)
    target = cfg.eval.eval_swe_count
    collected = 0
    seen_ids: set[str] = set()
    for ex in swe_stream:
        lang = ex.get("language", "unknown")
        bucket = by_lang.setdefault(lang, [])
        if len(bucket) >= cap_per_lang:
            continue
        if ex["instance_id"] in seen_ids:
            continue
        bucket.append(ex)
        seen_ids.add(ex["instance_id"])
        collected += 1
        if collected >= target * 4:
            break

    flat: list[dict] = []
    for items in by_lang.values():
        flat.extend(items)
    rng.shuffle(flat)
    swe_subset = flat[: cfg.eval.eval_swe_count]
    log_event(
        "datasets",
        f"Prepared eval subsets: codecontests={len(cc_subset)}, swe={len(swe_subset)}.",
    )

    return EvalSubsets(codecontests_valid=cc_subset, swe_held_out=swe_subset)


def load_training_streams(cfg: PipelineConfig, eval_subsets: EvalSubsets) -> TrainingStreams:
    """
    :param cfg: Pipeline config.
    :param eval_subsets: Loaded eval subsets, used to derive the exclusion set.
    """
    _hf_authenticate(cfg)
    held_out_ids = {ex["instance_id"] for ex in eval_subsets.swe_held_out}
    log_event(
        "datasets",
        f"Loading training streams with {len(held_out_ids)} held-out SWE instances excluded.",
    )

    swe_v2 = load_dataset(cfg.data.swe_rebench_v2_id, split="train", streaming=True)
    codecontests = load_dataset(cfg.data.codecontests_o_id, split="train", streaming=True)
    log_event("datasets", "Training streams are ready.")

    return TrainingStreams(
        swe_v2=swe_v2,
        codecontests=codecontests,
        held_out_swe_ids=held_out_ids,
    )

def codecontests_difficulty(example: dict) -> float:
    """
    Heuristic difficulty score for a CodeContests-O problem in [0, 1].
    Larger descriptions are treated as harder.
    """
    desc = example.get("description", "") or ""
    return min(1.0, len(desc) / 9000.0)

def swe_difficulty(example: dict) -> float:
    """
    Heuristic difficulty score for a SWE-rebench-V2 instance in [0, 1].
    Uses modified files / lines count from the `meta` field when available.
    """
    meta = example.get("meta") or {}
    files = meta.get("modified_files", 1) or 1
    lines = meta.get("modified_lines", 10) or 10
    score = (files / 10.0) * 0.5 + (lines / 500.0) * 0.5
    return min(1.0, score)
