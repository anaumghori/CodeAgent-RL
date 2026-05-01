from dataclasses import dataclass
from collections import defaultdict
import torch


@dataclass
class TrainingSequence:
    """
    A single training sequence assembled from one rollout segment.

    `input_ids` is the full token sequence (prompt + model output + tool responses).
    `loss_mask` is 1 for model-generated tokens (those that should produce gradient)
    and 0 elsewhere. `old_logprobs` are the per-token log-probs recorded by vLLM
    at sampling time, aligned to positions 1..T (predicting `input_ids[1:]`).
    `advantage` is the per-rollout scalar computed by `rewards/advantage.py`.
    """
    input_ids: list[int]
    loss_mask: list[int]
    old_logprobs: list[float]
    advantage: float
    policy_version: int
    rollout_id: str = ""


@dataclass
class PackedMicrobatch:
    """A padded microbatch ready for the forward pass."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    rank: int


def _pad_to(values: list, length: int, pad_value) -> list:
    return values + [pad_value] * (length - len(values))


def pack_sequences(
    sequences: list[TrainingSequence],
    num_ranks: int,
    max_seq_length: int,
) -> list[list[PackedMicrobatch]]:
    """
    Distribute training sequences across `num_ranks` data-parallel ranks
    using greedy bin-packing on quadratic attention cost, then pad each
    per-rank microbatch (one sequence per microbatch — `micro_batch_size=1`).

    :param sequences: All training sequences for a single optimizer step.
    :param num_ranks: Number of training (DP) ranks.
    :param max_seq_length: Hard cap; longer sequences are truncated.
    :returns: For each rank, an ordered list of `PackedMicrobatch` items.
    """
    truncated: list[TrainingSequence] = []
    for s in sequences:
        if len(s.input_ids) > max_seq_length:
            cut = max_seq_length
            truncated.append(TrainingSequence(
                input_ids=s.input_ids[:cut],
                loss_mask=s.loss_mask[:cut],
                old_logprobs=s.old_logprobs[: max(0, cut - 1)],
                advantage=s.advantage,
                policy_version=s.policy_version,
                rollout_id=s.rollout_id,
            ))
        else:
            truncated.append(s)

    # Keep all segments from the same rollout on the same rank so each GPU
    # trains on different samples rather than splitting one rollout across ranks.
    rollouts: dict[str, list[TrainingSequence]] = defaultdict(list)
    for sequence in truncated:
        rollouts[sequence.rollout_id].append(sequence)

    rollout_groups = sorted(
        rollouts.values(),
        key=lambda group: sum(len(sequence.input_ids) ** 2 for sequence in group),
        reverse=True,
    )
    rank_buckets: list[list[TrainingSequence]] = [[] for _ in range(num_ranks)]
    rank_costs = [0.0] * num_ranks
    for group in rollout_groups:
        idx = min(range(num_ranks), key=lambda i: rank_costs[i])
        rank_buckets[idx].extend(group)
        rank_costs[idx] += sum(len(sequence.input_ids) ** 2 for sequence in group)

    # Build per-rank microbatch lists. Pad all microbatches in a rank to the
    # rank's longest sequence so the trainer can stack them into a tensor.
    per_rank: list[list[PackedMicrobatch]] = []
    for rank, bucket in enumerate(rank_buckets):
        microbatches: list[PackedMicrobatch] = []
        for s in bucket:
            T = len(s.input_ids)
            attn = [1] * T
            input_ids = torch.tensor([s.input_ids], dtype=torch.long)
            attention = torch.tensor([attn], dtype=torch.long)
            # The loss applies to predictions of positions 1..T-1, length T-1.
            loss_mask = torch.tensor([s.loss_mask[1:T]], dtype=torch.float32)
            old_lp = torch.tensor([_pad_to(s.old_logprobs, T - 1, 0.0)], dtype=torch.float32)
            adv = torch.tensor([s.advantage], dtype=torch.float32)
            microbatches.append(PackedMicrobatch(
                input_ids=input_ids,
                attention_mask=attention,
                loss_mask=loss_mask,
                old_logprobs=old_lp,
                advantages=adv,
                rank=rank,
            ))
        per_rank.append(microbatches)
    return per_rank
