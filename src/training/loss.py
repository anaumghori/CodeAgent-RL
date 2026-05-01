import torch


def compute_per_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute log π(a_t | s_t) for every position. Logits at position t predict
    the token at position t+1, so we shift logits left by one and gather the
    log-prob of the actual `input_ids[t+1]` token.

    :param logits: (B, T, V)
    :param input_ids: (B, T)
    :returns: (B, T-1)
    """
    shifted_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def compute_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_coeff: float = 0.02,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Dr. GRPO loss for a single microbatch.

    Implements the clipped surrogate objective with the k₁ KL estimator
    (-log r) and unbiased token-level normalisation: the loss is divided by
    the total number of model-generated tokens in the batch (no per-response
    length normalisation, no group std normalisation).

    :param logits: (B, T, V) model output.
    :param input_ids: (B, T) token IDs.
    :param old_logprobs: (B, T-1) log-probs from the sampling policy.
    :param advantages: (B,) per-rollout advantages.
    :param loss_mask: (B, T-1) 1 for model-generated tokens, 0 elsewhere.
    :param kl_coeff: KL penalty coefficient β.
    :param clip_epsilon: PPO clip threshold ε.
    :returns: (scalar loss tensor, diagnostics dict)
    """
    new_logprobs = compute_per_token_logprobs(logits, input_ids)

    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)  # (B, 1)
    pg_loss1 = ratio * adv
    pg_loss2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    pg_loss = -torch.min(pg_loss1, pg_loss2)

    # k₁ KL estimator: KL ≈ old_logprobs - new_logprobs (= -log r)
    kl_penalty = kl_coeff * (old_logprobs - new_logprobs)

    per_token = pg_loss + kl_penalty
    total_tokens = loss_mask.sum().clamp(min=1.0)
    loss = (per_token * loss_mask).sum() / total_tokens

    with torch.no_grad():
        masked_pg = (pg_loss * loss_mask).sum() / total_tokens
        masked_kl = (kl_penalty * loss_mask).sum() / total_tokens
        clipped = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
        clip_fraction = (clipped * loss_mask).sum() / total_tokens
        mean_ratio = (ratio * loss_mask).sum() / total_tokens
        mean_kl = ((old_logprobs - new_logprobs) * loss_mask).sum() / total_tokens

    diagnostics = {
        "pg_loss": float(masked_pg.detach().item()),
        "kl_loss": float(masked_kl.detach().item()),
        "clip_fraction": float(clip_fraction.detach().item()),
        "mean_ratio": float(mean_ratio.detach().item()),
        "mean_kl": float(mean_kl.detach().item()),
    }
    return loss, diagnostics
