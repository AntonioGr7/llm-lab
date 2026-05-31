"""The GRPO loss and training step.

The math, once and for all:

**GRPO** (Shao et al., 2024 — DeepSeekMath; refined by DeepSeek-R1, 2025).

For each prompt q, sample G completions {o_i} from the current policy.
Compute a verifiable reward r_i per completion, then standardize within
the group:

    A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + eps)

This is the "group-relative" advantage — no value head, no critic network,
just within-prompt z-scoring. The baseline (mean reward) is the same idea
PPO's value head learns to predict, but GRPO replaces "learn this with a
neural net" with "compute it directly from the group". Cheap, stable, and
the dominant variance-reduction technique now.

The token-level objective for completion o_i with advantage A_i:

    L_t = -min( ρ_t · A_i, clip(ρ_t, 1-ε, 1+ε) · A_i )  +  β · KL_t

where:
    ρ_t = exp( log π_θ(o_t|q,o_<t) - log π_θ_old(o_t|q,o_<t) )    importance ratio
    KL_t = π_ref/π_θ - log(π_ref/π_θ) - 1                          k3 KL estimator
         = exp(δ) - δ - 1   where  δ = log π_ref - log π_θ

The loss is averaged over all (i, t) in the rollout's response tokens.

Three things worth pointing out about this objective:

1. **The PPO clip.** For mu_epochs=K=1 (one gradient step per rollout),
   ρ_t = 1 at the gradient step and the clip is a no-op. For K>1 the
   policy has moved by the time the second gradient step starts, the
   ratio diverges from 1, and the clip prevents the update from being
   off-policy by too much. We default K=1 (cheapest, most stable) but
   the math/code is correct for K>1.

2. **The k3 KL estimator** (Schulman 2020). The naive KL estimate
   `mean(log π_θ - log π_ref)` is unbiased but very high variance.
   The k3 estimator `exp(δ) - δ - 1` (where δ = log π_ref - log π_θ)
   is biased low but variance-reduced and ALWAYS NON-NEGATIVE — which
   matters because the KL is part of the loss being minimized; a
   negative KL estimate would push the policy AWAY from the reference,
   the opposite of what we want.

3. **Per-token vs sequence-level mean.** R1 averages the per-token loss
   over the response tokens of each completion, then averages over
   completions. That's `(L · response_mask).sum() / response_mask.sum()`
   — equal-weight per token, not per completion. Long responses dominate;
   that's by design (R1 reports response length GROWS as reasoning
   emerges).

The metrics we report at each step:
    - loss           : the scalar minimized
    - policy_loss    : the clipped-surrogate term, averaged over tokens
    - kl_loss        : β · KL averaged over tokens
    - mean_kl        : KL averaged over tokens (no β)
    - clip_frac      : fraction of tokens where ρ_t got clipped
    - mean_reward    : average total reward in the rollout (logged)
    - mean_accuracy  : average accuracy component (the headline curve)
    - mean_format    : average format component
    - mean_advantage : should center near 0 by construction
    - mean_response_length : tokens generated per completion
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import RLConfig
from rollout import Rollout, per_token_logps


# =============================================================================
# GRPO loss
# =============================================================================

def compute_grpo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rl_cfg: RLConfig,
) -> tuple[torch.Tensor, dict]:
    """Compute the GRPO loss given per-token log-probs.

    Args:
        current_logps:  Float[N, T] — policy log-probs at the gradient step
                        (carries gradient).
        old_logps:      Float[N, T] — policy log-probs at rollout time (detached).
        ref_logps:      Float[N, T] — reference log-probs (detached).
        advantages:     Float[N]    — group-normalized advantage per completion.
        response_mask:  Bool[N, T]  — True at positions we score.
        rl_cfg:         hyperparameters (kl_beta, clip_ratio, ...).

    Returns:
        (loss, metrics) where loss is a scalar and metrics is a dict of
        floats (already detached, ready to log).
    """
    mask_f = response_mask.float()
    n_tokens = mask_f.sum().clamp(min=1.0)

    # ---- 1. Importance ratio -------------------------------------------------
    log_ratio = current_logps - old_logps         # zero out non-response below via mask
    ratio = torch.exp(log_ratio)

    # ---- 2. Clipped surrogate policy loss ------------------------------------
    # Broadcast per-completion advantage across the response-token dimension.
    A = advantages.unsqueeze(1)                   # [N, 1]

    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - rl_cfg.clip_ratio, 1.0 + rl_cfg.clip_ratio) * A
    # We MAXIMIZE the advantage-weighted ratio, i.e. minimize its negation:
    policy_loss_per_token = -torch.min(surr1, surr2)              # [N, T]
    policy_loss = (policy_loss_per_token * mask_f).sum() / n_tokens

    # Clip-frac diagnostic: at each token, did the surrogate get clipped?
    # When A>0, clipping kicks in when ratio > 1+ε; when A<0, when ratio < 1-ε.
    # Reading `(surr1 != surr2) & ratio_outside_clip` precisely captures both.
    clipped = ((ratio > 1.0 + rl_cfg.clip_ratio) | (ratio < 1.0 - rl_cfg.clip_ratio))
    clip_frac = ((clipped.float() * mask_f).sum() / n_tokens).detach()

    # ---- 3. Per-token KL (k3 estimator) --------------------------------------
    # KL_t = exp(δ_t) - δ_t - 1   with   δ_t = log π_ref - log π_θ
    # This is non-negative by construction (Bregman divergence of exp) and
    # variance-reduced vs the naive estimator.
    delta = ref_logps - current_logps
    kl_per_token = torch.exp(delta) - delta - 1.0
    # Clamp at 0 in case of numerical wobble (e.g. delta ~ 0 in BF16).
    kl_per_token = torch.clamp(kl_per_token, min=0.0)
    kl_mean = (kl_per_token * mask_f).sum() / n_tokens
    kl_loss = rl_cfg.kl_beta * kl_mean

    # ---- 4. Total loss -------------------------------------------------------
    loss = policy_loss + kl_loss

    metrics = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "mean_kl": kl_mean.detach(),
        "clip_frac": clip_frac,
        "mean_advantage": (advantages.mean()).detach(),
        "ratio_mean": ((ratio * mask_f).sum() / n_tokens).detach(),
    }
    return loss, metrics


# =============================================================================
# train_step — one optimizer update per rollout
# =============================================================================

def train_step(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    rl_cfg: RLConfig,
    grad_clip: float,
    dtype: str = "bf16",
    device: str = "cuda",
) -> dict:
    """One full optimizer step on a rollout's data.

    Pipeline:
      1. Forward the policy on (input_ids, attention_mask) WITH grad.
      2. Gather per-token logps at the `labels` positions.
      3. Compute GRPO loss against the rollout's cached old/ref logps and
         advantages.
      4. Backward + grad clip + optimizer step.

    For K=1 (the default), this happens once per rollout. For K>1, you'd
    call this K times with the SAME rollout — the ratio diverges from 1
    after the first call and the clip starts mattering. The rollout's
    `old_logps` are CACHED at rollout time and don't get re-fetched, which
    is what makes the K>1 ratio meaningful.

    Returns:
        Dict of metrics ready to log. Includes the loss decomposition, the
        clip fraction, mean KL, plus reward-side stats pulled from the
        rollout (mean reward, accuracy, format, response length).
    """
    optimizer.zero_grad(set_to_none=True)

    input_ids = rollout.input_ids.to(device, non_blocking=True)
    labels = rollout.labels.to(device, non_blocking=True)
    attention_mask = rollout.attention_mask.to(device, non_blocking=True)
    old_logps = rollout.old_logps.to(device, non_blocking=True)
    ref_logps = rollout.ref_logps.to(device, non_blocking=True)
    advantages = rollout.advantages.to(device, non_blocking=True)
    response_mask = rollout.response_mask.to(device, non_blocking=True)

    # Autocast for BF16 compute on the gradient forward (mirrors M17's loop).
    _DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}
    from contextlib import nullcontext
    ctx = (torch.autocast(device_type=device, dtype=_DTYPE[dtype])
           if dtype != "fp32" else nullcontext())
    with ctx:
        policy_out = policy(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = policy_out.logits if hasattr(policy_out, "logits") else policy_out
    current_logps = per_token_logps(policy_logits, labels)

    loss, metrics = compute_grpo_loss(
        current_logps=current_logps,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=response_mask,
        rl_cfg=rl_cfg,
    )

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
    optimizer.step()

    # Reward-side metrics — pulled from the rollout, not the loss.
    n = len(rollout.rewards)
    mean_total = sum(r.total for r in rollout.rewards) / max(n, 1)
    mean_format = sum(r.format for r in rollout.rewards) / max(n, 1)
    mean_accuracy = sum(r.accuracy for r in rollout.rewards) / max(n, 1)
    mean_response_length = rollout.response_lengths.float().mean().item()

    out = {k: float(v.item()) if hasattr(v, "item") else float(v) for k, v in metrics.items()}
    out["grad_norm"] = float(grad_norm)
    out["mean_reward"] = mean_total
    out["mean_format"] = mean_format
    out["mean_accuracy"] = mean_accuracy
    out["mean_response_length"] = mean_response_length
    return out


# =============================================================================
# Offline smoke test (no model needed)
# =============================================================================

if __name__ == "__main__":
    print("--- GRPO loss smoke test (synthetic logps, no model) ---")
    torch.manual_seed(0)

    N, T = 4, 8
    # Construct: K=1 case where current == old → ratio = 1, clip is no-op.
    # Advantages alternate ±1.5 across completions so the loss should be
    # significantly negative-of-policy_loss (the model "ranks" rewarded
    # tokens up by the advantage's sign).
    current = torch.randn(N, T) * 0.1
    old = current.clone()                                        # K=1
    ref = current + 0.1                                          # small drift
    advantages = torch.tensor([1.5, -1.5, 1.5, -1.5])
    response_mask = torch.ones(N, T, dtype=torch.bool)

    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2, mu_epochs=1)
    loss, m = compute_grpo_loss(current, old, ref, advantages, response_mask, rl_cfg)
    print(f"  K=1 (ratio=1):")
    print(f"    loss         = {loss.item():+.4f}")
    print(f"    policy_loss  = {m['policy_loss'].item():+.4f}  (expect ~0 since A balanced)")
    print(f"    kl_loss      = {m['kl_loss'].item():+.4f}  (β·KL — small drift)")
    print(f"    mean_kl      = {m['mean_kl'].item():+.4f}")
    print(f"    clip_frac    = {m['clip_frac'].item():.3f}  (expect 0 at K=1, ratio=1)")
    print(f"    ratio_mean   = {m['ratio_mean'].item():.4f}  (expect 1.0)")

    # Now simulate K>1 step 2: policy has moved, ratio diverges from 1.
    current_moved = current + 0.3                                # policy update bumped logps
    loss2, m2 = compute_grpo_loss(current_moved, old, ref, advantages, response_mask, rl_cfg)
    print(f"\n  K=2 step (policy moved +0.3 in logp space):")
    print(f"    ratio_mean   = {m2['ratio_mean'].item():.4f}  (>1 because logp bumped)")
    print(f"    clip_frac    = {m2['clip_frac'].item():.3f}  (some tokens clipped)")
    print(f"    policy_loss  = {m2['policy_loss'].item():+.4f}")
    print(f"    kl_loss      = {m2['kl_loss'].item():+.4f}  (KL grows — policy further from ref)")

    # Degenerate group (std = 0 → all advantages = 0): loss should be pure KL.
    print(f"\n  Degenerate group (advantages all 0 — std-collapse case):")
    zero_adv = torch.zeros(N)
    loss3, m3 = compute_grpo_loss(current, old, ref, zero_adv, response_mask, rl_cfg)
    print(f"    policy_loss  = {m3['policy_loss'].item():+.4f}  (expect 0 — A=0)")
    print(f"    kl_loss      = {m3['kl_loss'].item():+.4f}  (β·KL still nonzero)")
    print(f"    loss         = {loss3.item():+.4f}  (= kl_loss)")
