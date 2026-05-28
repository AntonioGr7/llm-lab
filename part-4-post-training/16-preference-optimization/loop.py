"""The DPO / IPO loss and training step.

The math, once and for all:

**DPO** (Rafailov et al., 2023). The optimal policy under KL-constrained
reward maximization is `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`. Rearranging,
the implicit reward is `r(x,y) = β · log(π(y|x) / π_ref(y|x)) + Z(x)`.
Plug into the Bradley-Terry preference model `P(y_w > y_l | x) = σ(r_w - r_l)`;
the prompt-dependent normalizer `Z(x)` cancels. Loss is the NLL of the
preferred-over-rejected event:

    L_DPO = -log σ( β · [ (log π(y_w|x) - log π_ref(y_w|x))     <- chosen logratio
                       - (log π(y_l|x) - log π_ref(y_l|x)) ] )  <- rejected logratio

Computing requires four log-probabilities per pair: log π and log π_ref on
both the chosen and rejected responses. The reference is frozen so its two
log-probs have no gradient.

**IPO** (Azar et al., 2024 — "A General Theoretical Paradigm for...").
Same logratio difference, squared-loss link instead of log-sigmoid:

    L_IPO = ( margin - 1/(2β) )^2

DPO's sigmoid can blow up on unanimous-preference pairs (margin → ∞
contributes vanishing loss but huge gradient via the sigmoid's tail);
IPO's squared form is bounded and more robust to label noise. One-line
switch in `compute_dpo_loss`.

**cDPO** (a.k.a. label-smoothed DPO). Treat labels as `(1-α)` correct,
`α` flipped. The loss becomes a convex combination of the DPO loss with
the chosen and the *rejected* preferred — symmetric under label noise:

    L_cDPO = -(1-α) log σ(β · margin) - α log σ(-β · margin)

Implementation notes:

- All log-prob sums are gathered with `torch.gather` on shifted labels
  (the data pipeline already did the +1 shift; see Module 15's data.py).
  Positions outside the assistant response have `labels == IGNORE_INDEX`
  and contribute zero to the sum.
- We cast logits to FP32 *before* log_softmax. BF16 log_softmax over a
  150k-vocab is numerically dicey at the tails; cheap FP32 cast removes
  the risk and adds <1% wallclock.
- We do ONE forward pass per model with chosen+rejected concatenated into
  a 2B batch. Cheaper than two separate calls (kernel launch overhead),
  exactly equivalent to two calls under standard attention semantics.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PreferenceConfig

IGNORE_INDEX = -100

_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}


# =============================================================================
# Log-prob gather
# =============================================================================

def gather_response_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log π(y_t | y_<t) over positions where `labels != IGNORE_INDEX`.

    Args:
        logits: [B, S, V] — model output at every position.
        labels: [B, S]    — target token at every position; IGNORE_INDEX where we don't score.

    Returns:
        [B] — per-example sum of log-probabilities over response tokens.

    Numerical detail: log_softmax in FP32. BF16 over a 150k vocab gets shaky
    in the low-probability tail (matters for the rejected response, which the
    policy WILL push toward zero probability). The cast is free relative to
    the full transformer forward.
    """
    mask = labels != IGNORE_INDEX                   # [B, S], 1 where this position contributes
    # We need `safe_labels` to be valid indices for `gather` even at masked positions.
    safe_labels = labels.masked_fill(~mask, 0)      # [B, S]

    log_probs = torch.log_softmax(logits.float(), dim=-1)   # [B, S, V]
    per_token = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token = per_token * mask.float()            # zero out non-response positions

    return per_token.sum(dim=-1)                    # [B]


# =============================================================================
# The DPO / IPO loss
# =============================================================================

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str = "dpo",
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute the DPO / IPO / cDPO loss on a batch of log-prob 4-tuples.

    Returns:
        (loss, metrics) where
            loss is a scalar (mean over the batch).
            metrics is a dict of detached per-pair tensors for logging:
                - "chosen_rewards":   β · (log π(y_w) - log π_ref(y_w))
                - "rejected_rewards": β · (log π(y_l) - log π_ref(y_l))
                - "margin":           chosen_logratio - rejected_logratio  (no β)
                - "accuracy":         1.0 if margin > 0 else 0.0

    `margin > 0` means the policy ranks chosen above rejected. At init,
    margin ≈ 0 (policy == reference); over training, margin grows and
    accuracy → 1. The pair `(chosen_rewards, rejected_rewards)` is your
    canonical training plot: chosen rises (more probable than reference),
    rejected falls (less probable). Both should move; if only one moves,
    your LR or β is wrong.
    """
    if reference_free:
        # Ablation: drop the KL anchor. The model WILL drift; this is here for
        # sanity tests and for the notebook's β-sweep showing what the KL
        # actually buys you.
        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

    chosen_logratios = policy_chosen_logps - ref_chosen_logps       # [B]
    rejected_logratios = policy_rejected_logps - ref_rejected_logps  # [B]
    margin = chosen_logratios - rejected_logratios                  # [B]

    if loss_type == "dpo":
        if label_smoothing > 0:
            # cDPO: convex combination of "chosen is preferred" and "rejected is preferred"
            losses = (
                -(1.0 - label_smoothing) * F.logsigmoid(beta * margin)
                - label_smoothing * F.logsigmoid(-beta * margin)
            )
        else:
            losses = -F.logsigmoid(beta * margin)
    elif loss_type == "ipo":
        # Azar et al. — squared loss on (margin - 1/(2β)). Note this means the
        # "target" margin is 1/(2β); for β=0.1 that's 5. The model is asked to
        # achieve a SPECIFIC log-ratio gap rather than maximizing it indefinitely.
        losses = (margin - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}, expected 'dpo' or 'ipo'")

    loss = losses.mean()

    metrics = {
        "chosen_rewards": (beta * chosen_logratios).detach(),
        "rejected_rewards": (beta * rejected_logratios).detach(),
        "margin": margin.detach(),
        "accuracy": (margin > 0).float().detach(),
    }
    return loss, metrics


# =============================================================================
# Forward pass on policy + reference
# =============================================================================

def forward_dpo(
    policy: nn.Module,
    reference: nn.Module,
    batch: dict,
    pref_cfg: PreferenceConfig,
    dtype: str = "bf16",
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """One forward pass — policy + reference — on a (chosen, rejected) batch.

    The 2B trick: concatenate chosen and rejected along the batch dim so each
    model sees ONE forward call instead of two. This saves kernel-launch
    overhead and matches what TRL's DPOTrainer does. Mathematically identical
    to two separate forwards (no cross-sample attention under causal masking).
    """
    chosen_ids = batch["chosen_input_ids"]
    rejected_ids = batch["rejected_input_ids"]
    B = chosen_ids.shape[0]

    input_ids = torch.cat([chosen_ids, rejected_ids], dim=0).to(device, non_blocking=True)
    labels = torch.cat(
        [batch["chosen_labels"], batch["rejected_labels"]], dim=0
    ).to(device, non_blocking=True)
    attn_mask = torch.cat(
        [batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0
    ).to(device, non_blocking=True)

    # ---- Policy forward (under autocast — matches Module 15) ----
    ctx = (torch.autocast(device_type=device, dtype=_DTYPE[dtype])
           if dtype != "fp32" else nullcontext())
    with ctx:
        policy_out = policy(input_ids=input_ids, attention_mask=attn_mask)
        policy_logits = policy_out.logits if hasattr(policy_out, "logits") else policy_out
    policy_logps = gather_response_logps(policy_logits, labels)   # [2B]
    policy_chosen_logps = policy_logps[:B]
    policy_rejected_logps = policy_logps[B:]

    # ---- Reference forward (no grad; the reference is already BF16) ----
    # No autocast needed: the model's parameters are BF16, so the forward runs
    # in BF16 natively. Wrapping in `torch.no_grad()` prevents any autograd
    # graph from being kept around even though `requires_grad_(False)` is set
    # — belt-and-braces, and a tiny memory saving.
    with torch.no_grad():
        ref_out = reference(input_ids=input_ids, attention_mask=attn_mask)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out
        ref_logps = gather_response_logps(ref_logits, labels)
    ref_chosen_logps = ref_logps[:B]
    ref_rejected_logps = ref_logps[B:]

    return compute_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=pref_cfg.beta,
        loss_type=pref_cfg.loss_type,
        label_smoothing=pref_cfg.label_smoothing,
        reference_free=pref_cfg.reference_free,
    )


# =============================================================================
# train_step — one optimizer update = grad_accum micro-batches
# =============================================================================

def train_step(
    policy: nn.Module,
    reference: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_iter: Iterator[dict],
    grad_accum: int,
    grad_clip: float,
    pref_cfg: PreferenceConfig,
    dtype: str = "bf16",
    device: str = "cuda",
) -> dict:
    """One full optimizer step = `grad_accum` micro-batches.

    Same structural shape as Module 15's train_step (FSDP no_sync on all
    but the last micro-batch, grad clip, optimizer step). The differences:

    - We run TWO forwards per micro-batch (policy + reference); only the
      policy carries gradient. The reference forward is fast (BF16, no
      autograd graph) — typically 10-15% of step time.
    - We aggregate the richer DPO metrics dict, not just loss/grad_norm.
      Returns a dict of floats so the train loop can log everything.

    Loss-averaging caveat (same as SFT): each micro-batch's loss is `mean()`
    over its pairs, then divided by `grad_accum`. If pair-count varies
    across micro-batches (it doesn't with `drop_last=True`), the per-pair
    loss is slightly biased. Standard practice — fine for typical data.
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    sum_chosen_r = 0.0
    sum_rejected_r = 0.0
    sum_margin = 0.0
    sum_acc = 0.0
    n_micro = 0

    for accum_idx in range(grad_accum):
        batch = next(batch_iter)
        loss, metrics = forward_dpo(
            policy=policy,
            reference=reference,
            batch=batch,
            pref_cfg=pref_cfg,
            dtype=dtype,
            device=device,
        )
        loss = loss / grad_accum

        is_last = (accum_idx == grad_accum - 1)
        if not is_last and hasattr(policy, "no_sync"):
            with policy.no_sync():
                loss.backward()
        else:
            loss.backward()

        total_loss += loss.detach().float().item()
        sum_chosen_r += metrics["chosen_rewards"].mean().float().item()
        sum_rejected_r += metrics["rejected_rewards"].mean().float().item()
        sum_margin += metrics["margin"].mean().float().item()
        sum_acc += metrics["accuracy"].mean().float().item()
        n_micro += 1

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {
        "loss": total_loss,
        "chosen_rewards": sum_chosen_r / n_micro,
        "rejected_rewards": sum_rejected_r / n_micro,
        "margin": sum_margin / n_micro,
        "accuracy": sum_acc / n_micro,
        "grad_norm": float(grad_norm),
    }


# =============================================================================
# Offline smoke test of the loss math (no model needed)
# =============================================================================

if __name__ == "__main__":
    print("--- DPO loss smoke test (synthetic logps, no model) ---")
    torch.manual_seed(0)

    # Synthesize a batch of 4 pairs. Set things up so the policy is "better
    # than reference on chosen" (logratio > 0) and "worse than reference on
    # rejected" (logratio < 0). DPO loss should be low (we're already aligned).
    policy_chosen = torch.tensor([-10.0, -12.0, -8.0, -9.0])
    policy_rejected = torch.tensor([-14.0, -16.0, -13.0, -15.0])
    ref_chosen = torch.tensor([-11.0, -13.0, -9.0, -10.0])
    ref_rejected = torch.tensor([-13.0, -15.0, -12.0, -14.0])

    pref_cfg = PreferenceConfig(beta=0.1, loss_type="dpo")
    loss, metrics = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        beta=pref_cfg.beta, loss_type=pref_cfg.loss_type,
    )
    print(f"  DPO loss (aligned):  {loss.item():.4f}  "
          f"(expect small — policy already prefers chosen)")
    print(f"  margin mean:         {metrics['margin'].mean().item():.3f}  "
          f"(expect positive)")
    print(f"  accuracy:            {metrics['accuracy'].mean().item():.1%}  "
          f"(expect 100%)")

    # Now flip the signs — policy is MISaligned (worse on chosen, better on rejected).
    loss_bad, m_bad = compute_dpo_loss(
        policy_rejected, policy_chosen, ref_chosen, ref_rejected,
        beta=pref_cfg.beta, loss_type=pref_cfg.loss_type,
    )
    print(f"\n  DPO loss (misaligned): {loss_bad.item():.4f}  (expect much higher)")
    print(f"  margin mean:           {m_bad['margin'].mean().item():.3f}  (expect negative)")
    print(f"  accuracy:              {m_bad['accuracy'].mean().item():.1%}  (expect 0%)")

    # IPO variant on the aligned batch
    loss_ipo, _ = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        beta=pref_cfg.beta, loss_type="ipo",
    )
    print(f"\n  IPO loss (aligned):  {loss_ipo.item():.4f}  "
          f"(squared distance from target margin 1/(2β) = 5.0)")

    # Reference-free DPO — should produce a different loss (policy's logps alone)
    loss_rf, _ = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        beta=pref_cfg.beta, loss_type="dpo", reference_free=True,
    )
    print(f"\n  Reference-free DPO:  {loss_rf.item():.4f}  "
          f"(no KL anchor — never use in real training)")
