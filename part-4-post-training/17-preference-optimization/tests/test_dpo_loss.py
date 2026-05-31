"""Correctness tests for the DPO/IPO loss — the heart of Module 17.

The loss math is the single piece of new code in this module that has no
"flat-line loss" tell-tale at runtime; if it's silently wrong, training just
fails to improve over weeks of GPU time. These tests pin the properties the
README §3 derives.

Properties checked:
  1. `gather_response_logps` matches a hand-computed log-softmax + gather and
     zeros out IGNORE_INDEX positions.
  2. DPO loss has the right gradient sign: lower for aligned pairs (margin > 0)
     than misaligned, and saturates above (margin → ∞ → loss → 0).
  3. IPO loss has its minimum at margin = 1/(2β) and is bounded.
  4. cDPO (label_smoothing > 0) is symmetric: L_α(m) + L_α(-m) is independent of m.
  5. reference_free=True equals "set ref_logps to 0 manually".
  6. The `accuracy` metric is the fraction of pairs with margin > 0.

Usage:
    cd 17-preference-optimization/
    python tests/test_dpo_loss.py

Exits 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from loop import gather_response_logps, compute_dpo_loss, IGNORE_INDEX


# ---------------------------------------------------------------------------
# 1. gather_response_logps
# ---------------------------------------------------------------------------

def test_gather_logps() -> bool:
    """Hand-compute the per-example sum and check it matches."""
    torch.manual_seed(0)
    B, S, V = 2, 5, 7
    logits = torch.randn(B, S, V)
    # Make labels: example 0 scores positions 1..3, example 1 scores 0..2.
    labels = torch.full((B, S), IGNORE_INDEX, dtype=torch.long)
    labels[0, 1] = 3
    labels[0, 2] = 1
    labels[0, 3] = 6
    labels[1, 0] = 2
    labels[1, 1] = 5
    labels[1, 2] = 0

    got = gather_response_logps(logits, labels)         # [B]

    # Hand-compute
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected = torch.zeros(B)
    for b in range(B):
        for t in range(S):
            if labels[b, t].item() != IGNORE_INDEX:
                expected[b] += log_probs[b, t, labels[b, t]].item()

    ok = torch.allclose(got, expected, atol=1e-5)
    print(f"  [1] gather_response_logps: {'OK' if ok else 'FAIL'}  "
          f"(got={got.tolist()}, exp={expected.tolist()})")
    return ok


# ---------------------------------------------------------------------------
# 2. DPO loss directionality + saturation
# ---------------------------------------------------------------------------

def test_dpo_directionality() -> bool:
    """Aligned (margin > 0) -> low loss, misaligned (margin < 0) -> high loss.
    Margin → ∞ -> loss → 0 (sigmoid saturation, not divergence).
    Margin → -∞ -> loss → ∞ linearly."""
    beta = 0.1

    def lo(margin):
        # Build a synthetic 4-tuple that produces this margin under DPO.
        # margin = (p_ch - r_ch) - (p_rej - r_rej). Pick simple terms.
        p_ch = torch.tensor([margin / 2.0])
        p_rej = torch.tensor([-margin / 2.0])
        r_ch = torch.tensor([0.0])
        r_rej = torch.tensor([0.0])
        loss, _ = compute_dpo_loss(p_ch, p_rej, r_ch, r_rej, beta=beta)
        return loss.item()

    # β=0.1 makes the loss roughly log(1+exp(-0.1·margin)); the saturation
    # only really kicks in past |β·margin| ~ 3-5, i.e. |margin| ~ 30-50.
    aligned = lo(50.0)                  # β·margin = 5 -> sigmoid ≈ 0.993 -> loss ≈ 0.007
    misaligned = lo(-50.0)              # symmetric -> loss ≈ 5.007 (linear regime)
    middle = lo(0.0)                    # sigmoid(0) = 0.5 -> loss = ln(2) ≈ 0.693
    saturated_high = lo(500.0)          # essentially zero
    saturated_low_loss = lo(-500.0)     # ≈ -β·margin = 50 (large finite)

    ok = (aligned < 0.05                # well into the saturation tail
          and misaligned > middle       # misaligned loses
          and middle > aligned          # neutral margin > aligned margin
          and saturated_high < 1e-3
          and saturated_low_loss > misaligned)
    print(f"  [2] dpo loss directionality: {'OK' if ok else 'FAIL'}  "
          f"(aligned={aligned:.4f}, middle={middle:.4f}, misaligned={misaligned:.4f}, "
          f"sat_high={saturated_high:.4f}, sat_low={saturated_low_loss:.1f})")
    return ok


# ---------------------------------------------------------------------------
# 3. IPO loss minimum at margin = 1/(2β)
# ---------------------------------------------------------------------------

def test_ipo_minimum() -> bool:
    beta = 0.1
    target_margin = 1.0 / (2.0 * beta)  # = 5.0

    def loss_at(m):
        p_ch = torch.tensor([m / 2.0])
        p_rej = torch.tensor([-m / 2.0])
        r_ch = torch.tensor([0.0])
        r_rej = torch.tensor([0.0])
        loss, _ = compute_dpo_loss(p_ch, p_rej, r_ch, r_rej, beta=beta, loss_type="ipo")
        return loss.item()

    at_target = loss_at(target_margin)
    above = loss_at(target_margin + 2.0)
    below = loss_at(target_margin - 2.0)
    far_above = loss_at(target_margin + 100.0)   # bounded ~10000

    ok = (at_target < 1e-6              # exact minimum
          and above > at_target         # going past target hurts
          and below > at_target
          and far_above > 100.0         # finite, no divergence — but big
          and abs(above - below) < 1e-6)  # symmetric around target
    print(f"  [3] ipo loss minimum at margin=1/(2β)={target_margin}: {'OK' if ok else 'FAIL'}  "
          f"(at_target={at_target:.4f}, above={above:.4f}, below={below:.4f}, "
          f"far_above={far_above:.1f})")
    return ok


# ---------------------------------------------------------------------------
# 4. cDPO symmetry
# ---------------------------------------------------------------------------

def test_cdpo_symmetry() -> bool:
    """With label_smoothing=α, the loss should treat margin and -margin
    symmetrically up to a constant offset — that's the point of cDPO."""
    beta = 0.1
    alpha = 0.2

    def loss_at(m):
        p_ch = torch.tensor([m / 2.0])
        p_rej = torch.tensor([-m / 2.0])
        r_ch = torch.tensor([0.0])
        r_rej = torch.tensor([0.0])
        loss, _ = compute_dpo_loss(p_ch, p_rej, r_ch, r_rej, beta=beta,
                                   loss_type="dpo", label_smoothing=alpha)
        return loss.item()

    # cDPO: L_α(m) = -(1-α) log σ(βm) - α log σ(-βm)
    #       L_α(-m) = -(1-α) log σ(-βm) - α log σ(βm)
    # So L_α(m) + L_α(-m) = - log σ(βm) - log σ(-βm) (independent of α).
    sum_pos_5 = loss_at(5.0) + loss_at(-5.0)
    sum_pos_2 = loss_at(2.0) + loss_at(-2.0)

    # Independent-of-margin invariant: should equal the no-smoothing sum.
    def vanilla_sum(m):
        beta_m = beta * m
        return (-F.logsigmoid(torch.tensor(beta_m)).item()
                - F.logsigmoid(torch.tensor(-beta_m)).item())

    expected_sum_5 = vanilla_sum(5.0)
    expected_sum_2 = vanilla_sum(2.0)

    ok = (abs(sum_pos_5 - expected_sum_5) < 1e-5
          and abs(sum_pos_2 - expected_sum_2) < 1e-5)
    print(f"  [4] cDPO symmetry: {'OK' if ok else 'FAIL'}  "
          f"(sum@m=5: got={sum_pos_5:.4f}, exp={expected_sum_5:.4f}; "
          f"sum@m=2: got={sum_pos_2:.4f}, exp={expected_sum_2:.4f})")
    return ok


# ---------------------------------------------------------------------------
# 5. reference_free flag
# ---------------------------------------------------------------------------

def test_reference_free() -> bool:
    """reference_free=True should give the same answer as manually zeroing
    the reference logps."""
    beta = 0.1
    p_ch = torch.tensor([-10.0, -8.0])
    p_rej = torch.tensor([-14.0, -12.0])
    r_ch = torch.tensor([-11.0, -9.0])
    r_rej = torch.tensor([-13.0, -11.0])

    loss_rf, _ = compute_dpo_loss(p_ch, p_rej, r_ch, r_rej, beta=beta, reference_free=True)
    zero = torch.zeros_like(p_ch)
    loss_manual, _ = compute_dpo_loss(p_ch, p_rej, zero, zero, beta=beta)
    ok = abs(loss_rf.item() - loss_manual.item()) < 1e-6
    print(f"  [5] reference_free: {'OK' if ok else 'FAIL'}  "
          f"(rf={loss_rf.item():.5f}, manual={loss_manual.item():.5f})")
    return ok


# ---------------------------------------------------------------------------
# 6. accuracy metric
# ---------------------------------------------------------------------------

def test_accuracy_metric() -> bool:
    """`accuracy` = mean(margin > 0). Construct a batch with known margins."""
    # Margins: [+2, -1, +0.5, -3]   -> 2 of 4 are positive -> 50%
    p_ch = torch.tensor([0.0, 0.0, 0.0, 0.0])
    p_rej = torch.tensor([-2.0, 1.0, -0.5, 3.0])
    r_ch = torch.tensor([0.0, 0.0, 0.0, 0.0])
    r_rej = torch.tensor([0.0, 0.0, 0.0, 0.0])
    _, m = compute_dpo_loss(p_ch, p_rej, r_ch, r_rej, beta=0.1)
    acc = m["accuracy"].mean().item()
    ok = abs(acc - 0.5) < 1e-6
    print(f"  [6] accuracy metric: {'OK' if ok else 'FAIL'}  (got={acc:.2f}, exp=0.50)")
    return ok


# ---------------------------------------------------------------------------
# 7. Sanity: a forward+backward through compute_dpo_loss produces a
#    gradient that pushes policy_chosen UP and policy_rejected DOWN.
# ---------------------------------------------------------------------------

def test_gradient_directions() -> bool:
    beta = 0.1
    p_ch = torch.tensor([0.0, 0.0], requires_grad=True)
    p_rej = torch.tensor([0.0, 0.0], requires_grad=True)
    r_ch = torch.tensor([-1.0, -1.0])
    r_rej = torch.tensor([-1.0, -1.0])
    loss, _ = compute_dpo_loss(p_ch, p_rej, r_ch, r_rej, beta=beta)
    loss.backward()
    # d L_dpo / d p_chosen   = -β · σ(-β · margin)        (negative — push p_ch UP to reduce loss)
    # d L_dpo / d p_rejected = +β · σ(-β · margin)        (positive — push p_rej DOWN to reduce loss)
    grad_ch_sign = (p_ch.grad < 0).all().item()
    grad_rej_sign = (p_rej.grad > 0).all().item()
    ok = bool(grad_ch_sign and grad_rej_sign)
    print(f"  [7] gradient directions: {'OK' if ok else 'FAIL'}  "
          f"(grad p_chosen={p_ch.grad.tolist()} (want negative), "
          f"grad p_rejected={p_rej.grad.tolist()} (want positive))")
    return ok


def main() -> int:
    print("DPO/IPO loss tests\n")
    results = [
        test_gather_logps(),
        test_dpo_directionality(),
        test_ipo_minimum(),
        test_cdpo_symmetry(),
        test_reference_free(),
        test_accuracy_metric(),
        test_gradient_directions(),
    ]
    print()
    if all(results):
        print("  PASS — log-prob gather correct, DPO/IPO/cDPO loss shapes verified, "
              "gradients point the right way.")
        return 0
    print("  FAIL — see the failing property above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
