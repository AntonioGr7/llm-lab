"""Tests for the GRPO loss math (loop.py) + the rollout pieces that
don't need a model (rollout.group_normalize_advantages,
rollout._build_labels_and_masks).

All tests run on CPU with synthetic tensors; <1s total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RLConfig
from loop import compute_grpo_loss
from rollout import (
    per_token_logps,
    group_normalize_advantages,
    _build_labels_and_masks,
    IGNORE_INDEX,
)


# =============================================================================
# group_normalize_advantages
# =============================================================================

def test_group_normalization_zero_mean_within_each_group():
    """Within each group of G, the mean should be zero by construction."""
    G = 4
    # 3 groups, distinct internal patterns
    rewards = torch.tensor([
        1.0, 1.0, 0.0, 0.0,   # group 0: 2 highs, 2 lows
        0.5, 0.5, 0.5, 0.5,   # group 1: degenerate
        -1.0, 2.0, 0.0, 1.0,  # group 2: spread
    ])
    adv = group_normalize_advantages(rewards, group_size=G, eps=1e-6)
    for p in range(3):
        chunk = adv[p*G:(p+1)*G]
        # Mean ~ 0 (within float precision)
        assert abs(chunk.mean().item()) < 1e-4, (p, chunk.tolist())
    # Group 1 should be all zeros (std=0 in the denominator, numerator is 0)
    assert torch.allclose(adv[4:8], torch.zeros(4), atol=1e-4)
    print("  test_group_normalization_zero_mean_within_each_group: PASS")


def test_group_normalization_degenerate_group_safe():
    """A group where all rewards are equal should not produce NaN/inf."""
    rewards = torch.tensor([0.7] * 4)
    adv = group_normalize_advantages(rewards, group_size=4, eps=1e-6)
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-4)
    print("  test_group_normalization_degenerate_group_safe: PASS")


# =============================================================================
# per_token_logps (the gather)
# =============================================================================

def test_per_token_logps_mask_and_gather():
    """Verify per_token_logps gathers correctly and zeros at IGNORE_INDEX."""
    torch.manual_seed(0)
    N, T, V = 2, 5, 8
    logits = torch.randn(N, T, V)
    labels = torch.tensor([
        [1, 2, 3, IGNORE_INDEX, IGNORE_INDEX],
        [IGNORE_INDEX, 4, 5, 6, IGNORE_INDEX],
    ])
    out = per_token_logps(logits, labels)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Position-wise checks
    assert torch.allclose(out[0, 0], log_probs[0, 0, 1])
    assert torch.allclose(out[0, 1], log_probs[0, 1, 2])
    assert torch.allclose(out[0, 2], log_probs[0, 2, 3])
    assert out[0, 3].item() == 0.0
    assert out[0, 4].item() == 0.0
    assert out[1, 0].item() == 0.0
    assert torch.allclose(out[1, 1], log_probs[1, 1, 4])
    assert torch.allclose(out[1, 2], log_probs[1, 2, 5])
    assert torch.allclose(out[1, 3], log_probs[1, 3, 6])
    assert out[1, 4].item() == 0.0
    print("  test_per_token_logps_mask_and_gather: PASS")


# =============================================================================
# _build_labels_and_masks (the geometry)
# =============================================================================

def test_build_labels_and_masks_geometry():
    """Geometry: prompt (left-padded) + completion (right-padded after EOS).

    Setup: S_full=10, prompt buffer 6, max_new 4. Two rows with varying
    REAL prompt lengths and completion lengths.
    """
    # full_ids layout for each row: [left_pad (S_prompt_buf - prompt_len)] +
    # [real prompt (prompt_len)] + [completion (completion_len)] + [pad rest]
    # S_prompt_buf=6, max_new=4, so S_full=10.
    full_ids = torch.tensor([
        # row 0: prompt_len=3, completion_len=4 (full)
        [0, 0, 0, 11, 12, 13, 21, 22, 23, 24],
        # row 1: prompt_len=2, completion_len=2 (early eos)
        [0, 0, 0, 0, 14, 15, 31, 32, 0, 0],
    ])
    prompt_lens = torch.tensor([3, 2])
    completion_lens = torch.tensor([4, 2])
    inp, lab, attn = _build_labels_and_masks(full_ids, prompt_lens, completion_lens, pad_id=0)
    assert inp.shape == (2, 9)
    assert lab.shape == (2, 9)
    assert attn.shape == (2, 9)

    # Row 0: completion lives at positions [6, 10) of full_ids. After the
    # shift, labels = full_ids[:, 1:], so labels positions [5, 9) hold the
    # completion tokens 21..24. Everything outside should be IGNORE_INDEX.
    expected_lab_row0 = [IGNORE_INDEX] * 5 + [21, 22, 23, 24]
    assert lab[0].tolist() == expected_lab_row0

    # Row 1: completion length 2, so labels positions [5, 7) hold 31, 32.
    expected_lab_row1 = [IGNORE_INDEX] * 5 + [31, 32] + [IGNORE_INDEX] * 2
    assert lab[1].tolist() == expected_lab_row1

    # Attention mask row 0: real prompt + completion = positions [3, 9)
    # (after the shift, S_full-1 = 9 caps it). So 3 zeros then 6 ones.
    assert attn[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    # Row 1: prompt_len=2, completion_len=2. Real span in full_ids [4, 8).
    # After dropping the last column for the shift, this is positions [4, 8)
    # in `attn` — 4 ones at indices 4..7.
    assert attn[1].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0]

    print("  test_build_labels_and_masks_geometry: PASS")


# =============================================================================
# GRPO loss properties
# =============================================================================

def _build_synth_loss_inputs(N=4, T=8, drift=0.0, advantages=None, seed=0):
    """Helper: synth (current, old, ref) per-token logps + masks."""
    torch.manual_seed(seed)
    current = torch.randn(N, T) * 0.1
    old = current.clone()                                  # K=1 contract
    ref = current + 0.05                                   # small ref drift
    if drift != 0.0:
        current = current + drift                          # simulate post-step
    if advantages is None:
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])[:N]
    mask = torch.ones(N, T, dtype=torch.bool)
    return current, old, ref, advantages, mask


def test_grpo_loss_K1_ratio_is_one():
    """At K=1 (old == current), the importance ratio is exactly 1
    everywhere, so the surrogate equals A · 1 (no clipping)."""
    current, old, ref, A, mask = _build_synth_loss_inputs(drift=0.0)
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2, mu_epochs=1)
    _, m = compute_grpo_loss(current, old, ref, A, mask, rl_cfg)
    # ratio mean should be 1.0
    assert abs(m["ratio_mean"].item() - 1.0) < 1e-5
    # clip_frac should be 0.0 (no token is outside [1-ε, 1+ε])
    assert abs(m["clip_frac"].item() - 0.0) < 1e-9
    print("  test_grpo_loss_K1_ratio_is_one: PASS")


def test_grpo_loss_balanced_advantages_zero_policy_loss():
    """If A=[+1,-1,+1,-1], the policy_loss at ratio=1 is (sum -ratio*A) / N·T
    where sum across rows balances to 0 → policy_loss = 0."""
    current, old, ref, A, mask = _build_synth_loss_inputs(drift=0.0)
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2, mu_epochs=1)
    _, m = compute_grpo_loss(current, old, ref, A, mask, rl_cfg)
    assert abs(m["policy_loss"].item()) < 1e-5
    print("  test_grpo_loss_balanced_advantages_zero_policy_loss: PASS")


def test_grpo_loss_positive_advantage_negative_policy_loss():
    """At ratio=1 with all-positive advantages, -A·1 is negative — so policy_loss < 0."""
    current, old, ref, _, mask = _build_synth_loss_inputs(drift=0.0)
    A = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2, mu_epochs=1)
    _, m = compute_grpo_loss(current, old, ref, A, mask, rl_cfg)
    assert m["policy_loss"].item() < 0
    # And specifically equal to -mean(A) (since ratio=1 everywhere)
    assert abs(m["policy_loss"].item() - (-1.0)) < 1e-5
    print("  test_grpo_loss_positive_advantage_negative_policy_loss: PASS")


def test_grpo_loss_K2_clip_kicks_in():
    """After a fake gradient step that bumped current_logps by 0.5, the
    ratio shifts to e^0.5 ≈ 1.65 — well outside the ε=0.2 clip window —
    so clip_frac should jump from 0 toward 1."""
    current, old, ref, A, mask = _build_synth_loss_inputs(drift=0.5)
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2, mu_epochs=2)
    _, m = compute_grpo_loss(current, old, ref, A, mask, rl_cfg)
    assert m["ratio_mean"].item() > 1.2     # exp(0.5) ≈ 1.65, with mask broadcasting
    assert m["clip_frac"].item() > 0.5      # most tokens should be clipped
    print("  test_grpo_loss_K2_clip_kicks_in: PASS")


def test_grpo_loss_kl_nonneg_k3():
    """The k3 KL estimator should be NON-NEGATIVE token-wise (and so should
    the per-batch mean). Verify by passing a deliberately diverged ref."""
    torch.manual_seed(7)
    N, T = 4, 6
    current = torch.randn(N, T) * 0.1
    old = current.clone()
    ref = current - 1.0       # big drift between current and ref
    A = torch.tensor([1.0, -1.0, 1.0, -1.0])
    mask = torch.ones(N, T, dtype=torch.bool)
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2)
    _, m = compute_grpo_loss(current, old, ref, A, mask, rl_cfg)
    assert m["mean_kl"].item() >= 0.0
    assert m["kl_loss"].item() >= 0.0
    print("  test_grpo_loss_kl_nonneg_k3: PASS")


def test_grpo_loss_degenerate_group_no_policy_signal():
    """If all advantages within a group are 0 (degenerate group),
    policy_loss is exactly 0 and only the KL term remains."""
    current, old, ref, _, mask = _build_synth_loss_inputs(drift=0.0)
    A = torch.zeros(4)
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2)
    loss, m = compute_grpo_loss(current, old, ref, A, mask, rl_cfg)
    assert abs(m["policy_loss"].item()) < 1e-9
    assert abs(loss.item() - m["kl_loss"].item()) < 1e-5
    print("  test_grpo_loss_degenerate_group_no_policy_signal: PASS")


def test_grpo_loss_mask_zeros_out_pad_positions():
    """Padding-position contributions should NOT enter the loss."""
    torch.manual_seed(11)
    N, T = 4, 8
    current = torch.randn(N, T) * 0.1
    old = current.clone()
    ref = current + 0.05
    A = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rl_cfg = RLConfig(group_size=4, kl_beta=0.04, clip_ratio=0.2)

    # All tokens valid
    full_mask = torch.ones(N, T, dtype=torch.bool)
    _, m_full = compute_grpo_loss(current, old, ref, A, full_mask, rl_cfg)

    # Half-masked (last T/2 columns are padding). The averaged-over-tokens
    # quantities should be IDENTICAL (we average over valid tokens only).
    half_mask = torch.zeros(N, T, dtype=torch.bool)
    half_mask[:, :T // 2] = True
    # Bring the "trash" half to weird values to prove it's ignored
    current_trashed = current.clone()
    current_trashed[:, T // 2:] = 1e6
    _, m_half = compute_grpo_loss(current_trashed, old, ref, A, half_mask, rl_cfg)

    # The unmasked computation used all T positions; the masked one used only
    # T/2. But the means of the surviving positions agree exactly with the
    # unmasked's means over the first T/2 — i.e. mask correctly removes
    # contributions.
    _, m_first_half_only = compute_grpo_loss(
        current[:, :T // 2], old[:, :T // 2], ref[:, :T // 2],
        A, torch.ones(N, T // 2, dtype=torch.bool), rl_cfg,
    )
    assert abs(m_half["policy_loss"].item() - m_first_half_only["policy_loss"].item()) < 1e-4
    assert abs(m_half["mean_kl"].item() - m_first_half_only["mean_kl"].item()) < 1e-4
    print("  test_grpo_loss_mask_zeros_out_pad_positions: PASS")


if __name__ == "__main__":
    print("--- test_grpo_loss.py ---")
    test_group_normalization_zero_mean_within_each_group()
    test_group_normalization_degenerate_group_safe()
    test_per_token_logps_mask_and_gather()
    test_build_labels_and_masks_geometry()
    test_grpo_loss_K1_ratio_is_one()
    test_grpo_loss_balanced_advantages_zero_policy_loss()
    test_grpo_loss_positive_advantage_negative_policy_loss()
    test_grpo_loss_K2_clip_kicks_in()
    test_grpo_loss_kl_nonneg_k3()
    test_grpo_loss_degenerate_group_no_policy_signal()
    test_grpo_loss_mask_zeros_out_pad_positions()
    print("ALL TESTS PASSED")
