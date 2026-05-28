"""Tests for the distillation loss and rollout geometry.

All tests run on CPU with synthetic tensors; <2s total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DistillationConfig
from loop import compute_distill_loss
from rollout import _left_pad_batch, _build_messages_for_teacher, _build_messages_for_student
from make_tooluse_corpus import Example


# =============================================================================
# Loss properties
# =============================================================================

def test_identical_distributions_zero_loss():
    """KL(p || p) = 0 for any direction."""
    torch.manual_seed(0)
    B, C, V = 2, 6, 32
    log_p = torch.log_softmax(torch.randn(B, C, V), dim=-1)
    mask = torch.ones(B, C, dtype=torch.bool)
    for direction in ("forward", "reverse"):
        cfg = DistillationConfig(kl_direction=direction)
        loss, _ = compute_distill_loss(log_p, log_p, mask, cfg)
        assert loss.item() < 1e-6, (direction, loss.item())
    print("  test_identical_distributions_zero_loss: PASS")


def test_forward_kl_always_nonneg():
    """KL >= 0 by Gibbs' inequality. Verify across random pairs."""
    torch.manual_seed(1)
    B, C, V = 4, 8, 64
    mask = torch.ones(B, C, dtype=torch.bool)
    cfg = DistillationConfig(kl_direction="forward")
    for _ in range(5):
        s = torch.log_softmax(torch.randn(B, C, V), dim=-1)
        t = torch.log_softmax(torch.randn(B, C, V), dim=-1)
        loss, _ = compute_distill_loss(s, t, mask, cfg)
        assert loss.item() >= -1e-6, loss.item()
    print("  test_forward_kl_always_nonneg: PASS")


def test_reverse_kl_always_nonneg():
    torch.manual_seed(2)
    B, C, V = 4, 8, 64
    mask = torch.ones(B, C, dtype=torch.bool)
    cfg = DistillationConfig(kl_direction="reverse")
    for _ in range(5):
        s = torch.log_softmax(torch.randn(B, C, V), dim=-1)
        t = torch.log_softmax(torch.randn(B, C, V), dim=-1)
        loss, _ = compute_distill_loss(s, t, mask, cfg)
        assert loss.item() >= -1e-6, loss.item()
    print("  test_reverse_kl_always_nonneg: PASS")


def test_argmax_agreement_metric():
    """argmax_agreement should reflect token-by-token argmax match."""
    B, C, V = 1, 4, 8
    # Hand-craft so two argmaxes match, two don't.
    teacher_logits = torch.full((B, C, V), -10.0)
    teacher_logits[0, 0, 3] = 5.0    # argmax = 3
    teacher_logits[0, 1, 7] = 5.0    # argmax = 7
    teacher_logits[0, 2, 1] = 5.0    # argmax = 1
    teacher_logits[0, 3, 5] = 5.0    # argmax = 5
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    student_logits = torch.full((B, C, V), -10.0)
    student_logits[0, 0, 3] = 5.0    # match
    student_logits[0, 1, 0] = 5.0    # mismatch
    student_logits[0, 2, 1] = 5.0    # match
    student_logits[0, 3, 2] = 5.0    # mismatch
    student_log_probs = torch.log_softmax(student_logits, dim=-1)

    mask = torch.ones(B, C, dtype=torch.bool)
    cfg = DistillationConfig(kl_direction="forward")
    _, m = compute_distill_loss(student_log_probs, teacher_log_probs, mask, cfg)
    # 2 of 4 positions match → 50%.
    assert abs(m["argmax_agreement"].item() - 0.5) < 1e-6
    print("  test_argmax_agreement_metric: PASS")


def test_mask_correctly_excludes_pad_positions():
    """Same loss whether we mask half the tokens or just slice to the
    unmasked half explicitly."""
    torch.manual_seed(3)
    B, C, V = 2, 8, 32
    s = torch.log_softmax(torch.randn(B, C, V), dim=-1)
    t = torch.log_softmax(torch.randn(B, C, V), dim=-1)
    half_mask = torch.zeros(B, C, dtype=torch.bool)
    half_mask[:, :C // 2] = True

    # Trash the masked half — with FINITE values (loss code does not
    # nan-guard; rollout must produce finite log-probs everywhere).
    t_trash = t.clone()
    t_trash[:, C // 2:] = torch.log_softmax(torch.randn(B, C - C // 2, V), dim=-1)

    cfg = DistillationConfig(kl_direction="forward")
    loss_masked, _ = compute_distill_loss(s, t_trash, half_mask, cfg)
    loss_explicit, _ = compute_distill_loss(
        s[:, :C // 2], t[:, :C // 2],
        torch.ones(B, C // 2, dtype=torch.bool), cfg,
    )
    assert abs(loss_masked.item() - loss_explicit.item()) < 1e-5
    print("  test_mask_correctly_excludes_pad_positions: PASS")


def test_temperature_changes_loss():
    """Higher T softens both distributions; loss magnitude should differ."""
    torch.manual_seed(4)
    B, C, V = 2, 6, 32
    s = torch.log_softmax(torch.randn(B, C, V) * 5, dim=-1)   # sharp
    t = torch.log_softmax(torch.randn(B, C, V) * 5, dim=-1)
    mask = torch.ones(B, C, dtype=torch.bool)

    # We hand-apply the temperature scaling inline (the train_step
    # plumbing isn't called from this test).
    def kl_at_T(temperature):
        scaled_s = (s / temperature) - torch.logsumexp(s / temperature, dim=-1, keepdim=True)
        scaled_t = (t / temperature) - torch.logsumexp(t / temperature, dim=-1, keepdim=True)
        cfg = DistillationConfig(kl_direction="forward", temperature=temperature)
        loss, _ = compute_distill_loss(scaled_s, scaled_t, mask, cfg)
        return loss.item()

    loss_T1 = kl_at_T(1.0)
    loss_T2 = kl_at_T(2.0)
    # Higher T smooths the distributions -> KL between two-random-sharp ones
    # should shrink. (Not provable rigorously without specific p,q, but
    # holds for random sharp distributions.)
    assert loss_T2 < loss_T1, (loss_T1, loss_T2)
    print("  test_temperature_changes_loss: PASS")


# =============================================================================
# Rollout geometry
# =============================================================================

def test_left_pad_batch_preserves_real_tokens():
    """Real tokens land on the right; left positions are pad."""
    out_ids, out_attn, L_max = _left_pad_batch([[1, 2, 3], [4, 5]], pad_id=0)
    assert L_max == 3
    assert out_ids.tolist() == [[1, 2, 3], [0, 4, 5]]
    assert out_attn.tolist() == [[1, 1, 1], [0, 1, 1]]
    print("  test_left_pad_batch_preserves_real_tokens: PASS")


def test_build_messages_sdft_vs_on_policy():
    """SDFT teacher messages include K demo turns; on-policy don't."""
    demos = [
        Example(user="d0", tool="t0", args={"k": 0}),
        Example(user="d1", tool="t1", args={"k": 1}),
        Example(user="d2", tool="t2", args={"k": 2}),
    ]
    sdft_msgs = _build_messages_for_teacher("user prompt", demos, "sys", use_demos=True)
    op_msgs = _build_messages_for_teacher("user prompt", demos, "sys", use_demos=False)
    student_msgs = _build_messages_for_student("user prompt", "sys")

    # SDFT: system + 3 demos × (user, assistant) + final user = 8 turns
    assert len(sdft_msgs) == 1 + 3 * 2 + 1
    # On-policy teacher: system + user = 2 turns (same as student)
    assert len(op_msgs) == 2
    assert len(student_msgs) == 2
    # Roles of SDFT demo turns
    for i, d in enumerate(demos):
        assert sdft_msgs[1 + 2 * i]["role"] == "user"
        assert sdft_msgs[1 + 2 * i]["content"] == d.user
        assert sdft_msgs[1 + 2 * i + 1]["role"] == "assistant"
        # Assistant content should be the structured tool call.
        assert "<tool>" in sdft_msgs[1 + 2 * i + 1]["content"]
    print("  test_build_messages_sdft_vs_on_policy: PASS")


if __name__ == "__main__":
    print("--- test_distill_loss.py ---")
    test_identical_distributions_zero_loss()
    test_forward_kl_always_nonneg()
    test_reverse_kl_always_nonneg()
    test_argmax_agreement_metric()
    test_mask_correctly_excludes_pad_positions()
    test_temperature_changes_loss()
    test_left_pad_batch_preserves_real_tokens()
    test_build_messages_sdft_vs_on_policy()
    print("ALL TESTS PASSED")
