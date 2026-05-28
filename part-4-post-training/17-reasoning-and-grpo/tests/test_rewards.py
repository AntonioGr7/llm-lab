"""Tests for rewards.py — the GSM8K verifier.

These are PURE — no torch, no tokenizer, no datasets. Should pass in <1s.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `rewards.py` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RewardConfig
from rewards import (
    extract_answer, parse_int, format_reward, accuracy_reward, compute_rewards,
)


def test_parse_int_basic():
    assert parse_int("18") == 18
    assert parse_int("  18  ") == 18
    assert parse_int("18.0") == 18
    assert parse_int("$1,234") == 1234
    assert parse_int("The answer is 18 eggs") == 18
    assert parse_int("no number") is None
    assert parse_int("") is None
    assert parse_int("3.7") is None    # non-integer floats reject
    print("  test_parse_int_basic: PASS")


def test_extract_answer_last_tag_wins():
    # The LAST <answer> tag is the model's final commit.
    txt = "I think <answer>3</answer> ... wait no, <answer>4</answer>"
    assert extract_answer(txt) == "4"
    # No tag
    assert extract_answer("just text") is None
    # Single tag
    assert extract_answer("<answer>42</answer>") == "42"
    # Whitespace
    assert extract_answer("<answer>  42  </answer>") == "42"
    print("  test_extract_answer_last_tag_wins: PASS")


def test_format_reward_schema_check():
    cfg = RewardConfig()
    good = "<think>steps</think><answer>4</answer>"
    spaced = "<think>steps</think>  \n  <answer>4</answer>"
    no_think = "<answer>4</answer>"
    no_answer = "<think>steps</think>"
    bare = "just an answer of 4"
    assert format_reward(good, cfg.format_pattern) == 1.0
    assert format_reward(spaced, cfg.format_pattern) == 1.0
    assert format_reward(no_think, cfg.format_pattern) == 0.0
    assert format_reward(no_answer, cfg.format_pattern) == 0.0
    assert format_reward(bare, cfg.format_pattern) == 0.0
    print("  test_format_reward_schema_check: PASS")


def test_accuracy_reward_int_and_string_gt():
    # ground_truth as int
    assert accuracy_reward("<answer>18</answer>", 18) == 1.0
    assert accuracy_reward("<answer>17</answer>", 18) == 0.0
    # ground_truth as string
    assert accuracy_reward("<answer>18</answer>", "18") == 1.0
    assert accuracy_reward("<answer>18</answer>", "Some text 18 more text") == 1.0
    # No <answer> tag
    assert accuracy_reward("just 18", 18) == 0.0
    # Tag but unparseable contents
    assert accuracy_reward("<answer>not a number</answer>", 18) == 0.0
    # Bad ground truth
    assert accuracy_reward("<answer>18</answer>", "no number") == 0.0
    print("  test_accuracy_reward_int_and_string_gt: PASS")


def test_compute_rewards_weighted_sum():
    cfg = RewardConfig(w_format=0.1, w_accuracy=1.0)
    texts = [
        "<think>x</think><answer>4</answer>",   # schema + correct
        "<answer>4</answer>",                    # no schema, correct answer
        "<think>x</think><answer>5</answer>",   # schema + wrong
        "just text",                              # neither
    ]
    gts = [4, 4, 4, 4]
    brs = compute_rewards(texts, gts, cfg)

    # row 0: fmt=1, acc=1, total = 0.1 + 1.0 = 1.1
    assert abs(brs[0].total - 1.1) < 1e-9, brs[0]
    # row 1: fmt=0, acc=1, total = 0.0 + 1.0 = 1.0
    assert abs(brs[1].total - 1.0) < 1e-9, brs[1]
    # row 2: fmt=1, acc=0, total = 0.1
    assert abs(brs[2].total - 0.1) < 1e-9, brs[2]
    # row 3: fmt=0, acc=0, total = 0.0
    assert abs(brs[3].total - 0.0) < 1e-9, brs[3]
    print("  test_compute_rewards_weighted_sum: PASS")


def test_compute_rewards_length_mismatch_raises():
    cfg = RewardConfig()
    try:
        compute_rewards(["a", "b"], [1], cfg)
    except ValueError as e:
        assert "len" in str(e)
        print("  test_compute_rewards_length_mismatch_raises: PASS")
        return
    raise AssertionError("expected ValueError")


def test_revised_answer_last_wins_e2e():
    cfg = RewardConfig()
    # Policy emits <answer>3</answer> early then revises to <answer>4</answer>.
    # We score the LAST commitment.
    txt = "<think>2+2... I think 3? <answer>3</answer> wait, actually</think><answer>4</answer>"
    assert format_reward(txt, cfg.format_pattern) == 1.0
    assert accuracy_reward(txt, 4) == 1.0
    assert accuracy_reward(txt, 3) == 0.0
    print("  test_revised_answer_last_wins_e2e: PASS")


if __name__ == "__main__":
    print("--- test_rewards.py ---")
    test_parse_int_basic()
    test_extract_answer_last_tag_wins()
    test_format_reward_schema_check()
    test_accuracy_reward_int_and_string_gt()
    test_compute_rewards_weighted_sum()
    test_compute_rewards_length_mismatch_raises()
    test_revised_answer_last_wins_e2e()
    print("ALL TESTS PASSED")
