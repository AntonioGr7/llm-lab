"""Tests for judge.py — parsing, position-swap debiasing, length bias, kappa.

Uses small deterministic judges (no model) so the resolution logic is pinned
exactly. The whole point of the module is that these biases are CAUGHT, so the
tests construct judges that exhibit each bias on purpose and assert it shows up
in the right diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from judge import (
    DummyJudge, parse_winner, parse_score, pairwise_judge, aggregate_win_rate,
    agreement, PAIRWISE_TEMPLATE, _extract_ab_from_prompt,
)


# --- deterministic test judges --------------------------------------------

class AlwaysFirstJudge:
    """Pure position bias: always picks whichever answer is shown FIRST (A)."""
    def __call__(self, prompt: str) -> str:
        return json.dumps({"winner": "A", "rationale": "first"})


class KeywordJudge:
    """Content-based: picks the answer containing the keyword (no position bias)."""
    def __init__(self, kw):
        self.kw = kw
    def __call__(self, prompt: str) -> str:
        a, b = _extract_ab_from_prompt(prompt)
        if self.kw in a and self.kw not in b:
            return json.dumps({"winner": "A"})
        if self.kw in b and self.kw not in a:
            return json.dumps({"winner": "B"})
        return json.dumps({"winner": "tie"})


def test_parse_winner_robust():
    assert parse_winner('{"winner": "A"}') == "A"
    assert parse_winner('```json\n{"winner": "B"}\n```') == "B"
    assert parse_winner('{"winner": "tie"}') == "tie"
    assert parse_winner("garbage no json") == "tie"      # unparseable => tie
    assert parse_winner('here is my answer: B') == "B"    # bare-letter fallback
    print("  test_parse_winner_robust: PASS")


def test_parse_score_clamps():
    assert parse_score('{"score": 7}') == 7
    assert parse_score('{"score": 99}') == 10            # clamp high
    assert parse_score('{"score": 0}') == 1             # clamp low
    assert parse_score("I'd give it a 5") == 5
    assert parse_score("no number here") is None
    print("  test_parse_score_clamps: PASS")


def test_extract_ab_roundtrip():
    p = PAIRWISE_TEMPLATE.format(question="q?", answer_a="ALPHA", answer_b="BETA")
    a, b = _extract_ab_from_prompt(p)
    assert a == "ALPHA" and b == "BETA"
    print("  test_extract_ab_roundtrip: PASS")


def test_position_bias_caught_by_swap():
    # AlwaysFirst picks the first-shown answer both orders => hard disagreement
    # => resolved to tie + position_bias flag.
    judge = AlwaysFirstJudge()
    v = pairwise_judge(judge, "q?", "answer one", "answer two", swap=True)
    assert v.winner == "tie"
    assert v.position_bias is True
    print("  test_position_bias_caught_by_swap: PASS")


def test_content_judge_survives_swap():
    # KeywordJudge is order-independent: a consistent winner survives the swap
    # with no position-bias flag.
    judge = KeywordJudge("RIGHT")
    v = pairwise_judge(judge, "q?", "this is wrong", "this is RIGHT", swap=True)
    assert v.winner == "B"
    assert v.position_bias is False
    print("  test_content_judge_survives_swap: PASS")


def test_no_swap_takes_forward_verdict():
    judge = AlwaysFirstJudge()
    v = pairwise_judge(judge, "q?", "a", "b", swap=False)
    assert v.winner == "A" and v.position_bias is False
    print("  test_no_swap_takes_forward_verdict: PASS")


def test_length_bias_shows_in_correlation():
    # DummyJudge(rule="longer") prefers the longer answer regardless of order
    # (content-ish, not positional). We MIX which side is longer so the win
    # pattern has variance: where B is longer B wins, where A is longer A wins.
    # The "B won" indicator then correlates positively with (len_b - len_a).
    judge = DummyJudge(rule="longer")
    pairs = [("q1", "short", "a much longer answer here"),          # B longer -> B wins
             ("q2", "a considerably more verbose response", "tiny"),# A longer -> A wins
             ("q3", "no", "yes indeed absolutely certainly so"),    # B longer -> B wins
             ("q4", "definitely the most elaborate reply", "yep")]  # A longer -> A wins
    verdicts = [pairwise_judge(judge, q, a, b) for q, a, b in pairs]
    la = [len(a) for _, a, _ in pairs]
    lb = [len(b) for _, _, b in pairs]
    res = aggregate_win_rate(verdicts, la, lb)
    assert res.win_rate == 0.5                # half each, since longer side alternates
    assert res.length_win_corr > 0.5          # winning tracks being-longer
    assert res.position_bias_rate == 0.0      # not a position artifact
    print(f"  test_length_bias_shows_in_correlation: PASS "
          f"(win={res.win_rate}, len_corr={res.length_win_corr:+.2f})")


def test_win_rate_counts_ties_as_half():
    judge = AlwaysFirstJudge()   # everything resolves to tie under swap
    verdicts = [pairwise_judge(judge, "q", "a", "b") for _ in range(4)]
    res = aggregate_win_rate(verdicts)
    assert res.win_rate == 0.5
    assert res.ties == 4
    print("  test_win_rate_counts_ties_as_half: PASS")


def test_agreement_kappa():
    # Perfect agreement => kappa 1.
    ag = agreement(["A", "B", "tie"], ["A", "B", "tie"])
    assert abs(ag.cohen_kappa - 1.0) < 1e-9 and ag.trustworthy
    # A judge that always says "B" on an 80/20 B-skewed gold set: high raw
    # agreement but kappa ~0 (no skill beyond the base rate).
    judge = ["B"] * 10
    human = ["B"] * 8 + ["A"] * 2
    ag2 = agreement(judge, human)
    assert ag2.raw_agreement == 0.8
    assert ag2.cohen_kappa < 0.1 and not ag2.trustworthy
    print(f"  test_agreement_kappa: PASS (skewed raw={ag2.raw_agreement}, "
          f"kappa={ag2.cohen_kappa:.2f})")


def test_agreement_length_mismatch_raises():
    try:
        agreement(["A"], ["A", "B"])
    except ValueError:
        print("  test_agreement_length_mismatch_raises: PASS")
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_parse_winner_robust()
    test_parse_score_clamps()
    test_extract_ab_roundtrip()
    test_position_bias_caught_by_swap()
    test_content_judge_survives_swap()
    test_no_swap_takes_forward_verdict()
    test_length_bias_shows_in_correlation()
    test_win_rate_counts_ties_as_half()
    test_agreement_kappa()
    test_agreement_length_mismatch_raises()
    print("\nall judge tests PASS")
