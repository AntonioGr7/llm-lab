"""Tests for benchmarks.py — MC scoring/format, generative extraction, IFEval.

PURE (mock logprobs for MC, no model). The headline lesson — that the
normalization and prompt format change the answer — is asserted directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import (
    MCExample, IFExample, build_mc_prompt, score_mc, mc_accuracy,
    parse_gsm8k_ground_truth, extract_final_int, normalize_text_answer,
    exact_match, token_f1, check_constraints, CONSTRAINT_CHECKERS,
)


# --- Multiple choice -------------------------------------------------------

def test_mc_normalization_changes_the_answer():
    # Option 0 is short (1 tok, 3 bytes), option 1 is long (3 tok, 15 bytes).
    # raw favors the short option; per-token & per-byte favor the long one.
    lp = [-2.0, -3.6]
    toks = [1, 3]
    byts = [3, 15]
    assert score_mc(lp, toks, byts, "raw") == 0       # -2.0 > -3.6
    assert score_mc(lp, toks, byts, "token") == 1     # -2.0 vs -1.2
    assert score_mc(lp, toks, byts, "byte") == 1
    print("  test_mc_normalization_changes_the_answer: PASS")


def test_mc_prompt_format_styles():
    ex = MCExample("What color is the sky?", ["Red", "Blue", "Green", "Black"], 1)
    p_letters, conts_letters = build_mc_prompt(ex, style="letters")
    assert "A. Red" in p_letters and "B. Blue" in p_letters
    assert p_letters.rstrip().endswith("Answer:")
    assert conts_letters == [" A", " B", " C", " D"]
    p_cloze, conts_cloze = build_mc_prompt(ex, style="cloze")
    assert "A. Red" not in p_cloze            # no letters in cloze
    assert conts_cloze == [" Red", " Blue", " Green", " Black"]
    print("  test_mc_prompt_format_styles: PASS")


def test_mc_accuracy_flags():
    exs = [MCExample("q", ["a", "b"], 1), MCExample("q", ["a", "b"], 0)]
    flags = mc_accuracy([1, 1], exs)            # pred 1,1 ; gt 1,0
    assert flags == [1.0, 0.0]
    print("  test_mc_accuracy_flags: PASS")


def test_unknown_style_and_norm_raise():
    ex = MCExample("q", ["a", "b"], 0)
    for bad in ("foo",):
        try:
            build_mc_prompt(ex, style=bad); raise AssertionError
        except ValueError:
            pass
        try:
            score_mc([-1.0, -2.0], [1, 1], [1, 1], bad); raise AssertionError
        except ValueError:
            pass
    print("  test_unknown_style_and_norm_raise: PASS")


# --- Generative + verifier -------------------------------------------------

def test_gsm8k_ground_truth_parse():
    assert parse_gsm8k_ground_truth("steps\n#### 42") == 42
    assert parse_gsm8k_ground_truth("#### 1,234") == 1234
    assert parse_gsm8k_ground_truth("no hash") is None
    print("  test_gsm8k_ground_truth_parse: PASS")


def test_extract_final_int_prefers_boxed_then_last():
    assert extract_final_int("blah \\boxed{42} blah 99") == 42
    assert extract_final_int("first 7 then 13 then 21") == 21
    assert extract_final_int("the cost is $1,250 total") == 1250
    assert extract_final_int("no digits at all") is None
    assert extract_final_int("value 3.5 here") is None   # non-integer float rejects
    assert extract_final_int("the answer is 12.0") == 12  # integer-valued float ok
    print("  test_extract_final_int_prefers_boxed_then_last: PASS")


def test_text_normalization_and_match():
    assert normalize_text_answer("The Paris!") == "paris"
    assert exact_match("the  Paris.", "Paris") == 1.0
    assert exact_match("London", "Paris") == 0.0
    f1 = token_f1("the cat sat", "a cat sat down")
    assert 0.0 < f1 < 1.0
    assert token_f1("paris", "paris") == 1.0
    print(f"  test_text_normalization_and_match: PASS (f1={f1:.2f})")


# --- IFEval constraint checkers -------------------------------------------

def test_constraint_checkers_individually():
    assert CONSTRAINT_CHECKERS["json"]('{"a": 1}', None) is True
    assert CONSTRAINT_CHECKERS["json"]("not json", None) is False
    assert CONSTRAINT_CHECKERS["exact_bullets"]("- one\n- two\n- three", 3) is True
    assert CONSTRAINT_CHECKERS["exact_bullets"]("- one\n- two", 3) is False
    assert CONSTRAINT_CHECKERS["min_words"]("a b c d e", 5) is True
    assert CONSTRAINT_CHECKERS["max_words"]("a b c", 3) is True
    assert CONSTRAINT_CHECKERS["no_commas"]("clean text") is True
    assert CONSTRAINT_CHECKERS["no_commas"]("has, comma") is False
    assert CONSTRAINT_CHECKERS["keyword_present"]("see the horizon", "horizon") is True
    assert CONSTRAINT_CHECKERS["keyword_absent"]("clean", "{") is True
    assert CONSTRAINT_CHECKERS["all_caps"]("HELLO WORLD") is True
    assert CONSTRAINT_CHECKERS["all_caps"]("Hello") is False
    print("  test_constraint_checkers_individually: PASS")


def test_check_constraints_strict_and_loose():
    # Both satisfied => strict True, loose 1.0
    r = check_constraints('{"x":1}', [("json", None), ("max_words", 5)])
    assert r["prompt_level"] is True and r["instruction_level"] == 1.0
    # One of two satisfied => strict False, loose 0.5
    r2 = check_constraints("plain text many many many many words here now",
                           [("json", None), ("min_words", 3)])
    assert r2["prompt_level"] is False and abs(r2["instruction_level"] - 0.5) < 1e-9
    print("  test_check_constraints_strict_and_loose: PASS")


if __name__ == "__main__":
    test_mc_normalization_changes_the_answer()
    test_mc_prompt_format_styles()
    test_mc_accuracy_flags()
    test_unknown_style_and_norm_raise()
    test_gsm8k_ground_truth_parse()
    test_extract_final_int_prefers_boxed_then_last()
    test_text_normalization_and_match()
    test_constraint_checkers_individually()
    test_check_constraints_strict_and_loose()
    print("\nall benchmark tests PASS")
