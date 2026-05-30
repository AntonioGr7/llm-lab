"""Tests for contamination.py — n-gram overlap + canary. PURE, sub-second."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contamination import (
    normalize_for_matching, word_ngrams, build_corpus_ngram_index,
    ngram_overlap, contamination_report, find_canary, overfit_gap,
    BIG_BENCH_CANARY,
)


def test_normalize_and_ngrams():
    toks = normalize_for_matching("The Quick, brown FOX!")
    assert toks == ["the", "quick", "brown", "fox"]
    grams = word_ngrams(toks, 2)
    assert grams == [("the", "quick"), ("quick", "brown"), ("brown", "fox")]
    # Too-short returns empty
    assert word_ngrams(["a"], 2) == []
    print("  test_normalize_and_ngrams: PASS")


def test_verbatim_leak_scores_high_clean_scores_zero():
    corpus = ["the mitochondria is the powerhouse of the cell and makes atp"]
    idx = build_corpus_ngram_index(corpus, n=4)
    leaked = "The mitochondria is the powerhouse of the cell"
    clean = "A train leaves the station heading west at noon"
    assert ngram_overlap(leaked, idx, n=4) == 1.0
    assert ngram_overlap(clean, idx, n=4) == 0.0
    print("  test_verbatim_leak_scores_high_clean_scores_zero: PASS")


def test_short_item_returns_zero_overlap():
    idx = build_corpus_ngram_index(["some long corpus text here for grams"], n=13)
    # Item shorter than n => no n-grams => 0.0 (don't crash, don't false-flag)
    assert ngram_overlap("too short", idx, n=13) == 0.0
    print("  test_short_item_returns_zero_overlap: PASS")


def test_partial_overlap_is_fractional():
    corpus = ["alpha beta gamma delta epsilon"]
    idx = build_corpus_ngram_index(corpus, n=2)
    # "alpha beta" present, "beta zeta" absent => 1 of 2 bigrams => 0.5
    item = "alpha beta zeta"
    ov = ngram_overlap(item, idx, n=2)
    assert abs(ov - 0.5) < 1e-9
    print(f"  test_partial_overlap_is_fractional: PASS (ov={ov})")


def test_contamination_report_flags_above_threshold():
    corpus = ["janet has sixteen eggs per day she sells them at the market"]
    tests = [
        "janet has sixteen eggs per day she sells them at the market",  # leak
        "a completely unrelated novel sentence about astronomy and stars",  # clean
    ]
    rep = contamination_report(tests, corpus, n=4, threshold=0.5)
    assert 0 in rep.flagged_indices
    assert 1 not in rep.flagged_indices
    assert abs(rep.contamination_rate - 0.5) < 1e-9
    assert rep.overlaps[0] > rep.overlaps[1]
    assert "flagged" in rep.summary()
    print("  test_contamination_report_flags_above_threshold: PASS")


def test_canary_detection_zero_false_positive():
    corpus = ["clean doc one", f"leaked benchmark file {BIG_BENCH_CANARY} here",
              "clean doc two"]
    hits = find_canary(corpus)
    assert hits == [1]
    # Custom canary that appears nowhere => no hits
    assert find_canary(corpus, canary="nonexistent-guid-xyz") == []
    print("  test_canary_detection_zero_false_positive: PASS")


def test_overfit_gap():
    assert abs(overfit_gap(0.85, 0.72) - 0.13) < 1e-9
    print("  test_overfit_gap: PASS")


if __name__ == "__main__":
    test_normalize_and_ngrams()
    test_verbatim_leak_scores_high_clean_scores_zero()
    test_short_item_returns_zero_overlap()
    test_partial_overlap_is_fractional()
    test_contamination_report_flags_above_threshold()
    test_canary_detection_zero_false_positive()
    test_overfit_gap()
    print("\nall contamination tests PASS")
