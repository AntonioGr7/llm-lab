"""Tests for metrics.py — the statistics. PURE, no torch, sub-second.

The statistics are the part you most want to trust, so they get the most
adversarial tests: CI coverage by simulation, paired-vs-unpaired tightness,
the pass@k estimator against its closed form, and the sample-size sanity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import numpy as np

from metrics import (
    bootstrap_ci, wilson_interval, paired_bootstrap_diff, mcnemar_test,
    pass_at_k, avg_at_k, majority_at_k, min_n_for_halfwidth, CI,
)


def test_bootstrap_ci_brackets_point_and_is_ordered():
    vals = [1.0] * 60 + [0.0] * 40   # mean 0.6
    ci = bootstrap_ci(vals, seed=0)
    assert abs(ci.point - 0.6) < 1e-9
    assert ci.lo < ci.point < ci.hi
    assert 0.0 <= ci.lo and ci.hi <= 1.0
    print("  test_bootstrap_ci_brackets_point_and_is_ordered: PASS")


def test_bootstrap_ci_coverage_by_simulation():
    # Over many resampled datasets, the 95% CI should contain the true mean
    # ~95% of the time. We use a relaxed band [0.90, 0.99] to keep it fast/stable.
    rng = np.random.default_rng(0)
    true_p = 0.5
    n = 200
    covered = 0
    trials = 200
    for t in range(trials):
        data = (rng.random(n) < true_p).astype(float)
        ci = bootstrap_ci(data, n_resamples=1000, seed=t)
        if ci.lo <= true_p <= ci.hi:
            covered += 1
    rate = covered / trials
    assert 0.90 <= rate <= 1.0, f"coverage {rate} out of band"
    print(f"  test_bootstrap_ci_coverage_by_simulation: PASS (coverage={rate:.2f})")


def test_wilson_stays_in_unit_interval_at_extremes():
    # Wald would give [1,1] at 100%; Wilson gives a sensible (<1) lower bound.
    ci = wilson_interval(50, 50)
    assert ci.lo < 1.0 and ci.hi <= 1.0 and ci.point == 1.0
    ci0 = wilson_interval(0, 50)
    assert ci0.lo >= 0.0 and ci0.hi > 0.0
    print("  test_wilson_stays_in_unit_interval_at_extremes: PASS")


def test_paired_is_tighter_than_unpaired_when_correlated():
    # Two highly-correlated models: B = A with a few flips. The paired diff CI
    # should be MUCH tighter than the gap between the two marginal CIs.
    rng = np.random.default_rng(1)
    a = (rng.random(300) < 0.6).astype(float)
    b = a.copy()
    flip = rng.choice(300, size=12, replace=False)
    b[flip] = 1.0 - b[flip]
    cmp = paired_bootstrap_diff(a, b, seed=0)
    paired_width = cmp.diff_ci.hi - cmp.diff_ci.lo
    ci_a, ci_b = bootstrap_ci(a, seed=0), bootstrap_ci(b, seed=0)
    unpaired_width = (ci_a.half_width + ci_b.half_width) * 2
    assert paired_width < unpaired_width
    print(f"  test_paired_is_tighter_than_unpaired_when_correlated: PASS "
          f"(paired={paired_width:.3f} < unpaired~{unpaired_width:.3f})")


def test_paired_requires_equal_length():
    try:
        paired_bootstrap_diff([1.0, 0.0], [1.0])
    except ValueError:
        print("  test_paired_requires_equal_length: PASS")
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_mcnemar_symmetric_and_significant():
    assert mcnemar_test(0, 0) == 1.0                 # no discordant pairs
    assert mcnemar_test(5, 5) == 1.0                 # even split => p=1
    # 18 vs 2 discordant: clearly significant.
    p = mcnemar_test(18, 2)
    assert p < 0.05
    # symmetry
    assert abs(mcnemar_test(18, 2) - mcnemar_test(2, 18)) < 1e-12
    print(f"  test_mcnemar_symmetric_and_significant: PASS (p(18,2)={p:.4f})")


def test_mcnemar_matches_hand_computation():
    # n=4 discordant, k=1 won by the loser: two-sided exact binomial.
    # P(X<=1) for Bin(4,0.5) = (C(4,0)+C(4,1))/16 = 5/16; two-sided = 10/16.
    p = mcnemar_test(3, 1)
    assert abs(p - (10 / 16)) < 1e-12
    print("  test_mcnemar_matches_hand_computation: PASS")


def test_pass_at_k_closed_form_and_edges():
    # c=0 => never pass. c=n => always. Monotone increasing in k.
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    p1 = pass_at_k(10, 2, 1)
    p5 = pass_at_k(10, 2, 5)
    p10 = pass_at_k(10, 2, 10)
    assert p1 < p5 < p10 == 1.0
    # pass@1 with c correct of n equals c/n exactly.
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-12
    # closed form for n=4,c=1,k=2: 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-12
    print("  test_pass_at_k_closed_form_and_edges: PASS")


def test_pass_at_k_requires_n_ge_k():
    try:
        pass_at_k(3, 1, 5)
    except ValueError:
        print("  test_pass_at_k_requires_n_ge_k: PASS")
        return
    raise AssertionError("expected ValueError when k > n")


def test_avg_and_majority_at_k():
    assert abs(avg_at_k([1.0, 0.0, 1.0, 1.0]) - 0.75) < 1e-12
    # majority: 42 appears 3x, 7 appears 2x => modal 42 is correct
    assert majority_at_k([42, 42, 7, 42, 7], 42) is True
    # modal answer wrong even though a correct one is present
    assert majority_at_k([7, 7, 42], 42) is False
    assert majority_at_k([None, None], 1) is False
    print("  test_avg_and_majority_at_k: PASS")


def test_min_n_for_halfwidth_is_large():
    # The sobering fact: a +-2pt CI at p=0.6 needs >2000 problems.
    n = min_n_for_halfwidth(0.6, 0.02)
    assert n > 2000
    # Wider tolerance needs fewer.
    assert min_n_for_halfwidth(0.6, 0.07) < n
    print(f"  test_min_n_for_halfwidth_is_large: PASS (n={n} for +-2pt)")


if __name__ == "__main__":
    test_bootstrap_ci_brackets_point_and_is_ordered()
    test_bootstrap_ci_coverage_by_simulation()
    test_wilson_stays_in_unit_interval_at_extremes()
    test_paired_is_tighter_than_unpaired_when_correlated()
    test_paired_requires_equal_length()
    test_mcnemar_symmetric_and_significant()
    test_mcnemar_matches_hand_computation()
    test_pass_at_k_closed_form_and_edges()
    test_pass_at_k_requires_n_ge_k()
    test_avg_and_majority_at_k()
    test_min_n_for_halfwidth_is_large()
    print("\nall metrics tests PASS")
