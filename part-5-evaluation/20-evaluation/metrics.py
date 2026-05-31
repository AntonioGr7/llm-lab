"""The statistics every eval number needs but almost no leaderboard reports.

A benchmark score is an *estimate*. "62.3%" is shorthand for "we sampled some
problems, scored them, and the sample mean was 0.623." The questions that
actually decide whether your new model is better are statistical:

  - How wide is the error bar on 62.3%? (A 200-problem eval has a ±~7pt
    95% CI. Two models 3pt apart are *indistinguishable*.)
  - Model B beat model A by 1.5pt. Is that real, or resampling noise? The
    answer depends on the *paired* difference, not the two marginals — the
    same problems were scored, so the comparison is paired.
  - The model gets a generation right 1-in-3 tries. What's `pass@10`? You
    can't just run it 10× and eyeball — there's an unbiased estimator
    (Codex, Chen et al. 2021) that uses more samples to estimate fewer.

This module is PURE — numpy + stdlib `math` only, no torch, no scipy. Every
function here is offline-testable in milliseconds, which is the point: the
statistics are the part you most want to trust and least want to get wrong.

Anthropic's "Adding Error Bars to Evals" (Miller, 2024) is the one-paper
version of this module. The headline: report a confidence interval, and for
A-vs-B use the paired/clustered standard error, not two independent ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# =============================================================================
# Confidence intervals for a single score
# =============================================================================

@dataclass
class CI:
    """A point estimate with a confidence interval. `lo`/`hi` are the bounds."""
    point: float
    lo: float
    hi: float
    confidence: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] ({pct}% CI)"


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for a statistic of a sample.

    The bootstrap makes no distributional assumption: it resamples the data
    *with replacement* `n_resamples` times, recomputes the statistic on each
    resample, and reads the empirical percentiles. For a mean of binary
    scores this agrees closely with the normal approximation, but it also
    works for medians, win-rates, F1 — anything.

    Use this for the marginal score of ONE model. For comparing TWO models
    on the same problems, use `paired_bootstrap_diff` (the per-problem
    pairing tightens the interval substantially).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), confidence)
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = statistic_over_axis(arr[idx], statistic)
    alpha = 1.0 - confidence
    lo = float(np.quantile(resampled, alpha / 2.0))
    hi = float(np.quantile(resampled, 1.0 - alpha / 2.0))
    point = float(statistic(arr))
    return CI(point, lo, hi, confidence)


def statistic_over_axis(resamples: np.ndarray, statistic: Callable) -> np.ndarray:
    """Apply `statistic` across axis=1 of a [n_resamples, n] matrix.

    Fast path for the common case `statistic is np.mean` (vectorized);
    otherwise falls back to a python loop so arbitrary callables work.
    """
    if statistic is np.mean:
        return resamples.mean(axis=1)
    if statistic is np.median:
        return np.median(resamples, axis=1)
    return np.array([statistic(row) for row in resamples])


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> CI:
    """Wilson score interval for a binomial proportion.

    The closed-form, no-resampling CI for a proportion (accuracy, win-rate).
    Better than the textbook `p ± z·sqrt(p(1-p)/n)` (Wald) interval at the
    extremes — Wald gives nonsense like [1.0, 1.0] at 100% or negative lower
    bounds near 0%; Wilson stays in [0, 1] and is well-calibrated even for
    small n. Handy as a fast sanity check on a bootstrap mean.
    """
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), confidence)
    z = _z_for_confidence(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return CI(point=p, lo=max(0.0, center - margin), hi=min(1.0, center + margin),
              confidence=confidence)


def _z_for_confidence(confidence: float) -> float:
    """Two-sided normal critical value. Common cases tabulated; else invert."""
    table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054,
             0.99: 2.5758293035489004}
    key = round(confidence, 2)
    if key in table:
        return table[key]
    # Inverse standard-normal CDF via the rational approximation (Acklam).
    return _norm_ppf(1.0 - (1.0 - confidence) / 2.0)


def _norm_ppf(p: float) -> float:
    """Acklam's inverse-normal-CDF approximation (good to ~1e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# =============================================================================
# Comparing two models — the paired difference is what matters
# =============================================================================

@dataclass
class PairedComparison:
    """Result of comparing model A vs B on the SAME items."""
    mean_a: float
    mean_b: float
    diff: float                 # mean_b - mean_a
    diff_ci: CI                 # bootstrap CI on the paired difference
    p_value: float              # two-sided, from the sign/McNemar test on wins
    n: int
    n_b_better: int
    n_a_better: int
    n_tie: int

    @property
    def significant(self) -> bool:
        """True if the 95% CI on the difference excludes zero."""
        return self.diff_ci.lo > 0 or self.diff_ci.hi < 0


def paired_bootstrap_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two models scored on the SAME problems.

    The right way to ask "is B better than A". Because both models saw the
    same items, the per-item scores are *paired* — resample item INDICES
    (not the two arrays independently) so the correlation between A and B on
    each item is preserved. This usually gives a much tighter interval on
    the difference than comparing two independent marginal CIs, because
    problem difficulty (the dominant variance source) cancels.

    The p-value comes from `mcnemar_test` on the win/loss pattern: of the
    items where the two models *disagree*, is the split far enough from
    50/50 to be unlikely under "the models are equally good"?
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal-length arrays, got "
                         f"{a.shape} and {b.shape}")
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    alpha = 1.0 - confidence
    diff_ci = CI(
        point=float(b.mean() - a.mean()),
        lo=float(np.quantile(diffs, alpha / 2.0)),
        hi=float(np.quantile(diffs, 1.0 - alpha / 2.0)),
        confidence=confidence,
    )
    b_better = int(np.sum(b > a))
    a_better = int(np.sum(a > b))
    ties = int(n - b_better - a_better)
    p = mcnemar_test(b_better, a_better)
    return PairedComparison(
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        diff=diff_ci.point, diff_ci=diff_ci, p_value=p,
        n=n, n_b_better=b_better, n_a_better=a_better, n_tie=ties,
    )


def mcnemar_test(n_b_better: int, n_a_better: int) -> float:
    """Exact two-sided McNemar (sign) test on discordant pairs.

    Only the items where the models disagree carry information about which is
    better. Under the null "each discordant item is a 50/50 coin flip", the
    number won by B is Binomial(n_discordant, 0.5). Returns the two-sided
    exact binomial p-value. With zero discordant pairs, p = 1.0 (no evidence).
    """
    n = n_b_better + n_a_better
    if n == 0:
        return 1.0
    k = min(n_b_better, n_a_better)
    # Two-sided: P(X <= k) + P(X >= n-k) = 2 * P(X <= k) for the symmetric case.
    tail = sum(_binom_pmf(i, n, 0.5) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


# =============================================================================
# Sampling-based generative metrics: pass@k / avg@k / maj@k
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, the Codex paper).

    You generated `n` samples for a problem, `c` of them were correct. The
    NAIVE pass@k (generate k, did any pass?) is biased and high-variance for
    small k. The unbiased estimate of "probability that k samples contain at
    least one correct" given the n you actually drew is:

        pass@k = 1 - C(n-c, k) / C(n, k)

    i.e. 1 minus the probability that all k draws miss. This lets you draw
    n=100 once and report pass@1, pass@10, pass@100 from the same samples.
    Requires n >= k. If n - c < k there's no way to avoid a correct one, so
    pass@k = 1.
    """
    if k > n:
        raise ValueError(f"pass@k needs n >= k, got n={n}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def avg_at_k(correct_flags: Sequence[float]) -> float:
    """Mean correctness over k samples (a.k.a. avg@k / mean accuracy).

    The expected fraction of samples that are correct. This is what AIME-style
    "avg@64" reports — it is exactly pass@1 estimated from k samples, and is
    far more stable than a single greedy decode on tiny high-variance sets.
    """
    arr = np.asarray(correct_flags, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def majority_at_k(answers: Sequence, correct_answer) -> bool:
    """maj@k / self-consistency (Wang et al. 2022): is the modal answer right?

    Take the k sampled answers, pick the most frequent (majority vote), and
    check it against ground truth. This is how reasoning models report
    "cons@64" — it usually beats avg@k because errors are diffuse but the
    correct answer is the single most common attractor.
    """
    answers = [a for a in answers if a is not None]
    if not answers:
        return False
    # Most common; ties broken by first-seen order for determinism.
    counts: dict = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], -list(counts).index(kv[0])))
    return best[0] == correct_answer


# =============================================================================
# Quick sample-size intuition
# =============================================================================

def min_n_for_halfwidth(p: float, half_width: float, confidence: float = 0.95) -> int:
    """How many problems to get a CI of at most ±half_width on a proportion p.

    Normal approximation: n >= (z/half_width)^2 · p(1-p). The sobering result:
    a ±2pt CI on a ~60% accuracy needs ~2300 problems. Most public benchmarks
    are far smaller than that — which is exactly why 1-2pt leaderboard gaps
    are usually noise.
    """
    z = _z_for_confidence(confidence)
    return math.ceil((z / half_width) ** 2 * p * (1 - p))


if __name__ == "__main__":
    # Smoke: a 200-problem eval at 62% has a wide interval; B beating A by a
    # hair is not significant; pass@k climbs with k.
    rng = np.random.default_rng(0)
    a = (rng.random(200) < 0.60).astype(float)
    b = a.copy()
    # Make B win on 6 problems A lost, lose on 3 it won -> tiny real edge.
    lost = np.where(a == 0)[0]
    won = np.where(a == 1)[0]
    b[lost[:6]] = 1.0
    b[won[:3]] = 0.0
    ci_a = bootstrap_ci(a)
    print("model A accuracy:", ci_a)
    print("wilson check:   ", wilson_interval(int(a.sum()), a.size))
    cmp = paired_bootstrap_diff(a, b)
    print(f"B-A diff: {cmp.diff:+.3f}  CI [{cmp.diff_ci.lo:+.3f}, {cmp.diff_ci.hi:+.3f}]"
          f"  p={cmp.p_value:.3f}  significant={cmp.significant}")
    print("pass@1/5/10 for c=2,n=10:",
          [round(pass_at_k(10, 2, k), 3) for k in (1, 5, 10)])
    print(f"n needed for +/-2pt CI at p=0.6: {min_n_for_halfwidth(0.6, 0.02)}")
