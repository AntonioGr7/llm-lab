"""Offline sanity tests for the Module 22 scaling calculator.

Run: python tests/test_scaling.py   (from the module directory)
No GPU, no network — pure arithmetic.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaling import (  # noqa: E402
    training_flops, tokens_for_flops, chinchilla_optimal, CHINCHILLA,
    ScalingLaw, cost_and_time, extrapolate, overtrain_savings, LADDER,
    TOKENS_PER_PARAM, FLOPS_PER_PARAM_TOKEN,
)


def approx(a, b, rel=0.02):
    return abs(a - b) <= rel * abs(b)


def test_flops_rule():
    # 6ND, and round-trip via tokens_for_flops.
    assert training_flops(1e9, 1e9) == 6e18
    c = training_flops(7e10, 1.4e12)
    assert approx(c, 5.88e23)
    assert approx(tokens_for_flops(7e10, c), 1.4e12)


def test_chinchilla_reproduces_canonical_point():
    # Feeding Chinchilla's compute should return ~70B params / ~1.4T tokens.
    a = chinchilla_optimal(5.88e23)
    assert approx(a.n_params, 7e10, rel=0.03), a.n_params
    assert approx(a.n_tokens, 1.4e12, rel=0.03), a.n_tokens
    assert approx(a.tokens_per_param, 20.0)


def test_optimal_ratio_is_20_to_1():
    for c in [1e18, 1e21, 1e24, 1e26]:
        a = chinchilla_optimal(c)
        assert approx(a.n_tokens / a.n_params, TOKENS_PER_PARAM)
        # And the allocation actually spends the budget.
        assert approx(training_flops(a.n_params, a.n_tokens), c)


def test_optimal_grows_as_sqrt():
    # 100x compute -> ~10x params and ~10x tokens (both ∝ √C).
    a = chinchilla_optimal(1e22)
    b = chinchilla_optimal(1e24)
    assert approx(b.n_params / a.n_params, 10.0, rel=0.01)
    assert approx(b.n_tokens / a.n_tokens, 10.0, rel=0.01)


def test_chinchilla_optimal_rejects_nonpositive():
    for bad in [0, -1e20]:
        try:
            chinchilla_optimal(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for flops <= 0")


def test_loss_above_irreducible_and_monotonic():
    law = CHINCHILLA
    # Loss is always above the irreducible floor.
    assert law.loss(1e9, 1e11) > law.E
    # More params (fixed data) lowers loss; more data (fixed params) lowers loss.
    assert law.loss(1e10, 1e11) < law.loss(1e9, 1e11)
    assert law.loss(1e9, 1e12) < law.loss(1e9, 1e11)
    # As both -> infinity, loss -> E.
    assert approx(law.loss(1e30, 1e30), law.E, rel=1e-3)


def test_loss_at_optimal_decreases_with_compute():
    prev = math.inf
    for c in [1e18, 1e20, 1e22, 1e24, 1e26]:
        l = CHINCHILLA.loss_at_optimal(c)
        assert l < prev, (c, l, prev)
        assert l > CHINCHILLA.E
        prev = l


def test_custom_scaling_law_constants():
    law = ScalingLaw(E=2.0, A=100.0, alpha=0.5, B=100.0, beta=0.5)
    # With symmetric constants, loss(N, D) == loss(D, N).
    assert approx(law.loss(1e6, 1e8), law.loss(1e8, 1e6))


def test_cost_and_time():
    est = cost_and_time(1e24, gpu="H100", n_gpus=1000, mfu=0.5)
    # Realized = 990e12 * 0.5 = 4.95e14 FLOP/s/gpu.
    expected_gpu_seconds = 1e24 / (990e12 * 0.5)
    assert approx(est.gpu_hours, expected_gpu_seconds / 3600)
    # Wall-clock is gpu-hours / n_gpus.
    assert approx(est.wall_clock_hours, est.gpu_hours / 1000)
    # Cost = gpu_hours * price.
    assert approx(est.cost_usd, est.gpu_hours * 2.5)
    # More GPUs -> same cost, less wall-clock.
    est2 = cost_and_time(1e24, gpu="H100", n_gpus=2000, mfu=0.5)
    assert approx(est2.cost_usd, est.cost_usd)
    assert approx(est2.wall_clock_hours, est.wall_clock_hours / 2)


def test_cost_validation():
    for kw in [{"gpu": "nope"}, {"mfu": 0.0}, {"mfu": 1.5}, {"n_gpus": 0}]:
        try:
            cost_and_time(1e20, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kw}")


def test_extrapolate():
    demo = LADDER[0]  # course demo
    a = extrapolate(demo, 1e6)
    # Compute scaled by 1e6 -> params/tokens scaled by 1e3 (√).
    base = chinchilla_optimal(demo.flops)
    assert approx(a.n_params / base.n_params, 1000.0, rel=0.01)


def test_overtrain_trade():
    # A small model CAN reach a bigger model's compute-optimal loss with more data,
    # at extra training cost but cheaper inference.
    s = overtrain_savings(7e10, 8e9)
    assert s["reachable"]
    assert s["tokens_small"] > s["tokens_big"]          # needs more data
    assert s["extra_train_cost_x"] > 1.0                # pays more to train
    assert s["inference_flops_ratio"] < 1.0             # cheaper to serve
    # A model far too small to ever reach the target is flagged unreachable.
    tiny = overtrain_savings(7e10, 1e6)
    assert tiny["reachable"] is False


def test_ladder_flops_consistent():
    for p in LADDER:
        assert approx(p.flops, FLOPS_PER_PARAM_TOKEN * p.n_params * p.n_tokens)
        assert approx(p.tokens_per_param, p.n_tokens / p.n_params)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
