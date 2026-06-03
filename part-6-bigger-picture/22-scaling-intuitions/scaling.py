"""
scaling.py — a small, honest scaling-law calculator for Module 22.

Module 22 is the conceptual capstone of the course. It has no training code.
This module exists so the notebook can let you *play* with the numbers that
govern every frontier run — and so you can answer, for any compute budget, the
three questions a lab asks before it spends a dollar:

  1) How big a model, and how much data?     -> chinchilla_optimal(C)
  2) How good will it be?                     -> chinchilla_loss(N, D)
  3) What will it cost, and how long?         -> cost_and_time(flops, ...)

Everything here is pure Python + math (no torch, no network). It runs instantly
on CPU. The point is not precision to three decimals — the published constants
are themselves debated (see the replication note in the README). The point is to
build the *intuition* that lets you look at a $50M run and know, roughly, what
shape it has, and to see that the same arithmetic governs your $50 run.

References:
  Kaplan et al. 2020, "Scaling Laws for Neural Language Models" (the 6ND rule,
    the original power laws — which over-weighted model size).
  Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models"
    (Chinchilla — the ~20-tokens-per-parameter correction; constants below).
  Besiroglu et al. 2024, "Chinchilla Scaling: A replication attempt"
    (why you should not trust the parametric constants to two digits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# The one equation everyone uses: training FLOPs ~= 6 * N * D
# ---------------------------------------------------------------------------
#
# N = non-embedding parameters, D = training tokens. The factor of 6 is
# 2 (multiply-add) x 3 (one forward + two backward passes' worth of matmuls).
# It is astonishingly robust across architectures — dense or MoE-with-N-active,
# it holds. Memorize it. It is the single most useful number in the field.

FLOPS_PER_PARAM_TOKEN = 6.0


def training_flops(n_params: float, n_tokens: float) -> float:
    """Total training FLOPs for a dense model. C = 6 * N * D."""
    return FLOPS_PER_PARAM_TOKEN * n_params * n_tokens


def tokens_for_flops(n_params: float, flops: float) -> float:
    """Given a parameter count and a compute budget, how many tokens fit."""
    return flops / (FLOPS_PER_PARAM_TOKEN * n_params)


# ---------------------------------------------------------------------------
# Compute-optimal allocation (Chinchilla)
# ---------------------------------------------------------------------------
#
# Given a fixed compute budget C, how do you split it between a bigger model
# (more N) and more data (more D)? Chinchilla's headline answer: at the optimum
# you want roughly TOKENS_PER_PARAM tokens per parameter, and that ratio is
# ~20 near the scales they studied. With D = r*N and C = 6*N*D = 6*r*N^2:
#
#     N_opt = sqrt(C / (6 * r))      D_opt = r * N_opt
#
# This reproduces Chinchilla itself (70B params, 1.4T tokens, C ~= 5.9e23).

TOKENS_PER_PARAM = 20.0


@dataclass
class Allocation:
    """A compute-optimal allocation for a given budget."""

    flops: float          # total training compute, C
    n_params: float       # optimal model size, N
    n_tokens: float       # optimal token count, D
    tokens_per_param: float

    def __str__(self) -> str:
        return (
            f"C={_si(self.flops)}FLOPs  ->  "
            f"N={_si(self.n_params)}params, "
            f"D={_si(self.n_tokens)}tokens "
            f"({self.tokens_per_param:.0f} tok/param)"
        )


def chinchilla_optimal(flops: float, tokens_per_param: float = TOKENS_PER_PARAM) -> Allocation:
    """Compute-optimal (N, D) for a training budget of `flops`.

    Uses the robust ~20-tokens-per-parameter rule rather than the (debated)
    parametric power-law exponents. This is the version labs reason with on a
    whiteboard, and it is self-consistent with Chinchilla's own 70B/1.4T point.
    """
    if flops <= 0:
        raise ValueError("flops must be positive")
    n = math.sqrt(flops / (FLOPS_PER_PARAM_TOKEN * tokens_per_param))
    d = tokens_per_param * n
    return Allocation(flops=flops, n_params=n, n_tokens=d, tokens_per_param=tokens_per_param)


# ---------------------------------------------------------------------------
# Predicting loss: the parametric scaling law
# ---------------------------------------------------------------------------
#
# Hoffmann et al.'s "approach 3" fits a single surface to loss as a function of
# N and D:
#
#     L(N, D) = E + A / N^alpha + B / D^beta
#
# E is the irreducible loss (entropy of natural language, ~the Bayes error).
# The A/N^alpha term is the cost of a finite model; B/D^beta the cost of finite
# data. These constants are the *published* Chinchilla values. Treat the
# ABSOLUTE numbers with suspicion (the replication attempt found the paper's
# fit internally inconsistent) — but the SHAPE is right, and relative
# predictions ("how much does loss drop if I 100x compute?") are robust.

@dataclass
class ScalingLaw:
    E: float = 1.69       # irreducible loss (nats/token)
    A: float = 406.4
    alpha: float = 0.34
    B: float = 410.7
    beta: float = 0.28

    def loss(self, n_params: float, n_tokens: float) -> float:
        """Predicted cross-entropy loss (nats/token) for a model of size N
        trained on D tokens."""
        return self.E + self.A / (n_params ** self.alpha) + self.B / (n_tokens ** self.beta)

    def loss_at_optimal(self, flops: float, tokens_per_param: float = TOKENS_PER_PARAM) -> float:
        """Predicted loss if you spend `flops` compute-optimally."""
        a = chinchilla_optimal(flops, tokens_per_param)
        return self.loss(a.n_params, a.n_tokens)


CHINCHILLA = ScalingLaw()


# ---------------------------------------------------------------------------
# Cost and wall-clock
# ---------------------------------------------------------------------------
#
# FLOPs are free to compute on paper; the bill comes from how fast your
# hardware actually runs and how much you pay per hour. The bridge is MFU —
# Model FLOPs Utilization — the fraction of a GPU's peak you actually realize.
# Real large runs land around 0.3-0.5; toy runs much lower.

# Peak dense throughput, in FLOP/s, for common accelerators (bf16, no sparsity).
GPU_PEAK_FLOPS = {
    "A100": 312e12,    # A100-80GB SXM, bf16
    "H100": 990e12,    # H100 SXM, bf16
    "H200": 990e12,    # same compute as H100, more memory
    "B200": 2250e12,   # Blackwell, bf16 (approx)
    "4090": 165e12,    # consumer, bf16
}

# Rough on-demand rental price, USD/hour/GPU (varies wildly by provider/time).
GPU_PRICE_PER_HOUR = {
    "A100": 1.5,
    "H100": 2.5,
    "H200": 3.0,
    "B200": 5.0,
    "4090": 0.4,
}


@dataclass
class RunEstimate:
    flops: float
    gpu: str
    n_gpus: int
    mfu: float
    gpu_hours: float
    wall_clock_hours: float
    cost_usd: float

    def __str__(self) -> str:
        return (
            f"{_si(self.flops)}FLOPs on {self.n_gpus}x{self.gpu} @ MFU {self.mfu:.0%}: "
            f"{self.gpu_hours:,.0f} GPU-hours, "
            f"{self.wall_clock_hours:,.1f}h wall-clock, "
            f"${self.cost_usd:,.0f}"
        )


def cost_and_time(
    flops: float,
    gpu: str = "H100",
    n_gpus: int = 1,
    mfu: float = 0.4,
    price_per_hour: float | None = None,
) -> RunEstimate:
    """Estimate GPU-hours, wall-clock, and dollar cost to spend `flops`."""
    if gpu not in GPU_PEAK_FLOPS:
        raise ValueError(f"unknown gpu {gpu!r}; known: {sorted(GPU_PEAK_FLOPS)}")
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in (0, 1]")
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1")
    realized_per_gpu = GPU_PEAK_FLOPS[gpu] * mfu
    gpu_seconds = flops / realized_per_gpu
    gpu_hours = gpu_seconds / 3600.0
    wall_clock_hours = gpu_hours / n_gpus
    price = price_per_hour if price_per_hour is not None else GPU_PRICE_PER_HOUR[gpu]
    cost = gpu_hours * price
    return RunEstimate(
        flops=flops, gpu=gpu, n_gpus=n_gpus, mfu=mfu,
        gpu_hours=gpu_hours, wall_clock_hours=wall_clock_hours, cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# Extrapolation: from your $50 run to a $50M run
# ---------------------------------------------------------------------------

@dataclass
class ScalePoint:
    """A named point on the scaling ladder."""

    name: str
    n_params: float
    n_tokens: float

    @property
    def flops(self) -> float:
        return training_flops(self.n_params, self.n_tokens)

    @property
    def tokens_per_param(self) -> float:
        return self.n_tokens / self.n_params


# A ladder of real (approximate, public) points, for the notebook's "same
# arithmetic, 12 orders of magnitude apart" table.
LADDER = [
    ScalePoint("course demo (Part 3)", 1.5e8, 3e9),      # ~150M params, ~3B tokens
    ScalePoint("Chinchilla (70B)", 7e10, 1.4e12),
    ScalePoint("Llama 3 8B", 8e9, 15e12),                # famously over-trained
    ScalePoint("Llama 3 70B", 7e10, 15e12),
    ScalePoint("GPT-3 (175B)", 1.75e11, 3e11),           # famously under-trained
    ScalePoint("frontier ~2024 (est.)", 1e12, 15e12),
]


def extrapolate(
    base: ScalePoint,
    flops_multiplier: float,
    tokens_per_param: float = TOKENS_PER_PARAM,
) -> Allocation:
    """If you scaled `base`'s compute by `flops_multiplier` and spent it
    compute-optimally, what model + data would you train?"""
    return chinchilla_optimal(base.flops * flops_multiplier, tokens_per_param)


def overtrain_savings(
    n_params_optimal: float,
    n_params_smaller: float,
    target_loss_law: ScalingLaw = CHINCHILLA,
    tokens_per_param: float = TOKENS_PER_PARAM,
) -> dict:
    """The Llama trade: deliberately train a SMALLER model PAST its
    compute-optimal point to reach the same loss, paying more training compute
    to save *inference* compute forever after.

    Returns the extra training tokens/FLOPs the smaller model needs to match
    the bigger model's compute-optimal loss, and the per-token inference saving.
    """
    # Bigger model, trained compute-optimally.
    d_big = tokens_per_param * n_params_optimal
    target = target_loss_law.loss(n_params_optimal, d_big)
    # Smaller model: how many tokens to hit the same loss? Invert the data term.
    # target = E + A/N_s^alpha + B/D_s^beta  ->  solve for D_s.
    residual = target - target_loss_law.E - target_loss_law.A / (n_params_smaller ** target_loss_law.alpha)
    if residual <= 0:
        # Smaller model can't reach the target loss at any data budget.
        return {
            "reachable": False,
            "target_loss": target,
            "inference_flops_ratio": n_params_smaller / n_params_optimal,
        }
    d_small = (target_loss_law.B / residual) ** (1.0 / target_loss_law.beta)
    return {
        "reachable": True,
        "target_loss": target,
        "tokens_big": d_big,
        "tokens_small": d_small,
        "train_flops_big": training_flops(n_params_optimal, d_big),
        "train_flops_small": training_flops(n_params_smaller, d_small),
        "extra_train_cost_x": training_flops(n_params_smaller, d_small)
        / training_flops(n_params_optimal, d_big),
        "inference_flops_ratio": n_params_smaller / n_params_optimal,  # cheaper forever
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _si(x: float) -> str:
    """Human-readable SI-ish suffix: 1.2e9 -> '1.2G'."""
    if x == 0:
        return "0"
    suffixes = [
        (1e24, "Y"), (1e21, "Z"), (1e18, "E"), (1e15, "P"),
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
    ]
    for scale, suf in suffixes:
        if abs(x) >= scale:
            return f"{x / scale:.2f}{suf}"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FLOPs rule: a 70B model on 1.4T tokens =",
          _si(training_flops(7e10, 1.4e12)), "FLOPs")

    print("\nCompute-optimal allocations:")
    for c in [1e18, 1e21, 5.9e23, 1e25, 1e26]:
        print("  ", chinchilla_optimal(c))

    print("\nChinchilla-point sanity (should land near 70B / 1.4T):")
    print("  ", chinchilla_optimal(5.88e23))

    print("\nPredicted loss along the ladder:")
    for p in LADDER:
        print(f"   {p.name:28s} L={CHINCHILLA.loss(p.n_params, p.n_tokens):.3f} "
              f"({p.tokens_per_param:.0f} tok/param)")

    print("\nCost to spend 1e24 FLOPs:")
    print("  ", cost_and_time(1e24, gpu="H100", n_gpus=1024, mfu=0.4))

    print("\nExtrapolate course demo x 1e6 compute:")
    print("  ", extrapolate(LADDER[0], 1e6))

    print("\nOver-training trade (70B-optimal vs an 8B served forever):")
    s = overtrain_savings(7e10, 8e9)
    if s["reachable"]:
        print(f"   8B needs {_si(s['tokens_small'])}tokens (vs {_si(s['tokens_big'])} optimal),")
        print(f"   {s['extra_train_cost_x']:.1f}x the training compute,")
        print(f"   but {1/s['inference_flops_ratio']:.1f}x cheaper inference forever.")
