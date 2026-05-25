"""Rotary Position Embedding (RoPE) with context-extension scaling.

Canonical RoPE implementation for the course. Module 04 has a minimal version
([`../04-attention/rope.py`](../04-attention/rope.py)) used only to make MLA
runnable; this file is the one the rest of the course imports from.

Three frequency-interpolation schemes are supported:

- **None** — standard RoPE, no scaling. What you use at pretraining time.
- **"pi"** — Position Interpolation (Chen et al. 2023): linearly compress
  positions by the target/train ratio. Simple, requires fine-tuning.
- **"ntk"** — NTK-aware scaling (bloc97 2023): adjust the base instead of
  positions. No fine-tuning needed for modest extensions.
- **"yarn"** — YaRN (Peng et al. 2023): frequency-dependent ramp between
  PI and NTK regimes, with an attention temperature correction. State of
  the art for long-context extension in 2026.

For the math and the design rationale, see this module's README, Section 5.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import torch


# ---------------------------------------------------------------------------
# Standard RoPE
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Precompute the complex rotation table used by RoPE.

    Args:
        dim: rotary dimension (must be even). RoPE rotates pairs of features,
            so there will be `dim // 2` distinct frequencies.
        max_seq_len: maximum sequence position to precompute. For inference
            past this length, recompute with a larger value (or use a scaled
            variant below).
        base: the RoPE base $b$. 10,000 is the Vaswani-era default; modern
            long-context models use 500,000 or 1,000,000.

    Returns:
        Complex tensor of shape `(max_seq_len, dim // 2)`, dtype `complex64`.
        Element `[m, k]` is $e^{i \\cdot m \\cdot \\theta_k}$, where
        $\\theta_k = b^{-2k/\\text{dim}}$.
    """
    assert dim % 2 == 0, "RoPE dim must be even (it rotates pairs of features)"
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to `x`.

    Args:
        x: tensor whose last two dims are `(seq_len, dim)` with `dim` even.
            Typical shapes: `(B, T, H, d_head)` or `(B, T, d_rope)`.
        freqs_cis: complex rotation table from `precompute_freqs_cis*`,
            shape `(max_seq_len, dim/2)`. Sliced to `T` along the first dim.

    Returns:
        Same shape as `x`, real-valued, dtype matched to `x`.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    seq_len = x.shape[-2]
    freqs_cis = freqs_cis[:seq_len]
    # Broadcast freqs_cis (T, d/2) against x_complex's leading dims.
    shape = [1] * (x_complex.dim() - 2) + list(freqs_cis.shape)
    freqs_cis = freqs_cis.view(*shape)
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2).type_as(x)


# ---------------------------------------------------------------------------
# Context extension: PI, NTK, YaRN
# ---------------------------------------------------------------------------

def precompute_freqs_cis_pi(
    dim: int,
    max_seq_len: int,
    train_seq_len: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Position Interpolation (Chen et al. 2023).

    Compresses positions by the factor `s = max_seq_len / train_seq_len` so
    that the model sees the same angle range it saw during training. Usually
    needs ~1k–5k steps of fine-tuning to recover quality at the new length.

    Args:
        dim: rotary dimension.
        max_seq_len: target context length.
        train_seq_len: context length the model was originally trained at.
        base: RoPE base.
    """
    assert dim % 2 == 0
    s = max_seq_len / train_seq_len
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32) / s    # the scaling
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_ntk(
    dim: int,
    max_seq_len: int,
    train_seq_len: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """NTK-aware scaled RoPE (bloc97 2023).

    Instead of compressing positions (PI), adjust the *base* so the slowest
    pair widens enough to cover the new context, while the fastest pair is
    left almost untouched. Better zero-shot quality than PI for modest
    extensions; no derivation, just an empirical fit.

    Args:
        dim: rotary dimension.
        max_seq_len: target context length.
        train_seq_len: context length the model was originally trained at.
        base: original RoPE base.
    """
    assert dim % 2 == 0
    s = max_seq_len / train_seq_len
    base_scaled = base * s ** (dim / (dim - 2))
    freqs = 1.0 / (base_scaled ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_yarn(
    dim: int,
    max_seq_len: int,
    train_seq_len: int,
    base: float = 10_000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """YaRN (Peng et al. 2023): frequency-dependent ramped scaling.

    The idea: each pair has a wavelength $\\lambda_k = 2\\pi / \\theta_k$.
    Pairs that complete many cycles within `train_seq_len` are left alone
    (use the original frequency); pairs that complete few cycles are scaled
    by `1/s` (full PI); the in-between band is smoothly ramped.

    Args:
        dim: rotary dimension.
        max_seq_len: target context length.
        train_seq_len: training context length.
        base: original RoPE base.
        beta_fast: pairs that complete >= beta_fast cycles in train_seq_len
            are left untouched (high-frequency regime). YaRN default: 32.
        beta_slow: pairs that complete <= beta_slow cycles in train_seq_len
            are fully PI-scaled (low-frequency regime). YaRN default: 1.

    Returns:
        Tuple of:
        - Complex rotation table, shape `(max_seq_len, dim // 2)`.
        - Attention-logit temperature scale, a float. Multiply logits by
          this BEFORE softmax to keep softmax sharpness consistent with the
          shorter-context model. YaRN's formula: $\\sqrt{1 + 0.1 \\log s}$.
    """
    assert dim % 2 == 0
    s = max_seq_len / train_seq_len

    # Per-pair original frequencies and wavelengths.
    pair_indices = torch.arange(0, dim, 2, dtype=torch.float32)
    freqs_orig = 1.0 / (base ** (pair_indices / dim))                    # (d/2,)
    wavelengths = 2 * math.pi / freqs_orig                                # (d/2,)

    # Number of full cycles each pair makes within the trained context.
    cycles_in_train = train_seq_len / wavelengths                         # (d/2,)

    # Ramp: 0 for the slow end (cycles <= beta_slow), 1 for the fast end
    # (cycles >= beta_fast), linear in between.
    ramp = (cycles_in_train - beta_slow) / (beta_fast - beta_slow)
    ramp = ramp.clamp(0.0, 1.0)                                           # (d/2,)

    # Interpolate between PI-scaled (freqs_orig / s) and unchanged (freqs_orig).
    freqs = freqs_orig * (ramp + (1 - ramp) / s)

    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    rotation = torch.polar(torch.ones_like(freqs), freqs)

    # YaRN's attention temperature correction.
    attn_scale = math.sqrt(1.0 + 0.1 * math.log(s)) if s > 1 else 1.0

    return rotation, attn_scale


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

ScalingMode = Literal["none", "pi", "ntk", "yarn"]


def make_freqs_cis(
    dim: int,
    max_seq_len: int,
    *,
    base: float = 10_000.0,
    scaling: ScalingMode = "none",
    train_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Pick a freqs_cis table by scaling mode. Convenience for notebooks.

    For YaRN, this drops the attention-temperature scale; if you need it,
    call `precompute_freqs_cis_yarn` directly.
    """
    if scaling == "none":
        return precompute_freqs_cis(dim, max_seq_len, base=base)
    assert train_seq_len is not None, f"{scaling} requires train_seq_len"
    if scaling == "pi":
        return precompute_freqs_cis_pi(dim, max_seq_len, train_seq_len, base=base)
    if scaling == "ntk":
        return precompute_freqs_cis_ntk(dim, max_seq_len, train_seq_len, base=base)
    if scaling == "yarn":
        rotation, _ = precompute_freqs_cis_yarn(dim, max_seq_len, train_seq_len, base=base)
        return rotation
    raise ValueError(f"unknown scaling mode: {scaling}")


if __name__ == "__main__":
    # Sanity: the property that drove the whole design.
    # The inner product <RoPE(q, m), RoPE(k, n)> depends only on n - m.
    dim, T = 64, 128
    freqs_cis = precompute_freqs_cis(dim, T)
    q = torch.randn(dim)
    k = torch.randn(dim)

    # Pick three (m, n) pairs with the same relative offset n - m = 5.
    pairs = [(0, 5), (10, 15), (100, 105)]
    dots = []
    for m, n in pairs:
        qm = apply_rotary_emb(q.view(1, 1, dim), freqs_cis[m:m + 1]).view(dim)
        kn = apply_rotary_emb(k.view(1, 1, dim), freqs_cis[n:n + 1]).view(dim)
        dots.append((qm @ kn).item())

    print("RoPE relative-position property check")
    print(f"  inner products at offsets (5, 5, 5): {dots}")
    print(f"  max spread: {max(dots) - min(dots):.2e}  (should be ~1e-6)")

    # Quick YaRN check: temperature scale at 4x extension.
    _, temp = precompute_freqs_cis_yarn(dim, max_seq_len=8192, train_seq_len=2048)
    print(f"\nYaRN attention-temperature at 4x extension: {temp:.4f}")
