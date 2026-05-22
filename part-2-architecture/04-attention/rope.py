"""Minimal rotary position embedding helper.

Just enough RoPE for MLA's decoupled-RoPE path in `attention.py`. The full
treatment — what RoPE is doing mathematically, why frequency interpolation
extends context length, what makes YaRN different — lives in Module 05.
"""
import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    """Precompute the complex-valued rotation table used by RoPE.

    Args:
        dim: rotary dimension (must be even). RoPE rotates pairs of features.
        max_seq_len: maximum position to precompute. Real models extend this
            with frequency interpolation (NTK, YaRN); we keep it simple here.
        base: rope base. 10000 is the original Vaswani-style choice; modern
            models use 500000+ for longer-context extrapolation (Module 05).

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2), `complex64`.
    """
    assert dim % 2 == 0, "RoPE dim must be even (it rotates pairs of features)"
    # Per-pair frequencies: dim/2 of them, geometrically spaced.
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # Outer product with positions: (max_seq_len, dim/2)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Convert to complex exponentials (Euler form): e^{i * freq * t}
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to `x`.

    Args:
        x: tensor with last two dims (seq_len, dim). Typically used as
            (..., T, dim) where dim is even. Common shapes: (B, T, H, d_head)
            or (B, T, d_rope).
        freqs_cis: complex rotation table from `precompute_freqs_cis`,
            shape (max_seq_len, dim/2). Will be sliced to T positions.

    Returns:
        `x` with the same shape, real-valued, rotated according to position.
    """
    # Pair up adjacent features and view as complex: (..., T, dim) -> (..., T, dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape freqs_cis to broadcast against the leading dims of x.
    # freqs_cis is (T, dim/2); we want (..., T, dim/2).
    seq_len = x.shape[-2]
    freqs_cis = freqs_cis[:seq_len]
    # Insert singleton dims for every leading dim of x_complex except seq and feature.
    shape = [1] * (x_complex.dim() - 2) + list(freqs_cis.shape)
    freqs_cis = freqs_cis.view(*shape)

    # Multiply by the complex rotation and convert back to real.
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.type_as(x)
