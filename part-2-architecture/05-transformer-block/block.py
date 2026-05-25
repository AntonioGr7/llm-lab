"""The transformer block: RMSNorm + attention + RMSNorm + SwiGLU FFN.

What this file contains:

- `RMSNorm`: the norm every modern frontier model uses. Same as LayerNorm
  with the mean subtraction dropped.
- `SwiGLU`: the gated-FFN that replaced the vanilla MLP. Three matrices,
  hidden width $\\frac{8}{3} d_{\\text{model}}$ rounded to a hardware-friendly
  multiple.
- `TransformerBlock`: pre-norm block with a pluggable attention module.
  Hand it any attention from `../04-attention/attention.py` (MHA, GQA, MLA)
  and it composes the rest.

The bias-free convention from Module 04 holds: no `bias=True` anywhere.

See the module README for the math behind every line.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Equivalent to LayerNorm without mean centering and without a bias term:

        x_norm = (x / sqrt(mean(x**2) + eps)) * gamma

    The promotion to FP32 inside `forward` matters when the input is BF16:
    the squared mean can underflow at small magnitudes, and the resulting
    NaN/Inf is hard to debug. Promoting matches Llama / Qwen reference code.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rrms = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rrms).type_as(x) * self.gamma


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

def swiglu_hidden_dim(d_model: int, multiplier: float = 8 / 3, multiple_of: int = 64) -> int:
    """Pick the SwiGLU hidden dim using the 8/3 rule, rounded up to `multiple_of`.

    Total FFN params (W_g, W_v, W_o) at this hidden dim are ~$8 d^2$, matching
    a vanilla 4× FFN. `multiple_of` keeps the dim aligned for tensorcore matmuls.
    """
    d = int(d_model * multiplier)
    return ((d + multiple_of - 1) // multiple_of) * multiple_of


class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer 2020): `W_o( (W_v x) * SiLU(W_g x) )`.

    Three matrices: gate (`W_g`), value (`W_v`), output (`W_o`). The hidden
    width defaults to `swiglu_hidden_dim(d_model)` so total FFN params match
    a vanilla 4× MLP.
    """

    def __init__(self, d_model: int, d_ffn: int | None = None):
        super().__init__()
        d_ffn = d_ffn if d_ffn is not None else swiglu_hidden_dim(d_model)
        self.W_g = nn.Linear(d_model, d_ffn, bias=False)
        self.W_v = nn.Linear(d_model, d_ffn, bias=False)
        self.W_o = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_o(F.silu(self.W_g(x)) * self.W_v(x))


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with a pluggable attention module.

    Layout:

        x → RMSNorm → attn  → +x
          → RMSNorm → ffn   → +x → out

    Args:
        d_model: residual stream width.
        attention: any attention module from Module 04 (or compatible). Its
            `forward(x, freqs_cis)` signature is what the block expects.
        d_ffn: SwiGLU hidden width. Defaults to `swiglu_hidden_dim(d_model)`.
        norm_eps: epsilon for RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        d_ffn: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm_attn = RMSNorm(d_model, eps=norm_eps)
        self.attn = attention
        self.norm_ffn = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, d_ffn=d_ffn)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), freqs_cis)
        x = x + self.ffn(self.norm_ffn(x))
        return x


if __name__ == "__main__":
    # Smoke test: build one block on top of each Module-04 attention variant,
    # run a forward, report shape + param count.
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "04-attention"))
    from attention import MultiHeadAttention, GroupedQueryAttention, MultiHeadLatentAttention
    from rope import precompute_freqs_cis  # canonical (this module)

    B, T, D = 2, 16, 128
    H = 8
    x = torch.randn(B, T, D)

    print("--- TransformerBlock(MHA) ---")
    attn = MultiHeadAttention(d_model=D, n_heads=H, use_rope=True)
    block = TransformerBlock(d_model=D, attention=attn)
    freqs_cis = precompute_freqs_cis(D // H, T)
    out = block(x, freqs_cis)
    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in block.parameters()):,}")

    print("\n--- TransformerBlock(GQA, kv_heads=2) ---")
    attn = GroupedQueryAttention(d_model=D, n_heads=H, n_kv_heads=2, use_rope=True)
    block = TransformerBlock(d_model=D, attention=attn)
    out = block(x, freqs_cis)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in block.parameters()):,}")

    print("\n--- TransformerBlock(MLA) ---")
    attn = MultiHeadLatentAttention(
        d_model=D, n_heads=H, d_head=D // H, d_rope=16, d_kv_latent=32
    )
    block = TransformerBlock(d_model=D, attention=attn)
    freqs_cis_mla = precompute_freqs_cis(16, T)  # d_rope = 16
    out = block(x, freqs_cis_mla)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in block.parameters()):,}")

    # RMSNorm sanity: output has unit RMS along the feature dim (within eps).
    print("\n--- RMSNorm sanity ---")
    norm = RMSNorm(D)
    z = torch.randn(B, T, D) * 5.0
    z_norm = norm(z)
    rms = z_norm.pow(2).mean(-1).sqrt()
    print(f"input RMS  (sample): {z.pow(2).mean(-1).sqrt()[0, 0].item():.4f}")
    print(f"output RMS (sample): {rms[0, 0].item():.4f}  (should be ~1.0)")

    # SwiGLU param count vs vanilla 4x FFN.
    print("\n--- SwiGLU param count vs vanilla 4x MLP ---")
    d_ffn = swiglu_hidden_dim(D)
    swiglu_params = 3 * D * d_ffn
    vanilla_params = 2 * D * (4 * D)
    print(f"d_ffn (8/3 rule, multiple-of-64): {d_ffn}")
    print(f"SwiGLU params:   {swiglu_params:,}")
    print(f"Vanilla params:  {vanilla_params:,}")
    print(f"Ratio:           {swiglu_params / vanilla_params:.3f}  (target ~1.0)")
