"""Attention variants — Multi-Head, Grouped-Query, and Multi-Head Latent.

All forward passes use PyTorch's `scaled_dot_product_attention` for the actual
attention kernel, which dispatches to FlashAttention 2 on Ampere (A100) and
FlashAttention 3 on Hopper (H100) automatically. No custom CUDA needed.

The bias-free convention follows modern recipes (Llama, Qwen, DeepSeek);
no `bias=True` linear layers anywhere in attention.

For the math and the reasoning behind every line, see the module README.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import apply_rotary_emb


# ---------------------------------------------------------------------------
# QK-Norm helper
# ---------------------------------------------------------------------------
# This is the same RMSNorm as Module 05's block.py, duplicated here so that
# Module 04 stays standalone (it must not import forward from a later module).
# The one difference in *usage*: here it normalizes over the per-head dimension
# (d_head), not the residual width (d_model).

class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension.

    For QK-Norm we instantiate this with `d_head`, so a single learnable
    `gamma` of width `d_head` is shared across all heads and applied to every
    head's query (or key) vector independently. The FP32 promotion matters in
    BF16 training: the squared mean can underflow, producing NaN/Inf logits
    that are painful to trace. This matches Llama/Qwen reference code.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rrms = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rrms).type_as(x) * self.gamma


# ---------------------------------------------------------------------------
# Reference implementation — for understanding, not for production use.
# ---------------------------------------------------------------------------

def scaled_dot_product_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True
) -> torch.Tensor:
    """The textbook attention formula, computed explicitly.

    Use this only for shape/intuition checks. For real workloads, call
    `F.scaled_dot_product_attention` which gets you FlashAttention for free.

    Args:
        q, k: (..., T, d_qk)
        v:    (..., T, d_v)
        is_causal: whether to apply a causal mask.

    Returns:
        out: (..., T, d_v)
    """
    d_qk = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_qk)
    if is_causal:
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v


# ---------------------------------------------------------------------------
# Multi-Head Attention (Vaswani 2017)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional RoPE.

    Each head has its own Q, K, V projection. Total parameter count for the
    QKV projection: 3 * d_model * d_model. The output projection is one more
    d_model x d_model matrix.

    KV cache size per token: 2 * n_heads * d_head = 2 * d_model values.

    `qk_norm` adds per-head RMSNorm to Q and K before the dot product
    (Qwen3, Gemma 2/3, OLMo 2). It bounds the magnitude of the attention
    logits, which prevents the logit-blowup / attention-entropy-collapse
    instability that shows up at scale and in long-context / low-precision
    training. Cost is two tiny `d_head`-wide norms; see the README.
    """

    def __init__(
        self, d_model: int, n_heads: int, *, use_rope: bool = True, qk_norm: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.qk_norm = qk_norm

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Per-head QK-Norm: one gamma of width d_head, shared across heads.
        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Project to Q, K, V and split into heads.
        qkv = self.W_qkv(x)                                    # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)                         # each (B, T, D)
        q = q.view(B, T, H, dh).transpose(1, 2)                # (B, H, T, dh)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)

        # QK-Norm goes on the raw projected Q/K, before RoPE. RoPE is a rotation
        # and therefore norm-preserving, so norm-then-RoPE keeps the unit-scale
        # guarantee intact.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            assert freqs_cis is not None, "RoPE enabled but freqs_cis not provided"
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        # FlashAttention via PyTorch SDPA. is_causal=True applies the upper-triangular mask.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, dh)

        # Concatenate heads and apply output projection.
        out = out.transpose(1, 2).reshape(B, T, D)             # (B, T, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Grouped-Query Attention (Ainslie 2023; Llama 2/3)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """GQA: H query heads share K/V across G = n_kv_heads groups.

    Setting n_kv_heads == n_heads recovers MHA. Setting n_kv_heads == 1 gives
    Multi-Query Attention (MQA). Llama 2/3 use n_kv_heads = 8 for n_heads = 64.

    KV cache size per token: 2 * n_kv_heads * d_head. Ratio vs MHA: n_kv_heads / n_heads.

    `qk_norm` adds per-head RMSNorm to Q and K (see `MultiHeadAttention`).
    The K norm uses `n_kv_heads` heads' worth of a single `d_head`-wide gamma
    and is applied *before* the K/V broadcast, so it normalizes the stored
    keys, not their replicated copies.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        use_rope: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # query heads per shared KV head
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.qk_norm = qk_norm

        # Q gets full n_heads * d_head; KV get n_kv_heads * d_head each.
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        # Per-head QK-Norm: one gamma of width d_head, shared across heads.
        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        Hq, Hkv, dh = self.n_heads, self.n_kv_heads, self.d_head

        q = self.W_q(x).view(B, T, Hq,  dh).transpose(1, 2)    # (B, Hq,  T, dh)
        k = self.W_k(x).view(B, T, Hkv, dh).transpose(1, 2)    # (B, Hkv, T, dh)
        v = self.W_v(x).view(B, T, Hkv, dh).transpose(1, 2)    # (B, Hkv, T, dh)

        # QK-Norm before RoPE and before the K broadcast (normalizes stored keys).
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            assert freqs_cis is not None, "RoPE enabled but freqs_cis not provided"
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        # Broadcast K and V from Hkv to Hq groups (each KV head serves n_groups query heads).
        k = k.repeat_interleave(self.n_groups, dim=1)          # (B, Hq, T, dh)
        v = v.repeat_interleave(self.n_groups, dim=1)          # (B, Hq, T, dh)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, Hq * dh)
        return self.W_o(out)

    def kv_cache_size_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes of KV cache per token at the given dtype precision (default BF16 = 2 bytes)."""
        return 2 * self.n_kv_heads * self.d_head * dtype_bytes


# ---------------------------------------------------------------------------
# Multi-Head Latent Attention (DeepSeek V2/V3)
# ---------------------------------------------------------------------------

class MultiHeadLatentAttention(nn.Module):
    """MLA with decoupled RoPE, the DeepSeek V2/V3 attention design.

    KV cache per token: d_kv_latent + d_rope (shared across heads) values.
    For DeepSeek-V3's config (n_heads=128, d_head=128, d_kv_latent=512, d_rope=64),
    that's 576 values per token vs 32,768 for the MHA equivalent.

    See the module README, section 8, for the design rationale and the
    decoupled-RoPE trick.

    Args:
        d_model: residual stream width.
        n_heads: number of attention heads.
        d_head: per-head dimension for the "content" K/Q part (no RoPE).
        d_rope: per-head dimension for the RoPE Q; the RoPE K is shared
            across heads at this same width.
        d_kv_latent: latent dimension that the KV cache compresses into.
        d_q_latent: optional Q-path latent dimension. If set, queries are
            also computed through a low-rank bottleneck (DeepSeek-V3 does
            this; reduces parameter count). Defaults to None (direct projection).
        qk_norm: if True, apply per-head RMSNorm to the *content* halves of Q
            and K (`q_c`, `k_c`) before the dot product. The decoupled-RoPE
            halves (`q_r`, `k_r`) are left unnormalized: `k_r` is a single
            shared vector cached per token (not per-head), so head-wise QK-Norm
            doesn't apply to it cleanly, and RoPE already controls its scale.
            Note DeepSeek-V2/V3 themselves do *not* use QK-Norm; this is the
            natural way to graft it onto MLA if you want the stability benefit.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_rope: int,
        d_kv_latent: int,
        d_q_latent: Optional[int] = None,
        *,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert d_rope % 2 == 0, "d_rope must be even (RoPE rotates pairs of features)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_rope = d_rope
        self.d_kv_latent = d_kv_latent
        self.d_q_latent = d_q_latent
        self.qk_norm = qk_norm

        # Q path. Either direct (d_model -> n_heads * (d_head + d_rope))
        # or through a latent (d_model -> d_q_latent -> n_heads * (d_head + d_rope)).
        if d_q_latent is None:
            self.W_qc = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.W_qr = nn.Linear(d_model, n_heads * d_rope, bias=False)
        else:
            self.W_q_down = nn.Linear(d_model, d_q_latent, bias=False)
            self.W_qc = nn.Linear(d_q_latent, n_heads * d_head, bias=False)
            self.W_qr = nn.Linear(d_q_latent, n_heads * d_rope, bias=False)

        # KV path: latent (cached) -> content K and V.
        self.W_kv_down = nn.Linear(d_model, d_kv_latent, bias=False)
        self.W_kc = nn.Linear(d_kv_latent, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_kv_latent, n_heads * d_head, bias=False)

        # Shared RoPE K — computed directly from x, NOT from the latent.
        # This is the decoupled-RoPE trick: RoPE doesn't commute with the
        # KV-up projection, so we route the position info through a separate
        # small path that's also cached but never decompressed.
        self.W_kr = nn.Linear(d_model, d_rope, bias=False)

        # Output projection.
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Per-head QK-Norm on the content halves only (width d_head).
        if qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh, dr = self.n_heads, self.d_head, self.d_rope

        # --- Q path ---
        if self.d_q_latent is None:
            q_c = self.W_qc(x).view(B, T, H, dh).transpose(1, 2)   # (B, H, T, dh)
            q_r = self.W_qr(x).view(B, T, H, dr).transpose(1, 2)   # (B, H, T, dr)
        else:
            c_q = self.W_q_down(x)
            q_c = self.W_qc(c_q).view(B, T, H, dh).transpose(1, 2)
            q_r = self.W_qr(c_q).view(B, T, H, dr).transpose(1, 2)
        q_r = apply_rotary_emb(q_r, freqs_cis)                     # (B, H, T, dr)

        # --- KV path through the latent (this is what gets cached) ---
        c_kv = self.W_kv_down(x)                                   # (B, T, d_kv_latent)  ← cached
        k_c = self.W_kc(c_kv).view(B, T, H, dh).transpose(1, 2)    # (B, H, T, dh)
        v   = self.W_v(c_kv).view(B, T, H, dh).transpose(1, 2)     # (B, H, T, dh)

        # QK-Norm on the content halves only (the RoPE halves stay as-is — see __init__).
        if self.qk_norm:
            q_c = self.q_norm(q_c)
            k_c = self.k_norm(k_c)

        # --- Shared RoPE K (also cached, but a single d_rope vector per token, not per-head) ---
        k_r = self.W_kr(x)                                         # (B, T, dr)            ← cached
        k_r = apply_rotary_emb(k_r, freqs_cis)                     # (B, T, dr)
        k_r = k_r.unsqueeze(1).expand(-1, H, -1, -1)               # (B, H, T, dr)

        # --- Concatenate the two halves of Q and K along feature dim ---
        q = torch.cat([q_c, q_r], dim=-1)                          # (B, H, T, dh + dr)
        k = torch.cat([k_c, k_r], dim=-1)                          # (B, H, T, dh + dr)
        # v stays at (B, H, T, dh) — values don't have a RoPE component.

        # --- Attention ---
        # Note: SDPA happily handles Q/K with one feature width and V with another.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, dh)
        out = out.transpose(1, 2).reshape(B, T, H * dh)
        return self.W_o(out)

    def kv_cache_size_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes of KV cache per token at the given dtype (default BF16 = 2 bytes).

        For MLA, this is the sum of the latent and the shared RoPE key, NOT
        the full K/V which is reconstructed on the fly.
        """
        return (self.d_kv_latent + self.d_rope) * dtype_bytes


if __name__ == "__main__":
    # Smoke test: build one of each, run a forward pass on tiny tensors,
    # and report the per-token KV cache.
    from rope import precompute_freqs_cis

    B, T, D = 2, 8, 64
    H = 8
    x = torch.randn(B, T, D)

    print("--- MultiHeadAttention ---")
    mha = MultiHeadAttention(d_model=D, n_heads=H, use_rope=True)
    freqs_cis = precompute_freqs_cis(D // H, T)
    out = mha(x, freqs_cis)
    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in mha.parameters()):,}")
    print(f"KV cache per token (BF16): {2 * H * (D // H) * 2} bytes")

    print("\n--- MultiHeadAttention (qk_norm=True) ---")
    mha_qk = MultiHeadAttention(d_model=D, n_heads=H, use_rope=True, qk_norm=True)
    out = mha_qk(x, freqs_cis)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in mha_qk.parameters()):,}  (+2 * d_head gammas)")

    print("\n--- GroupedQueryAttention (n_kv_heads=2) ---")
    gqa = GroupedQueryAttention(d_model=D, n_heads=H, n_kv_heads=2)
    out = gqa(x, freqs_cis)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in gqa.parameters()):,}")
    print(f"KV cache per token (BF16): {gqa.kv_cache_size_per_token()} bytes")

    print("\n--- GroupedQueryAttention (n_kv_heads=2, qk_norm=True) ---")
    gqa_qk = GroupedQueryAttention(d_model=D, n_heads=H, n_kv_heads=2, qk_norm=True)
    out = gqa_qk(x, freqs_cis)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in gqa_qk.parameters()):,}  (+2 * d_head gammas)")

    print("\n--- MultiHeadLatentAttention ---")
    mla = MultiHeadLatentAttention(
        d_model=D, n_heads=H, d_head=D // H, d_rope=8, d_kv_latent=16
    )
    freqs_cis_mla = precompute_freqs_cis(8, T)  # d_rope = 8
    out = mla(x, freqs_cis_mla)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in mla.parameters()):,}")
    print(f"KV cache per token (BF16): {mla.kv_cache_size_per_token()} bytes")

    print("\n--- MultiHeadLatentAttention (qk_norm=True) ---")
    mla_qk = MultiHeadLatentAttention(
        d_model=D, n_heads=H, d_head=D // H, d_rope=8, d_kv_latent=16, qk_norm=True
    )
    out = mla_qk(x, freqs_cis_mla)
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in mla_qk.parameters()):,}  (+2 * d_head gammas, content halves only)")
