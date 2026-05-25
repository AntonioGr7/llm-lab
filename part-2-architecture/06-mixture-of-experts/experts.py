"""Mixture of Experts FFN — routed experts + optional shared expert.

What this file contains:

- `Expert`: a small SwiGLU FFN (identical shape to Module 05's SwiGLU, just
  parameterized to a per-expert hidden width).
- `Router`: top-$k$ routing with **aux-loss-free balancing** (DeepSeek-V3's
  bias-update mechanism). Aux-loss balancing is exposed for comparison via
  the `balancing` flag.
- `MoEFFN`: composes everything. Top-$k$ routed experts + an optional shared
  expert that runs every token. Drop-in replacement for Module 05's SwiGLU
  in a transformer block.

For the math and design rationale see the module README.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BalancingMode = Literal["aux_loss_free", "aux_loss", "none"]


# ---------------------------------------------------------------------------
# Expert: a small SwiGLU FFN
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """One MoE expert. Same form as Module 05's SwiGLU, just smaller width.

    In DeepSeek-V3 each routed expert has $d_{\\text{ffn}} \\approx d_{\\text{model}} / 2$
    instead of the dense $8/3 \\cdot d_{\\text{model}}$. The total FFN compute
    that runs per token is `top_k * expert_ffn`, which the design tunes to
    roughly match the dense model's $\\frac{8}{3} d_{\\text{model}}$.
    """

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.W_g = nn.Linear(d_model, d_ffn, bias=False)
        self.W_v = nn.Linear(d_model, d_ffn, bias=False)
        self.W_o = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_o(F.silu(self.W_g(x)) * self.W_v(x))


# ---------------------------------------------------------------------------
# Router with aux-loss-free balancing
# ---------------------------------------------------------------------------

@dataclass
class RouterOutput:
    """What a router returns: which experts each token uses, and how to
    weight them. Plus diagnostics for balancing and logging."""

    indices: torch.Tensor    # (n_tokens, top_k) int64 — which experts each token picks
    weights: torch.Tensor    # (n_tokens, top_k) float — combine weights for those experts
    utilization: torch.Tensor  # (n_experts,) float — fraction of tokens that picked each expert
    aux_loss: Optional[torch.Tensor] = None  # scalar; None when balancing="none" or "aux_loss_free"


class Router(nn.Module):
    """Top-$k$ router. Default: aux-loss-free balancing (DeepSeek-V3).

    Args:
        d_model: input dim.
        n_experts: how many experts to route over.
        top_k: how many to pick per token.
        balancing: "aux_loss_free" (default), "aux_loss", or "none".
        bias_update_rate: $u$ in the DeepSeek bias-update rule. Adjusted per
            optimizer step via `update_bias()` after the forward.
        aux_loss_coef: $\\alpha$ in the classical aux-loss term. Only used
            when `balancing == "aux_loss"`.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        balancing: BalancingMode = "aux_loss_free",
        bias_update_rate: float = 1e-3,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.balancing = balancing
        self.bias_update_rate = bias_update_rate
        self.aux_loss_coef = aux_loss_coef

        self.W_g = nn.Linear(d_model, n_experts, bias=False)

        # The aux-loss-free balancing bias is NOT a learned parameter — it's
        # updated outside the gradient graph. Buffer, not Parameter.
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """x: (n_tokens, d_model). Returns indices/weights for top-k routing."""
        scores = self.W_g(x)                          # (n_tokens, N)
        n_tokens = scores.shape[0]

        # Selection uses biased scores (only when aux-loss-free). The bias is
        # zero in the other modes, so this is a no-op for them.
        biased = scores + self.bias if self.balancing == "aux_loss_free" else scores

        # Pick top-k experts per token, then re-softmax over the chosen ones.
        # The COMBINE weights use the un-biased scores, so the bias doesn't
        # leak into the value-mixing — it only steers WHICH experts run.
        topk = torch.topk(biased, self.top_k, dim=-1)
        indices = topk.indices                         # (n_tokens, top_k)
        chosen_scores = scores.gather(-1, indices)     # (n_tokens, top_k) — un-biased
        weights = chosen_scores.softmax(dim=-1)

        # Utilization: fraction of tokens for which expert i is in the top-k.
        # One-hot the indices and sum over tokens.
        one_hot = torch.zeros_like(scores)             # (n_tokens, N)
        one_hot.scatter_(-1, indices, 1.0)
        utilization = one_hot.mean(dim=0)              # (N,)

        # Auxiliary loss (only when balancing == "aux_loss"). The classical
        # Switch / GShard formula: $\\alpha \\cdot N \\cdot \\sum_i f_i \\cdot P_i$
        aux_loss = None
        if self.balancing == "aux_loss":
            P = scores.softmax(dim=-1).mean(dim=0)     # (N,) — mean router prob
            aux_loss = self.aux_loss_coef * self.n_experts * (utilization * P).sum()

        return RouterOutput(
            indices=indices, weights=weights,
            utilization=utilization, aux_loss=aux_loss,
        )

    @torch.no_grad()
    def update_bias(self, utilization: torch.Tensor) -> None:
        """DeepSeek-V3 bias update rule. Call once per training step, after
        the optimizer step, with the utilization observed at this step.

        Over-used experts (utilization > target) get their bias decreased.
        Under-used experts get their bias increased. The bias steers the
        top-k selection without contaminating the gradient on `W_g`.
        """
        if self.balancing != "aux_loss_free":
            return
        target = self.top_k / self.n_experts
        error = utilization - target
        self.bias -= self.bias_update_rate * error.sign()


# ---------------------------------------------------------------------------
# The MoE FFN
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """Top-k routed experts + optional shared expert.

    Drop-in replacement for Module 05's SwiGLU in a transformer block. Returns
    the MoE FFN output plus a `RouterOutput` so the training loop can read the
    aux loss and apply the bias update.

    Args:
        d_model: residual stream width.
        n_experts: number of routed experts.
        top_k: experts per token.
        d_ffn_expert: hidden width per expert. DeepSeek-V3 uses ~$d_{\\text{model}}/2$.
        n_shared_experts: how many "always-on" experts (DeepSeek default: 1).
            Set to 0 to disable. The shared expert's hidden width is the same
            as a routed expert's (it's just always selected).
        balancing: passed through to the router.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        d_ffn_expert: int,
        n_shared_experts: int = 1,
        balancing: BalancingMode = "aux_loss_free",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_shared = n_shared_experts

        self.router = Router(d_model, n_experts, top_k, balancing=balancing)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ffn_expert) for _ in range(n_experts)
        ])
        if n_shared_experts > 0:
            # One shared "expert" with the same per-expert width, scaled to
            # `n_shared` widths if you want a wider shared path.
            self.shared = Expert(d_model, d_ffn_expert * n_shared_experts)
        else:
            self.shared = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterOutput]:
        """Forward pass. Returns (output, router_info).

        The router_info is for the training loop: read `.aux_loss` for the
        classical balancing path; call `router.update_bias(.utilization)`
        after the optimizer step for the aux-loss-free path.
        """
        B, T, D = x.shape
        N, k = self.n_experts, self.top_k

        # Flatten tokens — routing is per-token, batch/seq are equivalent here.
        x_flat = x.view(B * T, D)
        router_out = self.router(x_flat)
        indices = router_out.indices       # (B*T, k)
        weights = router_out.weights       # (B*T, k)

        # Run each expert on the tokens routed to it. This is the obvious
        # implementation; it's correct and easy to read. A production MoE
        # kernel batches better — see Megablocks / FasterMoE for grouped-GEMM
        # implementations that get full GPU utilization.
        out = torch.zeros_like(x_flat)

        # For each expert, gather the tokens that picked it, run them, scatter
        # the weighted outputs back. Looping over N is fine for small N (256
        # is the upper end); for huge N you'd use a fused kernel.
        flat_indices = indices.view(-1)                    # (B*T*k,)
        flat_weights = weights.view(-1)                    # (B*T*k,)
        token_ids = torch.arange(B * T, device=x.device).repeat_interleave(k)  # (B*T*k,)

        for expert_id in range(N):
            mask = flat_indices == expert_id
            if not mask.any():
                continue
            tok_idx = token_ids[mask]                      # which tokens
            wts = flat_weights[mask].unsqueeze(-1)         # combine weights
            expert_in = x_flat[tok_idx]                    # gather inputs
            expert_out = self.experts[expert_id](expert_in)
            # Scatter-add the weighted output back to the right token slot.
            out.index_add_(0, tok_idx, expert_out * wts)

        # Shared expert (always on).
        if self.shared is not None:
            out = out + self.shared(x_flat)

        return out.view(B, T, D), router_out


# ---------------------------------------------------------------------------
# A reference MoE transformer block (composes with Module 05's pieces)
# ---------------------------------------------------------------------------

class MoETransformerBlock(nn.Module):
    """Pre-norm transformer block with MoE FFN instead of dense SwiGLU.

    Mirrors Module 05's `TransformerBlock` exactly except the FFN sub-layer
    is an `MoEFFN`. We don't subclass that to avoid a hard dependency on
    Module 05's import path.

    Args:
        d_model: residual stream width.
        attention: pluggable attention module (from Module 04).
        moe: a pre-built `MoEFFN`.
        norm_eps: epsilon for RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        moe: MoEFFN,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Re-define RMSNorm here to keep this module standalone.
        self.norm_attn = _RMSNorm(d_model, eps=norm_eps)
        self.attn = attention
        self.norm_ffn = _RMSNorm(d_model, eps=norm_eps)
        self.moe = moe

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, RouterOutput]:
        x = x + self.attn(self.norm_attn(x), freqs_cis)
        moe_out, router_info = self.moe(self.norm_ffn(x))
        x = x + moe_out
        return x, router_info


class _RMSNorm(nn.Module):
    """Local RMSNorm copy to avoid cross-module imports."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rrms = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rrms).type_as(x) * self.gamma


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "04-attention"))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "05-transformer-block"))
    from attention import GroupedQueryAttention
    from rope import precompute_freqs_cis

    B, T, d_model = 2, 32, 128
    n_heads = 8
    x = torch.randn(B, T, d_model)

    print("--- MoEFFN forward (aux-loss-free, 8 experts, top-2, 1 shared) ---")
    moe = MoEFFN(d_model=d_model, n_experts=8, top_k=2, d_ffn_expert=64, n_shared_experts=1)
    out, info = moe(x)
    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"router util:  {info.utilization.tolist()}")
    print(f"target util:  {2/8:.3f}  (top_k / n_experts)")
    print(f"aux_loss:     {info.aux_loss}  (None for aux-loss-free)")
    print(f"params total: {sum(p.numel() for p in moe.parameters()):,}")
    print(f"params/expert (routed): {sum(p.numel() for p in moe.experts[0].parameters()):,}")

    print("\n--- Bias update on imbalanced traffic ---")
    fake_util = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    moe.router.update_bias(fake_util)
    print(f"bias after one update: {moe.router.bias.tolist()}")
    print("(over-used experts 0,1 went negative; under-used 2-7 went positive)")

    print("\n--- MoETransformerBlock end-to-end ---")
    attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=2)
    moe = MoEFFN(d_model=d_model, n_experts=8, top_k=2, d_ffn_expert=64)
    blk = MoETransformerBlock(d_model=d_model, attention=attn, moe=moe)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, T)
    out, info = blk(x, freqs_cis)
    print(f"output shape: {tuple(out.shape)}  (matches input — drop-in replacement for dense block)")
