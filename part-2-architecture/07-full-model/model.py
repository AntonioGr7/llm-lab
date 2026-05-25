"""The full `TransformerLM` — what Module 11's pretraining will train.

Configurable along two architectural axes:

- **Attention**: "mla" (default, DeepSeek-V3), "gqa", or "mha".
- **FFN**: "dense" (default, SwiGLU) or "moe" (Module 06's MoEFFN).

Everything else (RMSNorm, RoPE base, pre-norm, weight tying) is fixed at the
frontier-canonical 2026 recipe. The `ModelConfig` dataclass holds every
hyperparameter; pass one of these to `TransformerLM(cfg)` and you have a
working LM.

This file imports from sibling modules in `part-2-architecture/`:
- Module 04 (`attention.py`) for the attention classes
- Module 05 (`block.py`, `rope.py`) for RMSNorm, SwiGLU, RoPE
- Module 06 (`experts.py`) for MoEFFN (only when ffn_type="moe")

See the module README for the design rationale.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sibling modules importable. The notebook also does this; doing it
# here lets `python model.py` work standalone.
_PART2 = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PART2 / "04-attention"))
sys.path.insert(0, str(_PART2 / "05-transformer-block"))
sys.path.insert(0, str(_PART2 / "06-mixture-of-experts"))

from attention import MultiHeadAttention, GroupedQueryAttention, MultiHeadLatentAttention  # noqa: E402
from block import RMSNorm, SwiGLU, swiglu_hidden_dim  # noqa: E402
from rope import precompute_freqs_cis  # noqa: E402
from experts import MoEFFN  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AttentionType = Literal["mha", "gqa", "mla"]
FFNType = Literal["dense", "moe"]


@dataclass
class ModelConfig:
    """Everything a `TransformerLM` needs to know.

    Defaults match Module 11's pretraining target — a small dense
    DeepSeek-V3-shaped model that fits a single A100 80GB.
    """
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    max_seq_len: int = 2048

    # Attention
    attention_type: AttentionType = "mla"
    n_heads: int = 8
    n_kv_heads: int = 2          # only used when attention_type == "gqa"
    d_head: Optional[int] = None # MLA: per-head content dim. Defaults to d_model // n_heads.
    d_rope: int = 64             # MLA: per-head RoPE dim. Shared K_r at this width.
    d_kv_latent: int = 128       # MLA: cached latent dim.
    d_q_latent: Optional[int] = None  # MLA: optional query latent.
    rope_base: float = 10_000.0

    # FFN
    ffn_type: FFNType = "dense"
    d_ffn: Optional[int] = None  # dense SwiGLU; defaults to swiglu_hidden_dim(d_model)
    # MoE-only (ignored when ffn_type == "dense")
    n_experts: int = 16
    top_k: int = 2
    d_ffn_expert: int = 256
    n_shared_experts: int = 1
    moe_balancing: str = "aux_loss_free"

    # Head
    tie_weights: bool = True
    norm_eps: float = 1e-6


# ---------------------------------------------------------------------------
# Building blocks (assembled here rather than reusing Module 05's TransformerBlock
# because we need both dense and MoE variants under one roof)
# ---------------------------------------------------------------------------

def _make_attention(cfg: ModelConfig) -> nn.Module:
    """Instantiate the configured attention module."""
    if cfg.attention_type == "mha":
        return MultiHeadAttention(d_model=cfg.d_model, n_heads=cfg.n_heads, use_rope=True)
    if cfg.attention_type == "gqa":
        return GroupedQueryAttention(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, use_rope=True,
        )
    if cfg.attention_type == "mla":
        d_head = cfg.d_head if cfg.d_head is not None else cfg.d_model // cfg.n_heads
        return MultiHeadLatentAttention(
            d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=d_head,
            d_rope=cfg.d_rope, d_kv_latent=cfg.d_kv_latent, d_q_latent=cfg.d_q_latent,
        )
    raise ValueError(f"unknown attention_type: {cfg.attention_type}")


def _make_ffn(cfg: ModelConfig) -> nn.Module:
    """Instantiate the configured FFN."""
    if cfg.ffn_type == "dense":
        return SwiGLU(cfg.d_model, d_ffn=cfg.d_ffn)
    if cfg.ffn_type == "moe":
        return MoEFFN(
            d_model=cfg.d_model, n_experts=cfg.n_experts, top_k=cfg.top_k,
            d_ffn_expert=cfg.d_ffn_expert, n_shared_experts=cfg.n_shared_experts,
            balancing=cfg.moe_balancing,
        )
    raise ValueError(f"unknown ffn_type: {cfg.ffn_type}")


class TransformerBlock(nn.Module):
    """A pre-norm block. Carries enough flexibility to handle MoE returns.

    The MoE FFN returns (output, router_info); the dense FFN returns just
    output. The block normalizes the two into the same forward signature
    by detecting which is in use.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.is_moe = cfg.ffn_type == "moe"
        self.norm_attn = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = _make_attention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = _make_ffn(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self.attn(self.norm_attn(x), freqs_cis)
        if self.is_moe:
            ffn_out, router_info = self.ffn(self.norm_ffn(x))
            x = x + ffn_out
            return x, router_info
        x = x + self.ffn(self.norm_ffn(x))
        return x, None


# ---------------------------------------------------------------------------
# The full LM
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """Token-IDs-to-logits decoder-only transformer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Precompute the RoPE rotation table at max length. For MLA the
        # rotary dim is d_rope; for MHA/GQA it's d_head (= d_model / n_heads).
        if cfg.attention_type == "mla":
            rotary_dim = cfg.d_rope
        else:
            rotary_dim = cfg.d_model // cfg.n_heads
        freqs_cis = precompute_freqs_cis(rotary_dim, cfg.max_seq_len, base=cfg.rope_base)
        # Register as a buffer so it moves with .to(device) but doesn't train.
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Transformer blocks.
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # Final norm + unembedding.
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Initialize with the standard recipe. Call `mup_init(model, ...)`
        # from init.py afterward if you want muP-scaled init instead.
        self._standard_init()

        # Weight tying — share embedding and unembedding matrices. Done
        # AFTER init so the aliased tensor inherits the embedding's init.
        if cfg.tie_weights:
            self.unembed.weight = self.tok_emb.weight

    def _standard_init(self, hidden_std: float = 0.02) -> None:
        """In-place standard LM init: GPT-2 / Llama style. Embeddings and
        hidden linears both at `hidden_std` (0.02). RMSNorm gamma stays at 1.

        Equivalent to `standard_init(self)` from init.py — duplicated inline
        so building a model doesn't require importing init.py.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=hidden_std, a=-3*hidden_std, b=3*hidden_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_router_info: bool = False,
    ):
        """Args:
            input_ids: (B, T) int64.
            return_router_info: if True (and FFN is MoE), return per-layer
                RouterOutputs. Used by the training loop to apply the bias
                update and collect aux losses.

        Returns:
            logits: (B, T, vocab_size) float32.
            If return_router_info: also a list of `RouterOutput | None`,
            length n_layers.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"seq_len {T} > max_seq_len {self.cfg.max_seq_len}; "
            "recompute freqs_cis with a larger max_seq_len."
        )

        x = self.tok_emb(input_ids)                # (B, T, d_model)
        freqs_cis = self.freqs_cis[:T]             # slice for current length

        router_infos = []
        for block in self.blocks:
            x, router_info = block(x, freqs_cis)
            router_infos.append(router_info)

        x = self.final_norm(x)
        logits = self.unembed(x)                   # (B, T, vocab_size)

        if return_router_info:
            return logits, router_infos
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Naive autoregressive generation (no KV cache).

        For real inference use the HF transformers library or vLLM. This
        implementation is for sanity-checking the model and for the smoke
        tests in the notebook.
        """
        for _ in range(max_new_tokens):
            # Slice to the model's max len in case the prompt is already long.
            ids_in = input_ids[:, -self.cfg.max_seq_len:]
            logits = self.forward(ids_in)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, -1:].expand_as(logits)] = -float("inf")
            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
        return input_ids

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def count_params(self) -> dict[str, int]:
        """Parameter count broken down by group. Useful for ablations."""
        counts = {"embedding": 0, "attention": 0, "ffn": 0, "norm": 0, "head": 0, "total": 0}
        seen_ids = set()
        for name, p in self.named_parameters():
            # Skip aliased params (weight tying makes embed and unembed point
            # at the same tensor; we count it once).
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            counts["total"] += p.numel()
            if "tok_emb" in name:
                counts["embedding"] += p.numel()
            elif "unembed" in name:
                counts["head"] += p.numel()
            elif ".attn." in name:
                counts["attention"] += p.numel()
            elif ".ffn." in name:
                counts["ffn"] += p.numel()
            elif "norm" in name:
                counts["norm"] += p.numel()
        return counts


if __name__ == "__main__":
    # Tiny smoke test: build, forward, shape-check, loss-at-init, backward.
    torch.manual_seed(0)

    # Small DeepSeek-V3-shaped dense model.
    cfg = ModelConfig(
        vocab_size=2048, d_model=128, n_layers=4,
        attention_type="mla", n_heads=4, d_head=32,
        d_rope=16, d_kv_latent=32,
        ffn_type="dense", max_seq_len=256, tie_weights=True,
    )
    model = TransformerLM(cfg)
    print(f"--- Built {cfg.attention_type.upper()} + {cfg.ffn_type} model ---")

    counts = model.count_params()
    for k, v in counts.items():
        print(f"  {k:10s} {v/1e6:>7.3f}M")

    # Forward shape.
    B, T = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, cfg.vocab_size), f"wrong shape: {logits.shape}"
    print(f"\n  forward OK: {tuple(ids.shape)} -> {tuple(logits.shape)}")

    # Loss at init — should be ~ ln(vocab_size).
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    expected = torch.tensor(cfg.vocab_size).log().item()
    print(f"  loss at init: {loss.item():.3f}  (expected ~{expected:.3f})")
    assert abs(loss.item() - expected) < 1.0, "loss-at-init off by more than 1 nat"

    # Backward + step.
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.grad is None)
    print(f"  params with grad: {sum(1 for p in model.parameters() if p.grad is not None)}, with None: {n_none}")
    assert n_none == 0, "some parameters didn't receive gradients"

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step(); opt.zero_grad()
    loss2 = F.cross_entropy(model(ids).reshape(-1, cfg.vocab_size), targets.reshape(-1))
    print(f"  loss after one step: {loss2.item():.3f}  (was {loss.item():.3f})")
    assert loss2.item() < loss.item(), "loss didn't decrease after one optimizer step"

    # Generation smoke test.
    prompt = torch.tensor([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)
    print(f"  generate OK: {tuple(prompt.shape)} -> {tuple(out.shape)}")

    print("\nAll shape and sanity checks passed.")
