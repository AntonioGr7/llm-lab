"""Optimizer construction for Module 15 — SFT.

Adapted from Module 11's `optim.py`, trimmed to the single path SFT needs:
**AdamW** with a decay / no-decay param split.

The Muon path from Module 11 is intentionally dropped here. Muon is a
*pretraining* accelerator (Newton-Schulz orthogonalized momentum on 2D
weights, Module 08 §4); SFT is a short nudge from a good initialization
where AdamW with a small LR is the universal choice. Every frontier lab's
SFT stage uses AdamW. If you want to experiment with Muon for SFT, copy
`muon.py` + the `_build_muon` branch back in from Module 11 — the
`build_optimizer` dispatch is structured to make that a clean addition.

The decay / no-decay split is the production-canonical setup: weight decay
applies to 2D weight matrices but NOT to 1D tensors (biases, RMSNorm gains,
embeddings-as-1D-rows). With SFT's default `weight_decay=0.0` the split is a
no-op, but we keep it so that bumping `weight_decay` for a large-dataset SFT
run (README §5) does the right thing automatically.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import OptimizerConfig


def build_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig,
    param_groups: Optional[list[dict]] = None,
) -> torch.optim.Optimizer:
    """Build the AdamW optimizer for SFT.

    Args:
        model: the model to optimize. For SFT this is a full HF causal LM
            with every parameter trainable (full fine-tuning).
        cfg: hyperparameters (lr, betas, eps, weight_decay).
        param_groups: optional caller-provided groups. If `None`, the default
            decay (2D+) / no-decay (1D) split is built. Pass your own to do
            layer-wise LR decay or freeze subsets.
    """
    if cfg.type != "adamw":
        raise ValueError(
            f"Module 15 ships AdamW only; got optimizer.type={cfg.type!r}. "
            "See the module docstring for how to add Muon back."
        )
    return _build_adamw(model, cfg, param_groups)


def _build_adamw(
    model: nn.Module,
    cfg: OptimizerConfig,
    param_groups: Optional[list[dict]],
) -> torch.optim.Optimizer:
    if param_groups is None:
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 else decay).append(p)
        param_groups = [
            {"params": decay,    "weight_decay": cfg.weight_decay, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0,              "name": "no_decay"},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

    # `fused=True` is the fast CUDA path; harmless to request on CPU where
    # PyTorch falls back to the foreach implementation.
    return torch.optim.AdamW(
        param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
        fused=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Smoke test: build a tiny HF model, show the param-group split.
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=2048, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=128, use_cache=False,
    ))

    print("--- AdamW (SFT defaults: lr=1e-5, wd=0.0) ---")
    opt = build_optimizer(model, OptimizerConfig())
    for i, g in enumerate(opt.param_groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  group {i} '{g.get('name','?')}': lr={g['lr']:.2e}  "
              f"wd={g['weight_decay']}  n_params={n:,}")
