"""Optimizer construction for Part 3 pretraining.

Two paths:

- **AdamW** (default): the production-canonical setup. Two param groups
  (decay / no-decay) split by `p.ndim`. Used at every frontier lab in 2026.
- **Muon** (research path): Jordan 2024's Newton-Schulz orthogonalized
  momentum on 2D Linear weights, with AdamW falling back for embeddings,
  norms, and biases. Faster wall-clock on small/mid scale; not yet shipped
  in frontier production. See `muon.py` and Module 08 README § 4.

Both return objects that quack like `torch.optim.Optimizer`, so they
compose cleanly with any LRScheduler (Module 09).
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
    """Build the optimizer for `cfg.type`.

    Args:
        model: the model to optimize.
        cfg: hyperparameters.
        param_groups: optional caller-provided groups. If `None`, the
            default groups are built for the chosen optimizer type:
            - AdamW: decay (2D+) / no-decay (1D) split.
            - Muon: 2D Linear weights -> Muon; embeddings/norms/biases -> AdamW.
            If passed, used as-is — useful for muP-style per-group LR scaling
            from Module 09's discussion, or any custom split.
    """
    if cfg.type == "adamw":
        return _build_adamw(model, cfg, param_groups)
    if cfg.type == "muon":
        return _build_muon(model, cfg, param_groups)
    raise ValueError(f"unknown optimizer type: {cfg.type!r}")


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

    return torch.optim.AdamW(
        param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=True,
    )


def _build_muon(
    model: nn.Module,
    cfg: OptimizerConfig,
    param_groups: Optional[list[dict]],
) -> torch.optim.Optimizer:
    # Lazy import — only pay the cost when muon is actually requested.
    from muon import MuonAdamW, split_muon_adamw_groups

    if param_groups is None:
        param_groups = split_muon_adamw_groups(
            model,
            muon_lr=cfg.muon_lr,
            muon_momentum=cfg.muon_momentum,
            muon_ns_steps=cfg.muon_ns_steps,
            adamw_lr=cfg.lr,
            adamw_betas=cfg.betas,
            adamw_eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    return MuonAdamW(param_groups)


if __name__ == "__main__":
    # Smoke test both paths.
    from model import build_model
    from config import ModelConfig, OptimizerConfig

    model = build_model(ModelConfig(
        vocab_size=2048, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ffn=256, max_seq=128,
    ))

    print("--- AdamW (default) ---")
    opt_adamw = build_optimizer(model, OptimizerConfig(type="adamw", lr=3e-4))
    for i, g in enumerate(opt_adamw.param_groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  group {i} '{g.get('name','?')}': lr={g['lr']:.2e}  wd={g['weight_decay']}  n_params={n:,}")

    print("\n--- Muon (hybrid Muon + AdamW) ---")
    opt_muon = build_optimizer(model, OptimizerConfig(type="muon", lr=3e-4, muon_lr=2e-2))
    for i, g in enumerate(opt_muon.param_groups):
        n = sum(p.numel() for p in g["params"])
        otype = g.get("optimizer_type", "?")
        print(f"  group {i} '{g.get('name','?')}' ({otype}): lr={g['lr']:.2e}  n_params={n:,}")
