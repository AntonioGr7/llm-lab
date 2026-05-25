"""Initialization recipes for `TransformerLM` — standard and muP.

Two init paths are exposed:

- `standard_init(model, hidden_std=0.02)`: the GPT-2/Llama-style recipe.
  Embeddings get `std=1.0`; hidden linears get `std=0.02`; biases are
  zeroed (where present); RMSNorm `gamma` stays at 1.

- `mup_init(model, base_d, hidden_std=0.02)`: scales hidden and output
  weight init by the width multiplier $m = d_{\\text{model}} / d_{\\text{base}}$
  so that the optimal learning rate becomes width-invariant. See the
  module README, Section 4.

Both functions act in-place on `model.parameters()` and return `None`.
The matching learning-rate groups for muP are set up by `param_groups_mup`
in this file — call it when building the optimizer in Module 08.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncated_normal_(tensor: torch.Tensor, std: float, a: float = -3.0, b: float = 3.0):
    """In-place truncated normal init; truncates at ±3 standard deviations.

    Wrapper around `nn.init.trunc_normal_` to keep the call sites readable.
    """
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a * std, b=b * std)


def _zero_bias_(module: nn.Module):
    """If the module has a bias attribute and it's a parameter, zero it."""
    if hasattr(module, "bias") and isinstance(module.bias, torch.nn.Parameter):
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Standard init (GPT-2 / Llama style)
# ---------------------------------------------------------------------------

def standard_init(model: nn.Module, hidden_std: float = 0.02) -> None:
    """The default LM init (GPT-2 / Llama style).

    - Token embeddings: `hidden_std` (0.02) normal. Same scale as hidden
      linears — this is what keeps logits-at-init in the $\\ln(V)$ range.
    - All hidden linears (attention QKV/O, FFN gates/values): `hidden_std`
      truncated normal.
    - LM head (unembedding): `hidden_std` truncated normal (when untied;
      when tied, the alias to `tok_emb.weight` is what's used).
    - RMSNorm `gamma`: left at 1.0 (PyTorch default).
    - Biases (almost none in modern transformers, but if any): zeroed.
    """
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
        elif isinstance(module, nn.Linear):
            _truncated_normal_(module.weight, std=hidden_std)
            _zero_bias_(module)


# ---------------------------------------------------------------------------
# muP init (Yang & Hu, 2021)
# ---------------------------------------------------------------------------

def mup_init(
    model: nn.Module,
    d_model: int,
    base_d: int,
    hidden_std: float = 0.02,
) -> None:
    """muP-scaled init.

    Three weight groups, three scalings:

    - **Embeddings (input)**: same as standard. Std=1.0.
    - **Hidden weights**: std scaled by $1/\\sqrt{m}$ where $m = d/d_{\\text{base}}$.
    - **Output weights (LM head)**: std scaled by $1/m$.

    This pairs with the muP LR groups (`param_groups_mup`). Together they
    enforce mu-transfer: the optimal LR you tuned at `base_d` is also
    optimal at any wider `d_model`.

    Apply BEFORE weight tying. If your model ties weights, the LM head's
    own scaled init is overwritten by the alias to the embedding, which
    means the "output" scaling is moot — that's fine, and matches what
    Yang & Hu describe for tied-weight cases.

    Args:
        model: the `TransformerLM`. Must expose `tok_emb`, `unembed`, and
            inner linear layers via `model.modules()`.
        d_model: this run's width.
        base_d: the reference width you tuned hyperparameters at.
        hidden_std: the std at the base width (the value you tuned).
    """
    m = d_model / base_d
    hidden_scale = 1.0 / math.sqrt(m)
    output_scale = 1.0 / m

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            # Input — unchanged across widths.
            nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
        elif isinstance(module, nn.Linear):
            # Distinguish the LM head from hidden linears by name.
            if "unembed" in name or "lm_head" in name:
                _truncated_normal_(module.weight, std=hidden_std * output_scale)
            else:
                _truncated_normal_(module.weight, std=hidden_std * hidden_scale)
            _zero_bias_(module)


# ---------------------------------------------------------------------------
# muP parameter groups for the optimizer (used in Module 08)
# ---------------------------------------------------------------------------

def param_groups_mup(
    model: nn.Module,
    d_model: int,
    base_d: int,
    base_lr: float,
    weight_decay: float = 0.1,
) -> list[dict]:
    """Build optimizer param groups with muP LR scaling.

    Three groups:

    - **Input** (embeddings): LR = base_lr.
    - **Hidden** (linears inside blocks): LR = base_lr / m, where m = d/d_base.
    - **Output** (unembedding/LM head): LR = base_lr.

    Weight decay applies to all of them at the same rate. RMSNorm `gamma`
    parameters are explicitly excluded from weight decay — these are scale
    parameters, decaying them hurts training.

    Returns the `param_groups` list you'd pass to `torch.optim.AdamW`.
    """
    m = d_model / base_d
    hidden_lr = base_lr / m

    embedding_params: list[nn.Parameter] = []
    output_params:    list[nn.Parameter] = []
    hidden_params:    list[nn.Parameter] = []
    no_decay_params:  list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # RMSNorm.gamma — no weight decay.
        if "gamma" in name or name.endswith(".bias"):
            no_decay_params.append(p)
        elif "tok_emb" in name:
            embedding_params.append(p)
        elif "unembed" in name or "lm_head" in name:
            output_params.append(p)
        else:
            hidden_params.append(p)

    groups = [
        {"params": embedding_params, "lr": base_lr,   "weight_decay": weight_decay, "name": "embedding"},
        {"params": hidden_params,    "lr": hidden_lr, "weight_decay": weight_decay, "name": "hidden"},
        {"params": output_params,    "lr": base_lr,   "weight_decay": weight_decay, "name": "output"},
        {"params": no_decay_params,  "lr": base_lr,   "weight_decay": 0.0,           "name": "no_decay"},
    ]
    # Drop empty groups (e.g. tied-weight models have no separate output params).
    return [g for g in groups if len(g["params"]) > 0]


if __name__ == "__main__":
    # Sanity check: build a tiny model, apply each init, inspect the resulting
    # weight magnitudes by group.
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from model import TransformerLM, ModelConfig

    cfg = ModelConfig(
        vocab_size=2048, d_model=128, n_layers=2,
        n_heads=8, n_kv_heads=2, max_seq_len=128,
        tie_weights=False,  # disable so we can see the unembed init separately
    )
    model = TransformerLM(cfg)

    print("--- Standard init ---")
    standard_init(model)
    for kind, name in [("emb", "tok_emb"), ("hidden", "blocks.0"), ("output", "unembed")]:
        for n, p in model.named_parameters():
            if name in n and p.dim() >= 2:
                print(f"  {kind:8s} {n:40s}  std={p.std().item():.4f}")
                break

    print("\n--- muP init (d=128, base_d=64; m=2) ---")
    mup_init(model, d_model=128, base_d=64)
    for kind, name in [("emb", "tok_emb"), ("hidden", "blocks.0"), ("output", "unembed")]:
        for n, p in model.named_parameters():
            if name in n and p.dim() >= 2:
                print(f"  {kind:8s} {n:40s}  std={p.std().item():.4f}")
                break

    print("\n--- muP param groups (base_lr=3e-4) ---")
    groups = param_groups_mup(model, d_model=128, base_d=64, base_lr=3e-4)
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:10s} lr={g['lr']:.2e}  wd={g['weight_decay']}  n_params={n_params:,}")
