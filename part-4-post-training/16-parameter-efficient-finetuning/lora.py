"""LoRA from scratch — the one file in this module that is genuinely new.

Everything else here (data, loop, optimizer, schedule, sharding) is copied
from Module 15's full-FT SFT. LoRA changes exactly one thing: *which weights
the optimizer is allowed to move*. This file implements that change in ~3
concepts, all standard `torch`, no `peft` dependency:

    1. `LoRALinear`  — a drop-in wrapper around a frozen `nn.Linear` that adds
                       a trainable low-rank update  ΔW = (alpha/r) · B @ A.
    2. `inject_lora_adapters` — walk the model, swap every targeted `nn.Linear`
                       for a `LoRALinear`, and freeze everything except the
                       new A/B matrices.
    3. `merge_lora_weights` — fold ΔW back into the base weight so inference
                       runs at full speed with zero extra parameters.

The math (Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*):

    A pretrained linear layer computes  y = W x  (+ b),  W ∈ ℝ^(out×in).
    Full fine-tuning learns a dense update  W ← W + ΔW,  ΔW ∈ ℝ^(out×in)
    — that's `out×in` trainable numbers, the same size as W.

    LoRA hypothesizes ΔW is *low-rank*: ΔW = B @ A with  A ∈ ℝ^(r×in),
    B ∈ ℝ^(out×r),  r ≪ min(in,out).  Now you train only `r·(in+out)`
    numbers. For a 2048×2048 attention projection at r=8 that is 32,768
    instead of 4,194,304 — a 128× reduction in trainable params *for that
    layer*, and the optimizer state shrinks by the same factor.

Why it works: the "intrinsic dimensionality" results (Aghajanyan et al. 2020)
show that adapting a pretrained model to a downstream task lives in a
surprisingly low-dimensional subspace. LoRA bets the *update* needs far less
capacity than the weights themselves — and at SFT/instruction-tuning scale,
that bet holds: LoRA gets within a point or two of full FT on most tasks.

Two initialization details that are load-bearing:

    - `A` is initialized with Kaiming-uniform noise, `B` is initialized to
      **zero**. So at step 0, ΔW = B@A = 0 and the wrapped layer is *exactly*
      the base layer. Training starts from the pretrained model, not from a
      randomly perturbed one. (Reverse the roles — zero A, random B — and you
      get the same zero product; the convention is zero-B.)
    - `scaling = alpha / r`. Increasing `r` would otherwise change the
      magnitude of the update; dividing by `r` keeps the effective step size
      roughly constant as you sweep rank, so you don't have to re-tune the LR.
      (`alpha` is usually set to `r` or `2r`; we default to `2r`.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# The default set of projection names to adapt. These match Qwen3 / Llama /
# Mistral / Gemma decoder layers. Adapting all seven linear projections
# (attention q/k/v/o + MLP gate/up/down) is the "LoRA everywhere" recipe from
# QLoRA (Dettmers et al. 2023), which found that adapting *more* modules at a
# *smaller* rank beats adapting only q/v at a larger rank.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",          # SwiGLU MLP
)


# =============================================================================
# The wrapper
# =============================================================================

class LoRALinear(nn.Module):
    """Wrap a frozen `nn.Linear` with a trainable low-rank update.

    forward(x) = base(x) + scaling · (dropout(x) @ Aᵀ) @ Bᵀ

    The base layer (weight + optional bias) is kept as a frozen submodule, so
    a quantized base (bitsandbytes `Linear4bit` for QLoRA) drops straight in —
    `base(x)` dequantizes on the fly and the adapter math is unchanged.
    """

    def __init__(
        self,
        base: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be >= 1, got {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # in/out features: nn.Linear and bnb.Linear4bit both expose these.
        in_features = getattr(base, "in_features", None)
        out_features = getattr(base, "out_features", None)
        if in_features is None or out_features is None:
            raise TypeError(
                f"LoRALinear expects a base with in_features/out_features, "
                f"got {type(base).__name__}"
            )
        self.in_features = in_features
        self.out_features = out_features

        # Put the adapter on the same device/dtype as the base weight, but keep
        # adapters in a higher-precision compute dtype than a 4-bit base. We
        # read the base param's device; dtype defaults to float32 master so the
        # optimizer updates are stable (FSDP/autocast cast to bf16 for compute).
        base_param = next((p for p in base.parameters()), None)
        device = base_param.device if base_param is not None else None
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.reset_lora_parameters()

        # Freeze the base; only A/B train.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def reset_lora_parameters(self) -> None:
        """Kaiming-uniform A, zero B → ΔW = 0 at initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Compute the adapter branch in the adapter's own dtype, then cast the
        # result back to the base output's dtype before adding. This keeps the
        # low-rank update numerically clean even when base_out is bf16/4-bit.
        x_adapt = self.lora_dropout(x).to(self.lora_A.dtype)
        delta = (x_adapt @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * delta.to(base_out.dtype)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3g}")


# =============================================================================
# Config + injection
# =============================================================================

@dataclass
class LoRASpec:
    """The knobs `inject_lora_adapters` needs. Mirrors `config.LoRAConfig` but
    kept dependency-free here so `lora.py` is importable on its own."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES


def _iter_target_linears(model: nn.Module, targets: tuple[str, ...]):
    """Yield (parent_module, attr_name, child) for every Linear whose attribute
    name is in `targets`. We match on the *leaf* attribute name (e.g. `q_proj`)
    so the same spec works across architectures regardless of layer indexing."""
    for parent in model.modules():
        for attr, child in list(parent.named_children()):
            if attr in targets and _is_linear_like(child):
                yield parent, attr, child


def _is_linear_like(m: nn.Module) -> bool:
    """A plain nn.Linear, or a bitsandbytes 4-bit/8-bit Linear (QLoRA base)."""
    if isinstance(m, nn.Linear):
        return True
    return type(m).__name__ in ("Linear4bit", "Linear8bitLt")


def inject_lora_adapters(model: nn.Module, spec: LoRASpec) -> int:
    """Replace every targeted Linear in `model` with a `LoRALinear`, in place.

    Returns the number of layers adapted. After this call, ONLY the adapter
    A/B matrices require grad — call `inject_lora_adapters` then build the
    optimizer and its `requires_grad` filter does the rest (see `optim.py`).

    Idempotent guard: a layer that is already a `LoRALinear` is skipped.
    """
    n = 0
    for parent, attr, child in _iter_target_linears(model, spec.target_modules):
        if isinstance(child, LoRALinear):
            continue
        setattr(parent, attr, LoRALinear(child, r=spec.r, alpha=spec.alpha,
                                         dropout=spec.dropout))
        n += 1
    if n == 0:
        raise ValueError(
            f"inject_lora_adapters matched 0 layers for targets={spec.target_modules}. "
            "Check the names against `model.named_modules()` — they must be leaf "
            "attribute names like 'q_proj', not dotted paths."
        )
    mark_only_lora_trainable(model)
    return n


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter except `lora_A` / `lora_B`. Safe to call again."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.endswith("lora_A") or name.endswith("lora_B"))


# =============================================================================
# Merge (deployment)
# =============================================================================

@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> int:
    """Fold every adapter back into its base weight and unwrap the LoRALinear.

    After merging, `model` is a plain HF causal LM again — `W ← W + (alpha/r)·B@A`
    — with zero extra parameters and zero inference overhead. This is what you
    deploy. Returns the number of layers merged.

    Only standard `nn.Linear` bases can be merged in place (we write to
    `base.weight`). A 4-bit QLoRA base must be dequantized first; we raise a
    clear error pointing at that, because silently merging a bf16 adapter into a
    4-bit weight would either fail or quietly lose precision.
    """
    merged = 0
    for parent in model.modules():
        for attr, child in list(parent.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            base = child.base
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"merge_lora_weights can only fold into nn.Linear, but layer "
                    f"{attr!r} has a {type(base).__name__} base (quantized?). For "
                    "QLoRA, dequantize the base to fp16/bf16 first (see merge.py)."
                )
            delta = child.scaling * (child.lora_B @ child.lora_A)   # [out, in]
            base.weight.data += delta.to(base.weight.dtype)
            setattr(parent, attr, base)     # unwrap: parent.q_proj is a plain Linear again
            merged += 1
    return merged


# =============================================================================
# Adapter-only state dict (the tiny, portable artifact)
# =============================================================================

def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter tensors, keyed by their full parameter name.

    This is the artifact you actually ship: a few MB, not a few GB. The base
    model is whatever public checkpoint you started from, so a LoRA "model" is
    `(base name, adapter file)`. `load_lora_state_dict` reattaches it.
    """
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if n.endswith("lora_A") or n.endswith("lora_B")}


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """Load adapter tensors produced by `lora_state_dict` into an already-injected
    model (call `inject_lora_adapters` first so the A/B params exist)."""
    own = dict(model.named_parameters())
    missing = [k for k in sd if k not in own]
    if missing:
        raise KeyError(f"adapter keys not found in model (inject first?): {missing[:3]}...")
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].dtype).to(own[k].device))


# =============================================================================
# Accounting helpers (for the banner + notebook)
# =============================================================================

def trainable_summary(model: nn.Module) -> dict[str, float]:
    """Trainable vs total param counts and the LoRA savings ratio."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
        "reduction_x": total / max(trainable, 1),
    }


if __name__ == "__main__":
    # Offline smoke test on a tiny stack of Linears — no network, no GPU.
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)
            self.up_proj = nn.Linear(16, 32)
            self.untouched = nn.Linear(16, 16)

        def forward(self, x):
            return self.up_proj(self.q_proj(x))

    m = Tiny()
    x = torch.randn(4, 16)
    base_out = m(x).detach().clone()

    n = inject_lora_adapters(m, LoRASpec(r=4, alpha=8, dropout=0.0))
    print(f"adapted {n} layers (expected 2: q_proj, up_proj; untouched skipped)")

    # 1) zero-init B => identical output at step 0
    after_inject = m(x)
    assert torch.allclose(base_out, after_inject, atol=1e-5), "ΔW must be 0 at init"
    print("init: ΔW == 0  ✓  (output unchanged after injection)")

    # 2) only adapters train
    s = trainable_summary(m)
    print(f"trainable {s['trainable']:,} / {s['total']:,} "
          f"({s['trainable_pct']:.1f}%, {s['reduction_x']:.1f}× fewer)")
    assert s["trainable"] < s["total"]

    # 3) perturb adapters, then merge == adapter forward
    with torch.no_grad():
        for nm, p in m.named_parameters():
            if nm.endswith("lora_B"):
                p.add_(torch.randn_like(p) * 0.1)
    adapter_out = m(x).detach().clone()
    merged = merge_lora_weights(m)
    merged_out = m(x)
    assert torch.allclose(adapter_out, merged_out, atol=1e-5), "merge must preserve outputs"
    print(f"merged {merged} layers; merged forward == adapter forward  ✓")
    print("lora.py smoke OK")
