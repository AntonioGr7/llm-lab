"""Correctness tests for the LoRA implementation — the heart of Module 16.

Pin the properties the README claims, all offline (torch only, no network/GPU):

  1. Zero-init.        B=0 at init ⇒ ΔW=0 ⇒ a freshly injected model is
                       byte-for-byte the base model. Training starts from the
                       pretrained point, not a perturbed one.
  2. Targeting.        Only the named projections are wrapped; everything else
                       is untouched. Matching zero layers is an error.
  3. Freezing.         After injection only lora_A/lora_B require grad, and the
                       optimizer's requires_grad filter sees exactly those.
  4. Scaling.          The update is multiplied by alpha/r, so sweeping r at
                       fixed alpha/r leaves the effective step size unchanged.
  5. Merge equivalence. After training, folding ΔW into the base reproduces the
                       adapter forward exactly, and unwraps to a plain Linear.
  6. State-dict roundtrip. lora_state_dict ↔ load_lora_state_dict preserves the
                       adapter, and the file is tiny relative to the model.
  7. Base stays frozen through a real train_step; adapters move.

Usage:
    cd 16-parameter-efficient-finetuning/
    python tests/test_lora.py
Exits 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from lora import (  # noqa: E402
    LoRALinear, LoRASpec, inject_lora_adapters, mark_only_lora_trainable,
    merge_lora_weights, lora_state_dict, load_lora_state_dict, trainable_summary,
)


def _tiny_stack():
    """A 2-attr stack with one targeted and one untargeted Linear."""
    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.up_proj = nn.Linear(32, 48)
            self.lm_head = nn.Linear(48, 100)   # NOT a target

        def forward(self, x):
            return self.lm_head(self.up_proj(self.q_proj(x)))
    return Stack()


def test_zero_init():
    torch.manual_seed(0)
    m = _tiny_stack()
    x = torch.randn(4, 32)
    before = m(x).detach().clone()
    inject_lora_adapters(m, LoRASpec(r=4, alpha=8))
    after = m(x)
    assert torch.allclose(before, after, atol=1e-6), "ΔW must be zero at init"
    print("  1. zero-init: injected model == base model  ✓")


def test_targeting():
    m = _tiny_stack()
    n = inject_lora_adapters(m, LoRASpec(r=4, target_modules=("q_proj", "up_proj")))
    assert n == 2
    assert isinstance(m.q_proj, LoRALinear) and isinstance(m.up_proj, LoRALinear)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, LoRALinear)
    # matching zero layers is a loud error
    try:
        inject_lora_adapters(_tiny_stack(), LoRASpec(target_modules=("does_not_exist",)))
        raise AssertionError("expected ValueError for zero matched layers")
    except ValueError:
        pass
    print("  2. targeting: only named projections wrapped; zero-match raises  ✓")


def test_freezing():
    from config import OptimizerConfig
    from optim import build_optimizer
    m = _tiny_stack()
    inject_lora_adapters(m, LoRASpec(r=4))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_A", "lora_B")) for n in trainable), trainable
    opt = build_optimizer(m, OptimizerConfig())
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_opt == n_train, (n_opt, n_train)
    print("  3. freezing: only adapters require grad; optimizer matches  ✓")


def test_scaling():
    base = nn.Linear(16, 16, bias=False)
    # same alpha/r ratio at two ranks => same-magnitude update for matched A,B norms
    lin1 = LoRALinear(nn.Linear(16, 16, bias=False), r=4, alpha=8)    # scaling 2.0
    lin2 = LoRALinear(nn.Linear(16, 16, bias=False), r=8, alpha=16)   # scaling 2.0
    assert abs(lin1.scaling - 2.0) < 1e-9 and abs(lin2.scaling - 2.0) < 1e-9
    lin3 = LoRALinear(nn.Linear(16, 16, bias=False), r=4, alpha=4)    # scaling 1.0
    assert abs(lin3.scaling - 1.0) < 1e-9
    print("  4. scaling: ΔW multiplied by alpha/r  ✓")


def test_merge_equivalence():
    torch.manual_seed(1)
    m = _tiny_stack()
    inject_lora_adapters(m, LoRASpec(r=4, alpha=8))
    # perturb adapters so ΔW != 0
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("lora_B"):
                p.copy_(torch.randn_like(p) * 0.05)
            if n.endswith("lora_A"):
                p.copy_(torch.randn_like(p) * 0.05)
    x = torch.randn(4, 32)
    adapter_out = m(x).detach().clone()
    n = merge_lora_weights(m)
    assert n == 2
    assert isinstance(m.q_proj, nn.Linear) and not isinstance(m.q_proj, LoRALinear)
    merged_out = m(x)
    assert torch.allclose(adapter_out, merged_out, atol=1e-5), \
        (adapter_out - merged_out).abs().max().item()
    print("  5. merge: folded ΔW reproduces adapter forward; unwrapped to Linear  ✓")


def test_state_dict_roundtrip():
    torch.manual_seed(2)
    m = _tiny_stack()
    inject_lora_adapters(m, LoRASpec(r=4, alpha=8))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("lora_B"):
                p.copy_(torch.randn_like(p) * 0.1)
    sd = lora_state_dict(m)
    assert all(k.endswith(("lora_A", "lora_B")) for k in sd)
    # adapter is tiny vs the model
    adapter_n = sum(t.numel() for t in sd.values())
    total_n = sum(p.numel() for p in m.parameters())
    assert adapter_n < 0.5 * total_n
    # roundtrip into a fresh model with the SAME base weights (reseed so the
    # frozen base matches; load_lora_state_dict then supplies the adapters)
    torch.manual_seed(2)
    m2 = _tiny_stack()
    inject_lora_adapters(m2, LoRASpec(r=4, alpha=8))
    load_lora_state_dict(m2, sd)
    x = torch.randn(4, 32)
    assert torch.allclose(m(x), m2(x), atol=1e-6)
    print(f"  6. state-dict roundtrip: adapter {adapter_n}/{total_n} params, reload matches  ✓")


def test_train_step_freezes_base():
    from config import OptimizerConfig
    from optim import build_optimizer
    from loop import train_step
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(3)
    m = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=64,
        tie_word_embeddings=True, use_cache=False)).float()
    base_w = m.model.layers[0].self_attn.q_proj.weight.detach().clone()
    inject_lora_adapters(m, LoRASpec(r=4, alpha=8))
    opt = build_optimizer(m, OptimizerConfig(lr=1e-3))

    B, S = 2, 12
    ids = torch.randint(0, 256, (B, S))
    labels = ids.clone(); labels[:, :S // 2] = -100
    batch = {"input_ids": ids, "labels": labels,
             "attention_mask": torch.ones(B, S, dtype=torch.long)}

    def gen():
        while True:
            yield batch

    before = {k: v.clone() for k, v in lora_state_dict(m).items()}
    loss, gn = train_step(m, opt, gen(), grad_accum=2, grad_clip=1.0, dtype="fp32", device="cpu")
    after = lora_state_dict(m)

    now_w = m.model.layers[0].self_attn.q_proj.base.weight.detach()
    assert torch.allclose(base_w, now_w), "frozen base weight moved during train_step!"
    moved = sum((after[k] - before[k]).abs().sum().item() for k in after)
    assert moved > 0, "adapters did not update"
    assert loss == loss and gn == gn  # finite (not NaN)
    s = trainable_summary(m)
    print(f"  7. train_step: loss={loss:.3f} base frozen ✓ adapters moved ✓ "
          f"({s['trainable_pct']:.1f}% trainable)")


if __name__ == "__main__":
    tests = [
        test_zero_init, test_targeting, test_freezing, test_scaling,
        test_merge_equivalence, test_state_dict_roundtrip, test_train_step_freezes_base,
    ]
    print(f"Running {len(tests)} LoRA tests...")
    for t in tests:
        t()
    print("\nAll LoRA tests passed.")
