"""Merge a trained LoRA adapter into its base — the deployment step.

A LoRA "model" is really two things: a public base checkpoint and a tiny
adapter file. At inference you have two options:

  1. **Keep them separate** and add ΔW = (alpha/r)·B@A at every forward. Costs
     a little latency and memory, but lets you hot-swap adapters (serve ten
     fine-tunes off one base in memory).
  2. **Merge** ΔW into the base weight once, producing a single standard model
     with *zero* inference overhead. This is what you ship when you serve one
     fine-tune at scale.

`merge.py` does (2): load the base, load just the adapter tensors from a
training checkpoint, fold them in, and `save_pretrained` a plain HF model that
vLLM / TGI / `from_pretrained` load with no LoRA code in sight.

    python merge.py --config=configs/lora_qwen3_1.7b.yaml \\
        --checkpoint=results/checkpoints/step_00000600 --out=results/merged

    # also dump the standalone adapter (a few MB) for hot-swap serving:
    python merge.py --config=... --checkpoint=... --out=results/merged \\
        --adapter-out=results/adapter.pt

Runs single-process on CPU or one GPU — no torchrun. The base is loaded in
BF16 (not 4-bit) even for a QLoRA run, because you merge a bf16 adapter into a
full-precision base; the adapter tensors are identical regardless of how the
base was quantized during training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import load_yaml, apply_dotted_overrides
from model import build_model
from lora import merge_lora_weights, lora_state_dict, load_lora_state_dict
from data import load_tokenizer


def _load_adapter_tensors(model, ckpt_dir: str) -> dict[str, torch.Tensor]:
    """Pull ONLY the adapter (lora_A/lora_B) tensors out of a training checkpoint.

    Works for both checkpoint flavors `checkpoint.save` produces:
      - single-process `state.pt` (a plain torch.save payload), and
      - multi-rank DCP directory (sharded).

    We never load the base weights from the checkpoint — they're just the frozen
    pretrained weights we already loaded fresh. This is also why a QLoRA
    checkpoint (4-bit base) merges fine: we only read its bf16/fp32 adapters.
    """
    p = Path(ckpt_dir)
    full = model.state_dict()
    adapter_keys = [k for k in full if k.endswith("lora_A") or k.endswith("lora_B")]
    if not adapter_keys:
        raise RuntimeError("model has no LoRA adapters — did build_model inject them?")

    state_pt = p / "state.pt"
    if state_pt.exists():
        saved = torch.load(state_pt, map_location="cpu", weights_only=False)["model"]
        return {k: saved[k] for k in adapter_keys}

    # DCP directory: ask only for the adapter keys (nested under "model" exactly
    # as checkpoint.save wrote them). DCP fills the provided target tensors.
    import torch.distributed.checkpoint as dcp
    target = {"model": {k: full[k].clone() for k in adapter_keys}}
    dcp.load(target, checkpoint_id=str(p))
    return target["model"]


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    ap.add_argument("--config", type=str, default="configs/lora_qwen3_1.7b.yaml")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="training checkpoint dir saved by train.py (step_XXXXXXXX/)")
    ap.add_argument("--out", type=str, required=True,
                    help="output dir for the merged HF model")
    ap.add_argument("--adapter-out", type=str, default=None,
                    help="also save the standalone adapter (lora_A/B only) to this .pt path")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="merge device (cpu is fine and avoids VRAM contention)")
    args, extra = ap.parse_known_args()
    overrides = [t[2:] for t in extra if t.startswith("--") and "=" in t]

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)

    # Always load a full-precision (bf16) base for merging, even for a QLoRA run.
    cfg.lora.qlora = False
    print(f"[merge] loading base {cfg.model.name!r} (bf16) + injecting r={cfg.lora.r} adapters")
    model = build_model(cfg.model, cfg.lora).to(args.device)
    model.eval()

    print(f"[merge] reading adapter tensors from {args.checkpoint}")
    sd = _load_adapter_tensors(model, args.checkpoint)
    load_lora_state_dict(model, sd)

    if args.adapter_out:
        adapter_sd = lora_state_dict(model)
        n_bytes = sum(t.numel() * t.element_size() for t in adapter_sd.values())
        torch.save(adapter_sd, args.adapter_out)
        print(f"[merge] wrote standalone adapter -> {args.adapter_out} "
              f"({len(adapter_sd)} tensors, {n_bytes/1e6:.1f} MB)")

    n = merge_lora_weights(model)
    print(f"[merge] folded {n} adapters into the base weights")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    load_tokenizer(cfg.model.name).save_pretrained(args.out)
    print(f"[merge] saved merged model -> {args.out}\n"
          f"        load it anywhere with AutoModelForCausalLM.from_pretrained({args.out!r})")


if __name__ == "__main__":
    main()
