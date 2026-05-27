"""Export a trained checkpoint to a standard HuggingFace directory.

Pretraining saves DCP shards (and, via the README's Option B, a single
`state.pt`). Neither is the format inference engines want. This script
repackages a checkpoint into the universal HuggingFace layout —

    config.json   model.safetensors   tokenizer.json   tokenizer_config.json   ...

— which **vLLM**, **SGLang**, and **TGI** all load directly. No key remapping
is needed: `build_model` returns a genuine `Qwen3ForCausalLM`, so the saved
weights are already a standard Qwen3 state dict.

Usage:

    # from a DCP checkpoint dir (run under torchrun) ...
    torchrun --standalone --nproc_per_node=1 export_hf.py \
        --checkpoint=./results/checkpoints/step_00003000 \
        --out=./results/export/qwen3-demo

    # ... or from a converted state.pt (plain python, no torchrun) ...
    python export_hf.py \
        --checkpoint=./results/checkpoints/step_00003000 \
        --out=./results/export/qwen3-demo

Then serve it:

    vllm serve ./results/export/qwen3-demo --dtype bfloat16

The checkpoint must have been saved by `train.py`. Loading the DCP format
directly requires `torchrun` (the distributed branch of `checkpoint.load`);
to export with plain `python`, first convert to `state.pt` per the README's
"Loading a checkpoint on your laptop — Option B".
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from config import load_yaml, apply_dotted_overrides
from model import build_model
from checkpoint import load as load_ckpt
from fsdp_setup import init_distributed, cleanup_distributed

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _parse_args():
    p = argparse.ArgumentParser(description="Export a checkpoint to HuggingFace format")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to a DCP or single-process checkpoint directory")
    p.add_argument("--config", type=str, default="configs/demo_a100.yaml")
    p.add_argument("--out", type=str, default="./results/export/qwen3-demo",
                   help="destination directory for the HF model")
    p.add_argument("--dtype", type=str, default="bf16", choices=list(_DTYPES),
                   help="weight dtype to save; bf16 is the pragmatic serving default")
    args, extra = p.parse_known_args()
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
        else:
            raise SystemExit(f"unrecognized arg: {tok!r}")
    return args, overrides


def main():
    args, overrides = _parse_args()
    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)

    # init_distributed() is a no-op under plain `python` (returns rank 0); under
    # torchrun it sets up the process group so the DCP branch of load() works.
    init_distributed()

    model = build_model(cfg.model)                 # a genuine Qwen3ForCausalLM
    step = load_ckpt(model, None, args.checkpoint)  # weights-only (optimizer=None)
    print(f"[export] loaded checkpoint at step {step}", flush=True)

    model.config.use_cache = True                  # training set this False; inference wants the KV cache
    model = model.to(_DTYPES[args.dtype])
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg.data.tokenizer_name).save_pretrained(args.out)

    cleanup_distributed()
    print(f"[export] wrote HF model ({args.dtype}) to {args.out}", flush=True)
    print(f"[export] serve it with:  vllm serve {args.out} --dtype bfloat16", flush=True)


if __name__ == "__main__":
    main()
