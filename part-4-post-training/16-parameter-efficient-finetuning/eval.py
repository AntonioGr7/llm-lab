"""Evaluate a LoRA fine-tune.

Same two signals as Module 15's SFT eval — generation with the chat template
(the one that matters) and held-out assistant-token perplexity — but the model
can be loaded three ways:

  --base               the raw, un-fine-tuned base model (the "before")
  --checkpoint <dir>   base + LoRA adapters from a training checkpoint, adapters
                       kept *live* (not merged) — the "after"
  --merged <dir>       a merged model produced by merge.py (a plain HF model)

The headline LoRA check is the same as SFT: does the model follow the
instruction now? Run once with --base and once with --checkpoint on the same
prompts and read the difference. The point of this module is that you get that
difference for a fraction of the cost of full fine-tuning.

    # before / after on the same prompts
    python eval.py --base --prompts "Write a haiku about Python"
    python eval.py --checkpoint=results/checkpoints/step_00000600 \
        --prompts "Write a haiku about Python"

    # held-out assistant-token perplexity of the adapter
    python eval.py --checkpoint=results/checkpoints/step_00000600 --perplexity

Single rank. `--device=cpu` to avoid VRAM contention with a training job.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from config import load_yaml, apply_dotted_overrides
from model import build_model
from data import ChatDataset, load_tokenizer, IGNORE_INDEX
from lora import load_lora_state_dict
from merge import _load_adapter_tensors


# ---------------------------------------------------------------------------
# Generation with the chat template (verbatim from Module 15)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_chat(model, tok, prompts, system_prompt="", max_new_tokens=256,
                  temperature=0.7, top_k=40, device="cuda"):
    model.config.use_cache = True
    model.eval()
    out = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=False,
            ).to(device)
        except TypeError:
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(device)
        gen = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=max(temperature, 1e-6), top_k=top_k, pad_token_id=tok.pad_token_id,
        )
        completion = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        out.append((prompt, completion.strip()))
    return out


# ---------------------------------------------------------------------------
# Held-out assistant-token perplexity (verbatim from Module 15)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model, cfg, n_examples=200, device="cuda"):
    model.eval()
    from copy import deepcopy
    eval_data_cfg = deepcopy(cfg.data)
    eval_data_cfg.max_examples = n_examples
    ds = ChatDataset(eval_data_cfg, tokenizer_name=cfg.model.name)
    total_nll, total_tokens = 0.0, 0
    for i in range(len(ds)):
        s = ds[i]
        input_ids = s["input_ids"].unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)
        attn = s["attention_mask"].unsqueeze(0).to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                              ignore_index=IGNORE_INDEX, reduction="sum")
        total_nll += float(nll.item())
        total_tokens += int((labels != IGNORE_INDEX).sum().item())
    mean_nll = total_nll / max(total_tokens, 1)
    return {"loss": mean_nll, "perplexity": math.exp(mean_nll),
            "n_assistant_tokens": total_tokens, "n_examples": len(ds)}


# ---------------------------------------------------------------------------
# Model loading — the LoRA-specific part
# ---------------------------------------------------------------------------

def load_eval_model(cfg, args, device):
    """Three ways in, per the CLI flags. Returns the model on `device`."""
    from transformers import AutoModelForCausalLM
    if args.merged:
        print(f"[eval] merged model {args.merged!r}", flush=True)
        return AutoModelForCausalLM.from_pretrained(args.merged, dtype=torch.bfloat16).to(device)
    if args.base:
        print(f"[eval] base model {cfg.model.name!r} (no fine-tuning)", flush=True)
        return AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16).to(device)
    # --checkpoint: base + live LoRA adapters (force bf16 base for eval/merge parity)
    cfg.lora.qlora = False
    model = build_model(cfg.model, cfg.lora).to(device)
    sd = _load_adapter_tensors(model, args.checkpoint)
    load_lora_state_dict(model, sd)
    print(f"[eval] base + LoRA adapter from {args.checkpoint} "
          f"(r={cfg.lora.r}, adapters live, not merged)", flush=True)
    return model


_DEFAULT_PROMPTS = [
    "Write a haiku about Python.",
    "Explain recurrent neural networks to a 10-year-old.",
    "What's a good weeknight dinner I can make in 20 minutes?",
]


def _parse_args():
    p = argparse.ArgumentParser(description="Eval a LoRA fine-tune (generation + perplexity)")
    p.add_argument("--config", type=str, default="configs/lora_qwen3_1.7b.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="LoRA training checkpoint dir (base + adapters, kept live)")
    p.add_argument("--merged", type=str, default=None,
                   help="a merged model dir produced by merge.py")
    p.add_argument("--base", action="store_true",
                   help="eval the raw base model (the before/after contrast)")
    p.add_argument("--prompts", type=str, nargs="+", default=None)
    p.add_argument("--perplexity", action="store_true")
    p.add_argument("--n_eval_examples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args, extra = p.parse_known_args()
    overrides = [t[2:] for t in extra if t.startswith("--") and "=" in t]
    bad = [t for t in extra if not (t.startswith("--") and "=" in t)]
    if bad:
        raise SystemExit(f"unrecognized arg(s): {bad}")
    return args, overrides


def main():
    args, overrides = _parse_args()
    if not args.base and not args.checkpoint and not args.merged:
        raise SystemExit("pass one of --checkpoint=<dir>, --merged=<dir>, or --base")

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    model = load_eval_model(cfg, args, device)
    tok = load_tokenizer(cfg.model.name)

    prompts = args.prompts or _DEFAULT_PROMPTS
    print("[eval] generating ...", flush=True)
    for prompt, completion in generate_chat(
        model, tok, prompts, system_prompt=cfg.data.system_prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, device=device,
    ):
        print(f"\nUSER:      {prompt}")
        print(f"ASSISTANT: {completion}")
    print()

    if args.perplexity:
        print("[eval] held-out assistant-token perplexity ...", flush=True)
        r = evaluate_perplexity(model, cfg, n_examples=args.n_eval_examples, device=device)
        print(f"\n  loss (per assistant token)  {r['loss']:.4f}")
        print(f"  perplexity                  {r['perplexity']:.2f}")
        print(f"  assistant tokens scored     {r['n_assistant_tokens']:,} "
              f"over {r['n_examples']} examples\n")


if __name__ == "__main__":
    main()
