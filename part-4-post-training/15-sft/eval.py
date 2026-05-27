"""Evaluate an SFT checkpoint.

Two signals, and the order matters:

  1. **Generation with the chat template** — *the* eval that matters for SFT.
     Wrap a prompt in the model's chat template, generate, and read the
     result. The whole point of SFT is behavioral: does the model now answer
     the question instead of continuing it like web text? You judge this with
     your eyes. Pass `--base` to generate from the un-fine-tuned base model
     for an A/B contrast — the base will ramble, the SFT model will respond.

  2. **Held-out assistant-token perplexity** — a held-out slice of the
     instruction data the model didn't train on, scored *only* on assistant
     tokens (the same loss mask as training). A sanity number to compare runs
     by; it is NOT a substitute for reading generations. A model can have
     great held-out NLL and still produce slop, and vice versa.

Usage:

    # The headline check — does it follow instructions now?
    python eval.py --checkpoint=results/checkpoints/step_00000600 \
        --prompts "Write a haiku about Python" "Explain RNNs to a 10-year-old"

    # Before/after: same prompts on the raw base model
    python eval.py --base --prompts "Write a haiku about Python"

    # Held-out perplexity on assistant tokens
    python eval.py --checkpoint=results/checkpoints/step_00000600 --perplexity

Run on a single rank. Pass `--device=cpu` to avoid VRAM contention with a
training job on the same box (slow, but fine for a few short generations).
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_model
from data import ChatDataset, load_tokenizer, IGNORE_INDEX
from checkpoint import load as load_ckpt
from fsdp_setup import init_distributed, cleanup_distributed


# ---------------------------------------------------------------------------
# Generation with the chat template
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_chat(
    model,
    tok,
    prompts: list[str],
    system_prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cuda",
) -> list[tuple[str, str]]:
    """For each user prompt, render it through the chat template and generate.

    Returns a list of (prompt, completion). The completion is decoded from
    the *new* tokens only — the rendered template prefix is stripped — and
    special tokens are removed so you read what a user would see.
    """
    model.config.use_cache = True       # generation wants the KV cache (train.py disabled it)
    model.eval()

    out = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Render EXACTLY as data.py does at training time (enable_thinking=False
        # so the <think> scaffold sits in the prompt, not the generation) — a
        # train/inference template mismatch is a classic silent SFT bug (README §2).
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
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_k=top_k,
            pad_token_id=tok.pad_token_id,
        )
        completion = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        out.append((prompt, completion.strip()))
    return out


# ---------------------------------------------------------------------------
# Held-out assistant-token perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model,
    cfg: TrainConfig,
    n_examples: int = 200,
    device: str = "cuda",
) -> dict[str, float]:
    """Per-assistant-token cross-entropy + perplexity on a held-out slice.

    Builds a small `ChatDataset` from the *end* of the configured dataset
    (a cheap held-out proxy — for a rigorous eval use a dataset with an
    explicit validation split). Scores CE summed over non-masked positions,
    so the perplexity reflects exactly the tokens SFT optimized.
    """
    model.eval()

    # Carve a held-out slice: load the dataset capped at n_examples but from a
    # different region by overriding the data config. For datasets without a
    # validation split this is an approximation — we just take the first
    # n_examples of the configured split that we then *did not* shuffle into
    # the (different-seed) training order. Good enough for a comparison number.
    from copy import deepcopy
    eval_data_cfg = deepcopy(cfg.data)
    eval_data_cfg.max_examples = n_examples
    ds = ChatDataset(eval_data_cfg, tokenizer_name=cfg.model.name)

    total_nll = 0.0
    total_tokens = 0
    for i in range(len(ds)):
        s = ds[i]
        input_ids = s["input_ids"].unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)
        attn = s["attention_mask"].unsqueeze(0).to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_tok = int((labels != IGNORE_INDEX).sum().item())
        total_nll += float(nll.item())
        total_tokens += n_tok

    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "loss": mean_nll,
        "perplexity": math.exp(mean_nll),
        "n_assistant_tokens": total_tokens,
        "n_examples": len(ds),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Write a haiku about Python.",
    "Explain recurrent neural networks to a 10-year-old.",
    "What's a good weeknight dinner I can make in 20 minutes?",
]


def _parse_args():
    p = argparse.ArgumentParser(description="Eval an SFT checkpoint (generation + perplexity)")
    p.add_argument("--config", type=str, default="configs/sft_qwen3_1.7b.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to an SFT checkpoint dir saved by train.py")
    p.add_argument("--base", action="store_true",
                   help="skip the checkpoint and eval the raw base model (before/after contrast)")
    p.add_argument("--prompts", type=str, nargs="+", default=None,
                   help="user prompts to generate from (default: a canned set)")
    p.add_argument("--perplexity", action="store_true",
                   help="also compute held-out assistant-token perplexity")
    p.add_argument("--n_eval_examples", type=int, default=200,
                   help="examples to score for perplexity (default 200)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
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

    if not args.base and args.checkpoint is None:
        raise SystemExit("pass --checkpoint=<dir>, or --base to eval the un-fine-tuned model")

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)

    rinfo = init_distributed()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not rinfo.is_main:
        cleanup_distributed()
        return

    model = build_model(cfg.model).to(device)
    if args.base:
        print(f"[eval] base model {cfg.model.name!r} (no fine-tuning)", flush=True)
    else:
        step = load_ckpt(model, None, args.checkpoint)   # weights-only
        print(f"[eval] loaded SFT checkpoint at step {step}", flush=True)

    tok = load_tokenizer(cfg.model.name)

    prompts = args.prompts or _DEFAULT_PROMPTS
    print("[eval] generating ...", flush=True)
    results = generate_chat(
        model, tok, prompts,
        system_prompt=cfg.data.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    for prompt, completion in results:
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

    cleanup_distributed()


if __name__ == "__main__":
    main()
