"""Evaluate a GRPO checkpoint.

Two signals, both useful:

  1. **GSM8K accuracy** — the headline number. Generate one completion per
     held-out test problem (greedy, no sampling) and score against the
     ground-truth integer. Reports overall accuracy + format-compliance
     rate. After a successful R1-style run on Qwen3-1.7B you should see
     accuracy climb from base (~30-40%) to GRPO'd (~55-70%) on GSM8K test.

  2. **Generation A/B vs reference** — qualitative. Same prompt, two
     completions: one from the trained policy, one from the loaded
     reference (the same checkpoint frozen). You should *see* the
     reasoning structure emerge: longer responses, explicit `<think>`
     blocks, structured solution traces.

Usage:

    # Score 500 GSM8K test problems (greedy)
    python eval.py --checkpoint=results/checkpoints/step_00000300 \\
        --config=configs/grpo_qwen3_1.7b.yaml --gsm8k --n=500

    # Baseline — score the same with the reference (untrained) model
    python eval.py --base --config=configs/grpo_qwen3_1.7b.yaml --gsm8k --n=500

    # Generation A/B on a custom prompt
    python eval.py --checkpoint=results/checkpoints/step_00000300 \\
        --config=configs/grpo_qwen3_1.7b.yaml \\
        --prompts "If 6 cats catch 6 mice in 6 minutes, how many cats catch 100 mice in 50 minutes?" \\
        --also-reference

`eval.py` is intentionally separate from `train.py` so you can run it on
a held-out box (or after the training run finishes) without holding the
whole training stack in memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_policy, build_reference, generation_mode
from data import load_tokenizer, parse_gsm8k_ground_truth
from rewards import compute_rewards, format_reward, accuracy_reward
from checkpoint import load as load_ckpt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tok, prompt: str, system_prompt: str, max_new_tokens: int,
             greedy: bool, temperature: float, top_p: float, device: str) -> str:
    """Single-prompt generation. Returns the decoded completion text."""
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

    with generation_mode(model):
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(not greedy),
            temperature=temperature if not greedy else 1.0,
            top_p=top_p if not greedy else 1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    completion_ids = out[0, input_ids.shape[1]:]
    return tok.decode(completion_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# GSM8K accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_gsm8k(model, tok, cfg: TrainConfig, n: int, device: str) -> dict:
    """Score N held-out GSM8K problems and report aggregate metrics."""
    from datasets import load_dataset
    ds = load_dataset(cfg.data.source, cfg.data.subset, split="test")
    n = min(n, len(ds))

    total = 0
    n_correct = 0
    n_format = 0
    n_both = 0
    examples = []
    print(f"[eval] scoring {n} GSM8K test problems...")
    for i in range(n):
        ex = ds[i]
        gt = parse_gsm8k_ground_truth(ex["answer"])
        if gt is None:
            continue
        completion = generate(
            model, tok, ex["question"],
            system_prompt=cfg.data.system_prompt,
            max_new_tokens=cfg.rl.max_new_tokens,
            greedy=True,
            temperature=cfg.rl.temperature,
            top_p=cfg.rl.top_p,
            device=device,
        )
        rf = format_reward(completion, cfg.reward.format_pattern)
        ra = accuracy_reward(completion, gt)
        total += 1
        n_correct += int(ra > 0)
        n_format += int(rf > 0)
        n_both += int(rf > 0 and ra > 0)
        if i < 5:
            preview = completion.replace("\n", " ")[:160]
            print(f"  [{i}] gt={gt}  acc={ra:.0f}  fmt={rf:.0f}  -> {preview!r}")
        examples.append({"q": ex["question"], "gt": gt, "completion": completion,
                         "accuracy": ra, "format": rf})
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{n}  acc so far: {n_correct/(total):.1%}")

    metrics = {
        "n": total,
        "accuracy": n_correct / max(total, 1),
        "format_rate": n_format / max(total, 1),
        "format_and_correct": n_both / max(total, 1),
    }
    print(f"\n[eval] GSM8K results on {total} problems:")
    print(f"  accuracy:           {metrics['accuracy']:.1%}")
    print(f"  format compliance:  {metrics['format_rate']:.1%}")
    print(f"  format AND correct: {metrics['format_and_correct']:.1%}")
    return metrics, examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Module 18 GRPO eval")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="DCP checkpoint dir to load. If omitted, evals the BASE model.")
    p.add_argument("--base", action="store_true",
                   help="Eval the reference (untrained) model — equivalent to --checkpoint=''.")
    p.add_argument("--gsm8k", action="store_true",
                   help="Run the GSM8K test-set accuracy eval.")
    p.add_argument("--n", type=int, default=200,
                   help="Number of GSM8K test problems to score (default 200).")
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Generate completions on these prompts (alternative to --gsm8k).")
    p.add_argument("--also-reference", action="store_true",
                   help="With --prompts: also generate from the reference for A/B.")
    p.add_argument("--out", default=None,
                   help="Optional JSON path to dump the per-example results.")
    args, extra = p.parse_known_args(argv)
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
    return args, overrides


def main(argv=None):
    args, overrides = _parse_args(argv)
    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(cfg.model.name, padding_side="left")

    # Load policy. Two paths: --base (the untrained reference) or
    # --checkpoint (the trained policy).
    if args.base or args.checkpoint is None:
        print("[eval] using BASE model (no fine-tuning)")
        model = build_reference(cfg.model).to(device)
        model.eval()
    else:
        print(f"[eval] loading checkpoint {args.checkpoint}")
        model = build_policy(cfg.model).to(device)
        load_ckpt(model, optimizer=None, ckpt_dir=args.checkpoint)
        model.eval()

    # GSM8K accuracy eval
    if args.gsm8k:
        metrics, examples = eval_gsm8k(model, tok, cfg, args.n, device)
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"metrics": metrics, "examples": examples[:50]}, indent=2,
            ))
            print(f"[eval] wrote {args.out}")

    # Custom-prompt A/B
    if args.prompts:
        if args.also_reference:
            ref = build_reference(cfg.model).to(device)
            ref.eval()
        else:
            ref = None
        for p in args.prompts:
            print("\n" + "=" * 72)
            print(f"PROMPT: {p}")
            print("=" * 72)
            print(f"\n[policy]")
            txt = generate(
                model, tok, p,
                system_prompt=cfg.data.system_prompt,
                max_new_tokens=cfg.rl.max_new_tokens,
                greedy=False,
                temperature=cfg.rl.temperature,
                top_p=cfg.rl.top_p,
                device=device,
            )
            print(txt)
            if ref is not None:
                print(f"\n[reference]")
                txt = generate(
                    ref, tok, p,
                    system_prompt=cfg.data.system_prompt,
                    max_new_tokens=cfg.rl.max_new_tokens,
                    greedy=False,
                    temperature=cfg.rl.temperature,
                    top_p=cfg.rl.top_p,
                    device=device,
                )
                print(txt)


if __name__ == "__main__":
    main()
