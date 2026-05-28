"""Two-axis eval for Module 18.

This is THE diagnostic for "have your cake and eat it" distillation:

  - **New-skill axis** (tool-use): does the student now emit the
    correct tool-call schema with the right args? Measured on the
    held-out eval split of the synthetic tool-use corpus, scored by
    `make_tooluse_corpus.score_example` (schema_ok / tool_ok /
    args_ok / correct).
  - **Prior-skill axis** (GSM8K): is the student's general math
    reasoning preserved? Measured on a subset of GSM8K test by
    extracting the boxed/tagged answer and comparing to ground truth.

Plain SFT on a small corpus typically pushes the new-skill axis up
and the prior-skill axis DOWN (catastrophic forgetting). SDFT is
designed to push the new axis up while leaving the prior axis
essentially unchanged. The numerical proof of that is what this eval
reports.

Two entry points:

  - `quick_two_axis_eval(student, tokenizer, cfg, n_tooluse, n_gsm8k, device)`
    — called from train.py periodically. Lightweight; uses a small
    GSM8K slice and a subset of the tool-use eval examples.
  - `main()` — CLI entry. `python eval.py --config=... --checkpoint=...
    [--full] [--base] [--tooluse-only|--gsm8k-only]`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_student, generation_mode
from data import load_tokenizer, load_corpus, load_gsm8k_eval
from make_tooluse_corpus import Example, score_example, grade_corpus
from checkpoint import load as load_ckpt


_ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>")
_INT_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _parse_answer_int(text: str) -> int | None:
    """Pull the FIRST integer from the LAST `<answer>...</answer>` block.

    Reused for GSM8K scoring — same conventions as M17's eval.
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    inner = matches[-1].strip()
    m = _INT_RE.search(inner)
    if m is None:
        return None
    raw = m.group(0).replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        if any(c != "0" for c in frac):
            try:
                f = float(raw)
                if f == int(f):
                    return int(f)
                return None
            except ValueError:
                return None
        raw = whole
    try:
        return int(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Generation helper (greedy; eval should be deterministic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_one(model, tok, messages, max_new_tokens: int, device: str) -> str:
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
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    completion = out[0, input_ids.shape[1]:]
    return tok.decode(completion, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Tool-use eval (new-skill axis)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_tooluse(model, tok, cfg: TrainConfig, n: int, device: str) -> tuple[dict, list[dict]]:
    """Score N held-out tool-use prompts. Greedy decode."""
    corpus = load_corpus(cfg.data.corpus_dir)
    examples = corpus["eval"][:n]
    responses: list[str] = []
    per_ex: list[dict] = []
    for i, ex in enumerate(examples):
        messages = [
            {"role": "system", "content": cfg.data.system_prompt},
            {"role": "user", "content": ex.user},
        ]
        text = _generate_one(model, tok, messages, cfg.distill.max_new_tokens, device)
        responses.append(text)
        s = score_example(text, ex)
        per_ex.append({
            "user": ex.user, "gt_tool": ex.tool, "gt_args": ex.args,
            "completion": text, **s,
        })
    aggregate = grade_corpus(responses, examples)
    return aggregate, per_ex


# ---------------------------------------------------------------------------
# GSM8K eval (prior-skill axis)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_gsm8k(model, tok, cfg: TrainConfig, n: int, device: str) -> tuple[dict, list[dict]]:
    """Score N held-out GSM8K problems. Greedy decode."""
    problems = load_gsm8k_eval(cfg.data, n=n)
    n_correct = 0
    n_format = 0
    per_ex: list[dict] = []
    for prob in problems:
        messages = [
            {"role": "system", "content": cfg.data.gsm8k_system_prompt},
            {"role": "user", "content": prob["question"]},
        ]
        # GSM8K can need a longer trace than tool calls.
        text = _generate_one(model, tok, messages, max(cfg.distill.max_new_tokens, 512), device)
        pred = _parse_answer_int(text)
        is_correct = int(pred == prob["ground_truth"])
        has_answer_tag = int(_ANSWER_RE.search(text) is not None)
        n_correct += is_correct
        n_format += has_answer_tag
        per_ex.append({
            "question": prob["question"],
            "ground_truth": prob["ground_truth"],
            "pred": pred,
            "correct": is_correct,
            "completion": text,
        })
    n = max(len(problems), 1)
    return {
        "accuracy": n_correct / n,
        "format_rate": n_format / n,
        "n": len(problems),
    }, per_ex


# ---------------------------------------------------------------------------
# Combined entry — called from train.py periodically
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_two_axis_eval(student, tokenizer, cfg: TrainConfig,
                         n_tooluse: int, n_gsm8k: int, device: str) -> dict:
    """Lightweight two-axis eval for in-training reporting."""
    was_training = student.training
    student.eval()
    try:
        tu_agg, _ = eval_tooluse(student, tokenizer, cfg, n_tooluse, device)
        gsm_agg, _ = eval_gsm8k(student, tokenizer, cfg, n_gsm8k, device)
    finally:
        if was_training:
            student.train()
    return {
        "tooluse_correct": tu_agg["correct"],
        "tooluse_schema_ok": tu_agg["schema_ok"],
        "tooluse_tool_ok": tu_agg["tool_ok"],
        "tooluse_args_ok": tu_agg["args_ok"],
        "gsm8k_accuracy": gsm_agg["accuracy"],
        "gsm8k_format_rate": gsm_agg["format_rate"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Module 18 two-axis distillation eval")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="DCP checkpoint dir to load. Omit for the base model.")
    p.add_argument("--base", action="store_true",
                   help="Eval the untrained base — equivalent to --checkpoint=''.")
    p.add_argument("--full", action="store_true",
                   help="Full eval: 50 tool-use + 500 GSM8K. Default is "
                        "20 + 100.")
    p.add_argument("--tooluse-only", action="store_true")
    p.add_argument("--gsm8k-only", action="store_true")
    p.add_argument("--out", default=None,
                   help="Optional JSON path to dump per-example results.")
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

    model = build_student(cfg.model).to(device)
    if not (args.base or args.checkpoint is None):
        print(f"[eval] loading checkpoint {args.checkpoint}")
        load_ckpt(model, optimizer=None, ckpt_dir=args.checkpoint)
    else:
        print("[eval] using BASE model (no checkpoint loaded)")
    model.eval()

    n_tooluse = 50 if args.full else 20
    n_gsm8k = 500 if args.full else 100

    out: dict = {}
    if not args.gsm8k_only:
        print(f"\n[eval] tool-use, {n_tooluse} held-out prompts (greedy)")
        tu_agg, tu_per = eval_tooluse(model, tok, cfg, n_tooluse, device)
        print(f"  schema_ok:  {tu_agg['schema_ok']:.1%}")
        print(f"  tool_ok:    {tu_agg['tool_ok']:.1%}")
        print(f"  args_ok:    {tu_agg['args_ok']:.1%}")
        print(f"  correct:    {tu_agg['correct']:.1%}  (schema AND tool AND args)")
        out["tooluse"] = tu_agg
        # Show one example each of pass and fail.
        for ex in tu_per:
            if ex["correct"] > 0:
                print(f"\n  [PASS] {ex['user']!r}")
                print(f"         -> {ex['completion'][:200]!r}")
                break
        for ex in tu_per:
            if ex["correct"] == 0:
                print(f"\n  [FAIL] {ex['user']!r}")
                print(f"         gt: <tool>{ex['gt_tool']}</tool><args>{json.dumps(ex['gt_args'])}</args>")
                print(f"         -> {ex['completion'][:200]!r}")
                break

    if not args.tooluse_only:
        print(f"\n[eval] GSM8K, {n_gsm8k} test problems (greedy)")
        gsm_agg, gsm_per = eval_gsm8k(model, tok, cfg, n_gsm8k, device)
        print(f"  accuracy:   {gsm_agg['accuracy']:.1%}")
        print(f"  format_rate:{gsm_agg['format_rate']:.1%}")
        print(f"  n scored:   {gsm_agg['n']}")
        out["gsm8k"] = gsm_agg

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
