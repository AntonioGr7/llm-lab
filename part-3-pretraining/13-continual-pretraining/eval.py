"""Evaluate a continual-pretraining run on BOTH axes that matter.

A CPT run is never "good" on one number. You measure two things:

  1. **Acquisition** — closed-book QA on the HELD-OUT probe set
     (`make_corpus.py` wrote it). The questions are phrasings never seen in
     training, so a model can only score by having internalized the facts. The
     base model should be at ~chance (it never saw the fiction); the
     continually-pretrained model should be high. That gap is the whole point.

  2. **Retention** — perplexity on general text (the replay corpus), before vs
     after. CPT damages general capability; this is the bill. The lower the
     post-CPT ppl rise, the less you forgot. For *standard* benchmarks
     (MMLU/GSM8K) we print the lm-eval-harness command, exactly as Module 11
     does — those need an HF-format export.

Run it twice for the before/after contrast:

    # BEFORE — the base knows nothing about the fiction
    python eval.py --base --qa results/corpus/qa_heldout.jsonl

    # AFTER — acquisition + the forgetting bill
    python eval.py --checkpoint results/checkpoints/step_00000400 \\
        --qa results/corpus/qa_heldout.jsonl --forgetting

Single rank. `--device=cpu` to avoid VRAM contention with a running train job.
"""
from __future__ import annotations

import argparse
import json
import math
import re

import torch
import torch.nn.functional as F

from config import load_yaml, apply_dotted_overrides
from model import build_model
from checkpoint import load as load_ckpt
from fsdp_setup import init_distributed, cleanup_distributed


# ---------------------------------------------------------------------------
# Closed-book QA (acquisition)
# ---------------------------------------------------------------------------

# A 2-shot primer using GENERIC (non-fiction) facts. It only teaches the base
# model the "Q:/A: The answer is ..." format so it attempts an answer instead of
# continuing like web text — it leaks none of the fictional knowledge.
_PRIMER = (
    "Answer each question with a short factual answer.\n"
    "Q: What is the capital of France?\nA: The answer is Paris.\n"
    "Q: What color is a clear daytime sky?\nA: The answer is blue.\n"
)


def _norm(s: str) -> str:
    """Lowercase, drop commas, collapse whitespace — so '4,200' == '4200'."""
    return re.sub(r"\s+", " ", s.replace(",", "").lower()).strip()


def _match(gold: str, generation: str) -> bool:
    """Did the model's answer (first line) contain the gold answer?"""
    first_line = generation.strip().split("\n", 1)[0]
    return _norm(gold) in _norm(first_line)


@torch.no_grad()
def closed_book_qa(model, tok, probes, device, max_new_tokens=24, n_show=6):
    model.config.use_cache = True
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    correct, shown = 0, []
    for h in probes:
        prompt = _PRIMER + f"Q: {h['question']}\nA:"
        enc = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=pad_id)
        out = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        hit = _match(h["answer"], out)
        correct += int(hit)
        if len(shown) < n_show:
            shown.append((h["question"], h["answer"], out.strip().split("\n", 1)[0], hit))
    return correct / max(len(probes), 1), shown


# ---------------------------------------------------------------------------
# Retention perplexity (forgetting)
# ---------------------------------------------------------------------------

@torch.no_grad()
def retention_perplexity(model, prefix, seq_len, device, max_samples=200):
    """Mean per-token NLL + perplexity on a general-text indexed corpus.

    Compare base vs checkpoint: a CPT run that forgot general English will show
    a higher perplexity here. A cheap, built-in retention signal — for real
    benchmark numbers use the harness pointer below.
    """
    from indexed_dataset import IndexedDataset
    from indexed_data import PackedIndexedDataset
    ds = PackedIndexedDataset(IndexedDataset(prefix), seq_len=seq_len)
    n = min(len(ds), max_samples)
    model.config.use_cache = False
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(n):
        s = ds[i]
        input_ids = s["input_ids"].unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=input_ids).logits
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                              reduction="sum")
        total_nll += float(nll.item())
        total_tok += int(labels.numel())
    mean = total_nll / max(total_tok, 1)
    return {"loss": mean, "perplexity": math.exp(mean), "n_samples": n}


def _harness_pointer(target: str):
    print("\n[forgetting on standard benchmarks] export to HF format, then:")
    print("  # (export a DCP checkpoint with Module 11's export_hf.py first)")
    print(f"  lm_eval --model hf --model_args pretrained={target} \\")
    print("          --tasks mmlu,gsm8k --batch_size auto")
    print("  Run it for the base model AND the exported CPT model; the drop is")
    print("  your forgetting bill on real benchmarks.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Two-axis CPT eval (acquisition + retention)")
    p.add_argument("--config", type=str, default="configs/cpt_qwen3_0.6b.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="a CPT checkpoint dir from train.py")
    p.add_argument("--base", action="store_true",
                   help="eval the un-continued base model (cfg.model.name) — the BEFORE")
    p.add_argument("--qa", type=str, default=None,
                   help="held-out QA probe jsonl (default: cfg.data.qa_heldout)")
    p.add_argument("--forgetting", action="store_true",
                   help="also compute retention perplexity on the replay corpus + print harness cmd")
    p.add_argument("--max_qa", type=int, default=0, help="cap probes (0 = all)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args, extra = p.parse_known_args()
    overrides = [t[2:] for t in extra if t.startswith("--") and "=" in t]
    bad = [t for t in extra if not (t.startswith("--") and "=" in t)]
    if bad:
        raise SystemExit(f"unrecognized arg(s): {bad}")
    return args, overrides


def main():
    args, overrides = _parse_args()
    if not args.base and args.checkpoint is None:
        raise SystemExit("pass --checkpoint=<dir>, or --base to eval the un-continued model")

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)

    rinfo = init_distributed()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not rinfo.is_main:
        cleanup_distributed()
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name)

    model = build_model(cfg.model).to(device)
    if args.base:
        label = f"BASE {cfg.model.name!r} (no continual pretraining)"
    else:
        step = load_ckpt(model, None, args.checkpoint)   # weights-only
        label = f"CPT checkpoint @ step {step}"
    print(f"\n=== eval: {label} ===", flush=True)

    # ---- acquisition -----------------------------------------------------
    qa_path = args.qa or cfg.data.qa_heldout
    probes = [json.loads(l) for l in open(qa_path) if l.strip()]
    if args.max_qa:
        probes = probes[:args.max_qa]
    acc, shown = closed_book_qa(model, tok, probes, device)
    print(f"\n[acquisition] closed-book held-out QA: {acc:.1%} "
          f"({int(round(acc*len(probes)))}/{len(probes)})")
    for q, gold, got, hit in shown:
        print(f"  [{'✓' if hit else '✗'}] {q}")
        print(f"        gold={gold!r}  got={got!r}")

    # ---- retention -------------------------------------------------------
    if args.forgetting:
        r = retention_perplexity(model, cfg.data.replay_prefix, cfg.data.seq_len, device)
        print(f"\n[retention] general-text perplexity (replay corpus): "
              f"{r['perplexity']:.2f}  (loss {r['loss']:.4f}, {r['n_samples']} samples)")
        print("            compare BASE vs CPT: a rise = forgetting.")
        _harness_pointer(args.checkpoint or cfg.model.name)

    cleanup_distributed()


if __name__ == "__main__":
    main()
