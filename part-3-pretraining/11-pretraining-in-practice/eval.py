"""Base-model evaluation for the Module 11 demo run.

Three signals, none of them sufficient on their own:

  1. **Validation perplexity** on a held-out slice of FineWeb-Edu the model
     never trained on. The headline number you compare runs by.
  2. **Generation sanity check**: sample short completions from canonical
     prompts. Checks the model produces coherent English-shaped text. A
     pretrained base model will NOT follow instructions — that's post-
     training (Part 4). The bar is "this reads like the start of an article."
  3. **Pointer at lm-evaluation-harness**: the standard external benchmark
     harness. Prints the command to run for our checkpoint shape; we don't
     vendor harness scoring here because that's a separate, large dependency.

Usage:

    python eval.py --checkpoint=./results/checkpoints/step_00003000 \
                   --config=configs/demo.yaml \
                   --slice=valid              # perplexity
    python eval.py --checkpoint=... --generate # generation samples
    python eval.py --harness                   # lm-eval-harness command print

The checkpoint must have been saved by `train.py` (DCP or single-process).
"""
from __future__ import annotations

import argparse
import math

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_model
from data import FineWebEduDataset
from checkpoint import load as load_ckpt
from fsdp_setup import init_distributed, cleanup_distributed


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer_name: str,
    seq_len: int,
    n_tokens: int = 5_000_000,
    seed: int = 1234,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute per-token cross-entropy and perplexity on a held-out slice.

    The slice is the same `sample-10BT` corpus, but reshuffled with a
    different seed so we read a different cut. For a more rigorous holdout,
    point at FineWeb-Edu's `sample-100BT` and skip the training subset.

    Args:
        model: a model whose forward accepts (input_ids, labels) and
            returns an object with `.loss`.
        n_tokens: target number of tokens to score (5M is the standard).
        seed: shuffle seed — distinct from the training seed.

    Returns:
        dict with keys `loss`, `perplexity`, `n_tokens_scored`.
    """
    model.eval()
    ds = FineWebEduDataset(
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        subset="sample-10BT",
        shuffle_buffer=1_000,
        seed=seed,
    )

    total_loss = 0.0
    total_tokens = 0
    for sample in ds:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)   # (1, seq_len)
        labels = sample["labels"].unsqueeze(0).to(device)
        out = model(input_ids=input_ids, labels=labels)
        # out.loss is mean per-token CE for this sample.
        n = labels.numel()
        total_loss += out.loss.item() * n
        total_tokens += n
        if total_tokens >= n_tokens:
            break

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "n_tokens_scored": total_tokens,
    }


# ---------------------------------------------------------------------------
# Generation sanity check
# ---------------------------------------------------------------------------

_CANONICAL_PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "Once upon a time, in a small village,",
    "To compute the factorial of n, we can",
    "The most important property of water is",
]


@torch.no_grad()
def generate_samples(
    model,
    tokenizer_name: str,
    prompts: list[str] = None,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda",
) -> list[tuple[str, str]]:
    """Produce a short completion for each prompt.

    Sampling is top-k + temperature, not beam search — for a base model,
    sampling is the better diagnostic because beam search hides repetition
    pathologies.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()

    if prompts is None:
        prompts = _CANONICAL_PROMPTS

    out = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        for _ in range(max_new_tokens):
            logits = model(input_ids=ids).logits[:, -1, :]   # (1, V)
            logits = logits / max(temperature, 1e-6)
            # Top-k filter
            top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
            probs = torch.softmax(top_vals, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_id = top_idx.gather(-1, sampled)             # (1, 1)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == tok.eos_token_id:
                break
        completion = tok.decode(ids[0], skip_special_tokens=True)
        out.append((prompt, completion))
    return out


# ---------------------------------------------------------------------------
# Pointer at lm-evaluation-harness
# ---------------------------------------------------------------------------

_HARNESS_HINT = """\
To run the full lm-evaluation-harness on this checkpoint:

    pip install lm-eval
    lm_eval --model hf \\
            --model_args pretrained={ckpt_path},dtype=bfloat16 \\
            --tasks arc_easy,piqa,hellaswag,winogrande \\
            --batch_size 8 \\
            --output_path ./results/harness/

Expected for the 150M demo (well below an instruction-tuned model):
    arc_easy:    ~30-35%
    piqa:        ~58-62%
    hellaswag:   ~26-30%
    winogrande:  ~50-52%

Anything noticeably below those bands suggests undertraining or a tokenizer mismatch.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Eval the Module 11 demo checkpoint")
    p.add_argument("--config", type=str, default="configs/demo.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to a DCP or single-process checkpoint directory")
    p.add_argument("--slice", type=str, default=None, choices=["valid"],
                   help="if set, run held-out perplexity on this slice")
    p.add_argument("--n_eval_tokens", type=int, default=1_000_000,
                   help="how many tokens to score (default 1M; bump to 5M for a stable number)")
    p.add_argument("--generate", action="store_true",
                   help="also print short completions for canonical prompts")
    p.add_argument("--harness", action="store_true",
                   help="print the lm-eval-harness command and exit")
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

    if args.harness:
        ckpt = args.checkpoint or "./results/checkpoints/step_00003000"
        print(_HARNESS_HINT.format(ckpt_path=ckpt))
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for perplexity or generation evaluation")

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)

    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not rinfo.is_main:
        # For eval we typically run on a single rank; rank>0 can wait and exit.
        cleanup_distributed()
        return

    model = build_model(cfg.model).to(device)
    # Build a dummy optimizer so load_ckpt has someone to load opt state into.
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
    step = load_ckpt(model, dummy_opt, args.checkpoint)
    print(f"[eval] loaded checkpoint at step {step}", flush=True)

    if args.slice == "valid":
        print("[eval] perplexity on held-out FineWeb-Edu slice ...", flush=True)
        result = evaluate_perplexity(
            model,
            tokenizer_name=cfg.data.tokenizer_name,
            seq_len=cfg.data.seq_len,
            n_tokens=args.n_eval_tokens,
            seed=cfg.data.seed + 99_999,   # different slice from training
            device=device,
        )
        print(f"\n  loss              {result['loss']:.4f}")
        print(f"  perplexity        {result['perplexity']:.2f}")
        print(f"  n_tokens_scored   {result['n_tokens_scored']:,}\n")

    if args.generate:
        print("[eval] generation samples ...", flush=True)
        out = generate_samples(
            model, tokenizer_name=cfg.data.tokenizer_name, device=device,
        )
        for prompt, completion in out:
            print(f"\nPROMPT:     {prompt}")
            print(f"COMPLETION: {completion}")
        print()

    cleanup_distributed()


if __name__ == "__main__":
    main()
