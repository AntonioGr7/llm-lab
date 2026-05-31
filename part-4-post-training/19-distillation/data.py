"""Data pipeline for Module 19 — Distillation.

The corpus is built by `make_tooluse_corpus.py` and lives on disk as
three jsonl files (`demos.jsonl`, `train.jsonl`, `eval.jsonl`) under
`cfg.data.corpus_dir`. Each row has the schema:

    {"user": str, "tool": str, "args": dict}

This file's responsibilities:

  1. Load the three splits.
  2. Provide a `ToolUseDataset` over the train split (the prompts that
     drive the distillation step).
  3. Provide a `load_tokenizer` helper with LEFT padding (required for
     `model.generate`).
  4. Reuse Module 18's GSM8K loader for the prior-skill eval anchor
     (we ship a copy of `parse_gsm8k_ground_truth` rather than imports
     from sibling modules — framework-directory pattern).

The actual chat-template RENDERING of (demos + prompt) into token ids
lives in `rollout.py`, NOT here. Rendering depends on `distill.mode`
(SDFT prepends K demos; on_policy doesn't) and shares plumbing with the
generation step.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DataConfig
from make_tooluse_corpus import Example


# =============================================================================
# Tokenizer loader
# =============================================================================

def load_tokenizer(name: str, padding_side: str = "left"):
    """LEFT-padded tokenizer for generation. (Same as Module 18.)"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side=padding_side)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if tok.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {name!r} has no chat_template attribute. "
            "Pick a tokenizer that ships one."
        )
    return tok


# =============================================================================
# Corpus loading
# =============================================================================

def load_corpus(corpus_dir: str) -> dict[str, list[Example]]:
    """Read demos/train/eval jsonl files. Raises if any are missing."""
    path = Path(corpus_dir)
    out: dict[str, list[Example]] = {}
    for split in ("demos", "train", "eval"):
        f = path / f"{split}.jsonl"
        if not f.exists():
            raise FileNotFoundError(
                f"missing {f} — run `python make_tooluse_corpus.py --out={corpus_dir}` first."
            )
        examples: list[Example] = []
        with f.open() as fh:
            for line in fh:
                d = json.loads(line)
                examples.append(Example(**d))
        out[split] = examples
    return out


# =============================================================================
# Dataset over the train split
# =============================================================================

class ToolUseDataset(Dataset):
    """Pre-loaded tool-use prompts for distillation training.

    Each `__getitem__` returns the raw fields — the dataset stays
    lightweight (no tokenization here). The rollout phase consumes these
    and renders the chat template + token ids on the fly, because that
    rendering depends on `distill.mode` (SDFT renders demos in-context,
    on_policy does not).

    Output schema per row:
        {
          "user": str,
          "tool": str,
          "args": dict,
        }
    """

    def __init__(self, corpus_dir: str, max_examples: Optional[int] = None):
        super().__init__()
        corpus = load_corpus(corpus_dir)
        self.demonstrations: list[Example] = corpus["demos"]
        self.train_examples: list[Example] = corpus["train"]
        self.eval_examples: list[Example] = corpus["eval"]
        if max_examples is not None:
            self.train_examples = self.train_examples[:max_examples]

        self.n_demos = len(self.demonstrations)
        self.n_train = len(self.train_examples)
        self.n_eval = len(self.eval_examples)

    def __len__(self) -> int:
        return self.n_train

    def __getitem__(self, idx: int) -> dict:
        ex = self.train_examples[idx]
        return {"user": ex.user, "tool": ex.tool, "args": ex.args}


def _collate(batch: list[dict]) -> dict:
    """Custom collate — keep strings/dicts as Python lists (default_collate would crash)."""
    return {
        "user": [b["user"] for b in batch],
        "tool": [b["tool"] for b in batch],
        "args": [b["args"] for b in batch],
    }


def make_dataloader(cfg: DataConfig) -> DataLoader:
    """Build the tool-use prompt DataLoader.

    `batch_size = cfg.prompts_per_step` — how many prompts feed one
    distillation step. Each prompt expands to ONE teacher rollout (no
    group_size like GRPO; distillation samples one completion per
    prompt).
    """
    if cfg.num_workers > 0:
        import torch.multiprocessing as torch_mp
        torch_mp.set_sharing_strategy("file_system")

    dataset = ToolUseDataset(cfg.corpus_dir, max_examples=cfg.max_train)

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, seed=cfg.seed, drop_last=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=cfg.prompts_per_step,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context="forkserver" if cfg.num_workers > 0 else None,
    )
    # Stash the demonstrations on the loader so train.py / rollout.py can fetch them.
    loader.demonstrations = dataset.demonstrations
    loader.eval_examples = dataset.eval_examples
    return loader


def cycle(loader: DataLoader):
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# =============================================================================
# GSM8K prior-skill eval — copied verbatim from Module 18 (framework dir pattern)
# =============================================================================

_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")


def parse_gsm8k_ground_truth(answer_field: str) -> int | None:
    """Pull the integer after `#### ` from a GSM8K answer field. (Same as M18.)"""
    m = _GSM8K_FINAL_ANSWER_RE.search(answer_field)
    if m is None:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def load_gsm8k_eval(cfg: DataConfig, n: Optional[int] = None) -> list[dict]:
    """Load N GSM8K test problems for the prior-skill eval.

    Returns a list of {"question": str, "ground_truth": int}. Requires
    network access on first call (HF datasets caches locally afterward).
    """
    from datasets import load_dataset
    ds = load_dataset(cfg.gsm8k_source, cfg.gsm8k_subset, split=cfg.gsm8k_split)
    n = n or cfg.gsm8k_n_eval
    n = min(n, len(ds))

    out: list[dict] = []
    for i in range(n):
        ex = ds[i]
        gt = parse_gsm8k_ground_truth(ex["answer"])
        if gt is None:
            continue
        out.append({"question": ex["question"], "ground_truth": gt})
    return out


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("--- ToolUseDataset smoke test ---")
    # Assumes you've already run `make_tooluse_corpus.py --out=./data`.
    cfg = DataConfig(corpus_dir="./data", prompts_per_step=2)
    ds = ToolUseDataset(cfg.corpus_dir, max_examples=10)
    print(f"  demos: {ds.n_demos}, train: {ds.n_train}, eval: {ds.n_eval}")
    print(f"  demos[0]: user={ds.demonstrations[0].user!r}")
    print(f"            tool={ds.demonstrations[0].tool!r}, args={ds.demonstrations[0].args}")
    print(f"  train[0]: user={ds[0]['user']!r}")

    loader = make_dataloader(cfg)
    batch = next(iter(loader))
    print(f"\n  batch: {len(batch['user'])} prompts")
    print(f"  batch['user'][0]: {batch['user'][0]!r}")
    print(f"  loader.demonstrations has {len(loader.demonstrations)} demos")
