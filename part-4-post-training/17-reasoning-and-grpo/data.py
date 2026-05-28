"""Data pipeline for Module 17 — GRPO on GSM8K.

GRPO trains on `(prompt, ground_truth_answer)` pairs — no chosen/rejected
completions, no demonstrations. The model GENERATES its own completions
at training time (see `rollout.py`) and the reward function (see
`rewards.py`) scores them against the ground truth.

So this file's only job is:

1. Load GSM8K (`openai/gsm8k`, "main" config).
2. Parse out the ground-truth integer from the trailing `#### N` of each
   `answer` field.
3. Render the system+user prompt through the model's chat template with
   `add_generation_prompt=True` (so the prompt ENDS at the position the
   model would start generating from).
4. Return `{prompt_input_ids, prompt_attention_mask, ground_truth}` per
   row — no labels, no response masks. Those are constructed in the
   rollout phase after the model generates.

Same chat-template + Qwen3 gotchas apply as Module 15/16:
  - `return_dict=False` on `apply_chat_template` (transformers 5.x quirk).
  - `enable_thinking=False` to keep the train/inference template aligned;
    this is doubly important here because the policy LEARNS to emit
    `<think>...</think>` from scratch — we don't want the chat template
    to be pre-injecting a different scaffold.

Output contract per `__getitem__`:

    {
      "prompt_input_ids":      Long[seq_len],   left-padded
      "prompt_attention_mask": Long[seq_len],   left-padded (0s on left, 1s on right)
      "ground_truth":          int,             the integer answer
      "question":              str,             original text, for logging
    }

Why LEFT padding: HF's `model.generate()` requires left-padded prompts —
the generated tokens append on the right, and right padding would have
the model attending to PAD tokens during generation. With left padding,
all prompt-final tokens are at the same column across the batch.
"""
from __future__ import annotations

import re
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DataConfig


# =============================================================================
# Tokenizer loader
# =============================================================================

def load_tokenizer(name: str, padding_side: str = "left"):
    """Load an HF AutoTokenizer with left-padding for generation.

    GRPO needs left padding because `model.generate()` extends the prompt
    on the right and won't attend correctly to right-padded inputs.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side=padding_side)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if tok.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {name!r} has no chat_template attribute. Either pick a "
            "tokenizer that ships one (e.g. an SFT'd Qwen3) or set "
            "tokenizer.chat_template before loading."
        )
    return tok


# =============================================================================
# Ground-truth parsing
# =============================================================================

_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")


def parse_gsm8k_ground_truth(answer_field: str) -> int | None:
    """Pull the integer after `#### ` from a GSM8K answer field.

    GSM8K rows have an `answer` of the form:
        "Step-by-step reasoning...\\n#### 18"
    where 18 is the gold final answer. Returns the int or None for
    malformed rows (rare; we skip them in the dataset).
    """
    m = _GSM8K_FINAL_ANSWER_RE.search(answer_field)
    if m is None:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


# =============================================================================
# Chat-template rendering for the PROMPT only
# =============================================================================

def _render_prompt(
    tokenizer,
    question: str,
    system_prompt: str = "",
) -> list[int]:
    """Render `system + user(question)` with `add_generation_prompt=True`.

    The returned token sequence ENDS at the position the model would start
    generating from (just after `<|im_start|>assistant\\n` for Qwen3-style
    templates). `model.generate(input_ids=prompt_ids, ...)` will append
    completion tokens directly after.

    `return_dict=False` is load-bearing under transformers 5.x — see
    Module 15's data.py for the gory details.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    base = dict(tokenize=True, return_dict=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **base)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **base)


def _left_pad(ids: list[int], seq_len: int, pad_id: int) -> tuple[list[int], list[int]]:
    """Left-pad `ids` to `seq_len`. Returns `(padded_ids, attention_mask)`.

    Attention mask is 0 on padding (left) and 1 on real tokens (right).
    Raises if `len(ids) > seq_len` — callers should filter such rows
    upstream rather than truncating prompts (truncation would corrupt
    the schema directive in the system prompt).
    """
    n = len(ids)
    if n > seq_len:
        raise ValueError(f"prompt length {n} exceeds seq_len {seq_len}")
    n_pad = seq_len - n
    return ([pad_id] * n_pad + ids, [0] * n_pad + [1] * n)


# =============================================================================
# GSM8KDataset
# =============================================================================

class GSM8KDataset(Dataset):
    """Pre-tokenized GSM8K prompts with ground-truth integers.

    Loads `cfg.source` (`cfg.subset`, `cfg.split`) via `datasets.load_dataset`,
    parses out the integer answer, renders the chat-template prompt, and
    left-pads to `cfg.seq_len`. Rows are dropped if:
      - the `#### N` ground-truth pattern is missing or unparseable, or
      - the tokenized prompt exceeds `cfg.seq_len` (rare for GSM8K).

    For a 7.5k-row train split, in-memory tokenization runs in seconds and
    consumes <100MB RAM. If you swap to a bigger prompt corpus, switch to
    IterableDataset.
    """

    def __init__(self, cfg: DataConfig, tokenizer_name: str):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer(tokenizer_name, padding_side="left")

        from datasets import load_dataset
        # GSM8K is "openai/gsm8k" with config "main" or "socratic". Other math
        # datasets can override `cfg.source` + `cfg.subset` to "" if they
        # don't use a config name.
        if cfg.subset:
            raw = load_dataset(cfg.source, cfg.subset, split=cfg.split)
        else:
            raw = load_dataset(cfg.source, split=cfg.split)

        if cfg.max_examples is not None:
            raw = raw.select(range(min(cfg.max_examples, len(raw))))

        pad_id = self.tokenizer.pad_token_id

        self.samples: list[dict] = []
        n_skipped_gt = 0
        n_skipped_len = 0
        for ex in raw:
            question = ex.get("question") or ex.get("problem") or ""
            gt_field = ex.get("answer") or ex.get("solution") or ""
            if not question or not gt_field:
                n_skipped_gt += 1
                continue
            gt = parse_gsm8k_ground_truth(gt_field)
            if gt is None:
                n_skipped_gt += 1
                continue

            try:
                ids = _render_prompt(self.tokenizer, question, cfg.system_prompt)
            except Exception:
                n_skipped_gt += 1
                continue

            if len(ids) > cfg.seq_len:
                n_skipped_len += 1
                continue

            padded_ids, attn_mask = _left_pad(ids, cfg.seq_len, pad_id)
            self.samples.append({
                "prompt_input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "prompt_attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "ground_truth": gt,
                "question": question,
            })

        if not self.samples:
            raise RuntimeError(
                f"GSM8KDataset({cfg.source!r} / {cfg.subset!r}) produced 0 usable rows. "
                "Check the dataset name and that `#### N` ground truths are present."
            )

        self.n_total = len(raw)
        self.n_skipped_gt = n_skipped_gt
        self.n_skipped_len = n_skipped_len
        self.n_examples = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# =============================================================================
# DataLoader construction
# =============================================================================

def _collate(batch: list[dict]) -> dict:
    """Stack tensors, pass `ground_truth` and `question` through as lists.

    The default `torch.utils.data.default_collate` would try to stack the
    Python int `ground_truth` into a tensor (fine) and the string
    `question` into a tensor (crashes). Custom collate keeps both as
    Python lists, which is what the rollout and reward code expects.
    """
    return {
        "prompt_input_ids": torch.stack([b["prompt_input_ids"] for b in batch]),
        "prompt_attention_mask": torch.stack([b["prompt_attention_mask"] for b in batch]),
        "ground_truth": [b["ground_truth"] for b in batch],
        "question": [b["question"] for b in batch],
    }


def make_dataloader(cfg: DataConfig, tokenizer_name: str) -> DataLoader:
    """Build the GSM8K prompt DataLoader.

    `batch_size` is `cfg.prompts_per_step` — how many prompts feed one RL
    step (each prompt then expands to G completions in the rollout).
    `num_workers=0` is the right default: prompts are pre-tokenized and
    in memory, so worker overhead has no payoff.
    """
    if cfg.num_workers > 0:
        import torch.multiprocessing as torch_mp
        torch_mp.set_sharing_strategy("file_system")

    dataset = GSM8KDataset(cfg, tokenizer_name=tokenizer_name)

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
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


def cycle(loader: DataLoader):
    """Wrap a DataLoader to yield batches forever (set_epoch on
    DistributedSampler each pass)."""
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("--- GSM8KDataset smoke test ---")
    smoke_cfg = DataConfig(
        source="openai/gsm8k",
        subset="main",
        split="train",
        seq_len=512,
        prompts_per_step=2,
        num_workers=0,
        max_examples=8,
    )
    ds = GSM8KDataset(smoke_cfg, tokenizer_name="Qwen/Qwen3-0.6B")
    print(f"  loaded {len(ds)} prompts")
    print(f"  skipped {ds.n_skipped_gt} (no ground truth) + "
          f"{ds.n_skipped_len} (too long) of {ds.n_total} raw")

    s = ds[0]
    print(f"\n  sample[0] question: {s['question'][:80]!r}...")
    print(f"  sample[0] ground_truth: {s['ground_truth']}")
    print(f"  sample[0] prompt_input_ids shape: {tuple(s['prompt_input_ids'].shape)}")

    # Sanity: how much of the prompt buffer is padding?
    n_real = int(s["prompt_attention_mask"].sum().item())
    seq_len = s["prompt_input_ids"].shape[0]
    print(f"  sample[0] prompt fill: {n_real}/{seq_len} tokens "
          f"({100*n_real/seq_len:.1f}% real, rest is LEFT pad)")
