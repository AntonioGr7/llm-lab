"""Data pipeline for Module 17 — Preference Optimization.

DPO/IPO consume preference *pairs*: a prompt with two candidate completions,
one preferred ("chosen") and one not ("rejected"). The data pipeline's job:

1. Load a preference dataset (default: `HuggingFaceH4/ultrafeedback_binarized`).
2. For each pair, render `(prompt + chosen)` and `(prompt + rejected)` with
   the model's chat template — independently, because they're separate
   sequences the model will score.
3. Compute the assistant-only loss mask on each, exactly the same way
   Module 15 does. The DPO loss sums log π(y|x) ONLY over the response
   tokens; positions outside the response carry `IGNORE_INDEX` in `labels`
   and contribute zero to the sum (see `loop.gather_response_logps`).

The output contract per __getitem__:

    {
      "chosen_input_ids":      Long[seq_len],
      "chosen_labels":         Long[seq_len],   IGNORE_INDEX outside response
      "chosen_attention_mask": Long[seq_len],
      "rejected_input_ids":    Long[seq_len],
      "rejected_labels":       Long[seq_len],
      "rejected_attention_mask": Long[seq_len],
    }

In the training loop we'll stack chosen and rejected along the batch dim
into a single 2B-sized forward pass — see `loop.compute_dpo_loss`. The
collation is just default `torch.stack` (no padding needed; rows are
already padded to seq_len here).

Why no length normalization? DPO's standard formulation sums log-probs
over the entire response without dividing by length. This is the source
of DPO's known length bias (the model learns longer = preferred). SimPO
adds 1/|y| normalization; length-normalized DPO does the same. We keep
the standard sum here and discuss the bias in README §8. Add length
normalization as a one-liner in `gather_response_logps` if you want to
experiment with it.

The chat-template + diff-mask helpers are deliberately duplicated from
Module 15's data.py. This is the "framework directory" pattern — every
module copies the code it needs, no cross-module imports.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DataConfig


IGNORE_INDEX = -100   # F.cross_entropy default; we use the same sentinel for log-prob gathering


# =============================================================================
# Chat template + loss mask — copied from Module 15 (see that module's data.py
# docstring for the full rationale of the diff-trick and the Qwen3 gotchas)
# =============================================================================

def _chat_to_ids(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Render messages to a flat list of token ids. `return_dict=False` is
    load-bearing under transformers 5.x (otherwise the length arithmetic in
    the diff trick degenerates and the mask silently empties)."""
    base = dict(tokenize=True, return_dict=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **base)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **base)


def _mask_final_turn(
    messages: list[dict],
    tokenizer,
    seq_len: int,
) -> Optional[dict]:
    """Render a conversation whose LAST message is the assistant turn we want
    to score, and mask everything except that final turn.

    Returns `{input_ids, labels, attention_mask}` (Long, length `seq_len`) or
    `None` if the last message isn't an assistant turn / the template
    misbehaves. See Module 15's data.py for the full discussion.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None

    full = _chat_to_ids(tokenizer, messages)
    prefix = _chat_to_ids(tokenizer, messages[:-1], add_generation_prompt=True)

    if not (0 < len(prefix) < len(full) and full[:len(prefix)] == prefix):
        return None

    is_response = [False] * len(full)
    for p in range(len(prefix), len(full)):
        is_response[p] = True

    input_ids = full[:-1]
    raw_labels = full[1:]
    target_mask = is_response[1:]
    labels = [t if m else IGNORE_INDEX for t, m in zip(raw_labels, target_mask)]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]
        attention_mask = [1] * seq_len
    else:
        n_real = len(input_ids)
        n_pad = seq_len - n_real
        input_ids = input_ids + [pad_id] * n_pad
        labels = labels + [IGNORE_INDEX] * n_pad
        attention_mask = [1] * n_real + [0] * n_pad

    if all(l == IGNORE_INDEX for l in labels):
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# =============================================================================
# Preference-pair normalization
# =============================================================================

def _normalize_to_pair(
    example: dict,
    system_prompt: str = "",
) -> Optional[tuple[list[dict], list[dict]]]:
    """Convert a heterogeneous preference row into a `(chosen_messages,
    rejected_messages)` pair.

    Accepts three common shapes:

    1. **UltraFeedback-binarized / Zephyr style** (default):
           {"prompt": str, "chosen": [msgs...], "rejected": [msgs...]}
       `chosen` and `rejected` are each a complete conversation ending with
       the assistant turn we want to score. They share the same user turn.

    2. **String-completion style** (Anthropic HH-RLHF, some Orca-DPO splits):
           {"prompt": str, "chosen": str, "rejected": str}
       Wrap as 2-turn conversations.

    3. **`messages`/`rejected_messages` style** (a few research splits):
           {"messages": [chosen msgs...], "rejected_messages": [rejected msgs...]}
       Pass through.

    Returns `None` for malformed rows so the caller can skip them.
    """
    chosen_raw = example.get("chosen")
    rejected_raw = example.get("rejected")

    if isinstance(chosen_raw, list) and isinstance(rejected_raw, list):
        chosen_msgs = list(chosen_raw)
        rejected_msgs = list(rejected_raw)
    elif isinstance(chosen_raw, str) and isinstance(rejected_raw, str):
        prompt = example.get("prompt") or example.get("instruction") or ""
        if not prompt:
            return None
        chosen_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_raw},
        ]
        rejected_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_raw},
        ]
    elif "messages" in example and "rejected_messages" in example:
        chosen_msgs = list(example["messages"])
        rejected_msgs = list(example["rejected_messages"])
    else:
        return None

    # The chosen and rejected branches must each end in an assistant turn for
    # the diff trick to work. If a dataset gives us a chosen list whose last
    # turn is the *user* (some Orca-DPO splits put the assistant turn pre-last
    # with a trailing template token), skip the row.
    if (not chosen_msgs or chosen_msgs[-1].get("role") != "assistant"
            or not rejected_msgs or rejected_msgs[-1].get("role") != "assistant"):
        return None

    if system_prompt:
        if chosen_msgs[0].get("role") != "system":
            chosen_msgs = [{"role": "system", "content": system_prompt}] + chosen_msgs
        if rejected_msgs[0].get("role") != "system":
            rejected_msgs = [{"role": "system", "content": system_prompt}] + rejected_msgs

    return chosen_msgs, rejected_msgs


# =============================================================================
# Tokenizer loader (same shape as Module 15)
# =============================================================================

def load_tokenizer(name: str):
    """Load an HF AutoTokenizer with a sensible pad token. DPO always uses the
    policy's own tokenizer (chat template lives on it)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
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
# PreferenceDataset
# =============================================================================

class PreferenceDataset(Dataset):
    """Pre-tokenized preference dataset.

    Loads `cfg.source` via `datasets.load_dataset`, normalizes each row into
    `(chosen_messages, rejected_messages)`, then runs `_mask_final_turn` on
    each side at __init__ time. Tokenized tensors live in memory.

    For 60k-pair datasets like ultrafeedback_binarized (~120k masked
    renders), this fits in a few GB of RAM after tokenization. For bigger
    mixes, swap to an IterableDataset; for ultrafeedback-class datasets,
    in-memory is faster and simpler.

    Sanity attribute `frac_chosen_response_tokens` exposes the same
    safety-net we used in SFT — if this is near 0 or near 1, the mask is
    broken on this tokenizer.
    """

    def __init__(self, cfg: DataConfig, tokenizer_name: str):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer(tokenizer_name)

        from datasets import load_dataset
        raw = load_dataset(cfg.source, split=cfg.split)
        if cfg.max_examples is not None:
            raw = raw.select(range(min(cfg.max_examples, len(raw))))

        self.samples: list[dict] = []
        n_skipped_format = 0
        n_skipped_mask = 0
        for example in raw:
            norm = _normalize_to_pair(example, cfg.system_prompt)
            if norm is None:
                n_skipped_format += 1
                continue
            chosen_msgs, rejected_msgs = norm

            chosen = _mask_final_turn(chosen_msgs, self.tokenizer, cfg.seq_len)
            rejected = _mask_final_turn(rejected_msgs, self.tokenizer, cfg.seq_len)
            if chosen is None or rejected is None:
                n_skipped_mask += 1
                continue

            self.samples.append({
                "chosen_input_ids": chosen["input_ids"],
                "chosen_labels": chosen["labels"],
                "chosen_attention_mask": chosen["attention_mask"],
                "rejected_input_ids": rejected["input_ids"],
                "rejected_labels": rejected["labels"],
                "rejected_attention_mask": rejected["attention_mask"],
            })

        if not self.samples:
            raise RuntimeError(
                f"PreferenceDataset({cfg.source!r}) produced 0 usable pairs after "
                "filtering. Check the dataset shape (expected chosen/rejected as "
                "message lists or strings)."
            )

        self.n_total = len(raw)
        self.n_skipped_format = n_skipped_format
        self.n_skipped_mask = n_skipped_mask
        self.n_examples = len(self.samples)
        self.frac_chosen_response_tokens = self._compute_response_fraction("chosen_labels")
        self.frac_rejected_response_tokens = self._compute_response_fraction("rejected_labels")

    def _compute_response_fraction(self, key: str) -> float:
        n_target = 0
        n_total = 0
        for s in self.samples:
            n_target += int((s[key] != IGNORE_INDEX).sum().item())
            n_total += s[key].numel()
        return n_target / max(n_total, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# =============================================================================
# DataLoader construction
# =============================================================================

def make_dataloader(cfg: DataConfig, tokenizer_name: str) -> DataLoader:
    """Build the preference DataLoader. Same shape as Module 15's, with the
    SAME forkserver / file_system rationale: pre-tokenized in-memory dataset
    means num_workers=0 is right; >0 needs the file_system tensor-sharing
    strategy to avoid 'too many fds'."""
    if cfg.num_workers > 0:
        import torch.multiprocessing as torch_mp
        torch_mp.set_sharing_strategy("file_system")

    dataset = PreferenceDataset(cfg, tokenizer_name=tokenizer_name)

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
        batch_size=cfg.batch_size_per_device,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context="forkserver" if cfg.num_workers > 0 else None,
    )


def cycle(loader: DataLoader) -> "Iterator[dict]":
    """Wrap a DataLoader to yield batches forever. Same as Module 15."""
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
    print("--- PreferenceDataset smoke test ---")
    smoke_cfg = DataConfig(
        source="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        seq_len=512,
        batch_size_per_device=2,
        num_workers=0,
        max_examples=8,
    )
    ds = PreferenceDataset(smoke_cfg, tokenizer_name="Qwen/Qwen3-1.7B")
    print(f"  loaded {len(ds)} pairs")
    print(f"  skipped {ds.n_skipped_format} (format) + {ds.n_skipped_mask} (mask) of {ds.n_total} raw")
    print(f"  chosen   response-token fraction: {ds.frac_chosen_response_tokens:.1%}")
    print(f"  rejected response-token fraction: {ds.frac_rejected_response_tokens:.1%}")
    print(f"    (expect 0.2-0.6; near-0 or near-1 = broken mask on this tokenizer)")

    s = ds[0]
    print(f"\n  sample[0] keys: {sorted(s.keys())}")
    for k in ("chosen_input_ids", "rejected_input_ids"):
        print(f"  {k} shape: {tuple(s[k].shape)}")

    n_c = int((s["chosen_labels"] != IGNORE_INDEX).sum().item())
    n_r = int((s["rejected_labels"] != IGNORE_INDEX).sum().item())
    print(f"\n  sample[0] response tokens: chosen={n_c}, rejected={n_r}")
    print(f"  (these differ because the two responses have different lengths)")
