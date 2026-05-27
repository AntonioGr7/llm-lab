"""Data pipeline for Module 15 — Supervised Fine-Tuning.

Two responsibilities, both unique to SFT (this is where the new pedagogical
content of the module lives):

1. **Apply the chat template.** Every HF tokenizer ships a Jinja template
   (`tokenizer.chat_template`) that serializes a list of `{role, content}`
   messages into a single string with delimiters marking whose turn it is.
   Different model families use different delimiters (Qwen3: `<|im_start|>`,
   Llama 3: `<|start_header_id|>`, etc.) — we never write those by hand.
   `tokenizer.apply_chat_template(messages)` returns the token IDs directly;
   we use it everywhere.

2. **Compute the assistant-only loss mask.** We want cross-entropy on
   tokens the *assistant* generates, not tokens the *user* sends. The
   `_mask_final_turn` helper uses the "diff trick" — render the prefix
   `apply_chat_template(messages[:-1], add_generation_prompt=True)` and the
   full `apply_chat_template(messages)`; the tokens *after* the prefix are
   exactly the assistant's response. Mask everything in the prefix as `-100`
   (the F.cross_entropy ignore_index), including the fixed assistant header
   and any template scaffold (e.g. Qwen3's empty `<think></think>`).

   The diff is reliable only when the assistant turn is the LAST message —
   templates render an assistant turn differently mid-conversation vs. as the
   final turn (Qwen3 adds `<think>` only to the final one), so a naive
   loop-over-all-turns-into-one-render mis-masks. `_render_examples` therefore
   expands a k-turn conversation into k single-target examples. The technique
   itself is template-agnostic (Qwen3, Llama 3, Mistral, Gemma); the only
   per-template subtlety is suppressed by always masking up to the generation
   prompt. See `_mask_final_turn` for the full rationale.

The output contract (extends Module 11's data.py with an attention mask):

    batch["input_ids"]:      LongTensor[B, seq_len]
    batch["labels"]:         LongTensor[B, seq_len]  — -100 wherever loss is masked,
                                                       otherwise = input_ids shifted +1.
    batch["attention_mask"]: LongTensor[B, seq_len]  — 1 for real tokens, 0 for padding.
                                                       HF causal LMs use this to skip
                                                       attention to padding positions
                                                       (which we add to reach seq_len).

Map-style Dataset, not IterableDataset: at 10k examples the no_robots
dataset fits easily in memory after tokenization, and pre-tokenizing once
at __init__ means we pay zero tokenization cost per training step.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import DataConfig


# =============================================================================
# Chat template + loss mask — the core pedagogical content
# =============================================================================

IGNORE_INDEX = -100   # the F.cross_entropy default for masked positions


def _chat_to_ids(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Render messages to a flat `list[int]` of token ids.

    Two flags matter:

    1. **`return_dict=False` is load-bearing.** Modern transformers (5.x)
       returns a `BatchEncoding` (dict-like with `input_ids`/`attention_mask`)
       from `apply_chat_template(..., tokenize=True)` by default, NOT a bare
       list. The diff trick is all length arithmetic on flat token lists, so we
       force the flat-list return. Forget it and every `len(prefix)`/`len(full)`
       comparison degenerates and the mask silently empties — exactly the
       "flat-line loss" failure the README §3 warns about.

    2. **`enable_thinking=False`** (Qwen3). Qwen3's template injects an empty
       `<think>\\n\\n</think>` scaffold into the assistant turn. With thinking
       enabled the scaffold lands *after* the generation prompt, so the diff
       trick would train the model to emit empty think blocks — unwanted for
       non-reasoning SFT (reasoning is Module 17). With it disabled the scaffold
       moves *into* the generation-prompt prefix, so it is masked and only the
       real response carries loss. **`eval.py` renders the same way**, so train
       and inference see identical formatting (the train/inference template
       mismatch is the second-most-common silent SFT bug — README §2). The kwarg
       is Qwen-specific; templates that don't accept it raise `TypeError` and we
       retry without it (the same kwarg is used for both renders, so the
       strict-prefix property is preserved either way).
    """
    base = dict(tokenize=True, return_dict=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **base)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **base)


def _normalize_to_messages(example: dict, system_prompt: str = "") -> list[dict]:
    """Convert a heterogeneous SFT dataset row into the canonical messages format.

    Accepts three common shapes:

    1. `{"messages": [{"role": ..., "content": ...}, ...]}` — pass through.
       This is the modern standard (UltraChat, Tülu 3, ShareGPT, OpenAssistant).

    2. `{"prompt": "...", "response": "..."}` or `{"instruction": "...", "output": "..."}`
       — wrap as a 2-turn conversation. This covers Alpaca-style datasets and
       most of the older instruction-tuning corpora.

    3. `{"prompt": str, "messages": [...]}` (no_robots-style) — use `messages`,
       ignore `prompt` (it's a duplicate of the first user message).

    Prepends a system prompt if `system_prompt` is non-empty and no system
    message is already present.
    """
    if "messages" in example and example["messages"]:
        messages = list(example["messages"])
    elif "conversations" in example:
        # ShareGPT format: [{"from": "human"/"gpt", "value": "..."}, ...]
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [{"role": role_map.get(t["from"], t["from"]),
                     "content": t["value"]} for t in example["conversations"]]
    else:
        prompt = example.get("prompt") or example.get("instruction") or ""
        response = example.get("response") or example.get("output") or ""
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]

    if system_prompt and messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    return messages


def _mask_final_turn(
    messages: list[dict],
    tokenizer,
    seq_len: int,
) -> Optional[dict]:
    """Render a conversation whose LAST message is the assistant turn to train
    on, and mask everything except that final turn's response.

    Returns `{input_ids, labels, attention_mask}` (LongTensors, length
    `seq_len`), or `None` if the last message isn't an assistant turn or the
    template misbehaves.

    **The diff trick — the central technique of this module:**

    Render the prefix (`messages[:-1]` plus the "generation prompt" that signals
    'now the assistant responds') and the full sequence. The tokens at positions
    `[len(prefix), len(full))` are exactly the assistant's response — its text
    plus the closing `<|im_end|>`, which we DO want to predict so the model
    learns when to stop. Everything in `prefix` is masked, including the fixed
    `<|im_start|>assistant\\n` header and any template scaffold (e.g. Qwen3's
    empty `<think></think>`): the model is *given* those at inference via the
    generation prompt, so it shouldn't spend loss learning to emit them.

    **Why only the FINAL turn?** Templates render an assistant turn differently
    depending on whether it's the last message — Qwen3, for instance, only adds
    the `<think>` scaffold to the final assistant turn. So `render(messages[:i])`
    for an *intermediate* assistant turn does not line up token-for-token with
    its slice of the full multi-turn render, and the naive "loop over all turns
    and index into one `full`" approach silently mis-masks. Masking only the
    final turn sidesteps that entirely; `_render_examples` below turns a k-turn
    conversation into k of these single-target examples, so every assistant turn
    still gets trained — each in the exact left-context it will see at inference.

    Causal-LM shift: `labels[t]` is the target predicted from `input_ids[:t+1]`,
    so position `t` carries loss iff `full[t+1]` is an assistant-response token.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None

    full = _chat_to_ids(tokenizer, messages)
    prefix = _chat_to_ids(tokenizer, messages[:-1], add_generation_prompt=True)

    # The prefix must line up with the full render token-for-token; if a template
    # breaks that (rare), skip rather than emit a corrupt mask.
    if not (0 < len(prefix) < len(full) and full[:len(prefix)] == prefix):
        return None

    is_response = [False] * len(full)
    for p in range(len(prefix), len(full)):
        is_response[p] = True

    input_ids = full[:-1]
    raw_labels = full[1:]
    target_mask = is_response[1:]             # is the NEXT token a response token?
    labels = [t if m else IGNORE_INDEX for t, m in zip(raw_labels, target_mask)]

    # Pad or truncate to `seq_len`; `attention_mask` is 1 for real tokens, 0 for
    # padding so the model's self-attention skips the pad positions we add.
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

    # If truncation chopped off the entire response (a very long prompt), there's
    # nothing left to learn from — drop it.
    if all(l == IGNORE_INDEX for l in labels):
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _render_examples(
    messages: list[dict],
    tokenizer,
    seq_len: int,
) -> list[dict]:
    """Turn one conversation into one training example per assistant turn.

    A single-turn `(user, assistant)` conversation yields one example. A
    multi-turn conversation yields one example per assistant turn, each ending
    at that turn so `_mask_final_turn`'s diff trick is exact. Returns `[]` for a
    conversation with no assistant turns (a junk row), which the caller drops.
    """
    out: list[dict] = []
    for k, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        sample = _mask_final_turn(messages[:k + 1], tokenizer, seq_len)
        if sample is not None:
            out.append(sample)
    return out


# =============================================================================
# Tokenizer loader (simpler than Module 11's adapter — SFT always uses HF AutoTokenizer)
# =============================================================================

def load_tokenizer(name: str):
    """Load an HF AutoTokenizer. SFT always uses the model's own tokenizer
    (because the chat template is on the tokenizer object); no separate
    BPE-from-disk path like Module 11 needs."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        # Most causal LM tokenizers don't define pad. Use EOS for padding.
        tok.pad_token_id = tok.eos_token_id
    if tok.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {name!r} has no chat_template attribute. Either pick "
            "a tokenizer that ships one (e.g. Qwen/Qwen3-1.7B-Base) or set "
            "tokenizer.chat_template to a Jinja string before loading."
        )
    return tok


# =============================================================================
# ChatDataset — the map-style SFT dataset
# =============================================================================

class ChatDataset(Dataset):
    """Pre-tokenized SFT dataset with chat templates and assistant-only loss mask.

    Loads `cfg.source` via `datasets.load_dataset`, normalizes each row to
    the `{messages: [...]}` shape, then renders + masks each conversation at
    __init__ time. Tokenized tensors live in memory for the rest of training,
    so per-step cost is just tensor indexing + collation.

    For datasets that don't fit in memory (>1M examples, longer sequences),
    swap this for an IterableDataset following Module 11's pattern. For 10k
    no_robots-class datasets, in-memory is faster and simpler.
    """

    def __init__(
        self,
        cfg: DataConfig,
        tokenizer_name: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer(tokenizer_name)

        from datasets import load_dataset
        raw = load_dataset(cfg.source, split=cfg.split)
        if cfg.max_examples is not None:
            raw = raw.select(range(min(cfg.max_examples, len(raw))))

        # Pre-tokenize every row into one example per assistant turn (see
        # _render_examples). Rows that produce no assistant turns (junk /
        # malformed) contribute nothing and are counted as skipped.
        self.samples: list[dict] = []
        n_skipped = 0
        for example in raw:
            messages = _normalize_to_messages(example, cfg.system_prompt)
            examples = _render_examples(messages, self.tokenizer, cfg.seq_len)
            if not examples:
                n_skipped += 1
                continue
            self.samples.extend(examples)

        if not self.samples:
            raise RuntimeError(
                f"ChatDataset({cfg.source!r}) produced 0 usable samples after "
                "filtering. Check the dataset shape and chat_template_name."
            )

        # Useful for the rank-0 startup banner / debugging.
        self.n_total = len(raw)               # raw rows (conversations)
        self.n_skipped = n_skipped            # rows with no usable assistant turn
        self.n_examples = len(self.samples)   # training examples (>= rows for multi-turn)
        self.frac_assistant_tokens = self._compute_assistant_fraction()

    def _compute_assistant_fraction(self) -> float:
        """What fraction of tokens carry loss? Useful sanity check: should
        be in the 0.2-0.6 range for typical chat data. A value near 0 means
        the mask is broken (no targets); a value near 1 means the mask is
        broken (everything is a target)."""
        n_target = 0
        n_total = 0
        for s in self.samples:
            n_target += int((s["labels"] != IGNORE_INDEX).sum().item())
            n_total += s["labels"].numel()
        return n_target / max(n_total, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# =============================================================================
# DataLoader construction
# =============================================================================

def make_dataloader(cfg: DataConfig, tokenizer_name: str) -> DataLoader:
    """Build the SFT DataLoader. Uses `DistributedSampler` for per-rank sharding
    under torchrun (the standard pattern for map-style datasets). Falls back to
    no sampler when run outside a distributed context.

    On `num_workers`: `ChatDataset` pre-tokenizes the whole corpus into memory at
    __init__, so `__getitem__` is a pure index op — worker processes add IPC
    overhead for zero benefit. The default is therefore `num_workers=0`. If you
    *do* set it >0 (e.g. with the packing stretch goal, or a huge SFT mix where
    you move tokenization into `__getitem__`), we switch torch's tensor-sharing
    strategy to `file_system`: the default `file_descriptor` strategy passes one
    fd per shared tensor, and a few-thousand-example in-memory dataset overruns
    the forkserver's ~253-fd handoff limit ("ValueError: too many fds").
    """
    if cfg.num_workers > 0:
        # Share tensors by filename, not by fd — avoids the "too many fds"
        # forkserver overflow when many small pre-tokenized tensors cross the
        # process boundary. (Trade-off: file_system can leak /dev/shm files if a
        # worker is hard-killed; fine for our short runs.)
        import torch.multiprocessing as torch_mp
        torch_mp.set_sharing_strategy("file_system")

    dataset = ChatDataset(cfg, tokenizer_name=tokenizer_name)

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        shuffle = False           # DistributedSampler handles shuffling
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
        # forkserver: same rationale as Module 11/data.py — fork-after-threads
        # under wandb deadlocks; forkserver routes the fork through a clean process.
        multiprocessing_context="forkserver" if cfg.num_workers > 0 else None,
    )


def cycle(loader: DataLoader) -> "Iterator[dict]":
    """Wrap a DataLoader to yield batches forever. Mirrors Module 11."""
    # For DistributedSampler we need to advance epoch every wrap-around so the
    # shuffle order changes between epochs (otherwise rank-0 sees the same
    # samples in the same order every epoch — a subtle reproducibility bug).
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
    # CPU-only smoke test using a tiny dataset slice. Useful to confirm the
    # diff-mask logic works end-to-end on a real tokenizer.
    print("--- ChatDataset smoke test ---")
    smoke_cfg = DataConfig(
        source="HuggingFaceH4/no_robots",
        seq_len=512,
        batch_size_per_device=2,
        num_workers=0,
        max_examples=8,
    )
    ds = ChatDataset(smoke_cfg, tokenizer_name="Qwen/Qwen3-1.7B-Base")
    print(f"  loaded {len(ds)} samples ({ds.n_skipped} skipped of {ds.n_total} raw)")
    print(f"  assistant-token fraction: {ds.frac_assistant_tokens:.1%}")
    print(f"    (expect roughly 0.2-0.6; near-0 or near-1 means broken mask)")

    s = ds[0]
    print(f"\n  sample[0] shapes: input_ids={tuple(s['input_ids'].shape)}, "
          f"labels={tuple(s['labels'].shape)}")
    n_target = int((s["labels"] != IGNORE_INDEX).sum().item())
    n_total = s["labels"].numel()
    print(f"  sample[0] loss-target tokens: {n_target}/{n_total} "
          f"({n_target/n_total:.1%})")

    # Show a small slice of the mask alignment.
    print(f"\n  first 40 (input_id, label) pairs from sample[0]:")
    for t in range(40):
        ii = s["input_ids"][t].item()
        ll = s["labels"][t].item()
        marker = "  " if ll == IGNORE_INDEX else " *"
        print(f"    {t:3d}  input={ii:6d}  label={ll:6d}{marker}")
