"""Data pipeline for the FineWeb-Edu pretraining demo.

Two sources are supported, both yielding `{"input_ids", "labels"}` dicts:

- **fineweb_edu**: streams `HuggingFaceFW/fineweb-edu` (sample-10BT subset),
  tokenizes with the Qwen3 tokenizer, packs documents into fixed-length
  sequences. The dataset is never materialized on disk — we read what we
  consume. Per-rank sharding via `IterableDataset.shard(...)`.
- **synthetic**: random token IDs from a small vocab. For smoke tests,
  CI, and the $0 tier where you just want to exercise the loop.

The data contract every loop in this directory expects:

    batch["input_ids"]: LongTensor[B, seq_len]   — what the model sees
    batch["labels"]:    LongTensor[B, seq_len]   — labels[t] = input_ids[t+1]
                                                   (i.e., already shifted)

Why we shift in the dataset, not the loss: every frontier framework (Megatron,
TorchTitan, HF Trainer) shifts at data-time so the model's forward is just
`logits = model(input_ids); loss = CE(logits, labels)` with no awkward
slicing. The loss helper in `loop.py` follows that convention.
"""
from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

from config import DataConfig


# ===========================================================================
# Tokenizer loader — supports BOTH the HF Hub format (AutoTokenizer) AND a
# locally-trained `tokenizers` library JSON (what Module 03's train_bpe.py
# produces).
# ===========================================================================

class _TokenizerAdapter:
    """Thin wrapper that gives both tokenizer flavors the same surface:
    `.encode(text) -> list[int]` and `.eos_id: int`.

    The packing loop only needs those two. We bypass the `__call__` interface
    so the adapter works against either backend with zero conditional code in
    the dataset itself."""

    def __init__(self, encode_fn, eos_id: int, name: str):
        self._encode = encode_fn
        self.eos_id = eos_id
        self.name = name

    def encode(self, text: str) -> list[int]:
        return self._encode(text)


def load_tokenizer(name: str) -> _TokenizerAdapter:
    """Load a tokenizer by name. Two recognized formats:

    - **HuggingFace Hub ID** (e.g. `"Qwen/Qwen3-0.6B"`, `"meta-llama/Llama-3-8B"`):
      goes through `transformers.AutoTokenizer`. Requires internet on first
      call; cached under ~/.cache/huggingface after.
    - **Local tokenizer.json** (e.g. `"../../part-1-data/03-tokenization/results/tokenizer.json"`):
      a JSON file produced by Module 03's `train_bpe.py`. Goes through the
      `tokenizers` library directly. No internet needed.

    The detector: ends in `.json` → local file. HF Hub IDs never end in `.json`
    (they're repos, not files), so this is unambiguous.
    """
    looks_like_local_file = name.endswith(".json")

    if looks_like_local_file:
        # Course-trained tokenizer (Module 03). The convention is that the
        # EOS/EOT token is "<|endoftext|>" — see `part-1-data/03-tokenization/tokenizer.py`.
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(name)
        eos_id = tk.token_to_id("<|endoftext|>")
        if eos_id is None:
            # Fall back: maybe a different EOS naming convention.
            for candidate in ("<|eos|>", "</s>", "<eos>"):
                eos_id = tk.token_to_id(candidate)
                if eos_id is not None:
                    break
        if eos_id is None:
            raise RuntimeError(
                f"local tokenizer at {name!r} has no recognizable EOS token. "
                "Expected one of: '<|endoftext|>', '<|eos|>', '</s>', '<eos>'."
            )
        return _TokenizerAdapter(
            encode_fn=lambda text: tk.encode(text).ids,
            eos_id=eos_id, name=name,
        )

    # HF Hub tokenizer
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    eos_id = tk.eos_token_id
    if eos_id is None:
        raise RuntimeError(
            f"HF tokenizer {name!r} has no eos_token_id; we need it as a "
            "document separator for packing"
        )
    return _TokenizerAdapter(
        encode_fn=lambda text: tk(text, add_special_tokens=False)["input_ids"],
        eos_id=eos_id, name=name,
    )


# =============================================================================
# Synthetic dataset (fallback / smoke tests)
# =============================================================================

class SyntheticDataset(IterableDataset):
    """Random integer token IDs. For exercising the training loop without
    a real corpus. Each yielded sample is independent and uniformly
    distributed over `[0, vocab_size)`.

    Sharding: when running under `torch.distributed`, each rank yields a
    disjoint slice of samples, based on `WORLD_SIZE` and `RANK`.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_samples: int,
        seed: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        my_id = rank * num_workers + worker_id
        my_world = world * num_workers
        gen = torch.Generator().manual_seed(self.seed + my_id)

        for sample_idx in range(my_id, self.n_samples, my_world):
            # +1 token: input_ids and labels are 1-shifted slices of this block.
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,),
                                generator=gen, dtype=torch.long)
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


# =============================================================================
# FineWeb-Edu streaming + packing
# =============================================================================

class FineWebEduDataset(IterableDataset):
    """Streaming, tokenizing, packing pipeline over HuggingFaceFW/fineweb-edu.

    Per rank:
      1. Open the streaming dataset.
      2. Shard it by (rank, worker) so each (rank, worker) reads a
         disjoint slice.
      3. Shuffle a small buffer for locality breaking.
      4. Tokenize each document, append EOS.
      5. Pack into a rolling buffer; emit `(seq_len + 1)`-token blocks.
      6. Yield `{"input_ids": block[:-1], "labels": block[1:]}`.

    Args:
        tokenizer_name: HF tokenizer ID. The Qwen3 tokenizer
            (`Qwen/Qwen3-0.6B`) is the demo default — vocab 151,936.
        seq_len: length of the per-sample sequence the model sees.
        subset: which FineWeb-Edu sub-corpus to stream. `sample-10BT` is
            10B tokens of the public release — more than enough up to
            ~500M models. Use `default` for the full ~1.3T tokens.
        shuffle_buffer: in-memory buffer size for the streaming shuffle.
            Larger = better randomization, more RAM.
        seed: shuffle seed (deterministic per-(rank, worker)).
        max_documents: optional cap on documents per (rank, worker).
            Useful for smoke tests; `None` streams forever.
    """

    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        seq_len: int = 2048,
        subset: str = "sample-10BT",
        shuffle_buffer: int = 10_000,
        seed: int = 0,
        max_documents: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.subset = subset
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_documents = max_documents

    def _tokenizer(self) -> _TokenizerAdapter:
        # Lazy: only pay the load cost inside the worker that needs it.
        # The adapter handles both HF Hub IDs and local tokenizer.json paths.
        return load_tokenizer(self.tokenizer_name)

    def _stream(self, rank: int, world: int, worker_id: int, num_workers: int):
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        )
        # Two-level sharding: by rank, then by worker within a rank.
        total_shards = world * num_workers
        my_shard = rank * num_workers + worker_id
        ds = ds.shard(num_shards=total_shards, index=my_shard)
        # Shuffle a small in-memory buffer so consecutive blocks aren't from
        # the same URL or contiguous range of the stream.
        ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + my_shard)
        return ds

    def __iter__(self) -> Iterator[dict]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        tok = self._tokenizer()
        eos_id = tok.eos_id

        stream = self._stream(rank, world, worker_id, num_workers)
        block_len = self.seq_len + 1   # +1 for the 1-shift label trick

        buf: list[int] = []
        doc_count = 0
        for example in stream:
            text = example.get("text", "")
            if not text:
                continue
            ids = tok.encode(text)
            buf.extend(ids)
            buf.append(eos_id)
            doc_count += 1

            while len(buf) >= block_len:
                block = torch.tensor(buf[:block_len], dtype=torch.long)
                buf = buf[block_len:]
                yield {"input_ids": block[:-1], "labels": block[1:]}

            if self.max_documents is not None and doc_count >= self.max_documents:
                return


# =============================================================================
# DataLoader construction
# =============================================================================

def make_dataloader(cfg: DataConfig, vocab_size: int) -> DataLoader:
    """Build the DataLoader matching `cfg.source`.

    Both branches return a standard PyTorch DataLoader. The underlying
    IterableDataset handles per-rank + per-worker sharding internally;
    no DistributedSampler is needed (and wouldn't work for streaming
    datasets anyway).
    """
    if cfg.source == "synthetic":
        dataset: IterableDataset = SyntheticDataset(
            vocab_size=vocab_size, seq_len=cfg.seq_len,
            n_samples=cfg.synthetic_samples, seed=cfg.seed,
        )
    elif cfg.source == "fineweb_edu":
        dataset = FineWebEduDataset(
            tokenizer_name=cfg.tokenizer_name,
            seq_len=cfg.seq_len,
            subset=cfg.fineweb_subset,
            shuffle_buffer=cfg.shuffle_buffer,
            seed=cfg.seed,
        )
    else:
        raise ValueError(
            f"unknown data.source: {cfg.source!r}. "
            "Supported: 'synthetic', 'fineweb_edu'."
        )

    # multiprocessing_context: when num_workers > 0, DataLoader spawns workers
    # by forking the main process. If wandb.init() (or anything else) has already
    # started background threads, a plain fork() deadlocks — Python's multiprocessing
    # docs explicitly warn against fork-after-threads. "forkserver" forks workers
    # from a tiny helper process with no pre-existing threads, so it stays safe.
    mp_context = "forkserver" if cfg.num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_device,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=True,
        # IterableDataset is naturally one-pass; persistent workers avoid
        # re-spawning between phases of the outer loop, but they keep the
        # streaming connection open across DataLoader iterations.
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=mp_context,
    )


def cycle(loader: DataLoader) -> Iterator[dict]:
    """Wrap a DataLoader to yield batches forever.

    Pretraining is step-counted, not epoch-counted. This avoids StopIteration
    if the stream ever terminates (it shouldn't for fineweb-edu streaming,
    but synthetic finite datasets need this).
    """
    while True:
        for batch in loader:
            yield batch


# =============================================================================
# Smoke tests
# =============================================================================

if __name__ == "__main__":
    print("--- SyntheticDataset ---")
    syn_cfg = DataConfig(
        source="synthetic", seq_len=64, batch_size_per_device=4,
        synthetic_samples=20, num_workers=0,
    )
    loader = make_dataloader(syn_cfg, vocab_size=2048)
    it = cycle(loader)
    for i in range(3):
        b = next(it)
        print(f"  batch {i}: input_ids {tuple(b['input_ids'].shape)}  "
              f"labels {tuple(b['labels'].shape)}  "
              f"check shift: input[0,1:5]={b['input_ids'][0,1:5].tolist()}  "
              f"labels[0,0:4]={b['labels'][0,0:4].tolist()}")

    # The label-shift sanity check (the most common bug in homemade pipelines)
    b = next(it)
    assert torch.equal(b["input_ids"][:, 1:], b["labels"][:, :-1]), \
        "labels should be input_ids shifted by +1"
    print("  shift assert OK\n")

    print("--- FineWebEduDataset (skipped — requires network) ---")
    print("  To smoke-test: comment out the 'pass' below and run with `datasets`")
    print("  installed and internet access. We yield 1 batch then stop.")
    # from config import DataConfig
    # fw_cfg = DataConfig(
    #     source="fineweb_edu", seq_len=128, batch_size_per_device=2,
    #     num_workers=0, tokenizer_name="Qwen/Qwen3-0.6B",
    # )
    # fw_ds = FineWebEduDataset(seq_len=128, max_documents=20)
    # for b in fw_ds:
    #     print(f"  one packed block: input_ids shape={tuple(b['input_ids'].shape)}")
    #     print(f"    head tokens: {b['input_ids'][:16].tolist()}")
    #     break
