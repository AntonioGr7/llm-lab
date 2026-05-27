"""A Megatron-style memory-mapped indexed corpus: builder + reader.

This is the on-disk format the rest of Module 12 reads. It is a deliberately
simplified — but faithful — version of what Megatron-LM, nanotron, and litgpt
write to disk. Three files share a prefix:

    <prefix>.bin        raw token stream, one flat array of the chosen dtype
    <prefix>.idx.npy    int64 array: token count of each document (in order)
    <prefix>.meta.json  small header: dtype, total_tokens, n_docs, eos_id, ...

Why split across three files (Megatron packs all of it into one binary `.idx`
with a magic header): so you can `np.load` the index and `cat` the meta in a
notebook without a parser. The mechanics are identical; only the packaging is
friendlier for teaching.

The two ideas worth taking away:

1. **Tokenize once, to disk.** Module 11 re-tokenizes the whole corpus every
   time it streams it. Here the tokenizer runs once in `prepare_*.py`; training
   reads integers off disk. For a multi-epoch run that is the difference between
   paying the tokenizer's CPU cost once vs. every epoch.

2. **O(1) random access via mmap.** The `.bin` is `np.memmap`-ed, so reading
   document `i` (or, in `indexed_data.py`, packed sample `i`) is a pointer +
   slice into virtual memory — no scan, no decode, no full load into RAM. This
   is the property that makes O(1) bit-exact resume possible: you can jump
   straight to the sample you left off at instead of replaying the stream.

dtype choice (this matters and is easy to get wrong):
  - A token id must fit in the element type. `uint16` holds 0..65,535.
  - GPT-2's vocab is 50,257 → fits in uint16 → 2 bytes/token (Megatron's
    historical default).
  - Qwen3's vocab is 151,936 → does NOT fit in uint16 → must use `uint32` →
    4 bytes/token. That is why the FineWeb-Edu `sample-10BT` corpus is ~40 GB
    on disk with the Qwen3 tokenizer (10B tokens x 4 bytes), not ~20 GB.
`best_dtype_for_vocab` makes this choice for you, the same way Megatron's
`best_fitting_dtype` does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def best_dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Smallest unsigned integer dtype that can hold every token id.

    The token stream is by far the largest artifact in a pretraining setup, so
    halving the per-token byte width (uint16 vs uint32) literally halves your
    disk and your read bandwidth. Pick the smallest type that fits.
    """
    if vocab_size <= 2**8:
        return np.dtype(np.uint8)
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)   # GPT-2 (50,257) lives here
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)   # Qwen3 (151,936) needs this
    return np.dtype(np.int64)


# Map a numpy dtype to a short stable name we can write into meta.json and read
# back. (We avoid pickling the dtype object so the meta file stays plain JSON.)
_DTYPE_TO_NAME = {
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.int64): "int64",
}
_NAME_TO_DTYPE = {v: k for k, v in _DTYPE_TO_NAME.items()}


# =============================================================================
# Builder (writer) — used by prepare_*.py
# =============================================================================

class IndexedDatasetBuilder:
    """Append documents (lists of token ids), then `finalize()` to flush the
    three artifact files.

    Usage:
        b = IndexedDatasetBuilder("results/corpus/train", dtype=np.uint32)
        for ids in tokenized_docs:        # each `ids` already ends with EOS
            b.add_document(ids)
        b.finalize(vocab_size=151936, eos_id=151643, extra={"source": "fineweb"})

    The builder streams tokens straight to the `.bin` file as it goes, so peak
    memory is one document, not the whole corpus — this is what lets the real
    `prepare_fineweb_edu.py` write a 40 GB corpus on a laptop's worth of RAM.
    """

    def __init__(self, prefix: str, dtype: np.dtype):
        self.prefix = str(prefix)
        self.dtype = np.dtype(dtype)
        Path(self.prefix).parent.mkdir(parents=True, exist_ok=True)
        self._bin = open(self.prefix + ".bin", "wb")
        self._sizes: list[int] = []          # tokens per document, in order
        self._total_tokens = 0

    def add_document(self, ids) -> None:
        """Append one document's token ids to the stream.

        Convention (shared with Module 11's packer): the caller has already
        appended the EOS/document-separator token to `ids`, so consecutive
        documents are self-delimiting in the flat stream.
        """
        arr = np.asarray(ids, dtype=self.dtype)
        arr.tofile(self._bin)
        self._sizes.append(int(arr.size))
        self._total_tokens += int(arr.size)

    def finalize(self, vocab_size: int, eos_id: int, extra: dict | None = None) -> dict:
        """Close the `.bin`, write `.idx.npy` (doc sizes) and `.meta.json`."""
        self._bin.close()
        sizes = np.asarray(self._sizes, dtype=np.int64)
        np.save(self.prefix + ".idx.npy", sizes)

        meta = {
            "dtype": _DTYPE_TO_NAME[self.dtype],
            "total_tokens": self._total_tokens,
            "n_docs": len(self._sizes),
            "vocab_size": int(vocab_size),
            "eos_id": int(eos_id),
        }
        if extra:
            meta.update(extra)
        with open(self.prefix + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta


# =============================================================================
# Reader — mmap-backed, O(1) document access
# =============================================================================

class IndexedDataset:
    """Read-only view over an `IndexedDatasetBuilder` corpus.

    Opening it is cheap: we `np.memmap` the `.bin` (which maps the file into the
    process's address space without reading it) and load the small `.idx.npy`
    sizes array. Nothing token-sized is copied into RAM until you ask for it.

    Two access surfaces:
      - `flat` / `total_tokens`: the whole token stream as one (virtual) array.
        `indexed_data.py` slices fixed-length training samples out of this.
      - `__getitem__(doc_id)` / `__len__`: per-document access (the Megatron
        `MMapIndexedDataset` API). Returns the document's tokens as an ndarray.
    Both are O(1): a pointer computation plus a memory-mapped slice.
    """

    def __init__(self, prefix: str):
        self.prefix = str(prefix)
        with open(self.prefix + ".meta.json") as f:
            self.meta = json.load(f)
        self.dtype = _NAME_TO_DTYPE[self.meta["dtype"]]
        self.total_tokens = int(self.meta["total_tokens"])
        self.vocab_size = int(self.meta["vocab_size"])
        self.eos_id = int(self.meta["eos_id"])

        # Document index: sizes -> cumulative start offsets (in tokens).
        self._sizes = np.load(self.prefix + ".idx.npy")
        self._offsets = np.zeros(len(self._sizes) + 1, dtype=np.int64)
        np.cumsum(self._sizes, out=self._offsets[1:])
        assert int(self._offsets[-1]) == self.total_tokens, (
            "idx/meta disagree: sum of document sizes "
            f"({int(self._offsets[-1])}) != total_tokens ({self.total_tokens}). "
            "The corpus is corrupt or was written by a different version."
        )

        # The flat token stream, memory-mapped (not loaded). mode='r' is shared
        # read-only, so many DataLoader workers map the same physical pages.
        self.flat = np.memmap(self.prefix + ".bin", dtype=self.dtype, mode="r",
                              shape=(self.total_tokens,))

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, doc_id: int) -> np.ndarray:
        start = int(self._offsets[doc_id])
        end = int(self._offsets[doc_id + 1])
        return self.flat[start:end]


def open_corpus(prefix: str) -> IndexedDataset:
    """Convenience opener — mirrors the verb `prepare_*.py` documents."""
    return IndexedDataset(prefix)
