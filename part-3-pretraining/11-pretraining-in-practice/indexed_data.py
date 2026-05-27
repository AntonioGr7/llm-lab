"""The training-time data pipeline over an indexed corpus.

This is the file you drop into Module 11's framework. It exposes the same
`{"input_ids", "labels"}` contract Module 11's `data.py` does, but it is
**map-style** (O(1) random access) instead of streaming, which buys two things
Module 11 can't have:

  - **O(1) bit-exact resume.** The order training visits samples is a saved
    permutation. We track one integer — `consumed_samples` — and on resume we
    seek to that position by arithmetic and keep going. No replaying the first
    8 hours of the stream; no re-tokenizing; the resumed run sees *exactly* the
    samples the uninterrupted run would have, in the same order.
  - **A shuffle that covers the whole corpus**, not just a sliding buffer.
    Module 11's streaming shuffle only mixes a `shuffle_buffer`-sized window;
    here the permutation is global over every sample.

Three pieces:

  PackedIndexedDataset       map-style Dataset. Sample i = the i-th non-
                             overlapping (seq_len+1)-token block of the flat
                             corpus. __getitem__(i) is an O(1) mmap slice.
  ResumableDistributedSampler  yields per-rank micro-batches of sample indices
                             in permuted order; checkpoints to one integer.
  make_indexed_dataloader    wires the two into a torch DataLoader; returns
                             (loader, sampler) so train.py can persist sampler
                             state alongside the model checkpoint.

The +1 shift lives in the dataset (same convention as Module 11): a sample is a
`seq_len+1` block, `input_ids = block[:-1]`, `labels = block[1:]`.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from indexed_dataset import IndexedDataset


# =============================================================================
# Map-style packed dataset
# =============================================================================

class PackedIndexedDataset(Dataset):
    """Views the flat token stream as fixed-length training samples.

    A "sample" is a contiguous `block_len = seq_len + 1` window. We pack with a
    fixed stride and never look at document boundaries — exactly like Module
    11's packer, which concatenates EOS-separated documents and cuts every
    `seq_len+1` tokens. The trailing `total_tokens % block_len` tokens are
    dropped (a fraction of one sequence; irrelevant at any real scale).

    Random access is genuinely O(1): sample `i` is `flat[i*bl : (i+1)*bl]`, a
    slice of the memory-mapped corpus. Reading sample 9,000,000 costs the same
    as reading sample 0.
    """

    def __init__(self, corpus: IndexedDataset, seq_len: int):
        self.corpus = corpus
        self.seq_len = seq_len
        self.block_len = seq_len + 1
        self.num_samples = corpus.total_tokens // self.block_len
        if self.num_samples == 0:
            raise ValueError(
                f"corpus has {corpus.total_tokens} tokens but block_len is "
                f"{self.block_len} — not even one full sample. Use a longer "
                "corpus or a shorter seq_len."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> dict:
        start = i * self.block_len
        block = np.asarray(self.corpus.flat[start:start + self.block_len])
        # Copy out of the mmap into an owned int64 tensor (the model wants long).
        block = torch.from_numpy(block.astype(np.int64))
        return {"input_ids": block[:-1], "labels": block[1:]}


# =============================================================================
# Resumable, distributed, permutation-shuffled sampler
# =============================================================================

class ResumableDistributedSampler:
    """Yields this rank's micro-batches of sample indices, in permuted order,
    with O(1) checkpoint/resume.

    Designed to be passed to a DataLoader as `batch_sampler=` (it yields *lists*
    of indices — one micro-batch per `__next__`). It runs in the main process,
    so its `consumed_samples` counter is always checkpointable.

    The shuffle. Each epoch `e` has a permutation `perm_e` of `[0, num_samples)`,
    derived deterministically from `(seed, e)` (or loaded from a cached `.npy`).
    We drop the tail so the usable length is a multiple of `step_samples =
    micro_batch * world` and chop it into `step_samples`-sized chunks. Chunk `c`
    is one synchronized step across the data-parallel group; rank `r` takes the
    contiguous slice `chunk[r*micro_batch : (r+1)*micro_batch]`. Across ranks
    this tiles the chunk exactly — disjoint, complete, balanced.

    The resume trick. `consumed_samples` counts samples consumed *globally*
    (across all ranks). Everything we need on restart is recovered by
    arithmetic, never by replay:

        epoch         = consumed_samples // active_per_epoch
        chunk_to_skip = (consumed_samples % active_per_epoch) // step_samples

    We rebuild `perm_epoch`, fast-forward the chunk pointer to `chunk_to_skip`,
    and continue. Computing where to start is O(1); reaching the first sample is
    one mmap slice. Contrast Module 11, where resume re-opens the stream at
    position 0 and the model replays everything up to the crash.

    Args:
        num_samples:   len(PackedIndexedDataset).
        micro_batch:   per-rank micro-batch (one yield = this many indices).
        num_replicas:  data-parallel world size.
        rank:          this process's data-parallel rank.
        seed:          shuffle seed. Identical across ranks (they must agree on
                       the permutation, then carve disjoint slices of it).
        consumed_samples: where to resume (0 for a fresh run).
        perm_path:     optional cached epoch-0 permutation `.npy` (mmap-loaded
                       for huge corpora). When absent, epoch 0 is regenerated
                       from `seed` exactly as `perm_path` would have been built.
        shuffle:       False => identity order (debugging / curriculum data).
    """

    def __init__(
        self,
        num_samples: int,
        micro_batch: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        consumed_samples: int = 0,
        perm_path: Optional[str] = None,
        shuffle: bool = True,
    ):
        self.num_samples = int(num_samples)
        self.micro_batch = int(micro_batch)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.consumed_samples = int(consumed_samples)
        self.perm_path = perm_path
        self.shuffle = shuffle

        self.step_samples = self.micro_batch * self.num_replicas
        # Usable samples per epoch: drop the remainder so every step is full and
        # every rank gets micro_batch indices in every step.
        self.active_per_epoch = (self.num_samples // self.step_samples) * self.step_samples
        if self.active_per_epoch == 0:
            raise ValueError(
                f"num_samples={self.num_samples} < step_samples={self.step_samples}: "
                "not even one full global step fits. Lower micro_batch/world or "
                "use a bigger corpus."
            )

    # ---- permutation -----------------------------------------------------
    def _permutation(self, epoch: int) -> np.ndarray:
        """The sample order for `epoch`. Deterministic from (seed, epoch).

        `default_rng(seed + epoch).permutation(n)` is reproducible across runs
        and machines (NumPy's PCG64 stream is stable), which is what makes the
        resume bit-exact. Epoch 0 may instead be read from a cached `.npy`
        (mmap, so only the slices we touch are paged in) — see prepare_*.py's
        `--write-perm`. Both paths must produce the same array; they do, because
        the cache is written with this very expression.
        """
        if not self.shuffle:
            return np.arange(self.num_samples, dtype=np.int64)
        if epoch == 0 and self.perm_path is not None:
            return np.load(self.perm_path, mmap_mode="r")
        return np.random.default_rng(self.seed + epoch).permutation(self.num_samples)

    # ---- the index stream ------------------------------------------------
    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of this rank's micro-batches, starting wherever
        `consumed_samples` says. Each yield advances `consumed_samples` by
        `step_samples`, so a subsequent `iter()` (the next epoch) resumes
        seamlessly — `cycle()` below relies on this.
        """
        epoch = self.consumed_samples // self.active_per_epoch
        consumed_in_epoch = self.consumed_samples % self.active_per_epoch
        start_chunk = consumed_in_epoch // self.step_samples
        n_chunks = self.active_per_epoch // self.step_samples

        perm = self._permutation(epoch)
        for c in range(start_chunk, n_chunks):
            base = c * self.step_samples + self.rank * self.micro_batch
            # Read this rank's contiguous slice of the chunk. On an mmap'd perm
            # this pages in only these `micro_batch` entries.
            idx = np.asarray(perm[base:base + self.micro_batch]).tolist()
            self.consumed_samples += self.step_samples
            yield idx

    def __len__(self) -> int:
        """Micro-batches remaining in the current epoch (DataLoader queries this)."""
        consumed_in_epoch = self.consumed_samples % self.active_per_epoch
        start_chunk = consumed_in_epoch // self.step_samples
        return (self.active_per_epoch // self.step_samples) - start_chunk

    # ---- checkpoint surface ---------------------------------------------
    def state_dict(self) -> dict:
        """Everything needed to resume — just the counter plus the knobs that
        define the permutation, persisted so a config typo on resume is caught
        instead of silently reshuffling."""
        return {
            "consumed_samples": self.consumed_samples,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "step_samples": self.step_samples,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["num_samples"] != self.num_samples or state["seed"] != self.seed:
            raise ValueError(
                "data-state checkpoint does not match this corpus/seed: "
                f"saved (num_samples={state['num_samples']}, seed={state['seed']}) "
                f"vs current (num_samples={self.num_samples}, seed={self.seed}). "
                "Resuming would reshuffle and re-show data."
            )
        self.consumed_samples = int(state["consumed_samples"])


# =============================================================================
# DataLoader wiring
# =============================================================================

def _collate(batch: list[dict]) -> dict:
    """Stack a list of per-sample dicts into one batched dict."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_indexed_dataloader(
    prefix: str,
    seq_len: int,
    micro_batch: int,
    *,
    num_replicas: int = 1,
    rank: int = 0,
    seed: int = 0,
    consumed_samples: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
    perm_path: Optional[str] = None,
) -> tuple[DataLoader, ResumableDistributedSampler]:
    """Open an indexed corpus and build (DataLoader, sampler).

    Returns the sampler too, because the *sampler* owns the resume state. The
    Module-11 drop-in pattern (see README) is:

        loader, sampler = make_indexed_dataloader(...)
        # save:  ckpt["data_state"] = sampler.state_dict()
        # load:  sampler.load_state_dict(ckpt["data_state"])

    The dataset is map-style, so workers each hold the same read-only mmap and
    fetch O(1) slices — no per-worker stream to shard, unlike Module 11.
    """
    corpus = IndexedDataset(prefix)
    dataset = PackedIndexedDataset(corpus, seq_len=seq_len)
    sampler = ResumableDistributedSampler(
        num_samples=len(dataset),
        micro_batch=micro_batch,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
        consumed_samples=consumed_samples,
        perm_path=perm_path,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,          # yields per-rank micro-batches itself
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def cycle(loader: DataLoader) -> Iterator[dict]:
    """Yield batches forever, advancing epochs.

    Pretraining is step-counted. When one epoch's `batch_sampler` is exhausted,
    re-iterating the loader calls `iter(sampler)` again; because the sampler's
    `consumed_samples` is now sitting on an epoch boundary, the next epoch's
    permutation (`seed + epoch+1`) kicks in automatically.
    """
    while True:
        for batch in loader:
            yield batch
