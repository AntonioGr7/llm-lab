"""Data pipeline interface for Part 3.

This module ships a `SyntheticDataset` that produces random token IDs — enough
to exercise the training loop end-to-end without a real corpus. Module 11
replaces this with the FineWeb-Edu pipeline.

The contract every dataset must satisfy:

- Yields dicts with keys `input_ids` (LongTensor[seq_len]) and `labels`
  (LongTensor[seq_len], usually `input_ids.clone()` for autoregressive LM).
- Is an `IterableDataset` (so it can stream and shard).
- Handles per-rank data sharding internally based on `torch.distributed`.
"""
from __future__ import annotations

from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

from config import DataConfig


# ---------------------------------------------------------------------------
# Synthetic dataset — for Module 08's loop tests
# ---------------------------------------------------------------------------

class SyntheticDataset(IterableDataset):
    """Random integer token IDs. For exercising the training loop without
    real data. Each yielded sample is independent and uniformly distributed
    over `[0, vocab_size)`.

    Sharding: when running under `torch.distributed`, each rank yields a
    disjoint slice of samples (based on `WORLD_SIZE` and `RANK`).
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 0):
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

        # Each (rank, worker) pair gets a disjoint sample subset and seed.
        my_id = rank * num_workers + worker_id
        my_world = world * num_workers
        gen = torch.Generator().manual_seed(self.seed + my_id)

        for sample_idx in range(my_id, self.n_samples, my_world):
            ids = torch.randint(0, self.vocab_size, (self.seq_len,), generator=gen, dtype=torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def make_dataloader(cfg: DataConfig, vocab_size: int) -> DataLoader:
    """Build the DataLoader for `cfg.source`.

    Returns a standard PyTorch DataLoader. The dataset itself handles
    distributed sharding (IterableDataset), so we don't need
    `DistributedSampler` here.
    """
    if cfg.source == "synthetic":
        dataset = SyntheticDataset(
            vocab_size=vocab_size, seq_len=cfg.seq_len,
            n_samples=cfg.synthetic_samples, seed=cfg.seed,
        )
    else:
        raise ValueError(
            f"unknown data source '{cfg.source}'. Module 11 adds 'fineweb_edu'."
        )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_device,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def cycle(loader: DataLoader) -> Iterator[dict]:
    """Wrap a DataLoader to yield batches forever (re-iterating when exhausted).

    Pretraining loops are step-counted, not epoch-counted, so the loader
    needs to be infinite. This is the simplest way.
    """
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    # Smoke test the synthetic dataset.
    cfg = DataConfig(seq_len=32, batch_size_per_device=4, synthetic_samples=100, num_workers=0)
    loader = make_dataloader(cfg, vocab_size=2048)
    batch_iter = cycle(loader)
    for i in range(3):
        batch = next(batch_iter)
        print(f"batch {i}: input_ids {tuple(batch['input_ids'].shape)}  "
              f"labels {tuple(batch['labels'].shape)}  "
              f"sample[0,:8] = {batch['input_ids'][0,:8].tolist()}")
