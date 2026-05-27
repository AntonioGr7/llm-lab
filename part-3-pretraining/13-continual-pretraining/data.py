"""The continual-pretraining data pipeline: a replay mixture over two corpora.

The one new idea this module adds on top of Module 12's indexed pipeline is the
**replay mix** (README lever 1): interleave samples from your *domain* corpus
and a *general* corpus at a fixed ratio, so the model keeps seeing the broad
distribution while it learns the narrow one. That mix is the single most
effective defence against catastrophic forgetting.

Because Module 12's `PackedIndexedDataset` cuts the corpus into fixed-length
`seq_len+1` blocks, **every sample carries the same number of tokens** — so a
*token* ratio is exactly a *sample* ratio. `MixedDataLoader` therefore just
decides, per micro-batch, whether to draw from domain or replay, using an even
(Bresenham-style) interleave that hits `replay_ratio` over any window.

Resume is **derived from the optimizer step**, not from the live samplers. With
`num_workers>0` the underlying DataLoaders prefetch ahead, so a sampler's
`consumed_samples` runs ahead of what the loop actually trained on (the trap
Module 12's README §7c flags). We sidestep it entirely: the global micro-batch
counter at optimizer step `s` is `k = s * grad_accum`, the replay/domain split
of `[0, k)` is fixed by the ratio, and `seek(k)` positions both samplers by
arithmetic. Bit-exact, prefetch-independent, O(1).
"""
from __future__ import annotations

from typing import Iterator, Optional

from indexed_data import make_indexed_dataloader, cycle


def _n_replay_in(k: int, replay_ratio: float) -> int:
    """How many of the first `k` micro-batches are replay, at this ratio.

    Bresenham/largest-remainder: micro-batch j is replay iff
    `floor((j+1)*r) > floor(j*r)`. Summed over `[0, k)` that is exactly
    `floor(k*r)` — an even spread, no long runs of one source.
    """
    return int(k * replay_ratio)


class MixedDataLoader:
    """Interleave a domain and a replay indexed corpus at `replay_ratio`.

    Iterating yields `{"input_ids", "labels"}` micro-batches forever (epochs
    advance automatically inside each corpus, like Module 12's `cycle`). Call
    `seek(k)` *before* iterating to resume at global micro-batch `k`.

    `replay_ratio == 0` (or an empty `replay_prefix`) disables replay — the
    "domain only" ablation that makes forgetting visible.
    """

    def __init__(
        self,
        domain_prefix: str,
        replay_prefix: str,
        seq_len: int,
        micro_batch: int,
        replay_ratio: float,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        domain_perm_path: Optional[str] = None,
        replay_perm_path: Optional[str] = None,
    ):
        self.replay_ratio = float(replay_ratio)
        self.micro_batch = int(micro_batch)
        self.num_replicas = int(num_replicas)
        self.step_samples = self.micro_batch * self.num_replicas
        self._k = 0   # global micro-batch counter (advances once per drawn batch)

        self.domain_loader, self.domain_sampler = make_indexed_dataloader(
            domain_prefix, seq_len, micro_batch,
            num_replicas=num_replicas, rank=rank, seed=seed,
            num_workers=num_workers, pin_memory=pin_memory,
            perm_path=domain_perm_path or None,
        )

        self.use_replay = self.replay_ratio > 0.0 and bool(replay_prefix)
        if self.use_replay:
            # A different seed so domain and replay don't share a permutation.
            self.replay_loader, self.replay_sampler = make_indexed_dataloader(
                replay_prefix, seq_len, micro_batch,
                num_replicas=num_replicas, rank=rank, seed=seed + 1,
                num_workers=num_workers, pin_memory=pin_memory,
                perm_path=replay_perm_path or None,
            )
        else:
            self.replay_loader = self.replay_sampler = None

    # ---- ratio decision --------------------------------------------------
    def is_replay(self, k: int) -> bool:
        if not self.use_replay:
            return False
        return _n_replay_in(k + 1, self.replay_ratio) > _n_replay_in(k, self.replay_ratio)

    # ---- resume ----------------------------------------------------------
    def seek(self, k_start: int) -> None:
        """Position both samplers to resume at global micro-batch `k_start`.

        Call before `iter(self)`. The replay/domain split of `[0, k_start)` is
        fixed by the ratio, so we set each sampler's `consumed_samples` directly
        — no replay, no dependence on prefetch state.
        """
        self._k = int(k_start)
        n_replay = _n_replay_in(self._k, self.replay_ratio) if self.use_replay else 0
        n_domain = self._k - n_replay
        self.domain_sampler.consumed_samples = n_domain * self.step_samples
        if self.use_replay:
            self.replay_sampler.consumed_samples = n_replay * self.step_samples

    # ---- the mixed stream ------------------------------------------------
    def __iter__(self) -> Iterator[dict]:
        d_it = cycle(self.domain_loader)
        r_it = cycle(self.replay_loader) if self.use_replay else None
        while True:
            if self.is_replay(self._k):
                batch = next(r_it)
            else:
                batch = next(d_it)
            self._k += 1
            yield batch

    # ---- reporting (for the startup banner) ------------------------------
    @property
    def domain_samples(self) -> int:
        return self.domain_sampler.num_samples

    @property
    def replay_samples(self) -> int:
        return self.replay_sampler.num_samples if self.use_replay else 0


def make_mixed_loader(cfg_data, *, num_replicas: int = 1, rank: int = 0) -> MixedDataLoader:
    """Build the MixedDataLoader from a DataConfig (train.py's entry point)."""
    return MixedDataLoader(
        domain_prefix=cfg_data.domain_prefix,
        replay_prefix=cfg_data.replay_prefix,
        seq_len=cfg_data.seq_len,
        micro_batch=cfg_data.batch_size_per_device,
        replay_ratio=cfg_data.replay_ratio,
        num_replicas=num_replicas,
        rank=rank,
        seed=cfg_data.seed,
        num_workers=cfg_data.num_workers,
        pin_memory=cfg_data.pin_memory,
        domain_perm_path=cfg_data.domain_perm_path,
        replay_perm_path=cfg_data.replay_perm_path,
    )


if __name__ == "__main__":
    # Self-contained smoke: build two tiny synthetic indexed corpora and show
    # that the realized replay fraction matches the configured ratio.
    import tempfile
    from pathlib import Path
    import numpy as np
    from indexed_dataset import IndexedDatasetBuilder

    def _tiny_corpus(prefix, token, n_docs=200, doc_len=40):
        b = IndexedDatasetBuilder(prefix, dtype=np.uint16)
        for _ in range(n_docs):
            b.add_document([token] * doc_len)   # all tokens == `token` so we can tell sources apart
        b.finalize(vocab_size=16, eos_id=0)

    with tempfile.TemporaryDirectory() as d:
        _tiny_corpus(str(Path(d) / "domain"), token=5)
        _tiny_corpus(str(Path(d) / "replay"), token=9)
        loader = MixedDataLoader(str(Path(d) / "domain"), str(Path(d) / "replay"),
                                 seq_len=8, micro_batch=4, replay_ratio=0.25,
                                 num_workers=0)
        it = iter(loader)
        n_replay = n_total = 0
        for _ in range(400):
            batch = next(it)
            # a batch is "replay" if its tokens are the replay marker (9)
            is_r = bool((batch["input_ids"] == 9).any())
            n_replay += int(is_r); n_total += 1
        print(f"configured replay_ratio=0.25  realized={n_replay/n_total:.3f} "
              f"({n_replay}/{n_total})")
