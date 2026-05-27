"""Correctness tests for the indexed data pipeline.

This is the Module-12 analog of Module 11's `tests/test_resume.py`, but it
targets the *data iterator* — the thing Module 11 could not checkpoint. Three
properties, each one a claim the README makes:

  1. O(1) random access. Packed sample `i` equals the i-th non-overlapping
     (seq_len+1) block of the flat corpus, and the +1 label shift holds.
  2. Bit-exact resume. A run that checkpoints mid-stream, throws away the
     sampler, rebuilds it, and continues visits *exactly* the same samples in
     the same order as an uninterrupted run — including across epoch boundaries
     where the permutation reseeds.
  3. Distributed sharding. With world_size > 1 each rank gets a disjoint slice
     and the ranks together cover every usable sample exactly once per epoch.

If property 2 ever fails, a resumed pretraining run would silently re-show or
skip data — do not deploy the pipeline to a long run until it passes again.

Usage:
    cd 12-production-data-pipelines/
    python tests/test_indexed_resume.py

Exits 0 on pass, 1 on fail. No network; builds a tiny synthetic corpus.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# Make the module directory importable when running from tests/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import indexed_dataset as ix
from indexed_data import PackedIndexedDataset, ResumableDistributedSampler

SEED = 4242
SEQ_LEN = 15            # block_len = 16
MICRO = 2
CORPUS = str(_HERE.parent / "results" / "_test_corpus" / "tiny")


def build_corpus(n_docs: int = 60, vocab: int = 64) -> ix.IndexedDataset:
    Path(CORPUS).parent.mkdir(parents=True, exist_ok=True)
    eos = vocab - 1
    b = ix.IndexedDatasetBuilder(CORPUS, dtype=ix.best_dtype_for_vocab(vocab))
    rng = np.random.default_rng(0)
    for _ in range(n_docs):
        n = int(rng.integers(20, 50))
        b.add_document(rng.integers(0, vocab - 1, size=n).tolist() + [eos])
    b.finalize(vocab_size=vocab, eos_id=eos, extra={"source": "synthetic"})
    return ix.IndexedDataset(CORPUS)


def all_batches(sampler: ResumableDistributedSampler, n_steps: int) -> list[tuple]:
    """Pull `n_steps` micro-batches across however many epochs that spans."""
    out, taken = [], 0
    while taken < n_steps:
        for batch in iter(sampler):     # one epoch; next iter() => next epoch
            out.append(tuple(batch))
            taken += 1
            if taken >= n_steps:
                break
    return out


def test_o1_access(corpus) -> bool:
    ds = PackedIndexedDataset(corpus, seq_len=SEQ_LEN)
    bl = SEQ_LEN + 1
    ok = True
    for i in (0, 1, len(ds) // 2, len(ds) - 1):
        sample = ds[i]
        expected = np.asarray(corpus.flat[i * bl:(i + 1) * bl]).astype(np.int64)
        got = torch.cat([sample["input_ids"], sample["labels"][-1:]]).numpy()
        if not np.array_equal(got, expected):
            print(f"    sample {i}: bytes don't match the flat slice")
            ok = False
        if not torch.equal(sample["input_ids"][1:], sample["labels"][:-1]):
            print(f"    sample {i}: label shift broken")
            ok = False
    print(f"  [1] O(1) access + label shift: {'OK' if ok else 'FAIL'}  "
          f"(num_samples={len(ds)})")
    return ok


def test_bitexact_resume(corpus) -> bool:
    ds = PackedIndexedDataset(corpus, seq_len=SEQ_LEN)
    active = (len(ds) // MICRO) * MICRO
    steps_per_epoch = active // MICRO
    n_steps = steps_per_epoch * 2 + 3        # span >2 epochs
    save_at = steps_per_epoch + 2            # checkpoint mid second epoch

    def fresh(consumed=0):
        return ResumableDistributedSampler(
            num_samples=len(ds), micro_batch=MICRO, num_replicas=1, rank=0,
            seed=SEED, consumed_samples=consumed)

    ref = all_batches(fresh(), n_steps)

    smp = fresh()
    first = all_batches(smp, save_at)
    state = smp.state_dict()
    del smp
    smp2 = fresh(consumed=999)               # wrong start on purpose...
    smp2.load_state_dict(state)              # ...overwritten by the checkpoint
    rest = all_batches(smp2, n_steps - save_at)

    ok = (first + rest) == ref
    crossed_epoch = state["consumed_samples"] // active
    print(f"  [2] bit-exact resume: {'OK' if ok else 'FAIL'}  "
          f"({n_steps} steps, checkpoint after {save_at} at epoch {crossed_epoch})")
    return ok


def test_sharding(corpus) -> bool:
    ds = PackedIndexedDataset(corpus, seq_len=SEQ_LEN)
    world = 4
    active = (len(ds) // (MICRO * world)) * (MICRO * world)
    steps_per_epoch = active // (MICRO * world)
    seen = []
    for r in range(world):
        smp = ResumableDistributedSampler(
            num_samples=len(ds), micro_batch=MICRO, num_replicas=world, rank=r,
            seed=SEED)
        idxs = [i for batch in all_batches(smp, steps_per_epoch) for i in batch]
        seen.append(idxs)

    # With shuffle on, the ranks together cover the first `active` entries of
    # the epoch-0 permutation (the tail is dropped), NOT range(active).
    perm = ResumableDistributedSampler(
        num_samples=len(ds), micro_batch=MICRO, num_replicas=world, rank=0,
        seed=SEED)._permutation(0)
    expected = set(np.asarray(perm[:active]).tolist())

    flat = [i for s in seen for i in s]
    disjoint = len(flat) == len(set(flat))
    complete = set(flat) == expected
    balanced = len({len(s) for s in seen}) == 1
    ok = disjoint and complete and balanced
    print(f"  [3] distributed sharding (world={world}): {'OK' if ok else 'FAIL'}  "
          f"(disjoint={disjoint}, complete={complete}, balanced={balanced}, "
          f"covers {len(set(flat))}/{active})")
    return ok


def main() -> int:
    test_dir = Path(CORPUS).parent
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print(f"Indexed data pipeline tests  (seed={SEED}, seq_len={SEQ_LEN}, micro={MICRO})\n")
    corpus = build_corpus()

    results = [
        test_o1_access(corpus),
        test_bitexact_resume(corpus),
        test_sharding(corpus),
    ]

    shutil.rmtree(test_dir, ignore_errors=True)
    print()
    if all(results):
        print("  PASS — indexed pipeline is O(1), bit-exact on resume, and shards cleanly.")
        return 0
    print("  FAIL — see the failing property above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
