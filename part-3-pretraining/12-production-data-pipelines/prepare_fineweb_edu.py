"""Pre-tokenize a corpus into the indexed binary format read by indexed_data.py.

This is the "tokenize once, to disk" step. Run it before training; training
then reads integers off disk instead of re-tokenizing the corpus every epoch.

Two sources:

  --source synthetic   No network. Generates random documents with a tiny
                       vocab. This is the $0 / offline path used by the
                       notebook and tests — a "small portion" you can build in
                       seconds and play with.
  --source fineweb_edu Streams HuggingFaceFW/fineweb-edu and tokenizes with a
                       real tokenizer. Cap it with --num-docs for a small slice,
                       or omit the cap to build the full corpus.

Examples
--------
# Tiny offline corpus to play with (a few thousand random docs):
python prepare_fineweb_edu.py --source synthetic \\
    --output results/corpus/tiny --num-docs 4000 --vocab-size 512

# A small real slice (~20k FineWeb-Edu docs, Qwen3 tokenizer), 8 worker procs:
python prepare_fineweb_edu.py --source fineweb_edu \\
    --output results/corpus/fineweb_small \\
    --tokenizer Qwen/Qwen3-0.6B --num-docs 20000 --workers 8

# The full sample-10BT corpus (~40 GB on disk with Qwen3 — see README):
python prepare_fineweb_edu.py --source fineweb_edu \\
    --output results/corpus/fineweb_10bt --subset sample-10BT --workers 16

Disk + time (rules of thumb, Qwen3 tokenizer):
  - Bytes on disk = total_tokens x 4   (uint32, because vocab 151,936 > 65,535).
    sample-10BT  ~= 10B tokens  -> ~40 GB.
  - Tokenization is CPU-bound and embarrassingly parallel; throughput scales
    with --workers. The full sample-10BT is a few hours on a 16-core box, done
    once. Every epoch after that pays zero tokenization cost.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from indexed_dataset import IndexedDatasetBuilder, best_dtype_for_vocab


# =============================================================================
# Tokenizer (workers) — loaded lazily, once per worker process
# =============================================================================

_TOKENIZER = None
_EOS_ID = None


def _init_worker(tokenizer_name: str) -> None:
    """Pool initializer: load the tokenizer once per worker, not per document."""
    global _TOKENIZER, _EOS_ID
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    _EOS_ID = _TOKENIZER.eos_token_id
    if _EOS_ID is None:
        raise RuntimeError(
            f"tokenizer {tokenizer_name!r} has no eos_token_id; we need it as a "
            "document separator."
        )


def _encode_doc(text: str) -> list[int]:
    """Tokenize one document and append EOS (the document separator)."""
    ids = _TOKENIZER(text, add_special_tokens=False)["input_ids"]
    ids.append(_EOS_ID)
    return ids


# =============================================================================
# Sources
# =============================================================================

def _iter_fineweb(subset: str, num_docs: int | None):
    """Yield raw document strings from streaming FineWeb-Edu."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset,
                      split="train", streaming=True)
    for i, ex in enumerate(ds):
        if num_docs is not None and i >= num_docs:
            return
        text = ex.get("text", "")
        if text:
            yield text


def _build_synthetic(builder: IndexedDatasetBuilder, num_docs: int,
                     vocab_size: int, eos_id: int, seed: int) -> None:
    """Write `num_docs` random documents. No tokenizer, no network."""
    rng = np.random.default_rng(seed)
    for _ in range(num_docs):
        doc_len = int(rng.integers(50, 400))      # varied document lengths
        ids = rng.integers(0, vocab_size - 1, size=doc_len).tolist()
        ids.append(eos_id)                        # document separator
        builder.add_document(ids)


# =============================================================================
# Optional: materialize the epoch-0 permutation as a cached artifact
# =============================================================================

def _write_perm(prefix: str, seq_len: int, seed: int) -> str:
    """Write `<prefix>.s{seq_len}.seed{seed}.perm.npy` — the explicit shuffle.

    Optional. For small corpora the sampler regenerates this from `seed` in
    milliseconds, so you don't need it. For huge corpora, materializing it once
    lets the sampler `mmap` the permutation and page in only the slice it needs
    on resume — the same trick Megatron uses for its cached shuffle index. The
    number of *samples* depends on seq_len, which is why the filename encodes
    it.
    """
    from indexed_dataset import IndexedDataset
    corpus = IndexedDataset(prefix)
    block_len = seq_len + 1
    num_samples = corpus.total_tokens // block_len
    perm = np.random.default_rng(seed).permutation(num_samples)
    out = f"{prefix}.s{seq_len}.seed{seed}.perm.npy"
    np.save(out, perm)
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Pre-tokenize a corpus into the indexed binary format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", choices=["synthetic", "fineweb_edu"],
                   default="synthetic")
    p.add_argument("--output", required=True,
                   help="path prefix; writes <prefix>.bin/.idx.npy/.meta.json")
    p.add_argument("--num-docs", type=int, default=None,
                   help="cap on documents (None = all). Use a small value for a slice.")
    p.add_argument("--seed", type=int, default=0)
    # fineweb_edu
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--subset", type=str, default="sample-10BT")
    p.add_argument("--workers", type=int, default=8)
    # synthetic
    p.add_argument("--vocab-size", type=int, default=512,
                   help="(synthetic only) vocab to draw token ids from")
    # optional perm cache
    p.add_argument("--write-perm", action="store_true",
                   help="also materialize the epoch-0 permutation (needs --seq-len)")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="sequence length, only used by --write-perm")
    args = p.parse_args()

    t0 = time.time()
    print(f"[prepare] source={args.source}  output={args.output}")

    if args.source == "synthetic":
        eos_id = args.vocab_size - 1
        dtype = best_dtype_for_vocab(args.vocab_size)
        builder = IndexedDatasetBuilder(args.output, dtype=dtype)
        n = args.num_docs if args.num_docs is not None else 4000
        _build_synthetic(builder, num_docs=n, vocab_size=args.vocab_size,
                         eos_id=eos_id, seed=args.seed)
        meta = builder.finalize(vocab_size=args.vocab_size, eos_id=eos_id,
                                extra={"source": "synthetic", "seed": args.seed})
    else:
        # We don't know vocab until the tokenizer is loaded; do it up front so
        # we can pick the dtype before opening the builder.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size = tok.vocab_size
        eos_id = tok.eos_token_id
        dtype = best_dtype_for_vocab(vocab_size)
        print(f"[prepare] tokenizer={args.tokenizer}  vocab={vocab_size}  "
              f"dtype={dtype}  ({dtype.itemsize} bytes/token)")
        builder = IndexedDatasetBuilder(args.output, dtype=dtype)
        docs = _iter_fineweb(args.subset, args.num_docs)
        with Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.tokenizer,)) as pool:
            for k, ids in enumerate(pool.imap(_encode_doc, docs, chunksize=64)):
                builder.add_document(ids)
                if (k + 1) % 5000 == 0:
                    print(f"[prepare]   {k+1} docs, "
                          f"{builder._total_tokens/1e6:.1f}M tokens", flush=True)
        meta = builder.finalize(vocab_size=vocab_size, eos_id=eos_id,
                                extra={"source": "fineweb_edu",
                                       "subset": args.subset,
                                       "tokenizer": args.tokenizer})

    dt = time.time() - t0
    gb = meta["total_tokens"] * dtype.itemsize / 1e9
    print(f"[prepare] done in {dt:.1f}s")
    print(f"[prepare]   docs={meta['n_docs']:,}  tokens={meta['total_tokens']:,}  "
          f"dtype={meta['dtype']}  on-disk={gb:.2f} GB")
    print(f"[prepare]   wrote {args.output}.bin / .idx.npy / .meta.json")

    if args.write_perm:
        out = _write_perm(args.output, seq_len=args.seq_len, seed=args.seed)
        print(f"[prepare]   wrote permutation cache {out}")


if __name__ == "__main__":
    main()
