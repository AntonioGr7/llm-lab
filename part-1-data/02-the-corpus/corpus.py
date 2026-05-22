"""Streaming corpus loader for FineWeb-Edu.

The canonical data layer for every training script in this course. Use it as:

    stream = stream_fineweb_edu(rank=rank, world_size=world_size)
    for doc in stream:
        text = doc["text"]
        ...  # tokenize, batch, feed to model

Each rank sees a disjoint slice of the corpus. Documents are filtered to a
minimum educational-quality score and minimum length, then shuffled within a
buffer. The same (seed, rank, world_size) gives the same order.
"""
from typing import Iterable

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


FINEWEB_EDU = "HuggingFaceFW/fineweb-edu"

# FineWeb-Edu ships in subsets of varying size: "sample-10BT" (10B tokens),
# "sample-100BT", "sample-350BT", and "default" (~1.3T tokens, the full thing).
# 10BT is plenty for everything we do in the course.
DEFAULT_CONFIG = "sample-10BT"


def stream_fineweb_edu(
    rank: int = 0,
    world_size: int = 1,
    *,
    config: str = DEFAULT_CONFIG,
    split: str = "train",
    min_score: float = 3.0,
    min_chars: int = 200,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
) -> Iterable[dict]:
    """Stream FineWeb-Edu documents, sharded by rank and filtered for quality.

    Args:
        rank: Distributed rank (e.g. ``dist.get_rank()``). Defaults to 0 for
            single-process use (notebooks, local debugging).
        world_size: Total ranks. Defaults to 1.
        config: FineWeb-Edu subset. ``"sample-10BT"`` is the default; use
            ``"default"`` for the full ~1.3T-token corpus.
        split: Dataset split. Only ``"train"`` exists for FineWeb-Edu.
        min_score: Drop documents whose educational-quality score (0-5) is
            below this. 3.0 is a reasonable middle ground; raise to 4.0 for
            stricter filtering, lower to 2.0 to retain more.
        min_chars: Drop documents shorter than this many characters. Filters
            out boilerplate, error pages, and snippet-length noise.
        shuffle_buffer: Buffered-shuffle window size. Larger gives better
            mixing but uses more RAM. 10_000 is a sane default.
        seed: Shuffle seed. ``(seed, rank, world_size)`` determines order.

    Returns:
        An iterable of document dicts. Each doc has at least ``text``, ``id``,
        ``score``, ``url``, ``language``. See the FineWeb-Edu dataset card for
        the full schema.

    The ordering of operations matters:
        1. Load streaming
        2. Shard by node (each rank reads disjoint source files when possible)
        3. Filter (cheap, lazy, per-document)
        4. Shuffle (within ``shuffle_buffer``)
    Sharding before shuffling preserves the "each rank sees disjoint data"
    guarantee that distributed training depends on.
    """
    ds = load_dataset(FINEWEB_EDU, name=config, split=split, streaming=True)

    # split_dataset_by_node prefers to split underlying parquet shards across
    # ranks (efficient — each rank only downloads its files). When that isn't
    # possible, it falls back to per-example modular sharding. Either way the
    # ranks see disjoint documents.
    ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    ds = ds.filter(
        lambda doc: doc["score"] >= min_score and len(doc["text"]) >= min_chars
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    return ds


def measure_throughput(stream: Iterable[dict], n_docs: int = 1000) -> dict:
    """Iterate ``n_docs`` documents and report throughput statistics.

    Useful for diagnosing whether the data stream can keep up with the GPU.
    A loaded A100 typically wants ~1-2 GB/sec of tokens; if this function
    reports much less than that on a cloud pod, the stream will bottleneck
    training and you need bigger shuffle buffers, more dataloader workers,
    or local-disk pre-staging.
    """
    import time

    start = time.perf_counter()
    total_chars = 0
    total_docs = 0
    for doc in stream:
        total_chars += len(doc["text"])
        total_docs += 1
        if total_docs >= n_docs:
            break
    elapsed = time.perf_counter() - start
    return {
        "docs": total_docs,
        "elapsed_s": elapsed,
        "docs_per_s": total_docs / elapsed,
        "mb_per_s": total_chars / elapsed / 1e6,
        "avg_chars": total_chars / max(total_docs, 1),
    }


if __name__ == "__main__":
    # Quick sanity check: stream a handful of docs and print one.
    stream = stream_fineweb_edu(rank=0, world_size=1, shuffle_buffer=100)
    it = iter(stream)
    doc = next(it)
    print(f"score={doc['score']:.2f}  chars={len(doc['text'])}  url={doc.get('url', '?')}")
    print("---")
    print(doc["text"][:500])
    print("---")
    stats = measure_throughput(it, n_docs=200)
    print(f"throughput: {stats['docs_per_s']:.1f} docs/s, {stats['mb_per_s']:.2f} MB/s")
