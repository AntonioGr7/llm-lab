"""Byte-Pair Encoding, from scratch, in ~80 lines.

This file exists for understanding, not for production use. The encode loop is
O(n²) in the corpus length and there are no optimizations. For real training,
use `train_bpe.py` which wraps the HuggingFace `tokenizers` library (the same
algorithm, in Rust, ~100× faster).

The algorithm:
    1. Start with the 256 byte values as the initial vocabulary.
    2. Encode the corpus as a sequence of byte IDs.
    3. Find the most frequent adjacent pair.
    4. Replace it everywhere with a new token ID.
    5. Repeat until the vocabulary reaches the target size.

Train, encode, and decode are all here. The `__main__` block trains a 10-merge
BPE on a sample paragraph and prints the merges as they happen.
"""
from typing import Iterable


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count adjacent-pair frequencies in a sequence of token IDs."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text: str, vocab_size: int, verbose: bool = False) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:
    """Train a byte-level BPE on `text`.

    Returns:
        vocab: maps token ID -> the bytes that token represents.
        merges: maps (id_a, id_b) -> new_id, in the order they were learned.
            Encoding new text means applying these merges in order.
    """
    assert vocab_size >= 256, "vocab_size must be at least 256 (the byte alphabet)"

    ids: list[int] = list(text.encode("utf-8"))
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[int, int], int] = {}

    num_merges = vocab_size - 256
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break  # nothing left to merge
        pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {i+1:3d}: {pair} -> {new_id}  ({vocab[new_id]!r}, freq={stats[pair]})")

    return vocab, merges


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Encode `text` to token IDs using the learned merges, in priority order."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # Pick the pair with the lowest merge index (= learned earliest = highest priority).
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # no more applicable merges
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids: Iterable[int], vocab: dict[int, bytes]) -> str:
    """Decode token IDs back to a string (lossless if all IDs are in the vocab)."""
    tokens = b"".join(vocab[i] for i in ids)
    return tokens.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # A tiny demo: train 10 merges on a paragraph and watch what BPE learns.
    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The lazy dog sleeps. The quick brown fox runs. "
        "Foxes are quick. Dogs are lazy. The fox and the dog."
    )

    vocab, merges = train(sample, vocab_size=266, verbose=True)
    print()
    print(f"final vocab size: {len(vocab)}")
    print(f"learned merges:   {len(merges)}")

    encoded = encode(sample, merges)
    decoded = decode(encoded, vocab)
    print(f"\noriginal length (bytes):   {len(sample.encode('utf-8'))}")
    print(f"encoded length (tokens):   {len(encoded)}")
    print(f"compression ratio:         {len(sample.encode('utf-8')) / len(encoded):.2f} bytes/token")
    print(f"roundtrip ok:              {decoded == sample}")
