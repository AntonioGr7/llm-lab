"""Canonical tokenizer interface for the course.

Every later training script imports from here. The contract is intentionally
minimal — `load_tokenizer()` returns a `tokenizers.Tokenizer`, and the standard
HF tokenizer methods (`.encode`, `.decode`, `.get_vocab_size`) work as expected.

The trained tokenizer lives at `results/tokenizer.json`, produced by
`train_bpe.py`. If the file isn't there, run that first.
"""
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer


DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parent / "results" / "tokenizer.json"


# Special-token IDs. These match the order in train_bpe.DEFAULT_SPECIAL_TOKENS
# and the BpeTrainer's convention of placing specials at the top of the vocab.
EOT_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"


def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER_PATH) -> Tokenizer:
    """Load a trained tokenizer from disk.

    Args:
        path: path to a tokenizer JSON file produced by `train_bpe.py`.
            Defaults to `results/tokenizer.json` next to this module.

    Returns:
        A `tokenizers.Tokenizer`. Standard methods apply:
            tok.encode(text).ids       -> list[int]
            tok.decode(ids)            -> str
            tok.get_vocab_size()       -> int
            tok.token_to_id("<|pad|>") -> int

    Raises:
        FileNotFoundError: if the tokenizer file doesn't exist. Run
            `train_bpe.py` (or pass the notebook through Module 03) first.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer at {path}. Train one first: "
            f"`python train_bpe.py` from this module's directory."
        )
    return Tokenizer.from_file(str(path))


def compression_rate(tokenizer: Tokenizer, text: str) -> float:
    """Characters per token. Higher = better compression for this corpus.

    Useful for comparing tokenizers across languages/domains. Typical values
    on English prose: ~4.0 for well-trained BPE, ~1.0 for character-level.
    Chinese text on an English-only tokenizer drops to ~0.3-0.5 (each Chinese
    character takes 2-3 byte tokens).
    """
    n_tokens = len(tokenizer.encode(text).ids)
    if n_tokens == 0:
        return 0.0
    return len(text) / n_tokens


def encode_iter(tokenizer: Tokenizer, texts: Iterable[str], add_eot: bool = True) -> Iterable[int]:
    """Stream token IDs from an iterable of strings, optionally ending each doc with EOT.

    Convenience for the pretraining dataloader (Module 11): turns a stream of
    documents into a flat stream of token IDs.
    """
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    for text in texts:
        yield from tokenizer.encode(text).ids
        if add_eot:
            yield eot_id


if __name__ == "__main__":
    # Sanity check: load, encode, decode, report.
    tok = load_tokenizer()
    sample = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(sample).ids
    print(f"vocab size: {tok.get_vocab_size()}")
    print(f"sample:     {sample!r}")
    print(f"ids:        {ids}")
    print(f"decoded:    {tok.decode(ids)!r}")
    print(f"compression: {compression_rate(tok, sample):.2f} chars/token")
