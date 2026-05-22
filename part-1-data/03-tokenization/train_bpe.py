"""Train a byte-level BPE tokenizer on FineWeb-Edu.

Uses HuggingFace's `tokenizers` library (Rust under the hood, ~100x faster
than the from-scratch reference in `bpe_from_scratch.py`). The algorithm is
identical; the speed makes it practical to train on real corpus shards.

The configuration here matches what the course expects downstream:
    - 32k vocab (fits in uint16, sane for a ~125M-param pretraining demo)
    - GPT-2 byte-level regex + explicit digit splitting
    - 10 special tokens reserved at the top of the vocab

CLI usage:
    python train_bpe.py --num-docs 100000 --vocab-size 32000

Programmatic usage:
    from train_bpe import train_bpe
    tokenizer = train_bpe(text_iter, vocab_size=32_000, output_path="results/tokenizer.json")
"""
from itertools import islice
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# Reserved at the top of the vocab. Two are used at pretraining (<|endoftext|>,
# <|pad|>); the rest are placeholders so we don't have to resize the embedding
# table when post-training (SFT, chat templates) needs new tokens. Frontier
# models typically reserve 100-256 of these; 10 is plenty for the course.
DEFAULT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|reserved_0|>",
]

DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_OUTPUT = "results/tokenizer.json"


def build_tokenizer() -> Tokenizer:
    """Construct an untrained byte-level BPE tokenizer with our chosen pretokenizer.

    Pretokenizer stack (applied in order, before BPE training/encoding):
        1. Digits(individual_digits=True) — splits "1234" into "1","2","3","4".
           Modern recipe (Llama 3, DeepSeek). Prevents the model from memorizing
           multi-digit number tokens that hurt arithmetic.
        2. ByteLevel(use_regex=True) — GPT-2's regex splits text into runs of
           letters / digits / punctuation / whitespace, then maps each byte to
           a printable unicode codepoint so the BPE alphabet is human-readable.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_bpe(
    text_iterator: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    special_tokens: list[str] | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on `text_iterator`.

    Args:
        text_iterator: any iterable of strings (raw documents).
        vocab_size: target vocab. The actual count includes special tokens, so
            vocab_size=32000 with 10 special tokens leaves 31990 BPE merges.
        special_tokens: tokens to reserve at the top of the vocab. Defaults
            to DEFAULT_SPECIAL_TOKENS.
        output_path: if given, save the trained tokenizer here as JSON.
        show_progress: print a progress bar (only useful for human runs).

    Returns:
        The trained `tokenizers.Tokenizer`. Call `.encode(text).ids` to tokenize.
    """
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))

    return tokenizer


def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--num-docs", type=int, default=100_000,
        help="how many FineWeb-Edu documents to train on",
    )
    parser.add_argument(
        "--min-score", type=float, default=3.0,
        help="FineWeb-Edu quality filter threshold (0-5)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="where to save the trained tokenizer (relative to this script)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import the corpus stream from the sibling module folder.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "02-the-corpus"))
    from corpus import stream_fineweb_edu  # noqa: E402

    stream = stream_fineweb_edu(
        min_score=args.min_score,
        shuffle_buffer=1_000,
        seed=args.seed,
    )
    text_iter = (doc["text"] for doc in islice(stream, args.num_docs))

    output_path = here / args.output if not Path(args.output).is_absolute() else Path(args.output)

    print(
        f"training BPE: vocab={args.vocab_size}, "
        f"docs={args.num_docs}, min_score={args.min_score}, "
        f"output={output_path}"
    )
    train_bpe(text_iter, vocab_size=args.vocab_size, output_path=output_path)
    print(f"saved tokenizer to {output_path}")


if __name__ == "__main__":
    _cli()
