"""MixedDataLoader delivers the configured replay ratio + resumes by arithmetic.

Two synthetic indexed corpora are built whose tokens are constant markers (5 for
domain, 9 for replay), so we can read each batch's source straight off its
tokens. Then we check:

  - the realized replay fraction matches `replay_ratio`,
  - `replay_ratio == 0` truly disables replay,
  - `seek(k)` positions both underlying samplers to the exact arithmetic split,
  - the source sequence after `seek(k)` equals the tail of the uninterrupted run.

    python tests/test_mix_ratio.py        # or: pytest tests/test_mix_ratio.py
"""
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from indexed_dataset import IndexedDatasetBuilder  # noqa: E402
from data import MixedDataLoader, _n_replay_in       # noqa: E402

DOMAIN_MARKER, REPLAY_MARKER = 5, 9
MICRO, SEQ = 4, 8


def _make_corpus(prefix, marker, n_docs=400, doc_len=40):
    b = IndexedDatasetBuilder(prefix, dtype=np.uint16)
    for _ in range(n_docs):
        b.add_document([marker] * doc_len)
    b.finalize(vocab_size=16, eos_id=0)


def _corpus_dir(tmp):
    _make_corpus(f"{tmp}/domain", DOMAIN_MARKER)
    _make_corpus(f"{tmp}/replay", REPLAY_MARKER)
    return tmp


def _loader(tmp, ratio):
    return MixedDataLoader(f"{tmp}/domain", f"{tmp}/replay", seq_len=SEQ,
                           micro_batch=MICRO, replay_ratio=ratio, num_workers=0)


def _source_seq(loader, n):
    it = iter(loader)
    out = []
    for _ in range(n):
        b = next(it)["input_ids"]
        out.append(REPLAY_MARKER if bool((b == REPLAY_MARKER).any()) else DOMAIN_MARKER)
    return out


def test_realized_ratio_matches():
    for ratio in (0.0, 0.1, 0.25, 0.5):
        with tempfile.TemporaryDirectory() as d:
            _corpus_dir(d)
            seq = _source_seq(_loader(d, ratio), 400)
            frac = sum(s == REPLAY_MARKER for s in seq) / len(seq)
            assert abs(frac - ratio) < 0.02, f"ratio {ratio}: realized {frac:.3f}"


def test_zero_ratio_disables_replay():
    with tempfile.TemporaryDirectory() as d:
        _corpus_dir(d)
        loader = _loader(d, 0.0)
        assert not loader.use_replay
        assert all(s == DOMAIN_MARKER for s in _source_seq(loader, 100))


def test_seek_positions_samplers():
    with tempfile.TemporaryDirectory() as d:
        _corpus_dir(d)
        loader = _loader(d, 0.25)
        loader.seek(100)
        n_replay = _n_replay_in(100, 0.25)          # 25
        step_samples = MICRO                         # world=1
        assert loader.domain_sampler.consumed_samples == (100 - n_replay) * step_samples
        assert loader.replay_sampler.consumed_samples == n_replay * step_samples


def test_seek_matches_uninterrupted_tail():
    with tempfile.TemporaryDirectory() as d:
        _corpus_dir(d)
        full = _source_seq(_loader(d, 0.25), 80)     # one continuous run
        resumed = _loader(d, 0.25)
        resumed.seek(40)
        tail = _source_seq(resumed, 40)
        assert tail == full[40:80], "resumed source sequence diverged from the tail"


if __name__ == "__main__":
    test_realized_ratio_matches();        print("ok: realized replay fraction matches the ratio")
    test_zero_ratio_disables_replay();    print("ok: replay_ratio=0 disables replay")
    test_seek_positions_samplers();       print("ok: seek() positions both samplers by arithmetic")
    test_seek_matches_uninterrupted_tail();print("ok: seek(k) reproduces the uninterrupted tail")
    print("PASS test_mix_ratio")
