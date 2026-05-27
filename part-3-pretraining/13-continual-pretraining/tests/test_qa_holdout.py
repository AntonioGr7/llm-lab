"""The held-out QA probe set never leaks into the training corpus (Module 13).

The acquisition metric is only meaningful if the held-out *questions* are absent
from training — otherwise a model could score by memorizing a question string
instead of internalizing the fact. Conversely, the *answers* MUST be present
(that's the knowledge we're injecting). This test enforces both, on the raw
text, so it needs no tokenizer and no network.

    python tests/test_qa_holdout.py        # or: pytest tests/test_qa_holdout.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from make_corpus import (  # noqa: E402
    build_universe, render_documents, render_train_qa, build_heldout_qa,
    assert_no_leak,
)


def test_no_question_leak():
    u = build_universe(seed=0, n_entities=32)
    train = render_documents(u, seed=0, augment=4) + render_train_qa(u, seed=0)
    held = build_heldout_qa(u)
    assert_no_leak(train, held)             # raises on any leak
    blob = "\n".join(train)
    for h in held:                          # belt-and-suspenders
        assert h["question"] not in blob, f"leaked: {h['question']!r}"


def test_answers_are_present_in_training():
    """Every held-out answer (the fact itself) must appear in training."""
    u = build_universe(seed=1, n_entities=24)
    blob = "\n".join(render_documents(u, seed=1, augment=3) + render_train_qa(u, seed=1))
    missing = [h["answer"] for h in build_heldout_qa(u) if h["answer"] not in blob]
    assert not missing, f"{len(missing)} held-out answers absent from training: {missing[:3]}"


def test_generation_is_deterministic():
    assert build_universe(seed=7, n_entities=16) == build_universe(seed=7, n_entities=16)
    a = render_documents(build_universe(seed=7, n_entities=16), seed=7, augment=2)
    b = render_documents(build_universe(seed=7, n_entities=16), seed=7, augment=2)
    assert a == b


if __name__ == "__main__":
    test_no_question_leak();              print("ok: no held-out question leaks into training")
    test_answers_are_present_in_training();print("ok: every held-out answer is present in training")
    test_generation_is_deterministic();   print("ok: corpus generation is deterministic in seed")
    print("PASS test_qa_holdout")
