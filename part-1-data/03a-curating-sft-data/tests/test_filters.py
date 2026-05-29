"""
Offline correctness tests for the curation filters.

Designed to run with no network and no heavy deps (no fasttext, no
sentence-transformers). The MinHash test imports `datasketch` and skips
gracefully if it's not installed.

Run from this directory:

    python tests/test_filters.py

Exits 0 on success, nonzero on first failure.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filters import (  # noqa: E402
    ArtifactConfig,
    LangConfig,
    StructuralConfig,
    calque_filter,
    detect_language,
    empty_or_copy_filter,
    english_artifact_filter,
    function_word_filter,
    function_word_ratio,
    langid_filter,
    length_filter,
    ngram_repetition_score,
    repetition_filter,
    run_per_row_filters,
)
from judge import DummyJudge, JudgeScore, filter_by_score, score_rows  # noqa: E402


# ----------------------------------------------------------------------
# Tiny test harness (intentionally no pytest dep)
# ----------------------------------------------------------------------

_passed = 0
_failed = 0
_tests: list[tuple[str, callable]] = []


def check(name: str):
    """Register a test. Tests run inside ``if __name__ == '__main__'`` only —
    NOT at import time. This prevents spawn-mode multiprocessing workers
    (which re-import this module) from recursively re-running every test.
    """

    def deco(fn):
        _tests.append((name, fn))
        return fn

    return deco


def _run_all():
    global _passed, _failed
    for name, fn in _tests:
        try:
            fn()
            print(f"  PASS  {name}")
            _passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            traceback.print_exc()
            _failed += 1


def _row(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ----------------------------------------------------------------------
# Stage 1 — Structural
# ----------------------------------------------------------------------


@check("length_filter accepts a well-sized pair")
def t_length_ok():
    row = _row(
        "Riassumi questo testo in due frasi: il sole è una stella della nostra galassia.",
        "Il sole è una stella che fornisce energia. Si trova al centro del sistema solare.",
    )
    assert length_filter(row) is None


@check("length_filter rejects too-short assistant response")
def t_length_short_assistant():
    row = _row("Cosa è il sole?", "Stella.")
    assert length_filter(row) == "assistant_too_short"


@check("length_filter rejects too-long assistant response")
def t_length_long_assistant():
    row = _row("Scrivi una storia.", " ".join(["parola"] * 3000))
    assert length_filter(row) == "assistant_too_long"


@check("length_filter respects min_response_to_prompt_ratio")
def t_length_ratio():
    long_prompt = " ".join(["istruzione"] * 50)  # 50 words
    short_response = " ".join(["risposta"] * 10)  # 10 words = ratio 0.2 < 0.25
    cfg = StructuralConfig(min_response_to_prompt_ratio=0.25)
    assert length_filter(_row(long_prompt, short_response), cfg) == "response_much_shorter_than_prompt"


@check("empty_or_copy detects response == prompt")
def t_copy_bug():
    row = _row("Genera una frase di esempio.", "Genera una frase di esempio.")
    assert empty_or_copy_filter(row) == "response_equals_prompt"


@check("empty_or_copy detects empty fields")
def t_empty():
    assert empty_or_copy_filter(_row("", "ciao")) == "empty_field"
    assert empty_or_copy_filter(_row("ciao", "")) == "empty_field"


@check("ngram_repetition_score: clean text scores low")
def t_repetition_low():
    txt = "Il sole illumina il mondo, le stelle brillano nel cielo notturno e la luna splende."
    s = ngram_repetition_score(txt, n=3)
    assert s < 0.05, f"expected low score, got {s}"


@check("ngram_repetition_score: degenerate text scores high")
def t_repetition_high():
    txt = " ".join(["test"] * 50)
    s = ngram_repetition_score(txt, n=3)
    assert s > 0.5, f"expected high score, got {s}"


@check("repetition_filter triggers on degenerate output")
def t_repetition_filter():
    row = _row("Ripeti.", " ".join(["test"] * 30))
    assert repetition_filter(row) == "assistant_repetitive"


# ----------------------------------------------------------------------
# Stage 2 — Language fidelity
# ----------------------------------------------------------------------


@check("function_word_ratio: Italian text scores high")
def t_fnword_italian():
    txt = "Il gatto è sul tavolo e mangia il pesce mentre la luna brilla nel cielo."
    r = function_word_ratio(txt)
    assert r > 0.3, f"expected > 0.3 for Italian, got {r:.3f}"


@check("function_word_ratio: English text scores low")
def t_fnword_english():
    txt = "The cat eats fish under the moon while stars shine brightly above."
    r = function_word_ratio(txt)
    assert r < 0.10, f"expected < 0.10 for English, got {r:.3f}"


@check("function_word_filter rejects English assistant response")
def t_fnword_filter_english():
    row = _row("Traduci.", "The cat is on the table eating fish.")
    assert function_word_filter(row) == "low_italian_function_word_ratio"


@check("function_word_filter accepts Italian assistant response")
def t_fnword_filter_italian():
    row = _row("Descrivi.", "Il gatto si trova sul tavolo e mangia il pesce con appetito.")
    assert function_word_filter(row) is None


@check("langid_filter is a no-op when fasttext is unavailable")
def t_langid_no_model():
    # When the model is not loaded, detect_language returns ('?', 0.0) and
    # the filter conservatively returns None.
    lang, conf = detect_language("any text whatsoever")
    if lang == "?":
        row = _row("Hello.", "The cat eats fish.")
        assert langid_filter(row) is None
    else:
        # If the user happens to have fasttext installed, the filter should
        # correctly tag English as non-Italian.
        row = _row("Hello.", "The cat eats fish under the moon and stars.")
        assert langid_filter(row, LangConfig(target_lang="it")) is not None


@check("detect_language degrades gracefully when model.predict raises")
def t_langid_predict_raises():
    # Regression for the fasttext-wheel + NumPy 2.x bug: predict() raises
    # ValueError on np.array(probs, copy=False). detect_language should
    # catch that, fall back to model.predict (also broken), and finally
    # degrade to ('?', 0.0) so the filter pipeline keeps going.
    import filters as F
    saved = F._fasttext_model

    class BrokenModel:
        class _Binding:
            def predict(self, *a, **kw):
                raise ValueError("simulated C++ binding failure")
        f = _Binding()

        def predict(self, *a, **kw):
            raise ValueError("simulated NumPy 2 incompatibility")

    F._fasttext_model = BrokenModel()
    try:
        lang, conf = F.detect_language("ciao a tutti")
        assert lang == "?" and conf == 0.0
        # And the filter should be a no-op (None) instead of crashing.
        assert F.langid_filter(_row("ciao", "il gatto mangia il pesce")) is None
    finally:
        F._fasttext_model = saved


# ----------------------------------------------------------------------
# Stage 3 — Translation artifacts
# ----------------------------------------------------------------------


@check("english_artifact: untranslated 'I'm sorry'")
def t_artifact_sorry():
    row = _row("Cosa pensi della politica italiana attuale?", "I'm sorry, I cannot help with this.")
    res = english_artifact_filter(row)
    assert res is not None and "english_artifact" in res


@check("english_artifact: dollar amount")
def t_artifact_dollars():
    row = _row("Quanto costa una pizza a New York?", "Una pizza media costa $25 nelle pizzerie di Manhattan.")
    res = english_artifact_filter(row)
    assert res is not None and "us_dollar_amount" in res


@check("english_artifact: ChatGPT brand mention")
def t_artifact_brand():
    row = _row("Chi sei?", "Sono un assistente sviluppato da ChatGPT per aiutarti.")
    res = english_artifact_filter(row)
    assert res is not None and "chatgpt" in res


@check("english_artifact: clean Italian passes")
def t_artifact_clean():
    row = _row("Descrivi una giornata di primavera.", "Una giornata di primavera è caratterizzata da temperature miti e fiori in boccio.")
    assert english_artifact_filter(row) is None


@check("calque_filter: 'come modello linguistico'")
def t_calque_modello():
    row = _row("Opinione personale?", "Come modello linguistico, non posso fornire opinioni soggettive su questo tema.")
    res = calque_filter(row)
    assert res is not None and "modello_linguistico" in res


@check("calque_filter: 'fa senso'")
def t_calque_fa_senso():
    row = _row("Cosa ne pensi?", "Quella decisione non fa senso considerando tutte le variabili in gioco.")
    res = calque_filter(row)
    assert res is not None and "fa_senso" in res


@check("calque_filter: clean Italian passes")
def t_calque_clean():
    row = _row("Riassumi.", "La storia è ambientata in un piccolo paese di montagna e racconta la vita di una famiglia.")
    assert calque_filter(row) is None


# ----------------------------------------------------------------------
# Stage 4 — MinHash dedup (optional, skipped if datasketch missing)
# ----------------------------------------------------------------------


@check("minhash_dedup: detects near-duplicate prompts")
def t_minhash():
    try:
        from filters import minhash_dedup
        import datasketch  # noqa: F401
    except ImportError:
        print("       SKIP (datasketch not installed)")
        return
    rows = [
        _row("Riassumi in due frasi il seguente testo: la pizza è un piatto popolare italiano.", "..."),
        _row("Riassumi in due frasi il seguente testo: la pizza è un piatto popolare italiano!", "..."),  # near-dup
        _row("Traduci la frase 'the cat is on the table' in italiano.", "..."),
        _row("Scrivi una breve poesia sull'autunno e le foglie che cadono dagli alberi.", "..."),
    ]
    kept, n_dropped = minhash_dedup(rows, threshold=0.8)
    assert n_dropped == 1, f"expected 1 near-dup dropped, got {n_dropped}"
    assert len(kept) == 3


@check("minhash_dedup: resume after mid-stage crash == crash-free run")
def t_minhash_resume():
    try:
        from filters import minhash_dedup
        import datasketch  # noqa: F401
    except ImportError:
        print("       SKIP (datasketch not installed)")
        return
    import json
    import tempfile

    # Dataset with near-dups scattered throughout so dedup decisions depend on
    # the LSH state built from earlier rows — exactly what resume must reproduce.
    rows = []
    for i in range(300):
        rows.append(_row(f"Spiega il concetto numero {i} con un esempio pratico e chiaro.",
                         f"Ecco una spiegazione completa del concetto {i} con esempi utili."))
        if i % 5 == 0:
            rows.append(_row(f"Spiega il concetto numero {i} con un esempio pratico e chiaro!",
                             "dup"))  # near-dup of the row above

    ref_kept, ref_drop = minhash_dedup(rows, threshold=0.85)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cp, sp = d / "s4a.jsonl", d / "s4a.progress.json"

        # Streamed run with frequent markers must match the crash-free reference.
        k1, dr1 = minhash_dedup(rows, threshold=0.85, checkpoint_path=cp,
                                state_path=sp, checkpoint_every=30)
        assert k1 == ref_kept and dr1 == ref_drop, "streamed run diverged from reference"

        # Simulate a mid-stage crash at processed=150: rewrite the marker to that
        # point and append a half-written line to the rows file (truncate must
        # discard it), then resume against the full dataset.
        kept_150, _ = minhash_dedup(rows[:150], threshold=0.85)
        sp.write_text(json.dumps(
            {"processed": 150, "kept": len(kept_150), "dropped": 150 - len(kept_150)}))
        with cp.open("ab") as f:
            f.write(b'{"messages": [partial-broken-tail')

        k2, dr2 = minhash_dedup(rows, threshold=0.85, checkpoint_path=cp,
                                state_path=sp, checkpoint_every=30)
        assert k2 == ref_kept, f"resume kept diverged: {len(k2)} vs {len(ref_kept)}"
        assert dr2 == ref_drop, f"resume dropped diverged: {dr2} vs {ref_drop}"
        on_disk = [json.loads(l) for l in cp.read_text().splitlines()]
        assert on_disk == ref_kept, "on-disk checkpoint != returned rows"


# ----------------------------------------------------------------------
# Orchestrator — short-circuit semantics + drop-reason counting
# ----------------------------------------------------------------------


@check("run_per_row_filters: short-circuits on first reject")
def t_orchestrator_short_circuit():
    filters = [
        lambda r: empty_or_copy_filter(r),
        lambda r: length_filter(r),
        lambda r: english_artifact_filter(r),
    ]
    # This row should hit length_filter first (assistant_too_short), NOT english_artifact.
    rows = [_row("Hi", "I'm sorry")]
    kept, report = run_per_row_filters(rows, filters)
    assert len(kept) == 0
    assert report.drops.get("user_too_short") == 1
    # english_artifact should NOT have been recorded (short-circuited).
    assert report.drops.get("english_artifact:english_refusal_im_sorry") is None


@check("run_per_row_filters: mixed batch — counts add up")
def t_orchestrator_counts():
    filters = [empty_or_copy_filter, length_filter]
    rows = [
        _row("", "x"),  # empty
        _row("a", "b"),  # too short
        _row("Riassumi questo testo.", "La risposta è chiara e completa con molte parole utili e contenuto."),  # KEEP
    ]
    kept, report = run_per_row_filters(rows, filters)
    assert report.total == 3
    assert report.kept == 1
    assert sum(report.drops.values()) == 2


@check("run_per_row_filters_parallel: matches sequential output")
def t_parallel_matches_sequential():
    import json as _json
    from filters import (
        ArtifactConfig, LangConfig, StructuralConfig,
        run_per_row_filters_parallel,
    )
    s_cfg, l_cfg, a_cfg = StructuralConfig(), LangConfig(), ArtifactConfig()
    skip = {"langid": True}  # avoid fasttext model load in workers
    rows = [
        _row("", "x"),
        _row("a", "b"),
        _row("Riassumi questo testo brevemente.", "La risposta è chiara e completa con molte parole utili italiane."),
        _row("Cosa pensi?", "I'm sorry, as an AI, I cannot answer that question right now."),
        _row("Riassumi.", "Risposta sufficiente, contiene parole italiane in numero adeguato per superare il filtro."),
    ]
    seq_kept, seq_report = run_per_row_filters_parallel(
        rows, structural=s_cfg, lang=l_cfg, artifact=a_cfg, skip=skip, n_workers=1,
    )
    par_kept, par_report = run_per_row_filters_parallel(
        rows, structural=s_cfg, lang=l_cfg, artifact=a_cfg, skip=skip,
        n_workers=2, chunk_size=2,
    )
    assert seq_report.kept == par_report.kept, (seq_report.kept, par_report.kept)
    assert seq_report.total == par_report.total
    assert dict(seq_report.drops) == dict(par_report.drops)
    # Kept rows must be the same set (parallel may reorder).
    assert sorted(_json.dumps(r, sort_keys=True) for r in seq_kept) == \
           sorted(_json.dumps(r, sort_keys=True) for r in par_kept)


# ----------------------------------------------------------------------
# Judge — Dummy backend (no model load)
# ----------------------------------------------------------------------


@check("DummyJudge: deterministic and length-dependent")
def t_dummy_judge_basic():
    judge = DummyJudge()
    row_short = _row("?", "Boh.")
    row_long = _row("?", " ".join(["parola"] * 60))
    s_short = judge(row_short)
    s_long = judge(row_long)
    assert isinstance(s_short, JudgeScore)
    assert s_short.overall < s_long.overall
    assert s_long.overall == 5
    # Determinism.
    assert judge(row_short) == s_short


@check("score_rows + filter_by_score: end-to-end")
def t_score_and_filter():
    rows = [
        _row("Riassumi.", " ".join(["parola"] * 50)),  # high score
        _row("Riassumi.", "Boh."),  # low score
    ]
    scores = score_rows(rows, DummyJudge(), show_progress=False)
    assert len(scores) == 2
    kept, kept_scores = filter_by_score(rows, scores, min_overall=3, min_fluency_it=3)
    assert len(kept) == 1
    assert all(s.overall >= 3 for s in kept_scores)


@check("JudgeScore.from_json: parses bare JSON")
def t_judge_parse_bare():
    s = JudgeScore.from_json('{"fluency_it": 4, "relevance": 5, "correctness": 4, "overall": 4, "rationale": "ok"}')
    assert s.overall == 4 and s.rationale == "ok"


@check("JudgeScore.from_json: tolerates markdown fence")
def t_judge_parse_fenced():
    blob = "Ecco la mia valutazione:\n```json\n{\"fluency_it\": 5, \"relevance\": 5, \"correctness\": 5, \"overall\": 5, \"rationale\": \"perfetto\"}\n```"
    s = JudgeScore.from_json(blob)
    assert s.overall == 5


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


if __name__ == "__main__":
    _run_all()
    print(f"\n{'=' * 50}\nResults: {_passed} passed, {_failed} failed\n{'=' * 50}")
    sys.exit(0 if _failed == 0 else 1)
