"""Tests for the resumable LLM-as-judge orchestrator (score_rows_resumable).

Covers the "if the model fails, we start from where it failed" guarantee:
  - successful scores are flushed to the sidecar as they complete;
  - a crash mid-run loses nothing already written;
  - re-running with the same sidecar re-scores ONLY the unscored rows;
  - the circuit breaker aborts after N consecutive backend failures;
  - JSON parsing is robust to whole-object, fenced, and prose-wrapped output,
    and rejects an empty (thinking-model) response.

Pure offline — no server, no model. Run: python tests/test_judge_resume.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import (  # noqa: E402
    JudgeScore, BackendUnavailable, score_rows_resumable,
)

PASS = []


def check(name):
    def deco(fn):
        def wrapped():
            fn()
            print(f"PASS {name}")
            PASS.append(name)
        return wrapped
    return deco


def _rows(n):
    return [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i} " * (i + 1)}]}
            for i in range(n)]


# ---------------------------------------------------------------------------
# Parsing robustness — the bug that motivated this: a thinking model returns
# empty content, which must NOT silently become a valid score.
# ---------------------------------------------------------------------------

@check("from_json: bare object, fenced, prose-wrapped all parse")
def t_parse_variants():
    bare = '{"fluency_it": 5, "relevance": 4, "correctness": 3, "overall": 4, "rationale": "ok"}'
    s = JudgeScore.from_json(bare)
    assert (s.fluency_it, s.relevance, s.correctness, s.overall) == (5, 4, 3, 4)

    fenced = "```json\n" + bare + "\n```"
    assert JudgeScore.from_json(fenced).overall == 4

    prose = "Here is my evaluation:\n" + bare + "\nHope that helps!"
    assert JudgeScore.from_json(prose).overall == 4


@check("from_json: empty / whitespace / no-object raise (never a silent 0)")
def t_parse_empty_raises():
    for bad in ["", "   ", "\n", "I cannot evaluate this."]:
        try:
            JudgeScore.from_json(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


# ---------------------------------------------------------------------------
# Resume — the core guarantee.
# ---------------------------------------------------------------------------

class FlakyJudge:
    """Scores fine, except indices in `fail_idx` raise BackendUnavailable.
    Records every row index it was actually asked to score. `delay` simulates
    network latency so the consumer loop keeps pace with the worker pool (an
    instant judge lets a worker race ahead of the breaker, which never happens
    against a real HTTP endpoint)."""

    def __init__(self, fail_idx=(), delay=0.0):
        self.fail_idx = set(fail_idx)
        self.delay = delay
        self.seen = []

    def __call__(self, row):
        # Derive the index from the row content (thread-safe; a shared counter
        # would race).
        idx = int(row["messages"][0]["content"][1:])  # "q3" -> 3
        self.seen.append(idx)
        if self.delay:
            import time
            time.sleep(self.delay)
        if idx in self.fail_idx:
            raise BackendUnavailable(f"simulated down on row {idx}")
        n = len(row["messages"][1]["content"].split())
        return JudgeScore(n, n, n, n, rationale=f"row {idx}")


@check("resume: failed rows left None, written rows flushed to sidecar")
def t_partial_then_resume():
    with tempfile.TemporaryDirectory() as d:
        sidecar = Path(d) / "scores.jsonl"
        rows = _rows(10)

        # First pass: rows 3, 7 fail. Breaker disabled so the rest complete.
        j1 = FlakyJudge(fail_idx={3, 7})
        res1 = score_rows_resumable(rows, j1, sidecar, concurrent=1,
                                    show_progress=False, fail_fast_after=0)
        assert res1[3] is None and res1[7] is None, "failed rows must be None"
        assert all(res1[i] is not None for i in range(10) if i not in (3, 7))

        # Sidecar holds exactly the 8 successes — nothing for the failures.
        written = [json.loads(l)["idx"] for l in sidecar.read_text().splitlines()]
        assert sorted(written) == [0, 1, 2, 4, 5, 6, 8, 9], written

        # Second pass: backend healthy now. Only the 2 missing rows are re-scored.
        j2 = FlakyJudge(fail_idx=set())
        res2 = score_rows_resumable(rows, j2, sidecar, concurrent=1,
                                    show_progress=False, fail_fast_after=0)
        assert j2.seen == [3, 7] or sorted(j2.seen) == [3, 7], \
            f"resume should only re-score 3 and 7, got {j2.seen}"
        assert all(res2[i] is not None for i in range(10)), "all rows scored after resume"


@check("resume: a fully-scored sidecar re-scores nothing")
def t_resume_noop_when_complete():
    with tempfile.TemporaryDirectory() as d:
        sidecar = Path(d) / "scores.jsonl"
        rows = _rows(5)
        score_rows_resumable(rows, FlakyJudge(), sidecar, concurrent=1, show_progress=False)
        j2 = FlakyJudge()
        res = score_rows_resumable(rows, j2, sidecar, concurrent=1, show_progress=False)
        assert j2.seen == [], "no rows should be re-scored"
        assert all(s is not None for s in res)


@check("resume: corrupt sidecar lines are skipped, not fatal")
def t_resume_tolerates_corrupt_sidecar():
    with tempfile.TemporaryDirectory() as d:
        sidecar = Path(d) / "scores.jsonl"
        rows = _rows(4)
        # Hand-write a sidecar with one good record, one garbage line.
        sidecar.write_text(
            json.dumps({"idx": 0, "score": JudgeScore(5, 5, 5, 5, "ok").__dict__}) + "\n"
            "{ this is not valid json\n"
        )
        j = FlakyJudge()
        res = score_rows_resumable(rows, j, sidecar, concurrent=1, show_progress=False)
        assert 0 not in j.seen, "row 0 was already scored, should be skipped"
        assert sorted(j.seen) == [1, 2, 3]
        assert all(s is not None for s in res)


# ---------------------------------------------------------------------------
# Circuit breaker.
# ---------------------------------------------------------------------------

@check("circuit breaker: aborts after N consecutive backend failures")
def t_circuit_breaker_trips():
    with tempfile.TemporaryDirectory() as d:
        sidecar = Path(d) / "scores.jsonl"
        rows = _rows(200)
        # Every row fails -> consecutive failures pile up fast. A small per-call
        # delay simulates HTTP latency so the breaker can stop the bleed.
        j = FlakyJudge(fail_idx=set(range(200)), delay=0.005)
        try:
            score_rows_resumable(rows, j, sidecar, concurrent=2,
                                 show_progress=False, fail_fast_after=5)
        except BackendUnavailable as e:
            assert "resume" in str(e).lower()
        else:
            raise AssertionError("expected the breaker to raise BackendUnavailable")
        # Far fewer than 200 rows were attempted (breaker stopped the bleed early).
        assert len(j.seen) < 100, f"breaker should have stopped early, saw {len(j.seen)}"


@check("circuit breaker: a success resets the consecutive counter")
def t_breaker_resets_on_success():
    with tempfile.TemporaryDirectory() as d:
        sidecar = Path(d) / "scores.jsonl"
        rows = _rows(30)
        # Fail rows 0..3 (4 in a row), then row 4 succeeds (reset), then 5..8 fail.
        # With fail_fast_after=6, no run of 6 consecutive failures exists -> no abort.
        j = FlakyJudge(fail_idx={0, 1, 2, 3, 5, 6, 7, 8})
        res = score_rows_resumable(rows, j, sidecar, concurrent=1,
                                   show_progress=False, fail_fast_after=6)
        # It ran to completion; the 8 failed rows are None, the other 22 scored.
        assert sum(1 for s in res if s is None) == 8
        assert sum(1 for s in res if s is not None) == 22


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for t in tests:
        t()
    print(f"\n{len(PASS)}/{len(tests)} passed")
