"""Offline sanity tests for the Module 21 reading-the-literature toolkit.

Run: python tests/test_litkit.py   (from the module directory)
No GPU, no network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litkit import (  # noqa: E402
    TRIAGE_QUESTIONS, Triage, triage_questions,
    SIGNALS, score_claims,
    READING_LIST, Paper, filter_papers, reading_order, areas,
    CITATIONS, most_foundational, _descendants,
)


def test_triage_questions_three_passes():
    qs = triage_questions()
    assert len(qs) == len(TRIAGE_QUESTIONS) == 9
    keys = [k for k, _ in qs]
    assert keys[0] == "claim"           # first question of pass 1
    assert keys[-1] == "transfer"       # last question of pass 3
    assert len(set(keys)) == len(keys)  # unique keys


def test_triage_completeness_and_unanswered():
    t = Triage("p", {})
    assert t.completeness() == 0.0
    assert len(t.unanswered()) == 9
    # Answer everything.
    full = Triage("p", {k: "x" for k, _ in TRIAGE_QUESTIONS})
    assert full.completeness() == 1.0
    assert full.unanswered() == []


def test_triage_verdict_states():
    # Below 70% answered -> incomplete.
    partial = Triage("p", {"claim": "x", "delta": "y"})
    assert "incomplete" in partial.verdict()
    # Enough answered but missing a load-bearing one -> read deeper.
    missing_repro = Triage("p", {k: "x" for k, _ in TRIAGE_QUESTIONS if k != "repro"})
    assert "read deeper" in missing_repro.verdict()
    assert "repro" in missing_repro.verdict()
    # Everything answered -> judge on merits.
    full = Triage("p", {k: "x" for k, _ in TRIAGE_QUESTIONS})
    assert "judge" in full.verdict()


def test_score_claims_tallies():
    # Only counts answered signals.
    r = score_claims({"strong_baselines": True, "cherry_picked": True})
    assert r["signal"] == 1 and r["noise"] == 1 and r["net"] == 0
    assert r["n_checked"] == 2
    # A clean strong paper.
    good = score_claims({s.key: (s.polarity > 0) for s in SIGNALS})
    # green flags set True, red flags set False -> all signal, no noise.
    assert good["noise"] == 0
    assert good["signal"] == sum(1 for s in SIGNALS if s.polarity > 0)
    # An over-claimed paper: red flags True, green flags False -> all noise.
    bad = score_claims({s.key: (s.polarity < 0) for s in SIGNALS})
    assert bad["signal"] == 0
    assert bad["net"] < 0


def test_score_claims_ignores_unanswered():
    r = score_claims({})
    assert r == {"signal": 0, "noise": 0, "net": 0, "checked": [], "n_checked": 0}


def test_signals_have_polarity():
    for s in SIGNALS:
        assert s.polarity in (-1, 1)
    assert any(s.polarity > 0 for s in SIGNALS)
    assert any(s.polarity < 0 for s in SIGNALS)


def test_reading_list_well_formed():
    keys = [p.key for p in READING_LIST]
    assert len(set(keys)) == len(keys)            # unique keys
    for p in READING_LIST:
        assert isinstance(p, Paper)
        assert p.tier in {"foundational", "important", "frontier"}
        assert p.year >= 2017
        assert p.why                               # non-empty rationale


def test_filter_papers():
    found = filter_papers(tier="foundational")
    assert all(p.tier == "foundational" for p in found)
    assert len(found) >= 3
    scaling = filter_papers(area="scaling")
    assert all(p.area == "scaling" for p in scaling)
    recent = filter_papers(max_year=2020)
    assert all(p.year <= 2020 for p in recent)
    # Combined filter.
    combo = filter_papers(area="posttraining", tier="foundational")
    assert all(p.area == "posttraining" and p.tier == "foundational" for p in combo)


def test_reading_order_sorts_by_tier_then_year():
    ordered = reading_order()
    assert len(ordered) == len(READING_LIST)
    tier_rank = {"foundational": 0, "important": 1, "frontier": 2}
    ranks = [tier_rank[p.tier] for p in ordered]
    assert ranks == sorted(ranks)                  # tiers non-decreasing
    # Within each tier, years non-decreasing.
    for tier in tier_rank:
        years = [p.year for p in ordered if p.tier == tier]
        assert years == sorted(years)


def test_areas_first_appearance_order():
    a = areas()
    assert a[0] == "transformers"                  # attention is first in the list
    assert len(set(a)) == len(a)


def test_citation_graph_consistent():
    # Every cited parent exists as a node.
    for child, parents in CITATIONS.items():
        for p in parents:
            assert p in CITATIONS, f"{child} cites unknown {p}"
    # Reading-list keys and citation keys match.
    assert set(CITATIONS) == {p.key for p in READING_LIST}


def test_most_foundational_ranks_attention_first():
    ranked = most_foundational()
    assert ranked[0][0] == "attention"             # the root of the field
    assert ranked[0][1] >= ranked[1][1]            # sorted descending by reach
    # gpt3 should be near the top too.
    top_keys = [k for k, _ in ranked[:3]]
    assert "gpt3" in top_keys


def test_descendants():
    # attention's descendants include everything that transitively builds on it.
    desc = _descendants(CITATIONS, "attention")
    assert "gpt3" in desc
    assert "chinchilla" in desc                    # via kaplan -> gpt3
    assert "dpo" in desc                           # via instructgpt -> gpt3 -> attention
    # A leaf with nothing depending on it has zero descendants.
    assert _descendants(CITATIONS, "chinchilla_repro") == set()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
