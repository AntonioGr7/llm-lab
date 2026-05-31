"""Tests for make_tooluse_corpus.py — the verifier and the corpus generator.

Pure-Python, runs in <1s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_tooluse_corpus import (
    Example, build_corpus, parse_tool_call, score_example, grade_corpus,
)


def test_example_assistant_format():
    """Example.assistant should produce the exact <tool>X</tool><args>JSON</args> shape."""
    ex = Example(user="hi", tool="get_weather", args={"city": "Paris", "when": "today"})
    s = ex.assistant
    assert s.startswith("<tool>get_weather</tool>")
    assert s.endswith("</args>")
    # JSON keys are sorted for determinism.
    assert '"city"' in s and '"when"' in s
    # Re-parsing the assistant string should give back the same args.
    tool, args = parse_tool_call(s)
    assert tool == "get_weather"
    assert args == {"city": "Paris", "when": "today"}
    print("  test_example_assistant_format: PASS")


def test_parse_tool_call_normal():
    text = '<tool>send_email</tool><args>{"to": "x@y.com", "body": "hi"}</args>'
    tool, args = parse_tool_call(text)
    assert tool == "send_email"
    assert args == {"to": "x@y.com", "body": "hi"}
    print("  test_parse_tool_call_normal: PASS")


def test_parse_tool_call_missing_or_malformed():
    # No tool tag
    tool, args = parse_tool_call("just text")
    assert tool is None and args is None
    # Missing args
    tool, args = parse_tool_call("<tool>foo</tool>")
    assert tool == "foo" and args is None
    # Args not parseable
    tool, args = parse_tool_call("<tool>foo</tool><args>not json</args>")
    assert tool == "foo" and args is None
    # Args is a list, not a dict
    tool, args = parse_tool_call('<tool>foo</tool><args>[1, 2, 3]</args>')
    assert args is None
    print("  test_parse_tool_call_missing_or_malformed: PASS")


def test_score_example_correct():
    ex = Example(user="hi", tool="calculator", args={"expression": "2 + 2"})
    resp = '<tool>calculator</tool><args>{"expression": "2 + 2"}</args>'
    s = score_example(resp, ex)
    assert s["schema_ok"] == 1.0
    assert s["tool_ok"] == 1.0
    assert s["args_ok"] == 1.0
    assert s["correct"] == 1.0
    print("  test_score_example_correct: PASS")


def test_score_example_wrong_args():
    ex = Example(user="hi", tool="calculator", args={"expression": "2 + 2"})
    resp = '<tool>calculator</tool><args>{"expression": "2 + 3"}</args>'
    s = score_example(resp, ex)
    assert s["schema_ok"] == 1.0
    assert s["tool_ok"] == 1.0
    assert s["args_ok"] == 0.0   # expression mismatch
    assert s["correct"] == 0.0    # AND fails
    print("  test_score_example_wrong_args: PASS")


def test_score_example_wrong_tool():
    ex = Example(user="hi", tool="calculator", args={"expression": "2 + 2"})
    resp = '<tool>get_weather</tool><args>{"city": "Paris"}</args>'
    s = score_example(resp, ex)
    assert s["schema_ok"] == 1.0
    assert s["tool_ok"] == 0.0
    assert s["correct"] == 0.0
    print("  test_score_example_wrong_tool: PASS")


def test_build_corpus_disjoint_and_reproducible():
    """Same seed -> same corpus. Demos / train / eval should not be empty."""
    c1 = build_corpus(n_demos=8, n_train=50, n_eval=20, seed=42)
    c2 = build_corpus(n_demos=8, n_train=50, n_eval=20, seed=42)
    assert len(c1["demos"]) == 8
    assert len(c1["train"]) == 50
    assert len(c1["eval"]) == 20
    # Deterministic: same seed gives same first user prompt.
    assert c1["train"][0].user == c2["train"][0].user
    # Different seed gives different content.
    c3 = build_corpus(n_demos=8, n_train=50, n_eval=20, seed=999)
    assert c3["train"][0].user != c1["train"][0].user
    print("  test_build_corpus_disjoint_and_reproducible: PASS")


def test_eval_templates_differ_from_train_templates():
    """Eval split uses different surface phrasings so the model can't trivially
    memorize. We check by string substrings — not perfect but indicative."""
    corpus = build_corpus(n_demos=8, n_train=300, n_eval=30, seed=42)
    # Eval prompts tend to start with "Could you" / "I need" phrasings.
    eval_starts = [e.user.split(" ", 2)[0:2] for e in corpus["eval"]]
    eval_text = " ".join(e.user for e in corpus["eval"])
    train_text = " ".join(e.user for e in corpus["train"])
    assert "Could you" in eval_text, "eval split should contain 'Could you' phrasings"
    # Training prompts shouldn't start with 'Could' as commonly.
    n_could_in_eval = sum(1 for e in corpus["eval"] if e.user.startswith("Could"))
    n_could_in_train = sum(1 for e in corpus["train"] if e.user.startswith("Could"))
    eval_could_rate = n_could_in_eval / len(corpus["eval"])
    train_could_rate = n_could_in_train / len(corpus["train"])
    # Eval rate of "Could" starts should be much higher than train rate.
    assert eval_could_rate > train_could_rate + 0.2, (
        f"eval rate {eval_could_rate:.1%} not meaningfully > "
        f"train rate {train_could_rate:.1%}"
    )
    print("  test_eval_templates_differ_from_train_templates: PASS")


def test_grade_corpus_aggregates():
    examples = [
        Example(user="x", tool="a", args={"k": 1}),
        Example(user="y", tool="b", args={"k": 2}),
        Example(user="z", tool="c", args={"k": 3}),
    ]
    responses = [
        '<tool>a</tool><args>{"k": 1}</args>',   # correct
        '<tool>b</tool><args>{"k": 999}</args>', # tool ok, args wrong
        "garbage",                                # all wrong
    ]
    agg = grade_corpus(responses, examples)
    assert abs(agg["correct"] - 1/3) < 1e-9
    assert abs(agg["tool_ok"] - 2/3) < 1e-9
    assert abs(agg["schema_ok"] - 2/3) < 1e-9
    print("  test_grade_corpus_aggregates: PASS")


if __name__ == "__main__":
    print("--- test_tooluse_corpus.py ---")
    test_example_assistant_format()
    test_parse_tool_call_normal()
    test_parse_tool_call_missing_or_malformed()
    test_score_example_correct()
    test_score_example_wrong_args()
    test_score_example_wrong_tool()
    test_build_corpus_disjoint_and_reproducible()
    test_eval_templates_differ_from_train_templates()
    test_grade_corpus_aggregates()
    print("ALL TESTS PASSED")
