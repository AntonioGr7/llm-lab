"""Tests for prepare_tooluse_corpus.py — the Hermes/Glaive parser.

Does NOT download from HF — tests the parsing helpers directly against
hand-crafted row dicts that match the Hermes schema. Keeps the test
suite offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_tooluse_corpus import (
    _extract_user_query, _extract_tool_call, _summarize_tools,
    _format_user_message,
)


# =============================================================================
# _extract_user_query
# =============================================================================

def test_extract_user_query_basic():
    convs = [
        {"from": "system", "value": "ignore"},
        {"from": "human",  "value": "  Hi there  "},
        {"from": "gpt",    "value": "ignore"},
    ]
    assert _extract_user_query(convs) == "Hi there"
    print("  test_extract_user_query_basic: PASS")


def test_extract_user_query_missing():
    assert _extract_user_query([]) is None
    assert _extract_user_query([{"from": "gpt", "value": "no human"}]) is None
    print("  test_extract_user_query_missing: PASS")


# =============================================================================
# _extract_tool_call
# =============================================================================

def test_extract_tool_call_single():
    convs = [
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_x", "arguments": {"a": 1}}\n</tool_call>'},
    ]
    call = _extract_tool_call(convs)
    assert call == {"name": "get_x", "arguments": {"a": 1}}
    print("  test_extract_tool_call_single: PASS")


def test_extract_tool_call_zero_or_multi_returns_none():
    # No tool call
    assert _extract_tool_call([{"from": "gpt", "value": "just text"}]) is None
    # Two tool calls (we only keep single-call rows)
    multi = ('<tool_call>{"name": "a", "arguments": {}}</tool_call> '
             '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert _extract_tool_call([{"from": "gpt", "value": multi}]) is None
    print("  test_extract_tool_call_zero_or_multi_returns_none: PASS")


def test_extract_tool_call_malformed_json():
    convs = [{"from": "gpt", "value": "<tool_call>not json</tool_call>"}]
    assert _extract_tool_call(convs) is None
    print("  test_extract_tool_call_malformed_json: PASS")


# =============================================================================
# Tool-summary formatting
# =============================================================================

def test_summarize_tools_compact():
    tools = [
        {"type": "function", "function": {
            "name": "f1", "description": "First tool.",
            "parameters": {"type": "object", "properties": {"x": {"type": "int"}}},
        }},
        {"type": "function", "function": {
            "name": "f2", "description": "Second.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]
    out = _summarize_tools(tools)
    assert "- f1:" in out
    assert "- f2:" in out
    assert "First tool." in out
    assert "Second." in out
    print("  test_summarize_tools_compact: PASS")


def test_summarize_tools_truncates_long_descriptions():
    tools = [
        {"function": {"name": "huge", "description": "x" * 500, "parameters": {}}},
    ]
    out = _summarize_tools(tools, max_desc_chars=50)
    # Description should be truncated to ~50 chars + "..."
    assert "..." in out
    desc_line = [line for line in out.split("\n") if line.startswith("- huge:")][0]
    # The description portion shouldn't exceed ~55 chars (50 + a small margin for ellipsis).
    assert len(desc_line) < 100, len(desc_line)
    print("  test_summarize_tools_truncates_long_descriptions: PASS")


def test_format_user_message_structure():
    tools = [
        {"function": {"name": "ping", "description": "Ping.", "parameters": {}}},
    ]
    msg = _format_user_message("hello", tools)
    assert msg.startswith("Available tools:")
    assert "- ping:" in msg
    assert "Request: hello" in msg
    print("  test_format_user_message_structure: PASS")


# =============================================================================
# End-to-end on a hand-crafted row
# =============================================================================

def test_endtoend_parse_one_row_shape():
    """Simulate one Hermes row dict and verify the full extraction chain works."""
    import json
    fake_row = {
        "tools": json.dumps([
            {"type": "function", "function": {
                "name": "calc", "description": "Compute.",
                "parameters": {"type": "object", "properties": {
                    "expr": {"type": "string"},
                }, "required": ["expr"]},
            }},
        ]),
        "conversations": [
            {"from": "system", "value": "ignored"},
            {"from": "human",  "value": "What is 7 times 6?"},
            {"from": "gpt",    "value": '<tool_call>\n{"name": "calc", "arguments": {"expr": "7 * 6"}}\n</tool_call>'},
        ],
    }
    tools = json.loads(fake_row["tools"])
    query = _extract_user_query(fake_row["conversations"])
    call = _extract_tool_call(fake_row["conversations"])

    assert query == "What is 7 times 6?"
    assert call["name"] == "calc"
    assert call["arguments"] == {"expr": "7 * 6"}

    user_msg = _format_user_message(query, tools)
    assert "calc" in user_msg and "7 times 6" in user_msg
    print("  test_endtoend_parse_one_row_shape: PASS")


if __name__ == "__main__":
    print("--- test_prepare_corpus.py ---")
    test_extract_user_query_basic()
    test_extract_user_query_missing()
    test_extract_tool_call_single()
    test_extract_tool_call_zero_or_multi_returns_none()
    test_extract_tool_call_malformed_json()
    test_summarize_tools_compact()
    test_summarize_tools_truncates_long_descriptions()
    test_format_user_message_structure()
    test_endtoend_parse_one_row_shape()
    print("ALL TESTS PASSED")
