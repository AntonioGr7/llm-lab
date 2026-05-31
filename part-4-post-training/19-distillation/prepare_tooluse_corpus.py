"""Load + filter a REAL function-calling dataset into our Example schema.

Source: `NousResearch/hermes-function-calling-v1`, config `glaive_func_calling`
(5,209 raw rows; ~3,300 single-tool-call rows; Apache 2.0 via Glaive AI).
Ungated — no HF access request needed; downloads ~50 MB on first run and
caches locally.

This is the REAL-DATA path for Module 19's canonical SDFT demo. Companies
deploying SDFT in production will face data exactly like this:

  - Natural-language queries with real phrasing variation
  - Tool schemas with OpenAI-style nested types, descriptions, enums, optionals
  - Argument extraction requiring actual NLU (dates, units, entity names)
  - 4,000+ distinct tools across the dataset (we cap at <=4 per example for
    context-window manageability)

The OUTPUT schema is identical to `make_tooluse_corpus.py` — same
`Example(user, tool, args)` dataclass written to demos/train/eval jsonl
under `--out`. Downstream code (`data.py`, `rollout.py`, etc.) doesn't
care which generator wrote the corpus.

The only difference vs the synthetic generator: `user` is now a formatted
string `"Available tools: {schemas}\n\nRequest: {query}"` — the tool
definitions come IN-CONTEXT per example, matching how real tool-calling
systems work. This is what makes the SDFT demos generalize: the model
isn't memorizing 5 tools, it's learning the *pattern* of "read tool defs,
fill in args correctly."

Usage:

    # Canonical: 8 demos + 2000 train + 200 eval (~$0 in network cost)
    python prepare_tooluse_corpus.py --out=./data

    # Bigger pool for scaled-up runs
    python prepare_tooluse_corpus.py --out=./data_big --n-train=10000

    # Stay offline / synthetic (matches make_tooluse_corpus.py)
    python make_tooluse_corpus.py --out=./data_synthetic
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

from make_tooluse_corpus import Example


# =============================================================================
# Source dataset constants
# =============================================================================

_DATASET = "NousResearch/hermes-function-calling-v1"
_CONFIG = "glaive_func_calling"

_TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")


# =============================================================================
# Per-row parsing — defensive, skip whatever doesn't fit our schema
# =============================================================================

def _extract_user_query(conversations: list[dict]) -> str | None:
    """Pull the first 'human' turn's text. None if missing."""
    for t in conversations:
        if t.get("from") == "human":
            v = t.get("value", "")
            return v.strip() if v else None
    return None


def _extract_tool_call(conversations: list[dict]) -> dict | None:
    """Pull the SINGLE `<tool_call>{...}</tool_call>` block from the 'gpt' turn.

    Returns the parsed JSON dict (with keys `name` and `arguments`), or
    None if the turn has 0 or >1 tool calls, or the JSON doesn't parse.
    """
    for t in conversations:
        if t.get("from") == "gpt":
            matches = _TOOL_CALL_RE.findall(t.get("value", ""))
            if len(matches) != 1:
                return None
            try:
                return json.loads(matches[0].strip())
            except json.JSONDecodeError:
                return None
    return None


def _truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _summarize_tools(tools: list[dict], max_desc_chars: int = 200,
                     max_params_chars: int = 400) -> str:
    """Compact one-line-per-tool summary. Each tool gets:

        - {name}: {description}
          parameters: {json-encoded schema}

    Descriptions and parameter schemas are truncated to keep the in-context
    cost bounded.
    """
    lines = []
    for t in tools:
        # OpenAI-style: {"type": "function", "function": {...}}
        f = t.get("function", t)
        name = f.get("name", "?")
        desc = _truncate(f.get("description", "") or "", max_desc_chars)
        params = f.get("parameters", {})
        params_str = _truncate(json.dumps(params, separators=(",", ":")), max_params_chars)
        lines.append(f"- {name}: {desc}\n  parameters: {params_str}")
    return "\n".join(lines)


def _format_user_message(query: str, tools: list[dict]) -> str:
    """The full user message a row produces: tool definitions IN-CONTEXT, then the request."""
    return f"Available tools:\n{_summarize_tools(tools)}\n\nRequest: {query}"


# =============================================================================
# Top-level: load + filter + convert
# =============================================================================

def prepare_corpus(
    out_dir: str,
    n_demos: int = 8,
    n_train: int = 2000,
    n_eval: int = 200,
    max_tools_per_example: int = 4,
    seed: int = 42,
) -> dict[str, list[Example]]:
    """Build demos/train/eval splits from Hermes/Glaive and write to disk.

    Filtering pipeline:
      1. Drop rows where `tools` JSON doesn't parse, is empty, or has >max_tools.
      2. Drop rows with 0 or >1 `<tool_call>` blocks in the assistant turn.
      3. Drop rows where the called tool's `name` isn't in the available tools.
      4. Drop rows where `arguments` isn't a dict.

    Splits:
      - Demos: K=n_demos SHORTEST surviving examples — keeps the SDFT
        teacher's in-context cost bounded (8 short demos << 8 random demos).
      - Train: next n_train.
      - Eval:  next n_eval. Drawn from the same distribution as train, so the
        "schema_ok / tool_ok / args_ok / correct" rates measure generalization
        to fresh tools + phrasings, not memorization.

    Returns the in-memory dict {demos, train, eval} of Example lists; also
    writes them to `{out_dir}/{split}.jsonl`.
    """
    from datasets import load_dataset

    print(f"[prep] loading {_DATASET}::{_CONFIG} ...")
    ds = load_dataset(_DATASET, _CONFIG, split="train")
    print(f"[prep]   raw rows: {len(ds)}")

    examples: list[Example] = []
    n_bad_tools = n_multi_or_none_call = n_name_mismatch = n_bad_args = 0
    for row in ds:
        try:
            tools = json.loads(row["tools"])
        except (json.JSONDecodeError, KeyError, TypeError):
            n_bad_tools += 1
            continue
        if not isinstance(tools, list) or len(tools) == 0:
            n_bad_tools += 1
            continue
        if len(tools) > max_tools_per_example:
            n_bad_tools += 1
            continue

        query = _extract_user_query(row.get("conversations", []))
        if not query:
            n_multi_or_none_call += 1
            continue

        call = _extract_tool_call(row.get("conversations", []))
        if call is None:
            n_multi_or_none_call += 1
            continue

        tool_name = call.get("name")
        args = call.get("arguments")
        if not tool_name or not isinstance(args, dict):
            n_bad_args += 1
            continue

        # Verify the called tool name is actually in the row's available tools
        avail_names = []
        for t in tools:
            f = t.get("function", t)
            n = f.get("name")
            if n:
                avail_names.append(n)
        if tool_name not in avail_names:
            n_name_mismatch += 1
            continue

        examples.append(Example(
            user=_format_user_message(query, tools),
            tool=tool_name,
            args=args,
        ))

    print(f"[prep]   filtered: {len(examples)} usable single-call examples")
    print(f"[prep]     skipped: tools_bad={n_bad_tools}  "
          f"multi_or_none_call={n_multi_or_none_call}  "
          f"name_mismatch={n_name_mismatch}  bad_args={n_bad_args}")

    if len(examples) < n_demos + n_train + n_eval:
        raise RuntimeError(
            f"Not enough usable examples: need {n_demos + n_train + n_eval}, "
            f"got {len(examples)}. Try lowering --n-train / --n-eval, or "
            f"raising --max-tools-per-example."
        )

    # Demos = the K SHORTEST examples (compact in-context cost).
    examples_sorted_by_len = sorted(
        examples, key=lambda e: len(e.user) + len(e.assistant),
    )
    demos = examples_sorted_by_len[:n_demos]
    demo_ids = {id(e) for e in demos}

    # Train + eval drawn from a SHUFFLED remainder (deterministic via seed).
    remaining = [e for e in examples if id(e) not in demo_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    train = remaining[:n_train]
    eval_split = remaining[n_train : n_train + n_eval]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"demos": demos, "train": train, "eval": eval_split}
    for split_name, exs in splits.items():
        with (out / f"{split_name}.jsonl").open("w") as f:
            for ex in exs:
                f.write(json.dumps(asdict(ex)) + "\n")

    # Helpful summary stats
    avg_demo_chars = sum(len(d.user) + len(d.assistant) for d in demos) / max(len(demos), 1)
    avg_train_chars = sum(len(e.user) + len(e.assistant) for e in train) / max(len(train), 1)
    print(f"[prep]   wrote {out}/")
    print(f"[prep]     demos: {len(demos)}  avg chars: {avg_demo_chars:.0f}")
    print(f"[prep]     train: {len(train)}  avg chars: {avg_train_chars:.0f}")
    print(f"[prep]     eval:  {len(eval_split)}")

    return splits


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Prepare a tool-use corpus from real function-calling data."
    )
    p.add_argument("--out", default="./data", help="output directory")
    p.add_argument("--n-demos", type=int, default=8)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--max-tools-per-example", type=int, default=4,
                   help="cap on number of available tools per row (for context-budget)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    splits = prepare_corpus(
        out_dir=args.out, n_demos=args.n_demos, n_train=args.n_train,
        n_eval=args.n_eval, max_tools_per_example=args.max_tools_per_example,
        seed=args.seed,
    )

    print("\nExample demo (one of the K shortest):")
    d = splits["demos"][0]
    print(f"  user (first 300 chars): {d.user[:300]!r}...")
    print(f"  assistant: {d.assistant}")

    print("\nExample train (random):")
    t = splits["train"][0]
    print(f"  user (first 300 chars): {t.user[:300]!r}...")
    print(f"  assistant: {t.assistant}")


if __name__ == "__main__":
    main()
