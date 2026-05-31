"""Procedural tool-use corpus generator for Module 19.

Builds three artifacts in one shot, all reproducibly from a seed:

  1. **8 demonstrations** — the K examples that SDFT prepends in-context.
     These are the "training wheels" the demonstration-conditioned teacher
     sees; the student never sees them at training time.
  2. **~500 training prompts** — fresh tool-use requests for the SDFT
     training loop (or for plain on-policy distillation). Disjoint from
     the demonstrations and from the eval set by construction.
  3. **~50 held-out eval prompts** — for measuring new-skill accuracy.
     Same generator, different sub-templates so we test generalization
     (not memorization).

Each example is a JSON object with three fields:

    {
      "user": "Send an email to alice@acme.com saying the meeting is at 3pm",
      "tool": "send_email",
      "args": {"to": "alice@acme.com", "body": "the meeting is at 3pm"},
    }

The ground-truth assistant response is therefore:

    <tool>send_email</tool><args>{"to": "alice@acme.com", "body": "..."}</args>

Why procedural and not real-world data:

- $0 cost, no network, no LLM judge — runs in the unit-test loop.
- Ground truth is *constructive* (we built the example knowing the
  intended args), so the eval verifier is provably correct.
- Reproducibility — same seed gives the same corpus, so notebook
  runs and `results/` artifacts are stable.

What's lost: real-world prompts have more variation in phrasing, more
edge cases (ambiguous tool choice, multi-step plans, errors). For the
pedagogical demo this is the right trade. If you want to scale, swap to
`xlam-function-calling-60k` or similar — `data.py` accepts the same
output schema.

Five tools shipped:

  - `get_weather(city: str, when: str = "today")`
  - `calculator(expression: str)`
  - `search_database(query: str, limit: int = 10)`
  - `send_email(to: str, body: str, subject: str = "")`
  - `create_calendar_event(title: str, date: str, time: str)`
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# =============================================================================
# Word pools — small, deterministic, no real PII
# =============================================================================

_CITIES = ["Paris", "Tokyo", "Lagos", "Buenos Aires", "Berlin", "Mumbai",
           "Sydney", "Cairo", "Toronto", "Seoul"]
_WHENS = ["today", "tomorrow", "this weekend", "next Monday", "Friday"]
_EXPRESSIONS = [
    ("17 * 24", "17 * 24"),
    ("the area of a 12 by 7 rectangle", "12 * 7"),
    ("15% of 240", "0.15 * 240"),
    ("the square root of 144", "144 ** 0.5"),
    ("3 to the power of 5", "3 ** 5"),
    ("the sum of 199 and 348", "199 + 348"),
    ("twice 49", "2 * 49"),
    ("how many seconds in 3.5 hours", "3.5 * 3600"),
    ("(20 + 4) * 3 - 6", "(20 + 4) * 3 - 6"),
    ("9 squared minus 17", "9 ** 2 - 17"),
]
_QUERY_TOPICS = [
    "active enterprise accounts", "customers in Q3", "open support tickets",
    "products with price under 50", "users from Germany",
    "orders shipped last week", "team members in engineering",
    "events in the last 30 days", "blog posts by Lena",
]
_NAMES_FIRST = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "henry"]
_DOMAINS = ["acme.com", "example.org", "research.lab", "team.co"]
_EMAIL_BODIES = [
    "the meeting is at 3pm",
    "please review the attached doc",
    "I will be out tomorrow",
    "the project is approved",
    "can you send me the report",
    "thanks for the feedback",
]
_EVENT_TITLES = ["weekly standup", "client demo", "design review",
                 "1:1 with the team", "all-hands", "interview"]
_DATES = ["2026-06-12", "2026-07-01", "2026-08-15", "2026-09-23", "2026-10-08"]
_TIMES = ["09:00", "10:30", "13:00", "14:30", "16:00", "17:45"]


# =============================================================================
# Example builder
# =============================================================================

@dataclass
class Example:
    user: str
    tool: str
    args: dict[str, Any]

    @property
    def assistant(self) -> str:
        # Stable JSON serialization — sort keys so the string is deterministic.
        return f"<tool>{self.tool}</tool><args>{json.dumps(self.args, sort_keys=True)}</args>"


def _email_from(name: str, domain: str) -> str:
    return f"{name}@{domain}"


def _gen_weather(rng: random.Random, *, eval: bool = False) -> Example:
    city = rng.choice(_CITIES)
    when = rng.choice(_WHENS)
    # Vary the surface phrasing to avoid memorization.
    if eval:
        templates = [
            f"Could you check the weather forecast for {city} {when}?",
            f"I need the {when} weather in {city}.",
            f"How is the weather going to be in {city} {when}?",
        ]
    else:
        templates = [
            f"What's the weather in {city} {when}?",
            f"Tell me the {when} weather for {city}.",
            f"Weather forecast {city} {when}?",
        ]
    return Example(
        user=rng.choice(templates),
        tool="get_weather",
        args={"city": city, "when": when},
    )


def _gen_calculator(rng: random.Random, *, eval: bool = False) -> Example:
    phrase, expr = rng.choice(_EXPRESSIONS)
    if eval:
        templates = [
            f"Could you compute {phrase} for me?",
            f"I need to know: what's {phrase}?",
        ]
    else:
        templates = [
            f"What is {phrase}?",
            f"Compute {phrase}.",
            f"Please calculate {phrase}.",
        ]
    return Example(
        user=rng.choice(templates),
        tool="calculator",
        args={"expression": expr},
    )


def _gen_search(rng: random.Random, *, eval: bool = False) -> Example:
    query = rng.choice(_QUERY_TOPICS)
    limit = rng.choice([5, 10, 20, 25, 50])
    if eval:
        templates = [
            f"Could you find the top {limit} {query} in our system?",
            f"I need to see {limit} {query}.",
        ]
    else:
        templates = [
            f"Search the database for {query}, limit {limit}.",
            f"Find {limit} {query} in the database.",
            f"Show me {limit} {query}.",
        ]
    return Example(
        user=rng.choice(templates),
        tool="search_database",
        args={"query": query, "limit": limit},
    )


def _gen_email(rng: random.Random, *, eval: bool = False) -> Example:
    name = rng.choice(_NAMES_FIRST)
    domain = rng.choice(_DOMAINS)
    to_addr = _email_from(name, domain)
    body = rng.choice(_EMAIL_BODIES)
    if eval:
        templates = [
            f"Could you send a message to {to_addr} letting them know {body}?",
            f"Please email {to_addr} that {body}.",
        ]
    else:
        templates = [
            f"Send an email to {to_addr} saying {body}.",
            f"Email {to_addr}: {body}.",
            f"Let {to_addr} know that {body}.",
        ]
    return Example(
        user=rng.choice(templates),
        tool="send_email",
        args={"to": to_addr, "body": body},
    )


def _gen_calendar(rng: random.Random, *, eval: bool = False) -> Example:
    title = rng.choice(_EVENT_TITLES)
    date = rng.choice(_DATES)
    time = rng.choice(_TIMES)
    if eval:
        templates = [
            f"Could you add a {title} on {date} at {time} to my calendar?",
            f"I need a {title} scheduled for {date}, {time}.",
        ]
    else:
        templates = [
            f"Schedule a {title} on {date} at {time}.",
            f"Create a calendar event: {title} on {date}, {time}.",
            f"Add {title} to my calendar for {date} {time}.",
        ]
    return Example(
        user=rng.choice(templates),
        tool="create_calendar_event",
        args={"title": title, "date": date, "time": time},
    )


_GENERATORS = [
    _gen_weather,
    _gen_calculator,
    _gen_search,
    _gen_email,
    _gen_calendar,
]


# =============================================================================
# Public builder
# =============================================================================

def build_corpus(
    n_demos: int = 8,
    n_train: int = 500,
    n_eval: int = 50,
    seed: int = 42,
) -> dict[str, list[Example]]:
    """Build the three splits. Each split's RNG is seeded separately and
    deterministically so the splits are reproducible and disjoint."""
    out: dict[str, list[Example]] = {"demos": [], "train": [], "eval": []}

    # Demos: one tool spread per demo if n_demos >= 5, else round-robin.
    rng_demos = random.Random(seed)
    for i in range(n_demos):
        gen = _GENERATORS[i % len(_GENERATORS)]
        out["demos"].append(gen(rng_demos, eval=False))

    # Train: roughly balanced across tools.
    rng_train = random.Random(seed + 1)
    for i in range(n_train):
        gen = _GENERATORS[rng_train.randrange(len(_GENERATORS))]
        out["train"].append(gen(rng_train, eval=False))

    # Eval: uses the *eval* templates so the model can't trivially
    # memorize surface phrasing from training.
    rng_eval = random.Random(seed + 2)
    for i in range(n_eval):
        gen = _GENERATORS[rng_eval.randrange(len(_GENERATORS))]
        out["eval"].append(gen(rng_eval, eval=True))

    return out


def write_corpus(corpus: dict[str, list[Example]], out_dir: str) -> None:
    """Write each split to `{out_dir}/{split}.jsonl`."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for split, examples in corpus.items():
        with (path / f"{split}.jsonl").open("w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")


def read_corpus(in_dir: str) -> dict[str, list[Example]]:
    """Read the three splits back from disk."""
    path = Path(in_dir)
    out: dict[str, list[Example]] = {}
    for split in ("demos", "train", "eval"):
        examples: list[Example] = []
        with (path / f"{split}.jsonl").open() as f:
            for line in f:
                d = json.loads(line)
                examples.append(Example(**d))
        out[split] = examples
    return out


# =============================================================================
# Verifier — grades a model's response on a single eval example
# =============================================================================

_TOOL_RE = re.compile(r"<tool>([\s\S]*?)</tool>")
_ARGS_RE = re.compile(r"<args>([\s\S]*?)</args>")


def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """Pull `(tool_name, args_dict)` out of a model response.

    Returns `(None, None)` for either component that doesn't parse. The
    schema is intentionally strict — partial credit invites reward
    hacking. See Module 18 §4 for why.
    """
    t = _TOOL_RE.search(text)
    a = _ARGS_RE.search(text)
    tool = t.group(1).strip() if t else None
    args: dict | None = None
    if a:
        try:
            args = json.loads(a.group(1))
            if not isinstance(args, dict):
                args = None
        except json.JSONDecodeError:
            args = None
    return tool, args


def score_example(response: str, example: Example) -> dict[str, float]:
    """Three signals:
      schema_ok    — both <tool> and <args> tags present, args parses to dict.
      tool_ok      — tool name matches ground truth.
      args_ok      — args dict matches ground truth exactly (key-by-key).
    `correct` = schema_ok AND tool_ok AND args_ok.
    """
    tool, args = parse_tool_call(response)
    schema_ok = float(tool is not None and args is not None)
    tool_ok = float(tool == example.tool)
    args_ok = float(args == example.args)
    correct = float(schema_ok and tool_ok and args_ok)
    return {
        "schema_ok": schema_ok,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "correct": correct,
    }


def grade_corpus(responses: list[str], examples: list[Example]) -> dict[str, float]:
    """Aggregate `score_example` over a list of (response, example) pairs."""
    assert len(responses) == len(examples), (len(responses), len(examples))
    n = len(responses)
    totals = {"schema_ok": 0.0, "tool_ok": 0.0, "args_ok": 0.0, "correct": 0.0}
    for r, e in zip(responses, examples):
        s = score_example(r, e)
        for k in totals:
            totals[k] += s[k]
    return {k: v / max(n, 1) for k, v in totals.items()}


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Build the synthetic tool-use corpus.")
    p.add_argument("--out", default="./data", help="output directory")
    p.add_argument("--n-demos", type=int, default=8)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    corpus = build_corpus(
        n_demos=args.n_demos, n_train=args.n_train,
        n_eval=args.n_eval, seed=args.seed,
    )
    write_corpus(corpus, args.out)

    print(f"--- wrote corpus to {args.out}/ ---")
    for split, examples in corpus.items():
        print(f"  {split}: {len(examples)} examples")
    print("\nExample demo:")
    d = corpus["demos"][0]
    print(f"  user: {d.user}")
    print(f"  assistant: {d.assistant}")
    print("\nExample eval (note paraphrase relative to demo):")
    e = corpus["eval"][0]
    print(f"  user: {e.user}")
    print(f"  assistant (gt): {e.assistant}")


if __name__ == "__main__":
    main()
