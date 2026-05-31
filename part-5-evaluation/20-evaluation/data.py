"""Benchmark dataset loaders, each with a $0 offline synthetic fallback.

Every loader takes the same shape: try to pull the real HF dataset; if the
network/dataset isn't available (CI, notebook, no HF auth), fall back to a tiny
hand-written synthetic set with the SAME schema. This is the course-wide
discipline — the logic is exercised offline, the real numbers come from the
real sets on a connected box.

Real sets and their canonical configs:
  - MMLU      — `cais/mmlu`, 4-way MC over 57 subjects, 5-shot canonical.
  - GSM8K     — `openai/gsm8k`/main, grade-school math, 8-shot CoT canonical.
  - IFEval    — `google/IFEval`, verifiable instruction-following. (We ship a
                small built-in constraint set so the checkers are demoable
                without the dataset; the real set has ~540 prompts.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from benchmarks import MCExample, IFExample


# =============================================================================
# MMLU (multiple-choice)
# =============================================================================

_DEFAULT_MMLU_SUBJECTS = (
    "high_school_mathematics", "college_computer_science", "philosophy",
    "professional_medicine", "world_religions",
)


def load_mmlu(n: int, subjects: tuple[str, ...] = (), seed: int = 0) -> list[MCExample]:
    """Load up to `n` MMLU test items spread across `subjects`."""
    subjects = subjects or _DEFAULT_MMLU_SUBJECTS
    try:
        from datasets import load_dataset
        out: list[MCExample] = []
        per = max(1, n // len(subjects))
        for subj in subjects:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for i in range(min(per, len(ds))):
                ex = ds[i]
                out.append(MCExample(
                    question=ex["question"], choices=list(ex["choices"]),
                    answer_index=int(ex["answer"]), subject=subj))
        return out[:n]
    except Exception as e:
        print(f"[data] MMLU real load failed ({type(e).__name__}); using synthetic.")
        return _synthetic_mmlu(n)


def load_mmlu_fewshot(subjects: tuple[str, ...] = (), k: int = 5, seed: int = 0) -> list[MCExample]:
    """Few-shot exemplars (from the MMLU `dev` split, the canonical source)."""
    subjects = subjects or _DEFAULT_MMLU_SUBJECTS
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", subjects[0], split="dev")
        return [MCExample(ds[i]["question"], list(ds[i]["choices"]),
                          int(ds[i]["answer"]), subjects[0]) for i in range(min(k, len(ds)))]
    except Exception:
        return _synthetic_mmlu(k)


def _synthetic_mmlu(n: int) -> list[MCExample]:
    base = [
        MCExample("What is 2 + 2?", ["3", "4", "5", "22"], 1, "synthetic_math"),
        MCExample("Which planet is closest to the sun?",
                  ["Venus", "Earth", "Mercury", "Mars"], 2, "synthetic_science"),
        MCExample("Who wrote 'Hamlet'?",
                  ["Dickens", "Shakespeare", "Tolstoy", "Homer"], 1, "synthetic_lit"),
        MCExample("What is the capital of Japan?",
                  ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2, "synthetic_geo"),
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out[:n]


# =============================================================================
# GSM8K (generative + integer verifier)
# =============================================================================

def load_gsm8k(n: int, split: str = "test") -> list[dict]:
    """Return [{question, answer_field}] for up to n GSM8K problems."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [{"question": ds[i]["question"], "answer_field": ds[i]["answer"]}
                for i in range(min(n, len(ds)))]
    except Exception as e:
        print(f"[data] GSM8K real load failed ({type(e).__name__}); using synthetic.")
        return _synthetic_gsm8k(n)


def _synthetic_gsm8k(n: int) -> list[dict]:
    base = [
        {"question": "A box has 3 red and 4 blue balls. How many balls total?",
         "answer_field": "3 + 4 = 7\n#### 7"},
        {"question": "Tom has 5 apples and buys 8 more. How many does he have?",
         "answer_field": "5 + 8 = 13\n#### 13"},
        {"question": "A train travels 60 miles in 2 hours. Miles per hour?",
         "answer_field": "60 / 2 = 30\n#### 30"},
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out[:n]


# =============================================================================
# IFEval (instruction-following, verifiable constraints)
# =============================================================================

def load_ifeval(n: int) -> list[IFExample]:
    """A small built-in set of verifiable-constraint prompts.

    The real IFEval (google/IFEval) ships ~540 prompts with a richer checker
    registry; this built-in set demonstrates the same idea with the checkers
    in `benchmarks.CONSTRAINT_CHECKERS` so it runs with zero downloads.
    """
    items = [
        IFExample("List three primary colors as a bulleted list.",
                  [("exact_bullets", 3)]),
        IFExample("Reply with a JSON object containing your name.",
                  [("json", None)]),
        IFExample("Explain photosynthesis in at least 50 words.",
                  [("min_words", 50)]),
        IFExample("Summarize the water cycle in 20 words or fewer.",
                  [("max_words", 20)]),
        IFExample("Write a sentence about the ocean without using any commas.",
                  [("no_commas", None)]),
        IFExample("Describe a sunset. Include the word 'horizon'.",
                  [("keyword_present", "horizon")]),
        IFExample("Write a greeting in all capital letters.",
                  [("all_caps", None)]),
        IFExample("Give exactly two tips for studying, as bullets, in JSON-free text.",
                  [("exact_bullets", 2), ("keyword_absent", "{")]),
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(items[i % len(items)])
        i += 1
    return out[:n]


# =============================================================================
# Pairwise set (for LLM-as-judge)
# =============================================================================

def load_pairwise_set(path: str = "") -> list[dict]:
    """Load [{question, answer_a, answer_b}] pairs for the judge.

    `answer_a` is the BASELINE/reference; `answer_b` is the model under test
    (the win rate reported is B's). If `path` is empty, returns a built-in demo
    set crafted so the length/position biases are visible.
    """
    if path:
        rows = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    # Built-in: B is sometimes better, sometimes just longer (length-bias bait).
    return [
        {"question": "What is the capital of France?",
         "answer_a": "Paris.",
         "answer_b": "The capital of France is Paris, a city on the Seine."},
        {"question": "Is 17 prime?",
         "answer_a": "Yes, 17 is prime — it has no divisors other than 1 and 17.",
         "answer_b": "Yes."},
        {"question": "Give a tip for better sleep.",
         "answer_a": "Avoid screens before bed.",
         "answer_b": "Avoid screens for an hour before bed; the blue light "
                     "suppresses melatonin and delays sleep onset."},
        {"question": "What causes seasons?",
         "answer_a": "The Earth's distance from the sun changes.",   # WRONG
         "answer_b": "The tilt of Earth's axis relative to its orbit."},  # RIGHT
    ]
