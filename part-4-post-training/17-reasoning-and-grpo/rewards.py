"""Verifiable rewards for GSM8K-style math reasoning.

GRPO sidesteps the reward-model training stage of classical RLHF by using
a *verifiable* reward — a rule-based function that compares the model's
completion against a known ground-truth answer. For math, this is just
"did the parsed integer match?".

We ship two components, summed by `compute_rewards`:

  format_reward(text)   — schema bonus.  1.0 if `<think>...</think>
                          <answer>...</answer>` is present, else 0.0.
                          Pedagogically: this is the "training wheels"
                          term that pulls the policy toward the expected
                          response shape early in training, before
                          accuracy signal can kick in.

  accuracy_reward(text, ground_truth) — task signal. 1.0 if the integer
                          parsed from the model's `<answer>` block equals
                          ground truth, else 0.0.

The combined reward is `w_format · format + w_accuracy · accuracy`. R1's
recipe leans heavily on accuracy with a small format bonus; if the policy
never emits the schema, accuracy is always 0 and format gives the policy
something to learn from. As accuracy starts firing, format becomes
near-saturated and the gradient is dominated by accuracy — which is what
you want, asymptotically.

These functions are *pure* (no tokenizer, no model, no torch): they take
strings and return floats. That keeps them trivial to test (see
`tests/test_rewards.py`) and ports cleanly to other RL frameworks.

Why no partial credit on the reasoning trace: getting the answer right
with a wrong reasoning trace is still right; getting it wrong with a
beautiful trace is still wrong. R1 famously showed this: emergent
reasoning came out of optimizing the outcome (answer correctness), not
out of any reward shaping on the trace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import RewardConfig


# =============================================================================
# Answer extraction
# =============================================================================

# Match the LAST <answer>...</answer> tag in the response (re-emitting the
# tag is a common failure mode; the LAST emission is the policy's "final"
# answer in practice). Non-greedy to avoid swallowing nested content.
_ANSWER_TAG_RE = re.compile(r"<answer>([\s\S]*?)</answer>")

# Match any int (with optional sign) inside a (possibly cluttered) string.
# Handles commas in numbers ("1,234"), trailing units, .0/.00 suffixes.
_INT_IN_STRING_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the contents of the LAST `<answer>...</answer>` block.

    Returns the raw string between the tags (whitespace-stripped) or None
    if no `<answer>` tag is present. The caller decides how to parse the
    inner content — for GSM8K we want an integer (see `parse_int`).

    Why the LAST tag: the policy sometimes emits `<answer>X</answer>`
    mid-trace then revises to `<answer>Y</answer>` at the end. Y is the
    intended final answer; reading the first match rewards the wrong
    behavior (early commits). This is the convention TRL's gsm8k verifier
    uses too.
    """
    matches = _ANSWER_TAG_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_int(text: str) -> int | None:
    """Parse the FIRST integer-like substring out of `text`.

    Handles:
      "18"            -> 18
      "  18  "        -> 18
      "$1,234"        -> 1234        (strips commas)
      "18.0"          -> 18          (strips trailing decimal)
      "The answer is 18 eggs" -> 18
      "no number here"        -> None

    Returns None if nothing parseable is found.
    """
    if not text:
        return None
    m = _INT_IN_STRING_RE.search(text)
    if m is None:
        return None
    raw = m.group(0).replace(",", "")
    # Allow `.00` suffix but not non-integer floats.
    if "." in raw:
        whole, frac = raw.split(".", 1)
        if any(c != "0" for c in frac):
            # Non-integer float — try int(float(raw)) only if it's exact, else None.
            try:
                f = float(raw)
                if f == int(f):
                    return int(f)
                return None
            except ValueError:
                return None
        raw = whole
    try:
        return int(raw)
    except ValueError:
        return None


# =============================================================================
# Individual reward components
# =============================================================================

def format_reward(text: str, pattern: str) -> float:
    """1.0 if `text` matches the schema regex anywhere, else 0.0.

    Default pattern is the R1 schema: `<think>...</think>` followed by
    `<answer>...</answer>`. Both blocks must have non-empty contents
    (the default regex enforces this via `[\\s\\S]*?` inside the tags
    which DOES match empty strings — accept that simplification; the
    accuracy reward is what actually drives learning).
    """
    return 1.0 if re.search(pattern, text) else 0.0


def accuracy_reward(text: str, ground_truth: int | str) -> float:
    """1.0 if the parsed answer equals `ground_truth`, else 0.0.

    `ground_truth` can be an int or a string (we coerce). The model's
    answer is pulled via `extract_answer` then parsed via `parse_int`;
    a missing tag or unparseable content scores 0.
    """
    if isinstance(ground_truth, str):
        gt = parse_int(ground_truth)
    else:
        gt = int(ground_truth)
    if gt is None:
        return 0.0

    answer_text = extract_answer(text)
    if answer_text is None:
        return 0.0
    answer_int = parse_int(answer_text)
    if answer_int is None:
        return 0.0
    return 1.0 if answer_int == gt else 0.0


# =============================================================================
# Combined reward for a group of completions
# =============================================================================

@dataclass
class RewardBreakdown:
    """Per-completion reward with components broken out for logging."""
    total: float
    format: float
    accuracy: float


def compute_rewards(
    texts: list[str],
    ground_truths: list[int | str],
    reward_cfg: RewardConfig,
) -> list[RewardBreakdown]:
    """Compute the GRPO reward for a list of (completion, ground_truth) pairs.

    Returns a list of `RewardBreakdown` — one per completion — with the
    weighted-sum total as the first field and the component values as the
    next two. The training loop only consumes `.total`; the components are
    logged so you can see what's actually driving learning at each step.

    Length contract: `len(texts) == len(ground_truths)`. For G completions
    per prompt × P prompts, the caller flattens to G·P then unflattens
    after this returns (see `rollout.py`).
    """
    if len(texts) != len(ground_truths):
        raise ValueError(
            f"len(texts) ({len(texts)}) != len(ground_truths) ({len(ground_truths)})"
        )
    out: list[RewardBreakdown] = []
    for text, gt in zip(texts, ground_truths):
        rf = format_reward(text, reward_cfg.format_pattern)
        ra = accuracy_reward(text, gt)
        total = reward_cfg.w_format * rf + reward_cfg.w_accuracy * ra
        out.append(RewardBreakdown(total=total, format=rf, accuracy=ra))
    return out


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    cfg = RewardConfig()
    print("--- rewards.py smoke test ---")

    good = "<think>2+2=4</think><answer>4</answer>"
    bad_format = "The answer is 4"
    bad_answer = "<think>maths</think><answer>5</answer>"
    revised = "<answer>3</answer> hmm wait <answer>4</answer>"

    cases = [
        ("schema OK + correct",  good,        4),
        ("no schema + correct",  bad_format,  4),
        ("schema OK + wrong",    bad_answer,  4),
        ("revised final",        revised,     4),
        ("$ + commas",           "<think>x</think><answer>$1,234</answer>", 1234),
    ]
    print(f"\n{'case':<24} {'format':>7} {'acc':>5} {'total':>7}")
    for label, txt, gt in cases:
        br = compute_rewards([txt], [gt], cfg)[0]
        print(f"  {label:<22} {br.format:>7.1f} {br.accuracy:>5.1f} {br.total:>7.3f}")
