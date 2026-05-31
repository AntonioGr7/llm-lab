"""The three benchmark families, and where each one lies to you.

This module implements the *scoring logic* for the three shapes of automatic
benchmark — multiple-choice, generative-with-verifier, and
instruction-following — separately from the model that produces the
predictions (that lives in `model.py`). Keeping scoring pure makes the
failure modes testable and the lessons explicit:

  - **Multiple-choice (MMLU, ARC, HellaSwag, GPQA)** is scored by *likelihood*,
    not generation: rank the answer options by the model's log-prob and pick
    the argmax. The catch the leaderboards rarely mention: the *normalization*
    you choose (raw sum / per-token / per-byte) changes the accuracy by several
    points, and so does the *prompt format* (lettered "A/B/C/D" vs cloze
    completion). Same model, same data, different harness => different number.
    This is why the Open LLM Leaderboard pins an exact harness + format.

  - **Generative + verifier (GSM8K, MATH, HumanEval)** generates free text and
    checks it with a rule (parse the integer, run the unit tests, exact-match
    the normalized string). Honest where a verifier exists; the trap is answer
    *extraction* — a correct answer phrased unexpectedly scores 0, inflating
    the apparent gap between models that follow the format and ones that don't.

  - **Instruction-following (IFEval, Zhou et al. 2023)** sidesteps judges with
    *programmatically verifiable* constraints: "reply in JSON", "exactly 3
    bullets", "no commas", "at least 200 words". You can check these with code,
    so the score is objective — a rare island of trustworthy automatic eval.

All scoring here is PURE (no torch); the MC scorer consumes pre-computed
per-option log-probs so it's testable with hand-written numbers.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


# =============================================================================
# 1. Multiple-choice: likelihood scoring + the normalization/format knobs
# =============================================================================

@dataclass
class MCExample:
    question: str
    choices: list[str]
    answer_index: int       # ground-truth index into `choices`
    subject: str = ""


def build_mc_prompt(ex: MCExample, style: str = "letters", n_shot_block: str = "") -> tuple[str, list[str]]:
    """Render an MC question + return the per-option CONTINUATION strings.

    Two styles, to demonstrate format sensitivity:

      - "letters": classic MMLU format. Prompt lists "A. <c0>  B. <c1> ..."
        and the continuations are the letters " A" / " B" / .... The model is
        scored on the log-prob of each letter. Sensitive to whether the model
        has learned the convention that the letter follows "Answer:".

      - "cloze": no letters. The prompt is just the question stem and each
        continuation is the full answer TEXT. Scores the model on how likely
        each completion is. Removes the letter-convention confound but
        introduces a length confound (longer answers have lower raw logprob)
        — which is exactly what the normalization knob below is for.

    Returns (prompt, continuations). The model scores logprob(continuation |
    prompt) for each, and the scorer picks the argmax.
    """
    letters = [chr(ord("A") + i) for i in range(len(ex.choices))]
    if style == "letters":
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, ex.choices))
        prompt = f"{n_shot_block}{ex.question}\n{body}\nAnswer:"
        continuations = [f" {l}" for l in letters]
    elif style == "cloze":
        prompt = f"{n_shot_block}{ex.question}\nAnswer:"
        continuations = [f" {c}" for c in ex.choices]
    else:
        raise ValueError(f"unknown MC style: {style!r}")
    return prompt, continuations


def score_mc(option_logprobs: Sequence[float], option_token_counts: Sequence[int],
             option_byte_counts: Sequence[int], norm: str = "raw") -> int:
    """Pick the predicted option index from per-option log-probs.

    `option_logprobs[i]` is the SUM of token log-probs of continuation i.
    Three normalizations, each a defensible and DIFFERENT answer:

      - "raw"   : argmax of the summed logprob. Biased toward SHORT options
                  (fewer negative terms). Fine for "letters" (all length 1),
                  wrong for "cloze" (favors short answers).
      - "token" : divide by token count (per-token average logprob / perplexity).
                  The HellaSwag-style "acc_norm". Removes most length bias.
      - "byte"  : divide by UTF-8 byte length. Tokenizer-independent — lets you
                  compare models with different tokenizers fairly (the Pythia /
                  lm-eval "bits-per-byte" idea).

    Returns the argmax index under the chosen normalization.
    """
    lp = list(option_logprobs)
    if norm == "raw":
        scores = lp
    elif norm == "token":
        scores = [l / max(t, 1) for l, t in zip(lp, option_token_counts)]
    elif norm == "byte":
        scores = [l / max(b, 1) for l, b in zip(lp, option_byte_counts)]
    else:
        raise ValueError(f"unknown norm: {norm!r}")
    # argmax with first-seen tie-break
    best_i, best_s = 0, scores[0]
    for i, s in enumerate(scores):
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def mc_accuracy(predictions: Sequence[int], examples: Sequence[MCExample]) -> list[float]:
    """Per-item correctness flags (1.0/0.0) — feed to metrics.bootstrap_ci."""
    return [1.0 if p == ex.answer_index else 0.0 for p, ex in zip(predictions, examples)]


# =============================================================================
# 2. Generative + verifier: answer extraction is where the score leaks
# =============================================================================

_INT_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_GSM8K_HASH_RE = re.compile(r"####\s*(-?[\d,]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def parse_gsm8k_ground_truth(answer_field: str) -> Optional[int]:
    """GSM8K stores the gold answer after '#### '. Reused from Module 18."""
    m = _GSM8K_HASH_RE.search(answer_field)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def extract_final_int(text: str) -> Optional[int]:
    """Best-effort: prefer \\boxed{}, then the last integer in the text.

    Answer extraction is a LOSSY, model-favoring step. A model that emits a
    clean "#### 42" or "\\boxed{42}" is easy to score; a verbose model that
    buries "...so the answer is forty-two" scores 0 here. This asymmetry can
    *manufacture* a benchmark gap between two equally-capable models, one of
    which just formats better. The fix labs use: a fixed few-shot prompt that
    pins the output format, so extraction is reliable for everyone.
    """
    boxed = _BOXED_RE.findall(text)
    if boxed:
        m = _INT_RE.search(boxed[-1])
        if m:
            return _to_int(m.group(0))
    ints = _INT_RE.findall(text)
    return _to_int(ints[-1]) if ints else None


def _to_int(raw: str) -> Optional[int]:
    raw = raw.replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        if any(c != "0" for c in frac):
            return None
        raw = whole
    try:
        return int(raw)
    except ValueError:
        return None


def normalize_text_answer(s: str) -> str:
    """SQuAD-style normalization for short-answer EM/F1: lowercase, strip
    punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_text_answer(prediction) == normalize_text_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 (SQuAD) — partial credit for short-answer QA."""
    p = normalize_text_answer(prediction).split()
    g = normalize_text_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(g)
    return 2 * prec * rec / (prec + rec)


# =============================================================================
# 3. Instruction-following: programmatically verifiable constraints (IFEval)
# =============================================================================

# Each checker takes (response_text, arg) -> bool. Pure, no model needed. This
# is the trustworthy island: the score is a fact about the string, not an opinion.

def _check_json(resp: str, _arg=None) -> bool:
    resp = resp.strip()
    resp = re.sub(r"^```(?:json)?|```$", "", resp).strip()
    try:
        json.loads(resp)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_exact_bullets(resp: str, n: int) -> bool:
    bullets = [ln for ln in resp.splitlines() if re.match(r"\s*[-*•]\s+\S", ln)]
    return len(bullets) == n


def _check_min_words(resp: str, n: int) -> bool:
    return len(re.findall(r"\b\w+\b", resp)) >= n


def _check_max_words(resp: str, n: int) -> bool:
    return len(re.findall(r"\b\w+\b", resp)) <= n


def _check_no_commas(resp: str, _arg=None) -> bool:
    return "," not in resp


def _check_keyword_present(resp: str, kw: str) -> bool:
    return kw.lower() in resp.lower()


def _check_keyword_absent(resp: str, kw: str) -> bool:
    return kw.lower() not in resp.lower()


def _check_ends_with(resp: str, suffix: str) -> bool:
    return resp.rstrip().endswith(suffix)


def _check_all_caps(resp: str, _arg=None) -> bool:
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


CONSTRAINT_CHECKERS: dict[str, Callable[[str, object], bool]] = {
    "json": _check_json,
    "exact_bullets": _check_exact_bullets,
    "min_words": _check_min_words,
    "max_words": _check_max_words,
    "no_commas": _check_no_commas,
    "keyword_present": _check_keyword_present,
    "keyword_absent": _check_keyword_absent,
    "ends_with": _check_ends_with,
    "all_caps": _check_all_caps,
}


@dataclass
class IFExample:
    prompt: str
    constraints: list[tuple[str, object]]   # [(checker_name, arg), ...]


def check_constraints(response: str, constraints: Sequence[tuple[str, object]]) -> dict:
    """Run every constraint on a response. IFEval reports two granularities:

      - strict (prompt-level): ALL constraints satisfied for this prompt?
      - loose (instruction-level): fraction of individual constraints met.

    Both matter — prompt-level is the user-visible "did it obey", while the
    instruction-level rate is a smoother training signal.
    """
    results = []
    for name, arg in constraints:
        checker = CONSTRAINT_CHECKERS[name]
        results.append(bool(checker(response, arg)))
    return {
        "per_constraint": results,
        "prompt_level": all(results) if results else False,
        "instruction_level": sum(results) / len(results) if results else 0.0,
    }


if __name__ == "__main__":
    # MC: same logprobs, three norms can disagree on the cloze format.
    # Option 0 is short ("Yes"), option 1 is long but slightly higher per-token.
    lp = [-2.0, -3.6]          # raw favors option 0
    toks = [1, 3]
    byts = [3, 12]
    for norm in ("raw", "token", "byte"):
        print(f"  MC argmax under {norm:5s}: option {score_mc(lp, toks, byts, norm)}")
    # Generative extraction
    print("extract:", extract_final_int("steps... so the answer is \\boxed{42}."))
    print("EM:", exact_match("The Paris.", "paris"), "F1:",
          round(token_f1("the cat sat", "a cat sat down"), 2))
    # IFEval
    r = check_constraints('{"a": 1}', [("json", None), ("max_words", 5)])
    print("IF strict:", r["prompt_level"], "loose:", r["instruction_level"])
