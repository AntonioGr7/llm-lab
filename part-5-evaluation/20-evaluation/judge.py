"""LLM-as-judge — the workhorse of modern eval, and the easiest to fool.

Once outputs are open-ended (chat, summaries, code explanations) there is no
regex that scores them. So labs use a strong model as the grader. This is how
**MT-Bench**, **AlpacaEval**, **Arena-Hard**, and most internal "is the new
checkpoint better?" evals work. Done right it correlates ~0.8+ with human
preference at a fraction of the cost. Done naively it measures the judge's
biases instead of your model's quality.

The three biases this module is built to fight:

  1. **Position bias** — judges favor whichever answer is shown *first* (or,
     for some models, second). Fix: run each pair BOTH orders and only count a
     win if it survives the swap. Disagreement across orders => the judge is
     guessing => score it a tie. We surface the position-bias rate as a
     first-class diagnostic.

  2. **Verbosity / length bias** — judges reward longer answers regardless of
     quality. This is why raw AlpacaEval was gameable by padding; **AlpacaEval
     2.0 length-controlled** (Dubois et al. 2024) regresses length out. We
     report the length–win correlation so you can see the bias in your data.

  3. **Self-preference bias** — a model judging its own family rates it higher
     (Panickssery et al. 2024). Mitigation: judge with a *different* model
     family than the one you're evaluating, and calibrate against human labels.

The decisive practice: **calibrate the judge against human labels** on a small
gold set (`agreement` / Cohen's κ) before trusting it on thousands of pairs. A
judge with κ < ~0.4 vs humans is not measuring what you think.

The judging LOGIC (position-swap resolution, win-rate, length bias, κ) is pure
and offline-testable via `DummyJudge`. `LocalJudge` runs a real HF model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence


# =============================================================================
# The judge backend (pluggable)
# =============================================================================

class Judge(Protocol):
    """Anything that maps a judging prompt -> raw text response."""
    def __call__(self, prompt: str) -> str: ...


class DummyJudge:
    """Deterministic offline judge for tests + the $0 notebook path.

    Picks the winner by a transparent rule so behavior is fully predictable:
    by default it prefers the LONGER answer (which lets us *demonstrate* length
    bias on purpose), and it has a `prefer` keyword override. It also exhibits
    a configurable `position_bias`: with prob given, it ignores content and
    just picks by position — so the swap-resolution code has something to catch.
    """
    def __init__(self, rule: str = "longer", position_bias: float = 0.0,
                 prefer_keyword: Optional[str] = None):
        self.rule = rule
        self.position_bias = position_bias
        self.prefer_keyword = prefer_keyword
        self._call_count = 0

    def __call__(self, prompt: str) -> str:
        self._call_count += 1
        a, b = _extract_ab_from_prompt(prompt)
        # Deterministic "position bias": every Nth call just picks A.
        if self.position_bias > 0 and (self._call_count % max(1, round(1 / self.position_bias)) == 0):
            return json.dumps({"winner": "A", "rationale": "position-biased pick"})
        if self.prefer_keyword:
            a_has = self.prefer_keyword in a
            b_has = self.prefer_keyword in b
            if a_has != b_has:
                win = "A" if a_has else "B"
                return json.dumps({"winner": win, "rationale": "keyword present"})
        if self.rule == "longer":
            win = "A" if len(a) > len(b) else ("B" if len(b) > len(a) else "tie")
        else:  # "shorter"
            win = "A" if len(a) < len(b) else ("B" if len(b) < len(a) else "tie")
        return json.dumps({"winner": win, "rationale": f"{self.rule} answer"})


class LocalJudge:
    """A real HF causal LM as the judge. Loaded lazily so importing this module
    never touches GPU/network. Use a model OUTSIDE the family you're evaluating
    (self-preference bias). Greedy decode for reproducibility."""
    def __init__(self, model_name: str = "Qwen/Qwen3-4B", device: Optional[str] = None,
                 max_new_tokens: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device).eval()

    def __call__(self, prompt: str) -> str:
        import torch
        self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = self._tok.apply_chat_template(messages, add_generation_prompt=True,
                                                return_tensors="pt", enable_thinking=False)
        except TypeError:
            ids = self._tok.apply_chat_template(messages, add_generation_prompt=True,
                                                return_tensors="pt")
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=self.max_new_tokens,
                                       do_sample=False, pad_token_id=self._tok.pad_token_id)
        return self._tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# =============================================================================
# Prompt templates + parsing
# =============================================================================

PAIRWISE_TEMPLATE = """You are an impartial judge evaluating two AI assistant responses to a user question. Pick the response that is more helpful, correct, and complete. Do not let the order, length, or style sway you — judge substance.

[User question]
{question}

[Assistant A]
{answer_a}

[Assistant B]
{answer_b}

Respond with ONLY a JSON object: {{"winner": "A" | "B" | "tie", "rationale": "<one sentence>"}}."""

POINTWISE_TEMPLATE = """You are grading a single AI assistant response against a rubric. Score from 1 (terrible) to 10 (excellent).

[User question]
{question}

[Response]
{answer}

[Rubric]
{rubric}

Respond with ONLY a JSON object: {{"score": <int 1-10>, "rationale": "<one sentence>"}}."""

# Markers used to slice A/B back out of a rendered prompt (for DummyJudge/tests).
_A_MARK, _B_MARK = "[Assistant A]", "[Assistant B]"


def _extract_ab_from_prompt(prompt: str) -> tuple[str, str]:
    """Recover the A and B answer text from a rendered pairwise prompt."""
    a_start = prompt.find(_A_MARK)
    b_start = prompt.find(_B_MARK)
    if a_start == -1 or b_start == -1:
        return prompt, ""
    a = prompt[a_start + len(_A_MARK):b_start].strip()
    b_end = prompt.find("\n\nRespond with", b_start)
    b = prompt[b_start + len(_B_MARK):(b_end if b_end != -1 else len(prompt))].strip()
    return a, b


def _parse_json_obj(text: str) -> Optional[dict]:
    """Tolerant JSON extraction: strips markdown fences, finds the first {...}."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_winner(raw: str) -> str:
    """Pull a winner ∈ {A, B, tie} from a judge response. Defaults to 'tie'
    when unparseable (an unparseable verdict carries no information)."""
    obj = _parse_json_obj(raw)
    if obj and "winner" in obj:
        w = str(obj["winner"]).strip().upper()
        if w in ("A", "B"):
            return w
        return "tie"
    # Fallback: scan for a bare letter.
    m = re.search(r"\b(A|B|tie)\b", raw, re.IGNORECASE)
    if m:
        v = m.group(1).upper()
        return v if v in ("A", "B") else "tie"
    return "tie"


def parse_score(raw: str) -> Optional[int]:
    """Pull an integer 1-10 from a pointwise judge response."""
    obj = _parse_json_obj(raw)
    if obj and "score" in obj:
        try:
            return max(1, min(10, int(obj["score"])))
        except (ValueError, TypeError):
            return None
    m = re.search(r"\b([1-9]|10)\b", raw)
    return int(m.group(1)) if m else None


# =============================================================================
# Pairwise judging with position-swap debiasing
# =============================================================================

@dataclass
class PairwiseVerdict:
    """Outcome of a single position-swapped pairwise comparison.

    `winner` ∈ {"A", "B", "tie"} where A/B refer to the ORIGINAL (un-swapped)
    pair. `position_bias` is True when the two orders disagreed (the judge
    flipped when we flipped the answers) — those are counted as ties because
    the judge was responding to position, not substance.
    """
    winner: str
    position_bias: bool
    raw_forward: str = ""
    raw_swapped: str = ""


def pairwise_judge(judge: Judge, question: str, answer_a: str, answer_b: str,
                   swap: bool = True) -> PairwiseVerdict:
    """Judge A vs B, optionally running BOTH orders and resolving disagreement.

    Forward order: A shown first. Swapped order: B shown first (so the judge's
    "A" now refers to our B). A consistent winner survives both orders. If the
    judge picks the first-shown answer both times, the orders disagree -> we
    record position_bias and return a tie. This single trick removes most of
    the position-bias variance for free.
    """
    fwd_raw = judge(PAIRWISE_TEMPLATE.format(question=question, answer_a=answer_a,
                                             answer_b=answer_b))
    fwd = parse_winner(fwd_raw)
    if not swap:
        return PairwiseVerdict(winner=fwd, position_bias=False, raw_forward=fwd_raw)

    # Swap: B is now shown as "A". Translate the judge's verdict back to our labels.
    swp_raw = judge(PAIRWISE_TEMPLATE.format(question=question, answer_a=answer_b,
                                             answer_b=answer_a))
    swp_seen = parse_winner(swp_raw)
    swp = {"A": "B", "B": "A", "tie": "tie"}[swp_seen]  # back to original labels

    if fwd == swp:
        winner = fwd
        bias = False
    elif "tie" in (fwd, swp):
        # One order called it; the other tied. Take the decisive one but note it.
        winner = fwd if fwd != "tie" else swp
        bias = True
    else:
        # Hard disagreement (fwd=A, swp=B or vice versa) => pure position bias.
        winner = "tie"
        bias = True
    return PairwiseVerdict(winner=winner, position_bias=bias,
                           raw_forward=fwd_raw, raw_swapped=swp_raw)


@dataclass
class WinRateResult:
    """Aggregate of many pairwise verdicts. `win_rate` counts ties as 0.5 (the
    AlpacaEval/Arena convention) so a model that ties everything scores 0.5."""
    n: int
    wins_b: int            # B is the model under test by convention
    wins_a: int            # A is the baseline/reference
    ties: int
    position_bias_rate: float
    length_win_corr: float   # corr between "B won" and (len_b - len_a); >0 = length bias

    @property
    def win_rate(self) -> float:
        """Win rate of B (the model under test) vs A (the reference)."""
        return (self.wins_b + 0.5 * self.ties) / max(self.n, 1)


def aggregate_win_rate(verdicts: Sequence[PairwiseVerdict],
                       len_a: Optional[Sequence[int]] = None,
                       len_b: Optional[Sequence[int]] = None) -> WinRateResult:
    """Aggregate verdicts into a win rate + the bias diagnostics.

    Pass per-item answer lengths (in chars or tokens) to get the length–win
    correlation: a strongly positive value means your win rate is partly a
    length artifact and you should length-control (see README §LLM-as-judge).
    """
    n = len(verdicts)
    wins_b = sum(1 for v in verdicts if v.winner == "B")
    wins_a = sum(1 for v in verdicts if v.winner == "A")
    ties = sum(1 for v in verdicts if v.winner == "tie")
    pos_bias = sum(1 for v in verdicts if v.position_bias) / max(n, 1)

    corr = 0.0
    if len_a is not None and len_b is not None and n >= 2:
        b_won = [1.0 if v.winner == "B" else (0.5 if v.winner == "tie" else 0.0)
                 for v in verdicts]
        len_diff = [lb - la for la, lb in zip(len_a, len_b)]
        corr = _pearson(b_won, len_diff)
    return WinRateResult(n=n, wins_b=wins_b, wins_a=wins_a, ties=ties,
                         position_bias_rate=pos_bias, length_win_corr=corr)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; 0.0 if either series is constant."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx ** 0.5 * syy ** 0.5)


# =============================================================================
# Calibration against human labels
# =============================================================================

@dataclass
class Agreement:
    raw_agreement: float    # fraction of items where judge == human
    cohen_kappa: float      # chance-corrected agreement
    n: int

    @property
    def trustworthy(self) -> bool:
        """Rule of thumb: κ >= 0.4 (moderate) before trusting the judge at scale."""
        return self.cohen_kappa >= 0.4


def agreement(judge_labels: Sequence[str], human_labels: Sequence[str]) -> Agreement:
    """Cohen's κ between judge verdicts and human gold labels.

    Raw agreement overstates quality when the label distribution is skewed
    (if 80% of items are "B wins", a judge that always says "B" gets 80% raw
    agreement but is useless). κ corrects for agreement expected by chance.
    Run this on a small human-labeled gold set BEFORE deploying the judge.
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge and human label lists must be the same length")
    n = len(judge_labels)
    if n == 0:
        return Agreement(float("nan"), float("nan"), 0)
    po = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    labels = set(judge_labels) | set(human_labels)
    pe = 0.0
    for lab in labels:
        pj = sum(1 for j in judge_labels if j == lab) / n
        ph = sum(1 for h in human_labels if h == lab) / n
        pe += pj * ph
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)
    return Agreement(raw_agreement=po, cohen_kappa=kappa, n=n)


if __name__ == "__main__":
    # A judge that prefers longer answers, run with swap. The "good" short
    # answer loses to a padded long one -> length bias visible in the corr.
    judge = DummyJudge(rule="longer")
    pairs = [
        ("What is 2+2?", "4", "The answer is 4. " * 20),
        ("Capital of France?", "Paris is the capital of France.", "Paris."),
        ("Define entropy.", "Disorder.", "Entropy is a measure of disorder " * 10),
    ]
    verdicts = [pairwise_judge(judge, q, a, b) for q, a, b in pairs]
    la = [len(a) for _, a, _ in pairs]
    lb = [len(b) for _, _, b in pairs]
    res = aggregate_win_rate(verdicts, la, lb)
    print(f"win_rate(B)={res.win_rate:.2f}  pos_bias={res.position_bias_rate:.2f}  "
          f"length_win_corr={res.length_win_corr:+.2f}")
    # Calibration demo
    jl = ["A", "B", "B", "tie", "A"]
    hl = ["A", "B", "A", "tie", "A"]
    ag = agreement(jl, hl)
    print(f"agreement raw={ag.raw_agreement:.2f}  kappa={ag.cohen_kappa:.2f}  "
          f"trustworthy={ag.trustworthy}")
