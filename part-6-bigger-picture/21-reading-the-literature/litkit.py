"""
litkit.py — tools for reading the literature, for the Module 21 notebook.

Module 21 is about a skill, not an algorithm: how to read a paper in 30 minutes,
how to tell a genuine advance from noise, and how to find the few papers that
actually matter in a field that publishes hundreds a day. There's no model to
train here. What there *is*:

  1) A structured 30-minute triage         -> Triage / triage_questions()
  2) A signal-vs-noise red-flag scanner     -> SIGNALS / score_claims()
  3) A small, opinionated reading list as    -> READING_LIST + filters
     queryable data
  4) A toy citation graph + foundational-    -> CITATIONS / most_foundational()
     paper finder (how to discover the
     seminal works in an area)

Everything is pure Python (no torch, no network). It runs instantly on CPU. The
point is to turn "read more papers" — useless advice — into a repeatable
procedure you can actually run, and a dataset you can actually sort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. The 30-minute triage
# ---------------------------------------------------------------------------
#
# You cannot read every paper deeply. You CAN, in 30 minutes, decide whether a
# paper is worth a deep read — and extract its core claim either way. The
# procedure below is the three-pass method (abstract+figures -> claims+method
# -> reproduce-in-your-head), turned into explicit questions.

TRIAGE_QUESTIONS = [
    # pass 1 — 5 minutes: is this even relevant?
    ("claim", "What is the single concrete claim? (one sentence, no hedging)"),
    ("delta", "What does it do that the prior best did not? (the delta, specifically)"),
    # pass 2 — 15 minutes: is the claim supported?
    ("method", "What is the method, in two sentences a peer could re-implement from?"),
    ("baselines", "What are the baselines, and are they STRONG and current?"),
    ("evidence", "What experiment supports the claim? On what data, at what scale?"),
    ("ablations", "Which design choices are ablated? Which are asserted on faith?"),
    # pass 3 — 10 minutes: would it survive contact with reality?
    ("repro", "Is there code/data? Could YOU reproduce the headline result?"),
    ("limits", "What does the paper admit it does NOT do? What does it quietly avoid?"),
    ("transfer", "Does the result transfer to YOUR scale / setting, or only theirs?"),
]


@dataclass
class Triage:
    """A filled-in 30-minute triage of one paper."""

    title: str
    answers: dict = field(default_factory=dict)

    def unanswered(self) -> list[str]:
        """Triage questions you haven't answered yet — the gaps in your read."""
        return [k for k, _ in TRIAGE_QUESTIONS if not self.answers.get(k)]

    def completeness(self) -> float:
        """Fraction of triage questions answered. A read below ~0.7 is a skim,
        not a triage — you don't yet understand the paper well enough to judge it."""
        answered = sum(1 for k, _ in TRIAGE_QUESTIONS if self.answers.get(k))
        return answered / len(TRIAGE_QUESTIONS)

    def verdict(self) -> str:
        """A coarse recommendation from the triage state."""
        if self.completeness() < 0.7:
            return "incomplete — finish the triage before judging"
        # The three questions that most predict a paper's durability.
        load_bearing = ["baselines", "repro", "ablations"]
        weak = [k for k in load_bearing if not self.answers.get(k)]
        if weak:
            return f"read deeper — unresolved on: {', '.join(weak)}"
        return "triage complete — judge on the merits below"


def triage_questions() -> list[tuple[str, str]]:
    """Return the triage checklist (key, question) in reading order."""
    return list(TRIAGE_QUESTIONS)


# ---------------------------------------------------------------------------
# 2. Signal vs. noise
# ---------------------------------------------------------------------------
#
# Most papers are not fraudulent; they're just over-claimed. These are the
# recurring tells. Each SIGNAL is a yes/no question; a paper accrues "noise
# points" for each red flag and "signal points" for each green one. This is a
# heuristic, not a verdict — its value is forcing you to ask the questions.

@dataclass
class Signal:
    key: str
    question: str
    # +1 means "yes" is a GOOD sign; -1 means "yes" is a RED FLAG.
    polarity: int


SIGNALS = [
    Signal("strong_baselines", "Compared against the CURRENT strongest method, tuned fairly?", +1),
    Signal("released_code", "Is code and enough detail to reproduce released?", +1),
    Signal("error_bars", "Are results reported with error bars / multiple seeds?", +1),
    Signal("ablations", "Are the key design choices ablated?", +1),
    Signal("honest_limits", "Does it state its own limitations concretely?", +1),
    Signal("scales", "Is the effect shown at more than one scale?", +1),
    Signal("cherry_picked", "Does it lean on hand-picked qualitative examples?", -1),
    Signal("sota_on_one", "Does the headline rest on a single benchmark number?", -1),
    Signal("no_baseline_tuning", "Are baselines suspiciously weak / untuned?", -1),
    Signal("vague_method", "Is the method too vague to re-implement?", -1),
    Signal("benchmark_only", "Could the gain be contamination / overfitting to the eval?", -1),
]


def score_claims(answers: dict) -> dict:
    """Given {signal_key: bool}, return signal/noise tallies and a net score.

    Net > 0 leans trustworthy; net < 0 leans over-claimed. Unanswered signals
    are simply not counted — the score reflects only what you actually checked.
    """
    signal = noise = 0
    checked = []
    for s in SIGNALS:
        val = answers.get(s.key)
        if val is None:
            continue
        checked.append(s.key)
        if bool(val):
            if s.polarity > 0:
                signal += 1
            else:
                noise += 1
    return {
        "signal": signal,
        "noise": noise,
        "net": signal - noise,
        "checked": checked,
        "n_checked": len(checked),
    }


# ---------------------------------------------------------------------------
# 3. The reading list as data
# ---------------------------------------------------------------------------
#
# Opinionated, not exhaustive — the papers a graduate of THIS course should
# read to operate in the field. Tier: "foundational" (read in full),
# "important" (read carefully), "frontier" (track, read as needed). `course`
# links each to the module where the course used it, so you can re-derive the
# map of where the ideas live.

@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    authors: str
    year: int
    area: str          # transformers | scaling | pretraining | posttraining | eval | systems
    tier: str          # foundational | important | frontier
    course_module: Optional[int]
    why: str


READING_LIST = [
    Paper("attention", "Attention Is All You Need", "Vaswani et al.", 2017,
          "transformers", "foundational", 4,
          "The transformer. Everything else is a footnote to this."),
    Paper("gpt3", "Language Models are Few-Shot Learners", "Brown et al.", 2020,
          "scaling", "foundational", 0,
          "In-context learning emerges from scale. The paper that started the race."),
    Paper("kaplan", "Scaling Laws for Neural Language Models", "Kaplan et al.", 2020,
          "scaling", "foundational", 22,
          "C≈6ND and the power laws. The allocation conclusion was later corrected."),
    Paper("chinchilla", "Training Compute-Optimal Large Language Models", "Hoffmann et al.", 2022,
          "scaling", "foundational", 22,
          "~20 tokens/param. The correction to Kaplan. The most important scaling paper."),
    Paper("chinchilla_repro", "Chinchilla Scaling: A replication attempt", "Besiroglu et al.", 2024,
          "scaling", "important", 22,
          "Why you trust the shape and verify the constants. A model of adversarial reading."),
    Paper("instructgpt", "Training LMs to follow instructions with human feedback", "Ouyang et al.", 2022,
          "posttraining", "foundational", 14,
          "Defined the modern post-training stack. SFT then RLHF."),
    Paper("dpo", "Direct Preference Optimization", "Rafailov et al.", 2023,
          "posttraining", "foundational", 17,
          "Made RLHF practical: a classification loss, no reward model, no RL loop."),
    Paper("r1", "DeepSeek-R1", "DeepSeek-AI", 2025,
          "posttraining", "important", 18,
          "Reasoning via RL with verifiable rewards. GRPO, R1-Zero."),
    Paper("lora", "LoRA: Low-Rank Adaptation of Large Language Models", "Hu et al.", 2021,
          "posttraining", "important", 16,
          "Fine-tune 0.1% of the weights, keep ~all the quality. The default in practice."),
    Paper("qlora", "QLoRA: Efficient Finetuning of Quantized LLMs", "Dettmers et al.", 2023,
          "posttraining", "important", 16,
          "4-bit base + LoRA. Fine-tune a 65B model on one consumer GPU."),
    Paper("flashattn", "FlashAttention", "Dao et al.", 2022,
          "systems", "important", 4,
          "IO-aware exact attention. Why long context became affordable."),
    Paper("zero", "ZeRO: Memory Optimizations Toward Training Trillion-Param Models", "Rajbhandari et al.", 2019,
          "systems", "important", 10,
          "Shard optimizer/grad/params across data-parallel ranks. The basis of FSDP."),
    Paper("adamw", "Decoupled Weight Decay Regularization", "Loshchilov & Hutter", 2017,
          "pretraining", "important", 8,
          "AdamW. The optimizer you actually use, and why decoupling matters."),
    Paper("deepseekv3", "DeepSeek-V3 Technical Report", "DeepSeek-AI", 2024,
          "pretraining", "frontier", 6,
          "Aux-loss-free MoE balancing, FP8 training, a frontier recipe written down."),
    Paper("tulu3", "Tülu 3", "Lambert et al.", 2024,
          "posttraining", "frontier", 14,
          "The most thoroughly documented open end-to-end post-training recipe."),
    Paper("llama3", "The Llama 3 Herd of Models", "Meta", 2024,
          "scaling", "frontier", 22,
          "Over-training for cheap inference; honest systems engineering at scale."),
    Paper("sdft", "Self-Distillation Fine-Tuning", "Shenfeld et al.", 2026,
          "posttraining", "frontier", 19,
          "The model as its own teacher; fights catastrophic forgetting."),
    Paper("evalbars", "Adding Error Bars to Evals", "Miller (Anthropic)", 2024,
          "eval", "important", 20,
          "Most 1-2 point benchmark gaps are noise. How to report evals honestly."),
]


def filter_papers(area: Optional[str] = None, tier: Optional[str] = None,
                  max_year: Optional[int] = None) -> list[Paper]:
    """Query the reading list by area / tier / recency."""
    out = READING_LIST
    if area is not None:
        out = [p for p in out if p.area == area]
    if tier is not None:
        out = [p for p in out if p.tier == tier]
    if max_year is not None:
        out = [p for p in out if p.year <= max_year]
    return list(out)


def reading_order() -> list[Paper]:
    """The list sorted the way to actually read it: foundational first,
    then important, then frontier; within a tier, oldest first (ideas build)."""
    tier_rank = {"foundational": 0, "important": 1, "frontier": 2}
    return sorted(READING_LIST, key=lambda p: (tier_rank[p.tier], p.year))


def areas() -> list[str]:
    """Distinct areas in the reading list, in first-appearance order."""
    seen = []
    for p in READING_LIST:
        if p.area not in seen:
            seen.append(p.area)
    return seen


# ---------------------------------------------------------------------------
# 4. Finding the seminal papers: a toy citation graph
# ---------------------------------------------------------------------------
#
# When you enter a new area, the fastest way to find what matters is the
# citation graph: the papers everyone cites are the ones to read first. Here we
# encode a tiny dependency graph among the reading list (key -> the keys it
# builds directly on) and rank by how foundational each node is. "Foundational"
# = many things depend on it, directly or transitively.

CITATIONS = {
    # paper -> what it directly builds on
    "attention": [],
    "gpt3": ["attention"],
    "kaplan": ["gpt3", "attention"],
    "chinchilla": ["kaplan", "gpt3"],
    "chinchilla_repro": ["chinchilla"],
    "adamw": [],
    "flashattn": ["attention"],
    "zero": ["adamw"],
    "instructgpt": ["gpt3"],
    "dpo": ["instructgpt"],
    "r1": ["instructgpt", "dpo"],
    "lora": ["attention", "gpt3"],
    "qlora": ["lora", "zero"],
    "deepseekv3": ["attention", "zero", "flashattn"],
    "tulu3": ["instructgpt", "dpo"],
    "llama3": ["chinchilla", "deepseekv3"],
    "sdft": ["dpo", "instructgpt"],
    "evalbars": [],
}


def _descendants(graph: dict, node: str) -> set:
    """All nodes that depend on `node`, directly or transitively (reverse reach)."""
    # Build reverse adjacency once per call (tiny graph — fine).
    reverse: dict = {k: [] for k in graph}
    for child, parents in graph.items():
        for p in parents:
            reverse.setdefault(p, []).append(child)
    seen = set()
    stack = list(reverse.get(node, []))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(reverse.get(cur, []))
    return seen


def most_foundational(graph: dict = CITATIONS) -> list[tuple[str, int]]:
    """Rank papers by how many others (transitively) build on them.

    The papers at the top are where to start reading a new area: the more work
    depends on a paper, the more load-bearing it is. Ties broken alphabetically
    for determinism.
    """
    scored = [(k, len(_descendants(graph, k))) for k in graph]
    return sorted(scored, key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Triage checklist has", len(TRIAGE_QUESTIONS), "questions across 3 passes.")
    t = Triage("Some Paper", {"claim": "X beats Y", "delta": "first to do Z",
                              "method": "...", "baselines": "strong", "evidence": "...",
                              "ablations": "yes", "repro": "code released"})
    print("  completeness:", f"{t.completeness():.0%}", "| verdict:", t.verdict())
    print("  unanswered:", t.unanswered())

    print("\nSignal/noise on a cherry-picked, single-benchmark paper with no code:")
    print("  ", score_claims({"strong_baselines": False, "released_code": False,
                              "cherry_picked": True, "sota_on_one": True, "ablations": False}))

    print("\nReading list:", len(READING_LIST), "papers across areas:", areas())
    print("Foundational tier, read in order:")
    for p in filter_papers(tier="foundational"):
        print(f"   {p.year} {p.title}  [{p.area}]")

    print("\nMost foundational by citation reach:")
    for key, reach in most_foundational()[:5]:
        title = next(p.title for p in READING_LIST if p.key == key)
        print(f"   {reach:2d} depend on -> {title}")
