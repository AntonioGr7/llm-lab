"""The composed config for Module 19 — Evaluation.

Unlike the training modules, there's no optimizer or schedule here — evaluation
is forward-only. The config instead pins the things that, if left implicit,
make eval numbers irreproducible:

  - `model` — which checkpoint to score (a name or a DCP dir), and the dtype.
  - `benchmarks` — which suites to run, how many items each, the few-shot
    count, and the MC scoring knobs (`style`, `norm`) that the README shows
    can swing accuracy several points. PIN THESE — they are part of the score.
  - `generation` — decode settings for generative benchmarks. Greedy for
    deterministic single-shot; sampling + `n_samples` for pass@k / avg@k.
  - `judge` — the LLM-as-judge backend + whether to position-swap.
  - `contamination` — the n-gram size and flag threshold.

The same single-source-of-truth + YAML loader + dotted-CLI-override machinery
as the training modules, so `--benchmarks.mc_norm=token` works on the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    """Which model to evaluate.

    `name`: HF model id (the BASE / instruct model) OR the base to load a
        checkpoint onto.
    `checkpoint`: optional DCP checkpoint dir produced by Modules 13-18. If
        set, weights are loaded onto `name`'s architecture. If empty, `name`
        is evaluated as-is (the base/published model).
    `dtype`: bf16 on GPU; fp32 for CPU smoke. Eval is forward-only so bf16 is
        almost always fine and halves the memory.
    """
    name: str = "Qwen/Qwen3-1.7B"
    checkpoint: str = ""
    dtype: Literal["bf16", "fp32"] = "bf16"
    max_seq: int = 4096


# =============================================================================
# Benchmarks
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Which automatic benchmarks to run + the knobs that define the score.

    `suites`: subset of {"mmlu", "gsm8k", "ifeval"} to run.
    `n_per_suite`: cap on items per suite (the full sets are large; for a
        course demo a few hundred is plenty — but note the CI widens, see
        metrics.min_n_for_halfwidth).
    `n_shot`: few-shot examples prepended (MMLU canonical is 5-shot; GSM8K
        8-shot CoT). Few-shot count is part of the score — pin it.
    `mc_style`: "letters" (A/B/C/D, canonical MMLU) or "cloze" (answer-text
        continuation). Demonstrates format sensitivity.
    `mc_norm`: "raw" | "token" | "byte" — the likelihood normalization. Also
        part of the score.
    """
    suites: tuple[str, ...] = ("mmlu", "gsm8k", "ifeval")
    n_per_suite: int = 200
    n_shot: int = 0
    mc_style: Literal["letters", "cloze"] = "letters"
    mc_norm: Literal["raw", "token", "byte"] = "token"
    mmlu_subjects: tuple[str, ...] = ()      # () = a default spread
    seed: int = 0


# =============================================================================
# Generation (for generative benchmarks)
# =============================================================================

@dataclass
class GenerationConfig:
    """Decode settings for generative suites (GSM8K, IFEval).

    `greedy`: deterministic single decode — the default for a headline number.
    `n_samples`: >1 with `greedy=False` enables pass@k / avg@k / maj@k
        (sample n_samples completions per item). Set 0/1 for single-shot.
    """
    greedy: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    n_samples: int = 1                       # >1 => sampling-based metrics
    system_prompt: str = ""


# =============================================================================
# LLM-as-judge
# =============================================================================

@dataclass
class JudgeConfig:
    """Pairwise / pointwise LLM-as-judge settings.

    `backend`: "local" (HF model via judge.LocalJudge) or "dummy" (the
        deterministic offline judge — for the $0 path and tests).
    `model_name`: judge model. Use a DIFFERENT family than the model under
        test to avoid self-preference bias.
    `swap`: run each pair in both orders and resolve disagreement as a tie
        (position-bias control). Always leave on for real evals.
    `pairwise_set`: path to a JSONL of {question, answer_a, answer_b} pairs,
        or "" to use the small built-in demo set.
    """
    backend: Literal["local", "dummy"] = "dummy"
    model_name: str = "Qwen/Qwen3-4B"
    swap: bool = True
    max_new_tokens: int = 512
    pairwise_set: str = ""


# =============================================================================
# Contamination
# =============================================================================

@dataclass
class ContaminationConfig:
    """N-gram contamination scan settings.

    `corpus_glob`: glob of text/JSONL files forming the (sampled) training
        corpus to check against. "" disables the scan.
    `ngram`: n-gram size (13 = GPT-3 convention).
    `threshold`: overlap fraction at/above which an item is flagged.
    """
    corpus_glob: str = ""
    ngram: int = 13
    threshold: float = 0.5


# =============================================================================
# Run-level
# =============================================================================

@dataclass
class RunConfig:
    out: str = "./results/scorecard.json"
    confidence: float = 0.95
    bootstrap_resamples: int = 10_000
    seed: int = 0


# =============================================================================
# The full config
# =============================================================================

@dataclass
class EvalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def sync(self) -> None:
        valid = {"mmlu", "gsm8k", "ifeval"}
        bad = set(self.benchmarks.suites) - valid
        if bad:
            raise ValueError(f"unknown benchmark suite(s): {bad}; valid: {valid}")
        if self.benchmarks.n_shot < 0:
            raise ValueError("benchmarks.n_shot must be >= 0")
        if self.generation.n_samples >= 2 and self.generation.greedy:
            raise ValueError(
                "generation.n_samples >= 2 needs greedy=False (pass@k requires sampling)")
        if not (0 < self.run.confidence < 1):
            raise ValueError("run.confidence must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + dotted CLI overrides (same pattern as the training modules)
# =============================================================================

def load_yaml(path: str) -> EvalConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _from_dict(raw)
    cfg.sync()
    return cfg


def _from_dict(d: dict) -> EvalConfig:
    cfg = EvalConfig()
    for section_name, section_cfg in d.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"unknown config section: {section_name!r}")
        section = getattr(cfg, section_name)
        for k, v in (section_cfg or {}).items():
            if not hasattr(section, k):
                raise ValueError(f"unknown field: {section_name}.{k}")
            if isinstance(v, list):
                v = tuple(v)
            setattr(section, k, v)
    return cfg


def apply_dotted_overrides(cfg: EvalConfig, overrides: list[str]) -> None:
    import dataclasses
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be 'section.field=value', got {ov!r}")
        dotted, raw = ov.split("=", 1)
        if "." not in dotted:
            raise ValueError(f"override path must be dotted, got {dotted!r}")
        section_name, field_name = dotted.split(".", 1)
        if not hasattr(cfg, section_name):
            raise ValueError(f"unknown config section: {section_name!r}")
        section = getattr(cfg, section_name)
        if not hasattr(section, field_name):
            raise ValueError(f"unknown field: {section_name}.{field_name}")
        ann = {f.name: f.type for f in dataclasses.fields(section)}
        setattr(section, field_name, _coerce(raw, ann.get(field_name)))


def _coerce(raw: str, ann):
    s = raw.strip()
    ann_str = ann if isinstance(ann, str) else repr(ann)
    if "tuple" in ann_str.lower():
        # Comma-separated -> tuple of strings, e.g. "mmlu,gsm8k".
        return tuple(x.strip() for x in s.split(",") if x.strip())
    if "bool" in ann_str:
        if s.lower() in ("true", "1", "yes"):
            return True
        if s.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot coerce {raw!r} to bool")
    if "int" in ann_str:
        return int(float(s))
    if "float" in ann_str:
        return float(s)
    return s


if __name__ == "__main__":
    cfg = EvalConfig()
    cfg.sync()
    print("--- EvalConfig defaults (Module 19 Evaluation) ---")
    for section_name in ("model", "benchmarks", "generation", "judge",
                         "contamination", "run"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
