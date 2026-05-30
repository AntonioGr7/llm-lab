"""The harness — wire model + benchmarks + metrics into a scorecard.

This is the orchestration layer. Each `run_*` function takes a loaded model +
tokenizer + config, runs one benchmark family, and returns a result dict that
ALWAYS carries a confidence interval (via `metrics`) — because a benchmark
number without an error bar is not a measurement, it's a vibe.

The deliberate design choice: every headline number is paired with its CI and,
where two configs are run (e.g. two MC normalizations or two prompt formats),
the harness reports BOTH so the format-sensitivity lesson is visible in the
output, not buried.

`render_scorecard` produces the markdown table that goes in `results/` and
reads like a mini model card.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import benchmarks as B
import metrics as M
import model as Model
from judge import (Judge, DummyJudge, LocalJudge, pairwise_judge,
                   aggregate_win_rate)


# =============================================================================
# MMLU (multiple-choice via likelihood)
# =============================================================================

def run_mmlu(model, tok, examples, fewshot, style: str, norm: str,
             device: str, confidence: float, seed: int) -> dict:
    """Score MMLU examples; return accuracy + CI, plus a format/norm sensitivity
    sweep so the student SEES the score move when the harness changes."""
    fewshot_block = _build_mmlu_fewshot_block(fewshot, style) if fewshot else ""
    # Score every option's logprob once; reuse across normalizations.
    per_item_logprobs = []
    for ex in examples:
        prompt, conts = B.build_mc_prompt(ex, style=style, n_shot_block=fewshot_block)
        lps, ntoks, nbytes = [], [], []
        for c in conts:
            lp, nt, nb = Model.continuation_logprob(model, tok, prompt, c, device)
            lps.append(lp); ntoks.append(nt); nbytes.append(nb)
        per_item_logprobs.append((lps, ntoks, nbytes))

    def acc_for(norm_choice):
        preds = [B.score_mc(lp, nt, nb, norm_choice)
                 for (lp, nt, nb) in per_item_logprobs]
        flags = B.mc_accuracy(preds, examples)
        return flags

    flags = acc_for(norm)
    ci = M.bootstrap_ci(flags, confidence=confidence, seed=seed)
    # Sensitivity: report all three norms at the chosen style.
    sweep = {nm: M.bootstrap_ci(acc_for(nm), confidence=confidence, seed=seed).point
             for nm in ("raw", "token", "byte")}
    return {
        "suite": "mmlu", "n": len(examples), "style": style, "norm": norm,
        "accuracy": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
        "norm_sensitivity": sweep, "per_item": flags,
    }


def _build_mmlu_fewshot_block(fewshot, style: str) -> str:
    lines = []
    for ex in fewshot:
        prompt, _ = B.build_mc_prompt(ex, style=style)
        letter = chr(ord("A") + ex.answer_index)
        if style == "letters":
            lines.append(f"{prompt} {letter}\n")
        else:
            lines.append(f"{prompt} {ex.choices[ex.answer_index]}\n")
    return "\n".join(lines) + "\n" if lines else ""


# =============================================================================
# GSM8K (generative + integer verifier, optional sampling for pass@k)
# =============================================================================

def run_gsm8k(model, tok, problems, gen_cfg, device: str,
              confidence: float, seed: int) -> dict:
    """Score GSM8K. Single greedy decode -> accuracy + CI. With n_samples>1 and
    sampling -> also report pass@k / avg@k / maj@k from the same samples."""
    sys_prompt = gen_cfg.system_prompt or (
        "Solve the math problem step by step. End your answer with '#### <integer>'.")
    n_samples = max(1, gen_cfg.n_samples)
    greedy = gen_cfg.greedy

    correct_flags = []     # for the headline (first sample / greedy)
    pass_at = Counter()
    maj_flags = []
    for prob in problems:
        gt = B.parse_gsm8k_ground_truth(prob["answer_field"])
        if gt is None:
            continue
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prob["question"]}]
        sample_answers = []
        for s in range(n_samples):
            txt = Model.generate(
                model, tok, messages, gen_cfg.max_new_tokens,
                greedy=greedy and s == 0, temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p, device=device)
            sample_answers.append(B.extract_final_int(txt))
        c = sum(1 for a in sample_answers if a == gt)
        correct_flags.append(1.0 if sample_answers[0] == gt else 0.0)
        maj_flags.append(1.0 if M.majority_at_k(sample_answers, gt) else 0.0)
        if n_samples > 1:
            for k in (1, min(5, n_samples), n_samples):
                pass_at[k] += M.pass_at_k(n_samples, c, k)

    ci = M.bootstrap_ci(correct_flags, confidence=confidence, seed=seed)
    out = {"suite": "gsm8k", "n": len(correct_flags),
           "accuracy": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
           "per_item": correct_flags}
    if n_samples > 1:
        n_prob = max(1, len(correct_flags))
        out["pass_at_k"] = {k: pass_at[k] / n_prob for k in sorted(pass_at)}
        out["avg_at_k"] = M.avg_at_k(correct_flags)
        out["maj_at_k"] = sum(maj_flags) / n_prob
    return out


# =============================================================================
# IFEval (verifiable instruction-following)
# =============================================================================

def run_ifeval(model, tok, items, gen_cfg, device: str,
               confidence: float, seed: int) -> dict:
    """Generate a response per prompt; score programmatic constraints. Reports
    prompt-level (strict) accuracy + CI and instruction-level (loose) rate."""
    prompt_flags, instr_rates = [], []
    for ex in items:
        messages = [{"role": "user", "content": ex.prompt}]
        txt = Model.generate(model, tok, messages, gen_cfg.max_new_tokens,
                             greedy=True, temperature=1.0, top_p=1.0, device=device)
        res = B.check_constraints(txt, ex.constraints)
        prompt_flags.append(1.0 if res["prompt_level"] else 0.0)
        instr_rates.append(res["instruction_level"])
    ci = M.bootstrap_ci(prompt_flags, confidence=confidence, seed=seed)
    return {"suite": "ifeval", "n": len(items),
            "prompt_level": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "instruction_level": sum(instr_rates) / max(len(instr_rates), 1),
            "per_item": prompt_flags}


# =============================================================================
# LLM-as-judge (pairwise win rate with position-swap + bias diagnostics)
# =============================================================================

def make_judge(judge_cfg) -> Judge:
    if judge_cfg.backend == "dummy":
        return DummyJudge(rule="longer")
    return LocalJudge(judge_cfg.model_name, max_new_tokens=judge_cfg.max_new_tokens)


def run_judge_pairwise(judge: Judge, pairs, swap: bool,
                       confidence: float, seed: int) -> dict:
    """Pairwise-judge A (baseline) vs B (model under test). Returns win rate +
    CI + position-bias rate + length-win correlation."""
    verdicts, len_a, len_b = [], [], []
    for p in pairs:
        v = pairwise_judge(judge, p["question"], p["answer_a"], p["answer_b"], swap=swap)
        verdicts.append(v)
        len_a.append(len(p["answer_a"]))
        len_b.append(len(p["answer_b"]))
    res = aggregate_win_rate(verdicts, len_a, len_b)
    # CI on the win rate via bootstrap over per-pair outcomes (B-win=1, tie=0.5).
    outcomes = [1.0 if v.winner == "B" else (0.5 if v.winner == "tie" else 0.0)
                for v in verdicts]
    ci = M.bootstrap_ci(outcomes, confidence=confidence, seed=seed)
    return {"suite": "judge_pairwise", "n": res.n, "win_rate": res.win_rate,
            "ci_lo": ci.lo, "ci_hi": ci.hi,
            "position_bias_rate": res.position_bias_rate,
            "length_win_corr": res.length_win_corr,
            "wins_b": res.wins_b, "wins_a": res.wins_a, "ties": res.ties}


# =============================================================================
# Scorecard rendering
# =============================================================================

@dataclass
class Scorecard:
    model_name: str
    checkpoint: str = ""
    results: dict = field(default_factory=dict)
    contamination: Optional[dict] = None

    def to_markdown(self) -> str:
        lines = [f"# Scorecard — {self.model_name}"]
        if self.checkpoint:
            lines.append(f"_checkpoint: {self.checkpoint}_")
        lines.append("")
        lines.append("| Suite | Headline | 95% CI | Notes |")
        lines.append("|---|---|---|---|")
        for name, r in self.results.items():
            if name == "mmlu":
                note = (f"{r['style']}/{r['norm']}; norm sweep "
                        + "/".join(f"{k}={v:.2f}" for k, v in r["norm_sensitivity"].items()))
                lines.append(f"| MMLU | {r['accuracy']:.1%} | "
                             f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | {note} |")
            elif name == "gsm8k":
                note = ""
                if "pass_at_k" in r:
                    note = ("pass@k " + "/".join(f"{k}:{v:.2f}" for k, v in r["pass_at_k"].items())
                            + f"; maj@k={r['maj_at_k']:.2f}")
                lines.append(f"| GSM8K | {r['accuracy']:.1%} | "
                             f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | {note} |")
            elif name == "ifeval":
                lines.append(f"| IFEval | {r['prompt_level']:.1%} (strict) | "
                             f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | "
                             f"loose={r['instruction_level']:.1%} |")
            elif name == "judge_pairwise":
                note = (f"pos-bias={r['position_bias_rate']:.0%}, "
                        f"len-corr={r['length_win_corr']:+.2f}")
                lines.append(f"| Judge win-rate (vs baseline) | {r['win_rate']:.1%} | "
                             f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | {note} |")
        if self.contamination:
            c = self.contamination
            lines.append("")
            lines.append(f"**Contamination scan:** {c.get('summary', 'n/a')}")
        return "\n".join(lines)
