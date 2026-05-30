"""Module 19 evaluation CLI — produce a scorecard for one model/checkpoint.

Runs the automatic benchmark suites named in the config (MMLU / GSM8K /
IFEval), optionally an LLM-as-judge pairwise win-rate vs a baseline, and an
optional n-gram contamination scan, then writes a JSON + markdown scorecard
with confidence intervals on every number.

Examples:

    # Score the post-trained Qwen3-1.7B on all three suites
    python eval.py --config=configs/eval_qwen3_1.7b.yaml

    # Score a GRPO checkpoint from Module 17, GSM8K only, with pass@k
    python eval.py --config=configs/eval_qwen3_1.7b.yaml \\
        --model.checkpoint=../17-reasoning-and-grpo/results/checkpoints/step_00000300 \\
        --benchmarks.suites=gsm8k --generation.greedy=false \\
        --generation.n_samples=8

    # Compare two models head-to-head with a paired significance test
    python eval.py --config=... --model.name=A --out=a.json
    python eval.py --config=... --model.name=B --out=b.json
    python eval.py --compare a.json b.json        # prints the paired diff + p-value

    # Offline / $0 smoke (synthetic data, dummy judge, no GPU)
    python eval.py --config=configs/eval_demo.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_yaml, apply_dotted_overrides
import data as Data
import harness as H
import metrics as M


def run_eval(cfg) -> H.Scorecard:
    import torch
    import model as Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Model.load_tokenizer(cfg.model.name)
    print(f"[eval] loading model {cfg.model.name}"
          + (f" + checkpoint {cfg.model.checkpoint}" if cfg.model.checkpoint else ""))
    mdl = Model.build_model(cfg.model, device=device)

    card = H.Scorecard(model_name=cfg.model.name, checkpoint=cfg.model.checkpoint)
    bm = cfg.benchmarks

    if "mmlu" in bm.suites:
        print("[eval] MMLU ...")
        examples = Data.load_mmlu(bm.n_per_suite, bm.mmlu_subjects, bm.seed)
        fewshot = Data.load_mmlu_fewshot(bm.mmlu_subjects, bm.n_shot) if bm.n_shot else []
        card.results["mmlu"] = H.run_mmlu(
            mdl, tok, examples, fewshot, bm.mc_style, bm.mc_norm,
            device, cfg.run.confidence, cfg.run.seed)

    if "gsm8k" in bm.suites:
        print("[eval] GSM8K ...")
        problems = Data.load_gsm8k(bm.n_per_suite)
        card.results["gsm8k"] = H.run_gsm8k(
            mdl, tok, problems, cfg.generation, device,
            cfg.run.confidence, cfg.run.seed)

    if "ifeval" in bm.suites:
        print("[eval] IFEval ...")
        items = Data.load_ifeval(bm.n_per_suite)
        card.results["ifeval"] = H.run_ifeval(
            mdl, tok, items, cfg.generation, device,
            cfg.run.confidence, cfg.run.seed)

    return card


def run_judge(cfg) -> dict:
    judge = H.make_judge(cfg.judge)
    pairs = Data.load_pairwise_set(cfg.judge.pairwise_set)
    print(f"[eval] judge pairwise on {len(pairs)} pairs "
          f"(backend={cfg.judge.backend}, swap={cfg.judge.swap}) ...")
    return H.run_judge_pairwise(judge, pairs, cfg.judge.swap,
                                cfg.run.confidence, cfg.run.seed)


def run_contamination_scan(cfg) -> dict:
    import contamination as C
    import glob
    paths = glob.glob(cfg.contamination.corpus_glob)
    corpus = []
    for p in paths:
        txt = Path(p).read_text(errors="ignore")
        corpus.append(txt)
    # Use GSM8K questions as the "test set" to scan (illustrative target).
    problems = Data.load_gsm8k(cfg.benchmarks.n_per_suite)
    test_texts = [p["question"] for p in problems]
    rep = C.contamination_report(test_texts, corpus, cfg.contamination.ngram,
                                 cfg.contamination.threshold)
    canary = C.find_canary(corpus)
    summary = rep.summary() + (f"; CANARY found in {len(canary)} docs!" if canary else "")
    return {"summary": summary, "contamination_rate": rep.contamination_rate,
            "mean_overlap": rep.mean_overlap, "canary_docs": canary}


def compare(path_a: str, path_b: str) -> None:
    """Paired significance comparison between two saved scorecards."""
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    print(f"\nPaired comparison: A={path_a}  vs  B={path_b}")
    for suite in a.get("results", {}):
        ra, rb = a["results"].get(suite), b["results"].get(suite)
        if not ra or not rb or "per_item" not in ra or "per_item" not in rb:
            continue
        if len(ra["per_item"]) != len(rb["per_item"]):
            print(f"  [{suite}] skipped — different item counts "
                  f"({len(ra['per_item'])} vs {len(rb['per_item'])}); "
                  f"paired test needs the SAME items.")
            continue
        cmp = M.paired_bootstrap_diff(ra["per_item"], rb["per_item"])
        verdict = "SIGNIFICANT" if cmp.significant else "not significant"
        print(f"  [{suite}] A={cmp.mean_a:.1%}  B={cmp.mean_b:.1%}  "
              f"diff={cmp.diff:+.1%} CI[{cmp.diff_ci.lo:+.1%},{cmp.diff_ci.hi:+.1%}] "
              f"p={cmp.p_value:.3f}  -> {verdict}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Module 19 evaluation harness")
    p.add_argument("--config", default=None)
    p.add_argument("--judge", action="store_true", help="Run the LLM-as-judge pairwise eval.")
    p.add_argument("--contamination", action="store_true",
                   help="Run the n-gram contamination scan (needs contamination.corpus_glob).")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="Paired significance comparison of two saved scorecards; no model load.")
    p.add_argument("--out", default=None, help="Override run.out scorecard path.")
    args, extra = p.parse_known_args(argv)
    overrides = [t[2:] for t in extra if t.startswith("--") and "=" in t]
    return args, overrides


def main(argv=None):
    args, overrides = _parse_args(argv)
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    if not args.config:
        raise SystemExit("--config is required (or use --compare A.json B.json)")

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()
    if args.out:
        cfg.run.out = args.out

    card = run_eval(cfg)

    if args.judge:
        card.results["judge_pairwise"] = run_judge(cfg)

    if args.contamination and cfg.contamination.corpus_glob:
        card.contamination = run_contamination_scan(cfg)

    out_path = Path(cfg.run.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": card.model_name, "checkpoint": card.checkpoint,
               "results": card.results, "contamination": card.contamination,
               "config": cfg.to_dict()}
    out_path.write_text(json.dumps(payload, indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(card.to_markdown())
    print(f"\n[eval] wrote {out_path} and {md_path}\n")
    print(card.to_markdown())


if __name__ == "__main__":
    main()
