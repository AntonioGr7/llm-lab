"""
End-to-end curation pipeline.

Loads a source SFT dataset from HuggingFace (default: DeepMount00/OpenItalianData),
applies the filter cascade defined in `filters.py`, optionally runs the LLM-as-judge
stage from `judge.py`, and writes JSONL + a `report.json` documenting exactly what
was dropped and why.

Usage:

    # Small smoke run (no network if --source is a local jsonl)
    python prepare.py --limit 1000 --out demo.jsonl

    # Real run — 2.14M rows in, ~50-150k out
    python prepare.py \\
        --dataset DeepMount00/OpenItalianData \\
        --target-size 100000 \\
        --out italian-sft-curated.jsonl

You can disable any stage with the corresponding --skip-* flag (useful when you
don't have fasttext / sentence-transformers installed yet, or when you want to
A/B different filter combos and see how the survivor distribution changes).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from filters import (
    ArtifactConfig,
    FilterReport,
    LangConfig,
    StructuralConfig,
    Row,
    diversity_downsample,
    minhash_dedup,
    normalize_row,
    run_per_row_filters_parallel,
)


# ----------------------------------------------------------------------
# Loading & normalization
# ----------------------------------------------------------------------


def load_source(
    dataset: str,
    split: str = "train",
    streaming: bool = False,
    local_jsonl: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[Optional[int], Iterator[Row]]:
    """Return (total, iterator) of normalized rows.

    ``total`` is the expected row count (useful for tqdm ETAs). It is None
    when the source size is not cheaply knowable (streaming HF datasets).
    For non-streaming HF datasets and local JSONL we count up front.
    """
    if local_jsonl:
        path = Path(local_jsonl)
        # Cheap line count — JSONL is one-row-per-line by convention.
        with path.open("r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        if limit is not None:
            total = min(total, limit)

        def _iter_jsonl() -> Iterator[Row]:
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit is not None and i >= limit:
                        break
                    try:
                        row = normalize_row(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if row is not None:
                        yield row

        return total, _iter_jsonl()

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install `datasets` (`pip install datasets`) to load HF datasets") from e

    ds = load_dataset(dataset, split=split, streaming=streaming)
    total: Optional[int]
    if streaming:
        total = limit  # may be None
    else:
        total = len(ds)
        if limit is not None:
            total = min(total, limit)

    def _iter_hf() -> Iterator[Row]:
        for i, raw in enumerate(ds):
            if limit is not None and i >= limit:
                break
            candidate = raw
            if isinstance(raw, dict):
                for k in ("messages", "conversations", "conversation", "data"):
                    if k in raw and isinstance(raw[k], list):
                        candidate = raw[k]
                        break
            row = normalize_row(candidate)
            if row is not None:
                yield row

    return total, _iter_hf()


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[Row]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Row]:
    out: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_report(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def read_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_paths(out_path: Path) -> dict[str, Path]:
    """Compute the standard checkpoint file paths derived from --out."""
    stem = out_path.with_suffix("")  # strip .jsonl
    return {
        "stage_1_3_rows":   Path(f"{stem}.stage_1_3.jsonl"),
        "stage_1_3_report": Path(f"{stem}.stage_1_3.report.json"),
        "stage_4a_rows":    Path(f"{stem}.stage_4a.jsonl"),
        "stage_4a_report": Path(f"{stem}.stage_4a.report.json"),
        "stage_4a_progress": Path(f"{stem}.stage_4a.progress.json"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("source")
    src.add_argument("--dataset", default="DeepMount00/OpenItalianData",
                     help="HuggingFace dataset name (default: %(default)s)")
    src.add_argument("--split", default="train")
    src.add_argument("--streaming", action="store_true",
                     help="Stream from HF instead of full download (useful for large datasets)")
    src.add_argument("--local-jsonl", default=None,
                     help="Read from a local JSONL file instead of HF")
    src.add_argument("--limit", type=int, default=None,
                     help="Stop after N input rows (smoke testing)")

    out = p.add_argument_group("output")
    out.add_argument("--out", default="italian-sft-curated.jsonl")
    out.add_argument("--report-out", default=None,
                     help="Where to write the drop-reason report (default: alongside --out)")

    f1 = p.add_argument_group("stage 1 — structural")
    f1.add_argument("--min-user-words", type=int, default=3)
    f1.add_argument("--min-assistant-words", type=int, default=10)
    f1.add_argument("--max-assistant-words", type=int, default=2000)
    f1.add_argument("--min-ratio", type=float, default=0.25,
                    help="Min words(response)/words(prompt) — 0 to disable")
    f1.add_argument("--max-repetition", type=float, default=0.30)

    f2 = p.add_argument_group("stage 2 — language fidelity")
    f2.add_argument("--target-lang", default="it")
    f2.add_argument("--min-lang-conf", type=float, default=0.50)
    f2.add_argument("--min-fn-word-ratio", type=float, default=0.05)
    f2.add_argument("--skip-langid", action="store_true",
                    help="Disable fasttext language ID (use if model unavailable)")
    f2.add_argument("--skip-fnword", action="store_true")

    f3 = p.add_argument_group("stage 3 — translation artifacts")
    f3.add_argument("--skip-english-patterns", action="store_true")
    f3.add_argument("--skip-calques", action="store_true")

    f4 = p.add_argument_group("stage 4 — dedup & diversity")
    f4.add_argument("--minhash-threshold", type=float, default=0.85,
                    help="Jaccard threshold above which prompts are near-duplicates")
    f4.add_argument("--skip-dedup", action="store_true")
    f4.add_argument("--dedup-checkpoint-every", type=int, default=20_000,
                    help="Stream the stage-4a checkpoint + progress marker every N "
                         "input rows (default: %(default)s). A crash mid-dedup resumes "
                         "from the last marker. Smaller = more durable, more fsyncs.")
    f4.add_argument("--target-size", type=int, default=None,
                    help="If set, diversity-downsample to this many rows after dedup")
    f4.add_argument("--diversity-clusters", type=int, default=None,
                    help="Number of clusters for diversity sampling (default: sqrt(target_size))")
    f4.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    f4.add_argument("--skip-diversity", action="store_true")

    perf = p.add_argument_group("performance")
    perf.add_argument("--workers", type=int, default=-1,
                      help="Process-pool size for stage 1-3 (default: cpu_count - 1). "
                           "Pass 1 for single-threaded; useful for debugging.")
    perf.add_argument("--chunk-size", type=int, default=1000,
                      help="Rows per worker chunk (default: %(default)s). "
                           "Larger = less IPC overhead, worse load balance.")

    res = p.add_argument_group("resume")
    res.add_argument("--no-resume", action="store_true",
                     help="Ignore any existing stage checkpoints and rerun from scratch.")
    res.add_argument("--keep-checkpoints", action="store_true",
                     help="Don't delete stage_1_3 / stage_4a checkpoint files after a successful run.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    log = (lambda *a, **k: None) if args.quiet else print

    # Build filter configs.
    s_cfg = StructuralConfig(
        min_user_words=args.min_user_words,
        min_assistant_words=args.min_assistant_words,
        max_assistant_words=args.max_assistant_words,
        min_response_to_prompt_ratio=args.min_ratio,
        max_ngram_repetition=args.max_repetition,
    )
    l_cfg = LangConfig(
        target_lang=args.target_lang,
        min_lang_confidence=args.min_lang_conf,
        min_function_word_ratio=args.min_fn_word_ratio,
    )
    a_cfg = ArtifactConfig(
        drop_on_english_pattern=not args.skip_english_patterns,
        drop_on_calque=not args.skip_calques,
    )

    skip_flags = {
        "langid": args.skip_langid,
        "fnword": args.skip_fnword,
        "artifacts": args.skip_english_patterns,
        "calques": args.skip_calques,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_paths(out_path)

    # Pre-flight: detect existing checkpoints (unless --no-resume).
    have_stage_1_3 = (not args.no_resume) and ckpt["stage_1_3_rows"].exists()
    # Stage 4a now *streams* its rows file, so "rows file exists" no longer means
    # "4a finished" — a crash before the first marker leaves a partial rows file.
    # The report is written only after a clean completion, so it's the reliable
    # done-signal; the progress marker (deleted on success) signals partial work.
    stage_4a_partial = (not args.no_resume) and ckpt["stage_4a_progress"].exists()
    have_stage_4a = (
        (not args.no_resume)
        and ckpt["stage_4a_report"].exists()
        and ckpt["stage_4a_rows"].exists()
        and not stage_4a_partial
    )

    # Stage 1-3 — run or load from checkpoint.
    stage_1_3_meta: dict = {}
    if have_stage_4a:
        # Skip both 1-3 and 4a; we'll load 4a below.
        kept = []
        stage_1_3_meta = (read_report(ckpt["stage_1_3_report"]) if ckpt["stage_1_3_report"].exists() else {})
        log("=== Resume: stage 4a checkpoint found, skipping stages 1-3 + 4a ===")
        log(f"  -> {ckpt['stage_4a_rows']}")
    elif have_stage_1_3:
        log("=== Resume: stage 1-3 checkpoint found, skipping stages 1-3 ===")
        log(f"  loading {ckpt['stage_1_3_rows']}")
        t0 = time.time()
        kept = read_jsonl(ckpt["stage_1_3_rows"])
        stage_1_3_meta = read_report(ckpt["stage_1_3_report"]) if ckpt["stage_1_3_report"].exists() else {
            "kept": len(kept),
            "total": len(kept),
            "drops": {},
            "note": "stage_1_3 report not found — drop counts unknown (resume)",
        }
        log(f"  loaded {len(kept):,} rows in {time.time()-t0:.1f}s")
    else:
        log(f"=== Stage 1–3: per-row filters ===")
        log(f"  source         : {args.local_jsonl or args.dataset}")
        log(f"  target language: {args.target_lang}")
        log(f"  limit          : {args.limit}")
        log(f"  workers        : {args.workers if args.workers > 0 else 'auto'}")
        log(f"  chunk size     : {args.chunk_size}")

        t0 = time.time()
        total, source_iter = load_source(
            dataset=args.dataset, split=args.split, streaming=args.streaming,
            local_jsonl=args.local_jsonl, limit=args.limit,
        )
        if total is not None:
            log(f"  expected rows : {total:,}")
        kept, report = run_per_row_filters_parallel(
            source_iter,
            structural=s_cfg, lang=l_cfg, artifact=a_cfg, skip=skip_flags,
            n_workers=args.workers, chunk_size=args.chunk_size,
            progress=not args.quiet, total=total, desc="stage 1-3",
        )
        log(report.summary())
        log(f"  wallclock: {time.time() - t0:.1f}s")

        stage_1_3_meta = {
            "total": report.total,
            "kept": report.kept,
            "drops": dict(report.drops),
        }
        # Checkpoint: write rows + report.
        log(f"  writing checkpoint -> {ckpt['stage_1_3_rows']}")
        write_jsonl(ckpt["stage_1_3_rows"], kept)
        write_report(ckpt["stage_1_3_report"], stage_1_3_meta)

    # Stage 4a — MinHash dedup.
    stage4_drops: Counter = Counter()
    if have_stage_4a:
        log(f"\n=== Resume: loading stage 4a checkpoint ===")
        t0 = time.time()
        kept = read_jsonl(ckpt["stage_4a_rows"])
        log(f"  loaded {len(kept):,} rows in {time.time()-t0:.1f}s")
        if ckpt["stage_4a_report"].exists():
            stage4_drops.update(read_report(ckpt["stage_4a_report"]).get("drops", {}))
    elif not args.skip_dedup and kept:
        if stage_4a_partial:
            log(f"\n=== Resume: stage 4a partial checkpoint found, continuing dedup ===")
            log(f"  -> {ckpt['stage_4a_rows']} (progress: {ckpt['stage_4a_progress']})")
        else:
            log(f"\n=== Stage 4a: MinHash dedup (threshold={args.minhash_threshold}) ===")
        t0 = time.time()
        before = len(kept)
        # minhash_dedup streams kept rows to the stage_4a checkpoint and writes a
        # progress marker every --dedup-checkpoint-every rows, so a crash mid-stage
        # resumes from the last marker instead of restarting the whole pass.
        kept, n_dropped = minhash_dedup(
            kept, threshold=args.minhash_threshold, progress=not args.quiet,
            checkpoint_path=ckpt["stage_4a_rows"],
            state_path=ckpt["stage_4a_progress"],
            checkpoint_every=args.dedup_checkpoint_every,
        )
        stage4_drops["minhash_near_duplicate"] = n_dropped
        log(f"  {before:,} -> {len(kept):,} ({n_dropped:,} near-dup dropped, "
            f"{time.time() - t0:.1f}s)")
        # Stage 4a is complete: the rows file is already on disk (streamed). Write
        # the report and drop the progress marker so this counts as a *completed*
        # checkpoint on any future resume.
        write_report(ckpt["stage_4a_report"], {"drops": dict(stage4_drops)})
        ckpt["stage_4a_progress"].unlink(missing_ok=True)

    # Stage 4b — diversity downsample. (No checkpoint between 4a and 4b — 4b is
    # fast enough that re-running it from the 4a checkpoint is acceptable. The
    # expensive parts of 4b are embed + cluster, both of which would need
    # serializing the kmeans state to checkpoint mid-run.)
    if not args.skip_diversity and args.target_size is not None and len(kept) > args.target_size:
        log(f"\n=== Stage 4b: diversity downsample to {args.target_size:,} ===")
        t0 = time.time()
        before = len(kept)
        kept = diversity_downsample(
            kept,
            target_size=args.target_size,
            embedding_model=args.embedding_model,
            n_clusters=args.diversity_clusters,
            seed=args.seed,
            show_progress=not args.quiet,
        )
        stage4_drops["diversity_downsample"] = before - len(kept)
        log(f"  {before:,} -> {len(kept):,} ({time.time() - t0:.1f}s)")

    # Final write.
    n_written = write_jsonl(out_path, kept)
    log(f"\nWrote {n_written:,} rows -> {out_path}")

    report_path = Path(args.report_out) if args.report_out else out_path.with_suffix(".report.json")
    full_report = {
        "args": {k: v for k, v in vars(args).items()},
        "stage_1_3": stage_1_3_meta,
        "stage_4": dict(stage4_drops),
        "final_kept": n_written,
        "structural_config": asdict(s_cfg),
        "lang_config": asdict(l_cfg),
        "artifact_config": asdict(a_cfg),
        "resumed_from": (
            "stage_4a" if have_stage_4a
            else "stage_1_3" if have_stage_1_3
            else "scratch"
        ),
    }
    write_report(report_path, full_report)
    log(f"Wrote report -> {report_path}")

    # Cleanup checkpoints on success unless the user asked to keep them.
    if not args.keep_checkpoints:
        for k in ("stage_1_3_rows", "stage_1_3_report", "stage_4a_rows",
                  "stage_4a_report", "stage_4a_progress"):
            try:
                ckpt[k].unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
