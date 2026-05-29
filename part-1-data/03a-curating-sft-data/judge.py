"""
Stage 5 — LLM-as-judge (optional).

The cheap heuristic stages (1-4) cut the noise floor for ~free, but the last
quality lift comes from asking a strong model to actually read each (prompt,
response) pair and score it on a few axes. This is expensive but it is where
the biggest quality gains live, and the methodology generalizes far beyond
this module — every modern data-curation pipeline at frontier labs has some
version of "model-graded quality scoring" baked in.

This file exposes:

  - `JudgeScore` dataclass: per-row scores on 4 axes (fluency_it, relevance,
                            correctness, overall) plus a free-text rationale.
  - `LocalJudge`: a transformers-backed judge that loads a small Italian-capable
                  instruct model in-process (default Qwen3-1.7B).
  - `OpenAICompatibleJudge`: HTTP-based judge that talks to any OpenAI-compatible
                             chat-completions endpoint. Use with llama.cpp server,
                             vLLM's OpenAI server, Ollama, Together, OpenRouter, etc.
                             RECOMMENDED for small-VRAM setups — run llama.cpp with
                             a Q4 GGUF and you get the same quality at <2 GB VRAM.
  - `DummyJudge`: deterministic length-based score for tests.
  - `score_rows(rows, judge, ...)`: sequential batched scoring helper.
  - `score_rows_resumable(...)`: concurrent + resumable variant. Writes scores
                                  to a JSONL sidecar as they complete so a crash
                                  mid-run doesn't lose hours of work.
  - `filter_by_score(rows, scores, ...)`: top-N or threshold filter.
  - CLI entrypoint (``python judge.py --in ... --out ... --endpoint ...``)

The module is intentionally backend-agnostic — `Judge` is a Protocol. The CLI
is wired to ``OpenAICompatibleJudge`` because that's the most ergonomic local
path (separate llama.cpp process = no Python/CUDA conflicts).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

from filters import Row, assistant_text, normalize_row, user_text


SYSTEM_PROMPT = """Sei un revisore esperto di dataset per il fine-tuning di modelli linguistici italiani.

Riceverai una coppia (richiesta utente, risposta assistente). Il tuo compito è valutare la qualità della coppia su 4 assi, ciascuno con un punteggio intero da 0 a 5:

1. **fluency_it** — La risposta è in italiano fluente e naturale? (0 = non italiano o palesemente tradotto; 5 = scritto da un madrelingua)
2. **relevance** — La risposta affronta la richiesta dell'utente? (0 = risponde a qualcos'altro; 5 = aderente alla richiesta)
3. **correctness** — Il contenuto è fattualmente accurato e privo di errori evidenti? (0 = falso o senza senso; 5 = corretto e completo)
4. **overall** — Considerati tutti gli assi, quanto consiglieresti di includere questa coppia in un dataset di fine-tuning ad alta qualità? (0 = scarta; 5 = esempio eccellente)

Restituisci SOLO un oggetto JSON con questa esatta struttura, senza testo aggiuntivo:

{"fluency_it": int, "relevance": int, "correctness": int, "overall": int, "rationale": str}

Il campo `rationale` deve essere una frase breve (massimo 20 parole) che spieghi il tuo voto overall."""


USER_TEMPLATE = """RICHIESTA UTENTE:
{user}

RISPOSTA ASSISTENTE:
{assistant}

Valuta la coppia."""


@dataclass(frozen=True)
class JudgeScore:
    fluency_it: int
    relevance: int
    correctness: int
    overall: int
    rationale: str = ""

    @classmethod
    def from_json(cls, blob: str) -> "JudgeScore":
        """Parse the model's JSON output. Tolerates surrounding markdown fences."""
        # Strip ```json ... ``` fences if present.
        m = re.search(r"\{[^{}]*\}", blob, re.DOTALL)
        if m is None:
            raise ValueError(f"no JSON object in: {blob!r}")
        data = json.loads(m.group(0))
        return cls(
            fluency_it=int(data.get("fluency_it", 0)),
            relevance=int(data.get("relevance", 0)),
            correctness=int(data.get("correctness", 0)),
            overall=int(data.get("overall", 0)),
            rationale=str(data.get("rationale", "")),
        )


class Judge(Protocol):
    """Anything that maps a Row to a JudgeScore."""

    def __call__(self, row: Row) -> JudgeScore: ...


# ----------------------------------------------------------------------
# A trivially-deterministic judge used by tests.
# ----------------------------------------------------------------------


class DummyJudge:
    """Returns a score derived from the response length (deterministic, no model load).

    Useful for wiring tests without paying for or downloading anything.
    """

    def __call__(self, row: Row) -> JudgeScore:
        n_words = len(assistant_text(row).split())
        score = max(0, min(5, n_words // 8))
        return JudgeScore(
            fluency_it=score,
            relevance=score,
            correctness=score,
            overall=score,
            rationale=f"dummy: response is {n_words} words",
        )


# ----------------------------------------------------------------------
# Local-model judge (transformers).
# ----------------------------------------------------------------------


class LocalJudge:
    """Run a small Italian-capable instruct model locally and parse JSON output.

    Defaults to Qwen3-1.7B because it is multilingual and small enough to run
    on a single consumer GPU. Anything similar works — Mistral-7B-Instruct,
    Llama-3.1-8B-Instruct, Phi-3.5-mini-instruct, etc.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        device: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # We want JSON in the output, so set a pad token if missing.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _format_messages(self, row: Row) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    user=user_text(row).strip(),
                    assistant=assistant_text(row).strip(),
                ),
            },
        ]

    def __call__(self, row: Row) -> JudgeScore:
        import torch

        messages = self._format_messages(row)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        try:
            return JudgeScore.from_json(completion)
        except (ValueError, json.JSONDecodeError):
            # Be conservative: malformed output -> score 0 (will be dropped).
            return JudgeScore(0, 0, 0, 0, rationale=f"malformed: {completion[:80]}")


# ----------------------------------------------------------------------
# OpenAI-compatible HTTP judge (llama.cpp server, vLLM, Ollama, etc.)
# ----------------------------------------------------------------------


class OpenAICompatibleJudge:
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint.

    Tested against ``llama.cpp`` server (the recommended local path for small
    VRAM — see README for launch flags). Also works with vLLM's OpenAI server,
    Ollama (``/v1`` route), Together AI, OpenRouter, and the OpenAI API itself.

    Notes on JSON output:
      - We ask the model to return raw JSON in the prompt.
      - We *additionally* set ``response_format={"type": "json_object"}`` so
        servers that support structured output enforce well-formed JSON. Most
        modern OpenAI-compatible servers honor it (llama.cpp does); if yours
        doesn't, it's a no-op (the request still works).
      - On parse failure we return a 0-score with the raw text in ``rationale``,
        so the row is conservatively dropped without crashing the run.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/v1",
        model: str = "default",
        api_key: str = "no-key",
        timeout: float = 60.0,
        max_tokens: int = 200,
        temperature: float = 0.0,
        retries: int = 2,
    ):
        # Normalize endpoint — accept both "http://host:8080" and "http://host:8080/v1".
        ep = endpoint.rstrip("/")
        if not ep.endswith("/v1"):
            ep = ep + "/v1"
        self.url = ep + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def __call__(self, row: Row) -> JudgeScore:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    user=user_text(row).strip(),
                    assistant=assistant_text(row).strip(),
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # Structured-output hint. Honored by llama.cpp, vLLM, OpenAI; ignored
            # by some others. Cheap to ask, big win when supported.
            "response_format": {"type": "json_object"},
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._post(payload)
                completion = resp["choices"][0]["message"]["content"]
                return JudgeScore.from_json(completion)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))  # 0.5s, 1s, 2s
                continue
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                # Malformed response — conservative drop, don't retry (waste).
                return JudgeScore(0, 0, 0, 0, rationale=f"malformed: {str(e)[:80]}")
        return JudgeScore(0, 0, 0, 0, rationale=f"http_error: {str(last_err)[:80]}")


# ----------------------------------------------------------------------
# Batched orchestrator.
# ----------------------------------------------------------------------


def score_rows(
    rows: Iterable[Row], judge: Judge, show_progress: bool = True
) -> list[JudgeScore]:
    rows = list(rows)
    iterator: Iterable = rows
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(rows, desc="judging", unit="row")
        except ImportError:
            pass
    return [judge(r) for r in iterator]


def score_rows_resumable(
    rows: list[Row],
    judge: Judge,
    scores_path: Path,
    concurrent: int = 4,
    show_progress: bool = True,
    flush_every: int = 50,
) -> list[Optional[JudgeScore]]:
    """Score rows with bounded concurrency and incremental disk writes.

    Each score is appended to ``scores_path`` as ``{"idx": int, "score": {...}}``
    on completion. On re-invocation, scores already present for an index are
    NOT re-computed — pass the same ``scores_path`` and the function resumes
    where you left off.

    Returns a list of length len(rows), indexed in input order. Entries for
    rows whose scoring failed twice (after retries) will be ``None``.

    Threading is used (not multiprocessing) because the judge is I/O-bound
    on HTTP requests to llama.cpp. The server's own ``--parallel N`` slot
    count is the real concurrency knob — set ``concurrent`` to match it.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load existing scores (resume).
    existing: dict[int, JudgeScore] = {}
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    idx = int(rec["idx"])
                    s = rec["score"]
                    existing[idx] = JudgeScore(
                        fluency_it=int(s["fluency_it"]),
                        relevance=int(s["relevance"]),
                        correctness=int(s["correctness"]),
                        overall=int(s["overall"]),
                        rationale=str(s.get("rationale", "")),
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    results: list[Optional[JudgeScore]] = [existing.get(i) for i in range(len(rows))]
    todo = [i for i in range(len(rows)) if results[i] is None]

    if not todo:
        return results

    # Open scores file in append mode so we can stream out as we go.
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = scores_path.open("a", encoding="utf-8")

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=len(todo), desc="judging", unit="row")
        except ImportError:
            pass

    pending_writes = 0

    def _score_one(idx: int) -> tuple[int, JudgeScore]:
        return idx, judge(rows[idx])

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrent)) as pool:
            futures = {pool.submit(_score_one, i): i for i in todo}
            for fut in as_completed(futures):
                try:
                    idx, score = fut.result()
                    results[idx] = score
                    out_f.write(json.dumps({"idx": idx, "score": score.__dict__}, ensure_ascii=False) + "\n")
                    pending_writes += 1
                    if pending_writes >= flush_every:
                        out_f.flush()
                        pending_writes = 0
                except Exception as e:
                    # Judge raised something we didn't catch internally; leave
                    # results[idx] as None, log via tqdm if available.
                    idx = futures[fut]
                    if pbar is not None:
                        pbar.write(f"  row {idx}: {e.__class__.__name__}: {e}")
                if pbar is not None:
                    pbar.update(1)
    finally:
        out_f.flush()
        out_f.close()
        if pbar is not None:
            pbar.close()

    return results


def filter_by_score(
    rows: list[Row],
    scores: list[JudgeScore],
    min_overall: int = 3,
    min_fluency_it: int = 3,
    keep_top_n: Optional[int] = None,
) -> tuple[list[Row], list[JudgeScore]]:
    """Apply thresholds, then optionally trim to the top-N by overall score."""
    paired = [
        (r, s)
        for r, s in zip(rows, scores)
        if s.overall >= min_overall and s.fluency_it >= min_fluency_it
    ]
    if keep_top_n is not None and len(paired) > keep_top_n:
        paired.sort(key=lambda rs: (-rs[1].overall, -rs[1].fluency_it))
        paired = paired[:keep_top_n]
    if not paired:
        return [], []
    rows_out, scores_out = zip(*paired)
    return list(rows_out), list(scores_out)


# ----------------------------------------------------------------------
# CLI — stage 5 against an OpenAI-compatible endpoint.
# ----------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[Row]:
    """Load JSONL and normalize each row to {messages: [...]} shape.

    Tolerates: prepare.py's output (already normalized), raw OpenItalianData
    (list of turns), sharegpt-style, prompt+response pairs.
    """
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = normalize_row(raw)
            if row is not None:
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    import argparse
    from collections import Counter
    from dataclasses import asdict

    p = argparse.ArgumentParser(
        description="Stage 5 — LLM-as-judge filtering for SFT data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Score everything against a local llama.cpp server, keep top-N.
  python judge.py \\
      --in italian-sft-curated.jsonl \\
      --out italian-sft-judged.jsonl \\
      --endpoint http://localhost:8080/v1 \\
      --concurrent 4 \\
      --keep-top-n 100000

  # If you crash mid-run, just re-launch — scores already in .scores.jsonl
  # are NOT recomputed.

  # Score only, don't filter (useful for inspection / threshold tuning).
  python judge.py \\
      --in italian-sft-curated.jsonl \\
      --scores-out scores.jsonl \\
      --endpoint http://localhost:8080/v1 \\
      --no-filter
""",
    )
    p.add_argument("--in", dest="in_path", required=True,
                   help="Input JSONL of rows to judge (output of prepare.py)")
    p.add_argument("--out", default=None,
                   help="Output JSONL of rows that pass the judge filter. "
                        "Default: <in>.judged.jsonl")
    p.add_argument("--scores-out", default=None,
                   help="Where to write per-row judge scores (resumable). "
                        "Default: <out>.scores.jsonl")
    p.add_argument("--report-out", default=None,
                   help="Where to write summary report. Default: <out>.report.json")

    backend = p.add_argument_group("backend")
    backend.add_argument("--endpoint", default="http://localhost:8080/v1",
                         help="OpenAI-compatible API endpoint (default: %(default)s — llama.cpp's default)")
    backend.add_argument("--model", default="default",
                         help="Model name to pass in the request (llama.cpp ignores it but other servers may not)")
    backend.add_argument("--api-key", default="no-key",
                         help="Auth bearer token. llama.cpp doesn't require one; OpenAI/Together do.")
    backend.add_argument("--timeout", type=float, default=60.0)
    backend.add_argument("--retries", type=int, default=2)
    backend.add_argument("--max-tokens", type=int, default=200)
    backend.add_argument("--concurrent", type=int, default=4,
                         help="Concurrent HTTP requests. Match this to your server's --parallel slot count.")

    flt = p.add_argument_group("filtering")
    flt.add_argument("--min-overall", type=int, default=3,
                     help="Drop rows with overall score < this. 0 disables. (default: %(default)s)")
    flt.add_argument("--min-fluency-it", type=int, default=3,
                     help="Drop rows with fluency_it score < this. (default: %(default)s)")
    flt.add_argument("--keep-top-n", type=int, default=None,
                     help="After thresholding, keep only top-N by overall score.")
    flt.add_argument("--no-filter", action="store_true",
                     help="Score only; don't write filtered output (just scores + report).")

    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N rows (smoke testing)")
    p.add_argument("--dummy", action="store_true",
                   help="Use DummyJudge (offline) — for smoke testing without a server")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    log = (lambda *a, **k: None) if args.quiet else print

    in_path = Path(args.in_path)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".judged.jsonl")
    scores_path = Path(args.scores_out) if args.scores_out else out_path.with_suffix(".scores.jsonl")
    report_path = Path(args.report_out) if args.report_out else out_path.with_suffix(".report.json")

    log(f"=== Stage 5 — LLM-as-judge ===")
    log(f"  input        : {in_path}")
    log(f"  endpoint     : {args.endpoint}" if not args.dummy else "  backend      : DummyJudge (offline)")
    log(f"  concurrent   : {args.concurrent}")
    log(f"  scores file  : {scores_path}")

    rows = _load_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]
    log(f"  rows to score: {len(rows):,}")

    judge: Judge = DummyJudge() if args.dummy else OpenAICompatibleJudge(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        retries=args.retries,
    )

    t0 = time.time()
    scores = score_rows_resumable(
        rows, judge, scores_path=scores_path,
        concurrent=args.concurrent, show_progress=not args.quiet,
    )
    dt = time.time() - t0
    log(f"\n  wallclock: {dt:.1f}s ({len(rows) / max(dt, 1e-6):.1f} rows/sec)")

    # Aggregate report.
    n_scored = sum(1 for s in scores if s is not None)
    n_failed = len(scores) - n_scored
    overall_hist = Counter(s.overall for s in scores if s is not None)
    fluency_hist = Counter(s.fluency_it for s in scores if s is not None)
    log(f"\n  scored: {n_scored:,}   failed: {n_failed}")
    log("  overall histogram   : " + ", ".join(f"{k}:{overall_hist.get(k, 0):,}" for k in range(6)))
    log("  fluency_it histogram: " + ", ".join(f"{k}:{fluency_hist.get(k, 0):,}" for k in range(6)))

    if args.no_filter:
        n_written = 0
    else:
        valid = [(r, s) for r, s in zip(rows, scores) if s is not None]
        valid_rows, valid_scores = zip(*valid) if valid else ([], [])
        kept_rows, kept_scores = filter_by_score(
            list(valid_rows), list(valid_scores),
            min_overall=args.min_overall, min_fluency_it=args.min_fluency_it,
            keep_top_n=args.keep_top_n,
        )
        log(f"  after threshold (min_overall={args.min_overall}, min_fluency_it={args.min_fluency_it}): {len(kept_rows):,}")
        if args.keep_top_n is not None:
            log(f"  after top-{args.keep_top_n}: {len(kept_rows):,}")
        n_written = _write_jsonl(out_path, kept_rows)
        log(f"  wrote {n_written:,} rows -> {out_path}")

    full_report = {
        "args": {k: v for k, v in vars(args).items() if k not in ("in_path",) and not isinstance(v, Path)},
        "input": str(in_path),
        "input_rows": len(rows),
        "scored": n_scored,
        "failed": n_failed,
        "wallclock_s": round(dt, 2),
        "rows_per_sec": round(len(rows) / max(dt, 1e-6), 2),
        "overall_histogram": dict(overall_hist),
        "fluency_histogram": dict(fluency_hist),
        "filter_applied": not args.no_filter,
        "final_kept": n_written,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    log(f"  wrote report -> {report_path}")


if __name__ == "__main__":
    main()
