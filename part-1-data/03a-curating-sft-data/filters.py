"""
Filter cascade for Italian SFT data curation.

Stages, cheapest first:
  1. Structural        — length bounds, empties, copy bugs, n-gram repetition
  2. Language fidelity — fastText langid, Italian function-word ratio
  3. Artifact patterns — untranslated English entities, calques, leaked templates
  4. Dedup & diversity — MinHash near-dup, embedding-cluster downsample
  5. Model-based       — perplexity-band + LLM-as-judge (lives in judge.py)

Filters in stages 1–3 are PER-ROW: they accept a row dict and return either
None (keep) or a short string identifying the drop reason. Stages 4 are
BATCH operations that take a list of rows and return a filtered list +
a report of what got dropped.

The drop-reason convention makes the pipeline trivially auditable — at
the end of a run you have a histogram of why each filter rejected what
it rejected. That histogram is the point of the module: students should
look at it, sample 10 examples from each reason, and learn what their
dataset actually contains.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

# ----------------------------------------------------------------------
# Row schema and helpers
# ----------------------------------------------------------------------
# A "row" is a dict with a "messages" key holding a list of turns,
# each turn a dict with "role" ∈ {system, user, assistant} and "content".
# This matches the HuggingFace datasets convention used by ultrafeedback,
# no_robots, OpenItalianData, etc.

Row = dict
Filter = Callable[[Row], Optional[str]]


def normalize_row(raw) -> Optional[Row]:
    """Coerce a row from various source schemas into {messages: [...]} shape.

    Accepts:
      - a list of {role, content} dicts (OpenItalianData's apparent format)
      - a dict with a 'messages' key
      - a dict with 'conversations' key (sharegpt-style, with from/value)
      - prompt+response pairs ('prompt'/'response', 'instruction'/'output',
        'question'/'answer')
    Returns None if the row can't be coerced.
    """
    if isinstance(raw, list):
        msgs = [m for m in raw if isinstance(m, dict) and "role" in m and "content" in m]
        return {"messages": msgs} if msgs else None
    if not isinstance(raw, dict):
        return None
    if "messages" in raw and isinstance(raw["messages"], list):
        return {"messages": raw["messages"]}
    if "conversations" in raw and isinstance(raw["conversations"], list):
        msgs = []
        for turn in raw["conversations"]:
            if not isinstance(turn, dict):
                continue
            role_in = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("content")
            if role_in is None or content is None:
                continue
            role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role_in, role_in)
            msgs.append({"role": role, "content": content})
        return {"messages": msgs} if msgs else None
    for u_key, a_key in (("prompt", "response"), ("instruction", "output"), ("question", "answer")):
        if u_key in raw and a_key in raw:
            return {
                "messages": [
                    {"role": "user", "content": raw[u_key]},
                    {"role": "assistant", "content": raw[a_key]},
                ]
            }
    return None


def _turns_of(row: Row, role: str) -> list[str]:
    msgs = row.get("messages") or row.get("conversations") or []
    out = []
    for m in msgs:
        if m.get("role") == role:
            content = m.get("content")
            if isinstance(content, str):
                out.append(content)
    return out


def user_text(row: Row) -> str:
    return "\n".join(_turns_of(row, "user"))


def assistant_text(row: Row) -> str:
    return "\n".join(_turns_of(row, "assistant"))


def pair_text(row: Row) -> str:
    return user_text(row) + "\n" + assistant_text(row)


# ----------------------------------------------------------------------
# Stage 1 — Structural filters (free, pure-Python)
# ----------------------------------------------------------------------

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


@dataclass(frozen=True)
class StructuralConfig:
    min_user_words: int = 3
    min_assistant_words: int = 10
    max_assistant_words: int = 2000
    # If the assistant response is much shorter than the user prompt,
    # it is often a truncation artifact ("Sì.", "Non lo so.", etc.).
    # The ratio is words(response) / words(prompt). 0.0 disables.
    min_response_to_prompt_ratio: float = 0.25
    # N-gram repetition: if more than this fraction of n-grams in the
    # response are repeated, treat it as degenerate output.
    max_ngram_repetition: float = 0.30
    ngram_n: int = 3


def length_filter(row: Row, cfg: StructuralConfig = StructuralConfig()) -> Optional[str]:
    u_words = _word_count(user_text(row))
    a_words = _word_count(assistant_text(row))

    if u_words < cfg.min_user_words:
        return "user_too_short"
    if a_words < cfg.min_assistant_words:
        return "assistant_too_short"
    if a_words > cfg.max_assistant_words:
        return "assistant_too_long"
    if cfg.min_response_to_prompt_ratio > 0 and u_words > 0:
        if (a_words / u_words) < cfg.min_response_to_prompt_ratio:
            return "response_much_shorter_than_prompt"
    return None


def empty_or_copy_filter(row: Row, cfg: StructuralConfig = StructuralConfig()) -> Optional[str]:
    u, a = user_text(row).strip(), assistant_text(row).strip()
    if not u or not a:
        return "empty_field"
    # Common copy-bug: response is verbatim (or a long prefix of) the prompt.
    if u == a:
        return "response_equals_prompt"
    if len(a) >= 40 and a in u:
        return "response_substring_of_prompt"
    if len(u) >= 40 and u in a and len(a) < 2 * len(u):
        return "prompt_echoed_in_response"
    return None


def ngram_repetition_score(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are NOT unique. 0 = all unique; ~1 = degenerate."""
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1) - len(
        [c for c in counts.values() if c > 1]
    )
    # repeated = total occurrences minus one-per-distinct-repeated-gram
    return repeated / max(1, len(grams))


def repetition_filter(row: Row, cfg: StructuralConfig = StructuralConfig()) -> Optional[str]:
    if ngram_repetition_score(assistant_text(row), n=cfg.ngram_n) > cfg.max_ngram_repetition:
        return "assistant_repetitive"
    return None


# ----------------------------------------------------------------------
# Stage 2 — Italian language fidelity
# ----------------------------------------------------------------------

# A working set of Italian function words. ~50 entries that together
# typically make up ~25–40% of any normal Italian text. If a response
# has near-zero overlap, it is almost certainly not (fluent) Italian.
ITALIAN_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        # articles
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
        "l", "d", "all", "del", "della", "dei", "degli", "delle",
        "al", "alla", "ai", "agli", "alle",
        "dal", "dalla", "dai", "dagli", "dalle",
        "nel", "nella", "nei", "negli", "nelle",
        "sul", "sulla", "sui", "sugli", "sulle",
        # prepositions
        "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
        # conjunctions / connectives
        "e", "ed", "o", "od", "ma", "se", "che", "perché", "anche",
        "quando", "come", "mentre", "però", "quindi", "dunque",
        # pronouns / clitics
        "io", "tu", "lui", "lei", "noi", "voi", "loro",
        "mi", "ti", "si", "ci", "vi", "ne",
        "questo", "questa", "questi", "queste",
        "quello", "quella", "quelli", "quelle",
        # negation / affirmation
        "non", "sì", "no",
        # high-frequency verbs
        "è", "sono", "era", "erano", "stato", "essere",
        "ha", "hanno", "aveva", "avevano", "avere",
        "fa", "fanno", "fare",
    }
)


@dataclass(frozen=True)
class LangConfig:
    target_lang: str = "it"
    # fastText langid confidence threshold.
    min_lang_confidence: float = 0.50
    # Minimum fraction of tokens that must be Italian function words.
    # Around 0.10 in practice; below ~0.05 is almost always garbage.
    min_function_word_ratio: float = 0.05
    # Path to a locally-cached fastText lid.176 model. If None,
    # we'll attempt to download it on first use.
    fasttext_model_path: Optional[str] = None


_fasttext_model = None


def _get_fasttext():
    """Lazy-load the fastText lid.176 model. Returns None if unavailable."""
    global _fasttext_model
    if _fasttext_model is not None:
        return _fasttext_model
    try:
        import fasttext  # type: ignore
    except ImportError:
        return None
    # Look for a cached model alongside the module or in a standard cache dir.
    candidates = [
        Path.home() / ".cache" / "fasttext" / "lid.176.bin",
        Path.home() / ".cache" / "fasttext" / "lid.176.ftz",
        Path(__file__).parent / "lid.176.ftz",
    ]
    for p in candidates:
        if p.exists():
            _fasttext_model = fasttext.load_model(str(p))
            return _fasttext_model
    # Download a compressed variant (~1 MB) if missing.
    try:
        import urllib.request

        target = Path.home() / ".cache" / "fasttext" / "lid.176.ftz"
        target.parent.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        urllib.request.urlretrieve(url, target)
        _fasttext_model = fasttext.load_model(str(target))
        return _fasttext_model
    except Exception:
        return None


def detect_language(text: str) -> tuple[str, float]:
    """Return (lang_code, confidence). Falls back to ('?', 0.0) if unavailable.

    Note: `fasttext-wheel==0.9.2`'s Python wrapper calls
    `np.array(probs, copy=False)`, which raises under NumPy 2.x. We bypass
    it by calling the underlying C++ binding (`model.f.predict`) directly,
    which returns a list of (prob, label) tuples — same data, no NumPy
    conversion in the hot path.
    """
    model = _get_fasttext()
    if model is None:
        return ("?", 0.0)
    # fastText expects single-line input.
    text = text.replace("\n", " ").strip()
    if not text:
        return ("?", 0.0)
    try:
        predictions = model.f.predict(text, 1, 0.0, "strict")
    except Exception:
        # Fall back to the official wrapper; if it also fails (NumPy 2 bug,
        # truly broken model, etc.) degrade conservatively to no-op.
        try:
            labels, scores = model.predict(text, k=1)
        except Exception:
            return ("?", 0.0)
        if not labels:
            return ("?", 0.0)
        return (labels[0].replace("__label__", ""), float(scores[0]))
    if not predictions:
        return ("?", 0.0)
    prob, label = predictions[0]
    return (label.replace("__label__", ""), float(prob))


def langid_filter(row: Row, cfg: LangConfig = LangConfig()) -> Optional[str]:
    for field_name, text in [("user", user_text(row)), ("assistant", assistant_text(row))]:
        lang, conf = detect_language(text)
        if lang == "?":
            # Model unavailable — be conservative, do not drop.
            return None
        if lang != cfg.target_lang and conf >= cfg.min_lang_confidence:
            return f"{field_name}_not_{cfg.target_lang}"
    return None


def function_word_ratio(text: str, function_words: frozenset[str] = ITALIAN_FUNCTION_WORDS) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in function_words)
    return hits / len(words)


def function_word_filter(row: Row, cfg: LangConfig = LangConfig()) -> Optional[str]:
    # Only check the assistant text — the user prompt may be a translated
    # instruction tag like "Riassumi:" with very few function words.
    ratio = function_word_ratio(assistant_text(row))
    if ratio < cfg.min_function_word_ratio:
        return "low_italian_function_word_ratio"
    return None


# ----------------------------------------------------------------------
# Stage 3 — Translation-artifact patterns
# ----------------------------------------------------------------------

# Patterns that very strongly suggest the row is machine-translated
# from an English source and was not localized. None of these alone is
# proof; together they are a high-precision signal.
ENGLISH_ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # US-only units & currency forms (€/USD spelled out are fine in Italian).
    ("us_dollar_amount", re.compile(r"\$\s?\d")),
    ("fahrenheit", re.compile(r"\b\d{1,3}\s?°\s?F\b")),
    ("imperial_distance", re.compile(r"\b\d+\s?(mph|miles?|inches|feet|yards?)\b", re.IGNORECASE)),
    ("us_zip", re.compile(r"\b\d{5}(?:-\d{4})?\b(?!\s?(?:bytes|byte|caratteri|caratteri\.))")),
    # Untranslated English filler / refusal patterns.
    ("english_refusal_im_sorry", re.compile(r"\bI'?m\s+sorry\b", re.IGNORECASE)),
    ("english_refusal_i_cannot", re.compile(r"\bI\s+cannot\b", re.IGNORECASE)),
    ("english_as_an_ai", re.compile(r"\bas\s+an?\s+AI\b", re.IGNORECASE)),
    # Model brand mentions are a strong signal of "translated AI training data".
    ("model_brand_openai", re.compile(r"\bOpenAI\b")),
    ("model_brand_chatgpt", re.compile(r"\bChat\s?GPT\b", re.IGNORECASE)),
    ("model_brand_gpt_n", re.compile(r"\bGPT-\d\b")),
)

# Italian calques — phrasings that look like literal translations and
# are not idiomatic Italian. This list is intentionally conservative
# (false positives are worse than false negatives at this stage).
ITALIAN_CALQUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "fai/fa senso" (lit. "makes sense") instead of "ha senso".
    ("calque_fa_senso", re.compile(r"\b(fa|fai|faceva|farà)\s+senso\b", re.IGNORECASE)),
    # "come modello linguistico" — literal translation of "as a language model".
    ("calque_modello_linguistico", re.compile(r"\bcome\s+(un\s+)?modello\s+linguistico\b", re.IGNORECASE)),
    # "in ordine di" instead of "per/al fine di" — literal "in order to".
    ("calque_in_ordine_di", re.compile(r"\bin\s+ordine\s+(di|a)\b", re.IGNORECASE)),
    # "Sono solo un" (I'm just a) followed by AI/assistant terms.
    ("calque_sono_solo_un_ai", re.compile(r"\bsono\s+solo\s+un[oa]?\s+(IA|AI|assistente|modello|intelligenza\s+artificiale)\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class ArtifactConfig:
    drop_on_english_pattern: bool = True
    drop_on_calque: bool = True
    # If you'd rather warn than drop, set max_artifact_hits > 0 — rows are
    # only dropped when artifact count strictly exceeds this number.
    max_artifact_hits: int = 0


def english_artifact_filter(row: Row, cfg: ArtifactConfig = ArtifactConfig()) -> Optional[str]:
    if not cfg.drop_on_english_pattern:
        return None
    text = pair_text(row)
    hits: list[str] = []
    for name, pat in ENGLISH_ARTIFACT_PATTERNS:
        if pat.search(text):
            hits.append(name)
    if len(hits) > cfg.max_artifact_hits:
        return f"english_artifact:{hits[0]}"
    return None


def calque_filter(row: Row, cfg: ArtifactConfig = ArtifactConfig()) -> Optional[str]:
    if not cfg.drop_on_calque:
        return None
    text = pair_text(row)
    for name, pat in ITALIAN_CALQUES:
        if pat.search(text):
            return f"calque:{name}"
    return None


# ----------------------------------------------------------------------
# Pipeline orchestrator (stages 1-3)
# ----------------------------------------------------------------------


@dataclass
class FilterReport:
    """Counts of how many rows were dropped, indexed by drop reason."""

    drops: Counter = field(default_factory=Counter)
    kept: int = 0
    total: int = 0

    def record(self, reason: Optional[str]) -> bool:
        self.total += 1
        if reason is None:
            self.kept += 1
            return True
        self.drops[reason] += 1
        return False

    def summary(self) -> str:
        lines = [
            f"total seen:   {self.total:>10,}",
            f"kept:         {self.kept:>10,}  ({100*self.kept/max(1,self.total):5.1f}%)",
            f"dropped:      {self.total - self.kept:>10,}",
            "by reason:",
        ]
        for reason, n in self.drops.most_common():
            lines.append(f"  {reason:<40s} {n:>10,}")
        return "\n".join(lines)


def _process_chunk_worker(args: tuple) -> tuple[list[Row], dict[str, int]]:
    """Apply stage 1-3 filters to a chunk of rows. Runs in a worker process.

    Top-level (not a closure) so it's picklable. Filters are reconstructed
    inside the worker from the configs, so no lambdas cross the process
    boundary.
    """
    rows, structural, lang, artifact, skip = args
    skip = skip or {}
    filters: list[Filter] = [
        lambda r, _s=structural: empty_or_copy_filter(r, _s),
        lambda r, _s=structural: length_filter(r, _s),
        lambda r, _s=structural: repetition_filter(r, _s),
    ]
    if not skip.get("langid"):
        filters.append(lambda r, _l=lang: langid_filter(r, _l))
    if not skip.get("fnword"):
        filters.append(lambda r, _l=lang: function_word_filter(r, _l))
    if not skip.get("artifacts"):
        filters.append(lambda r, _a=artifact: english_artifact_filter(r, _a))
    if not skip.get("calques"):
        filters.append(lambda r, _a=artifact: calque_filter(r, _a))

    kept: list[Row] = []
    drops: Counter = Counter()
    for row in rows:
        reason: Optional[str] = None
        for f in filters:
            reason = f(row)
            if reason is not None:
                break
        if reason is None:
            kept.append(row)
        else:
            drops[reason] += 1
    return kept, dict(drops)


def run_per_row_filters_parallel(
    rows: Iterable[Row],
    structural: StructuralConfig,
    lang: LangConfig,
    artifact: ArtifactConfig,
    skip: Optional[dict] = None,
    n_workers: int = -1,
    chunk_size: int = 1000,
    progress: bool = False,
    total: Optional[int] = None,
    desc: str = "filtering",
) -> tuple[list[Row], FilterReport]:
    """Apply stage 1-3 filters in parallel across processes.

    Each worker lazy-loads its own fastText model on first langid call
    (~100ms one-time cost). On an N-core box you should see a ~(N-1)x
    speedup on stage 1-3 wallclock — the bottleneck is langid + regex,
    both CPU-bound. ``imap_unordered`` is used for max throughput, so
    output row order does not match input row order — this is fine
    because stages 4 (dedup) and 5 (judging) are order-invariant.

    For ``n_workers == 1`` this falls back to the sequential path with
    no Pool overhead.
    """
    import os
    from multiprocessing import get_context

    skip = skip or {}
    if n_workers < 1:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    if n_workers == 1:
        # Sequential fallback — reconstruct the filter list inline.
        filters: list[Filter] = [
            lambda r, _s=structural: empty_or_copy_filter(r, _s),
            lambda r, _s=structural: length_filter(r, _s),
            lambda r, _s=structural: repetition_filter(r, _s),
        ]
        if not skip.get("langid"):
            filters.append(lambda r, _l=lang: langid_filter(r, _l))
        if not skip.get("fnword"):
            filters.append(lambda r, _l=lang: function_word_filter(r, _l))
        if not skip.get("artifacts"):
            filters.append(lambda r, _a=artifact: english_artifact_filter(r, _a))
        if not skip.get("calques"):
            filters.append(lambda r, _a=artifact: calque_filter(r, _a))
        return run_per_row_filters(
            rows, filters, progress=progress, total=total, desc=desc,
        )

    def _chunked():
        chunk: list[Row] = []
        for r in rows:
            chunk.append(r)
            if len(chunk) >= chunk_size:
                yield (chunk, structural, lang, artifact, skip)
                chunk = []
        if chunk:
            yield (chunk, structural, lang, artifact, skip)

    report = FilterReport()
    kept_all: list[Row] = []

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=total, desc=desc, unit="row")
        except ImportError:
            pass

    # Spawn (not fork) for cross-platform safety + avoiding any fork-unsafe
    # state in imported modules. fork would skip the re-import overhead but
    # the savings are negligible vs the filter compute time.
    ctx = get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for kept_chunk, drops_chunk in pool.imap_unordered(
            _process_chunk_worker, _chunked(), chunksize=1
        ):
            kept_all.extend(kept_chunk)
            chunk_total = len(kept_chunk) + sum(drops_chunk.values())
            report.kept += len(kept_chunk)
            report.total += chunk_total
            for reason, n in drops_chunk.items():
                report.drops[reason] += n
            if pbar is not None:
                pbar.update(chunk_total)
                pbar.set_postfix(
                    kept=report.kept,
                    drop_rate=f"{100*(1-report.kept/max(1,report.total)):.1f}%",
                )

    if pbar is not None:
        pbar.close()

    return kept_all, report


def run_per_row_filters(
    rows: Iterable[Row],
    filters: list[Filter],
    report: Optional[FilterReport] = None,
    progress: bool = False,
    total: Optional[int] = None,
    desc: str = "filtering",
) -> tuple[list[Row], FilterReport]:
    """Apply a list of per-row filters in order, short-circuiting on first reject.

    Pass ``progress=True`` to show a tqdm bar (soft dep — silently degrades if
    tqdm is not installed). If you know the source size, pass ``total`` so the
    bar shows an ETA.
    """
    if report is None:
        report = FilterReport()
    iterator: Iterable = rows
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(rows, total=total, desc=desc, unit="row")
        except ImportError:
            pass
    kept: list[Row] = []
    for row in iterator:
        reason = None
        for f in filters:
            reason = f(row)
            if reason is not None:
                break
        if report.record(reason):
            kept.append(row)
        # Periodically refresh tqdm postfix with kept-rate so the user sees
        # the filter actually doing something useful, not just iterating.
        if progress and isinstance(iterator, object) and report.total % 5000 == 0:
            try:
                iterator.set_postfix(kept=report.kept, drop_rate=f"{100*(1-report.kept/max(1,report.total)):.1f}%")  # type: ignore[attr-defined]
            except Exception:
                pass
    return kept, report


# ----------------------------------------------------------------------
# Stage 4 — Dedup & diversity (batch operations)
# ----------------------------------------------------------------------


def _load_and_truncate_jsonl(path: Path, n_lines: int) -> list[Row]:
    """Read the first ``n_lines`` JSONL rows from ``path`` and truncate the file
    to exactly those lines.

    Used on resume: the progress marker promises ``n_lines`` durable kept rows,
    but a crash may have left extra (possibly half-written) lines appended after
    the last marker. We drop everything past ``n_lines`` so the on-disk file and
    the in-memory state agree before we continue.
    """
    rows: list[Row] = []
    with path.open("rb+") as f:
        offset = 0
        count = 0
        for raw in f:
            if count >= n_lines:
                break
            offset += len(raw)
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
            count += 1
        f.truncate(offset)
        f.flush()
        os.fsync(f.fileno())
    return rows


def minhash_dedup(
    rows: list[Row],
    threshold: float = 0.85,
    num_perm: int = 128,
    shingle_size: int = 5,
    progress: bool = False,
    desc: str = "minhash dedup",
    checkpoint_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    checkpoint_every: int = 50_000,
) -> tuple[list[Row], int]:
    """Drop rows whose USER prompt is a near-duplicate of an earlier row's user prompt.

    Returns (kept_rows, n_dropped). Uses MinHash + LSH; for 1M+ rows this is the
    only practical way to get O(N) approximate Jaccard.

    Sequential by design — LSH insert order matters for which of N near-dups
    is the one kept. ``progress=True`` shows a tqdm bar with running kept /
    dropped counts.

    **Crash-resume.** When ``checkpoint_path`` and ``state_path`` are both given,
    kept rows are streamed to ``checkpoint_path`` as they're found, and every
    ``checkpoint_every`` input rows a tiny ``state_path`` JSON marker is written
    (``{processed, kept, dropped}``) after the rows file is fsync'd. If the
    process dies mid-stage (WSL drop, OOM, reboot), calling again with the same
    paths resumes from the last marker instead of restarting the whole pass:

      1. the rows file is truncated back to the ``kept`` lines the marker
         promised (discarding any partial tail), then reloaded;
      2. the LSH index is rebuilt by re-MinHashing those kept rows — LSH queries
         are order-independent given the same inserted set, so the dedup
         decisions for the remaining input are identical to a crash-free run;
      3. iteration continues from input row ``processed``.

    The caller owns lifecycle: it deletes ``state_path`` once this returns
    cleanly (the rows file then stands as the completed stage checkpoint).
    """
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except ImportError as e:
        raise RuntimeError("datasketch is required for minhash_dedup (pip install datasketch)") from e

    def _minhash(text: str):
        """MinHash of ``text``'s shingles, or None if it has no shingles."""
        text = re.sub(r"\s+", " ", text.lower())
        if not text:
            return None
        if len(text) <= shingle_size:
            shingles = {text}
        else:
            shingles = {text[i : i + shingle_size] for i in range(len(text) - shingle_size + 1)}
        if not shingles:
            return None
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    resumable = checkpoint_path is not None and state_path is not None

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Row] = []
    dropped = 0
    start_index = 0
    fh = None  # binary append handle for the streamed checkpoint

    def _write_state(processed: int) -> None:
        """Atomically persist the progress marker after fsync'ing the rows file."""
        fh.flush()
        os.fsync(fh.fileno())
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as sf:
            json.dump({"processed": processed, "kept": len(kept), "dropped": dropped}, sf)
            sf.flush()
            os.fsync(sf.fileno())
        os.replace(tmp, state_path)

    if resumable and state_path.exists() and checkpoint_path.exists():
        # --- Resume path ---
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_index = int(state.get("processed", 0))
        dropped = int(state.get("dropped", 0))
        kept_count = int(state.get("kept", 0))
        if progress:
            print(
                f"  [dedup] resume: {start_index:,}/{len(rows):,} input rows already processed "
                f"({kept_count:,} kept, {dropped:,} dropped); rebuilding LSH index",
                flush=True,
            )
        kept = _load_and_truncate_jsonl(checkpoint_path, kept_count)
        rebuild_iter: Iterable = enumerate(kept)
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                rebuild_iter = tqdm(rebuild_iter, total=len(kept), desc="dedup rebuild", unit="row")
            except ImportError:
                pass
        for j, row in rebuild_iter:
            m = _minhash(user_text(row))
            if m is not None:
                lsh.insert(f"k_{j}", m)
        fh = checkpoint_path.open("ab")
    elif resumable:
        fh = checkpoint_path.open("wb")

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=len(rows), initial=start_index, desc=desc, unit="row")
        except ImportError:
            pass

    try:
        for i in range(start_index, len(rows)):
            row = rows[i]
            m = _minhash(user_text(row))
            keep = False
            if m is None:
                keep = True
            elif lsh.query(m):
                dropped += 1
            else:
                lsh.insert(f"row_{i}", m)
                keep = True

            if keep:
                kept.append(row)
                if fh is not None:
                    fh.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))

            if resumable and (i + 1) % checkpoint_every == 0:
                _write_state(i + 1)

            if pbar is not None:
                pbar.update(1)
                if (i + 1) % 1000 == 0:
                    pbar.set_postfix(kept=len(kept), dropped=dropped)

        if resumable:
            # Final marker so a resume after a clean run is a no-op rather than
            # a full re-pass (the caller will normally delete it on success).
            _write_state(len(rows))
    finally:
        if fh is not None:
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()

    if pbar is not None:
        pbar.set_postfix(kept=len(kept), dropped=dropped)
        pbar.close()
    return kept, dropped


def diversity_downsample(
    rows: list[Row],
    target_size: int,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    n_clusters: Optional[int] = None,
    seed: int = 0,
    batch_size: int = 128,
    show_progress: bool = False,
) -> list[Row]:
    """Embed user prompts, cluster, sample uniformly across clusters.

    The point is to avoid letting any one task type (e.g. data-to-text) dominate
    just because it was over-represented in the source dataset. If target_size
    is >= len(rows), returns rows unchanged.
    """
    if target_size >= len(rows):
        return list(rows)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers and scikit-learn are required for diversity_downsample"
        ) from e
    import numpy as np
    import time as _time

    if n_clusters is None:
        # Heuristic: roughly sqrt(target_size), capped to keep clustering tractable.
        n_clusters = min(max(8, int(target_size**0.5)), 256)

    if show_progress:
        print(f"  [diversity] loading embedding model: {embedding_model}", flush=True)
    _t0 = _time.time()
    model = SentenceTransformer(embedding_model)
    if show_progress:
        print(f"  [diversity] model loaded in {_time.time()-_t0:.1f}s", flush=True)

    texts = [user_text(r) for r in rows]
    if show_progress:
        print(f"  [diversity] embedding {len(texts):,} prompts (batch={batch_size})", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if show_progress:
        print(f"  [diversity] clustering {len(embeddings):,} vectors into {n_clusters} clusters", flush=True)
    _t0 = _time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, batch_size=min(1024, len(rows)), n_init=3
    )
    labels = kmeans.fit_predict(embeddings)
    if show_progress:
        print(f"  [diversity] kmeans done in {_time.time()-_t0:.1f}s; sampling target_size={target_size:,}", flush=True)

    # Per-cluster quotas: proportional to cluster size with a minimum floor so
    # small clusters aren't wiped out. Then trim by random sampling within
    # each cluster.
    rng = np.random.default_rng(seed)
    cluster_indices: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        cluster_indices.setdefault(int(lab), []).append(i)

    sizes = {c: len(ix) for c, ix in cluster_indices.items()}
    total = sum(sizes.values())
    # Allocate at least 1 per non-empty cluster, then proportionally distribute the rest.
    quotas = {c: max(1, int(target_size * (n / total))) for c, n in sizes.items()}
    overflow = sum(quotas.values()) - target_size
    if overflow > 0:
        # Trim from the largest clusters first.
        for c in sorted(quotas, key=lambda c: -sizes[c]):
            if overflow <= 0:
                break
            take = min(overflow, quotas[c] - 1)
            quotas[c] -= take
            overflow -= take

    picked: list[int] = []
    for c, ix in cluster_indices.items():
        k = min(quotas.get(c, 0), len(ix))
        if k <= 0:
            continue
        picked.extend(rng.choice(ix, size=k, replace=False).tolist())

    picked.sort()
    return [rows[i] for i in picked]


# ----------------------------------------------------------------------
# Default filter stack — what `prepare.py` uses by default.
# ----------------------------------------------------------------------


def default_per_row_filters(
    structural: StructuralConfig = StructuralConfig(),
    lang: LangConfig = LangConfig(),
    artifact: ArtifactConfig = ArtifactConfig(),
) -> list[Filter]:
    return [
        # Stage 1 — structural (fast).
        lambda r: empty_or_copy_filter(r, structural),
        lambda r: length_filter(r, structural),
        lambda r: repetition_filter(r, structural),
        # Stage 2 — language fidelity.
        lambda r: langid_filter(r, lang),
        lambda r: function_word_filter(r, lang),
        # Stage 3 — translation artifacts.
        lambda r: english_artifact_filter(r, artifact),
        lambda r: calque_filter(r, artifact),
    ]


if __name__ == "__main__":
    # Tiny smoke: invent a few rows, run the per-row cascade, print the report.
    rows = [
        {"messages": [
            {"role": "user", "content": "Riassumi il seguente testo in due frasi."},
            {"role": "assistant", "content": "Il testo descrive in modo conciso l'argomento principale e la sua importanza per il lettore."},
        ]},
        {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ]},
        {"messages": [
            {"role": "user", "content": "Spiega cosa fa Python."},
            {"role": "assistant", "content": "I'm sorry, as an AI language model, I cannot answer that."},
        ]},
        {"messages": [
            {"role": "user", "content": "Quanto costa?"},
            {"role": "assistant", "content": "Il prezzo è di $25 negli Stati Uniti, codice postale 90210."},
        ]},
        {"messages": [
            {"role": "user", "content": "Genera una frase."},
            {"role": "assistant", "content": "Genera una frase."},
        ]},
    ]
    filters = default_per_row_filters()
    kept, report = run_per_row_filters(rows, filters)
    print(report.summary())
    print(f"\nkept {len(kept)}/{len(rows)} rows")
