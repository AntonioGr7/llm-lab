# Module 03a — Curating SFT Data

> **EXTRA / OPTIONAL module.** Not on the critical path of the course. Take this module if you want to understand why your SFT data is at least as important as your training loop — and what to actually do about it.
>
> Part of [Part 1 — Data](../). Reading time: ~60 minutes. Compute cost: $0 for stages 1–4 (CPU-only). Stage 5 (LLM-as-judge) is optional and costs ~$5–15 with a small API, or $0 with a local GPU.

## The thesis

**100k high-quality SFT examples outperform 2M noisy ones.** This is no longer a hot take — LIMA showed it in 2023, Tülu 3's data ablations confirmed it at scale in 2024, every frontier post-training recipe since has reinforced it. The interesting decision in SFT data work is *what to throw away*, not what to keep.

This module walks through a complete curation pipeline on a real public dataset — [DeepMount00/OpenItalianData](https://huggingface.co/datasets/DeepMount00/OpenItalianData), 2.14M rows of machine-translated Italian SFT data. Most large public SFT datasets for languages other than English have the same shape: aggregations of English instruction data run through a translation system. They contain real signal, but also untranslated chunks, US-centric entities, calques no native speaker would produce, and brand mentions of the original model. The cascade in `filters.py` is precision-oriented removal of all of that.

The output is a clean ~50–150k subset, pushed to HuggingFace under your account, ready to feed an SFT trainer like Module 15.

## Why this module is optional

Two reasons:

1. **Module 15 (SFT) doesn't depend on this.** It uses `HuggingFaceH4/no_robots` (English, 10k clean rows) as its canonical dataset. You can complete the entire course without this module.
2. **The lesson is methodological.** The filters themselves are language-specific (the calque blocklist, the function-word set), but the *cascade pattern* — cheap heuristics first, model-based scoring last, drop-reason reporting throughout — is the part that generalizes. If you internalize the methodology, you can rewrite this module for any language or domain.

## What you'll be able to do at the end

- Take any aggregated machine-translated SFT dataset and produce a high-precision filtered subset, with full audit trail of what was dropped and why.
- Recognize the failure modes of automatic language identification, MinHash dedup, and embedding-cluster diversity — and know what to do when each one over- or under-fires.
- Build a held-out eval set in the target language and use it to measure whether your curation actually moved quality.
- Publish a curated dataset to HuggingFace with a proper dataset card and pinned revision.

## The source dataset

`DeepMount00/OpenItalianData`, ~2.14M rows. Each row is a single (user, assistant) pair in HuggingFace `messages` format:

```json
[
  {"role": "user", "content": "Genera una frase di circa quindici parole che descriva questi dati: Midsummer House ..."},
  {"role": "assistant", "content": "Il ristorante Midsummer House offre cucina cinese a prezzi moderati ..."}
]
```

A few things are obvious from reading samples:

- The provenance is *clearly* translated English instruction data — `Midsummer House`, `All Bar One` (a UK pub chain), US dollar amounts in restaurant prompts, etc. The dataset card upstream is honest about this.
- Many task families are over-represented — data-to-text generation (the example above is the E2E NLG dataset), simple QA, basic summarization. A naive sample of 100k rows would be 30%+ data-to-text, which is not what you want for an SFT mix.
- Translation quality is mostly fine but not uniform. Some rows have visible English residue. Some have idiomatic Italian; many have calques.

The cascade is designed against exactly these problems.

## The 5-stage cascade

Cheapest first. Each stage cuts the survivors before you pay for the next.

### Stage 1 — Structural

Pure-Python heuristics, ~10 µs per row, no model loads. Run first.

| Filter | What it catches |
|---|---|
| `empty_or_copy_filter` | Empty fields; response equals prompt (copy-bug); response is a substring of the prompt. |
| `length_filter` | Too-short responses (often refusals or truncations), too-long responses, responses much shorter than their prompts. |
| `repetition_filter` | Degenerate output — the same n-gram repeated dozens of times (sampling failures from the source model). |

The interesting knob here is `min_response_to_prompt_ratio`. If a 200-word prompt gets a 20-word response, it's *probably* truncated. But not always — `"Riassumi in 10 parole: <long text>"` is legitimate. The filter is precision-tunable via the ratio; the default 0.25 errs toward keeping borderline cases.

### Stage 2 — Language fidelity

Catches partially-untranslated rows. Two filters, both cheap.

**fastText `lid.176`** is Facebook's tiny (1 MB) language ID model. It's the single highest-yield filter on machine-translated data: if even part of the row didn't translate, the langid signal flips. The model auto-downloads to `~/.cache/fasttext/` on first use.

**Italian function-word ratio** is a model-free fallback. Any fluent Italian text has 25–40% function words (`il, di, che, è, non, …`); English text has ~0%. Even when langid is confident the text is Italian, an unusually low function-word ratio flags rows that are technically Italian but read like "translator output" (heavy on content words, light on connective tissue).

These two together catch ~80% of the obvious noise floor in a translated dataset like OpenItalianData.

### Stage 3 — Translation-artifact patterns

After langid, the surviving rows are syntactically Italian — but they can still be machine-translated junk. Two regex-based filters catch the highest-precision tells.

**English-artifact patterns** — untranslated filler (`I'm sorry`, `as an AI`), US-only entities (`$25`, ZIP codes, `°F`), brand mentions of the source model (`OpenAI`, `ChatGPT`, `GPT-4`). Each is a strong signal that the row is translated English; together they form a precision filter with very low false-positive rate.

**Italian calque blocklist** — phrasings that no native speaker would produce. `fa senso` (lit. "makes sense") instead of `ha senso`. `come modello linguistico` (lit. "as a language model"). `in ordine di` (lit. "in order to"). The shipped list is intentionally short — extending it is the canonical native-speaker exercise:

> 1. Run stages 1–3 on a slice of the dataset.
> 2. Sample 200 surviving rows. Read them.
> 3. Note every phrase that sounds "translated" to your ear.
> 4. Add to the blocklist in `filters.py`.
> 5. Repeat.

This is exactly how production data curation works at frontier labs, and there's no shortcut.

### Stage 4 — Dedup & diversity

After per-row filters, two remaining problems with the survivors:

**MinHash near-duplicate removal** on prompts. Aggregated datasets are duplicate-heavy: the same prompt template repeats with different fillers. MinHash + LSH gives O(N) approximate Jaccard on millions of rows. On OpenItalianData, expect 30–50% reduction at threshold ≈ 0.85.

**Embedding-cluster diversity downsampling**. Without this, one or two task types (the over-represented data-to-text family, for instance) will dominate your final set just because they dominated the source. We embed prompts with [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), cluster with MiniBatchKMeans, and sample uniformly across clusters. The result is a final 50–150k set with balanced task coverage.

### Stage 5 — LLM-as-judge (optional)

The cheap stages cut the noise floor for free. The last quality lift comes from asking a strong model to actually read each (prompt, response) pair and score it. `judge.py` implements this:

- A small Italian-capable instruct model scores each row on `fluency_it`, `relevance`, `correctness`, and `overall` (0–5 each), plus a free-text rationale.
- The scoring prompt is in Italian and requests structured JSON output (with `response_format={"type": "json_object"}` enforced when the server supports it — llama.cpp does).
- `filter_by_score` applies thresholds and/or top-N's the result.

Stage 5 is the most expensive step. On 150k rows it's ~$5–15 via API, or a few GPU-hours locally. It's optional because stages 1–4 already produce something usable; stage 5 is the difference between "usable" and "competitive".

#### Recommended setup: llama.cpp server with a quantized GGUF

For small-VRAM machines (4–8 GB) this is the most ergonomic path. The model runs in a separate process, so there's no Python/CUDA conflict with sentence-transformers or anything else in the pipeline, and the OpenAI-compatible HTTP API decouples judging from the rest of the code.

**1. Build llama.cpp + grab a Q4_K_M GGUF.**

```bash
# Build llama.cpp (with CUDA if your card supports it).
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j

# Grab a Q4_K_M quant of Qwen3-1.7B-Instruct (~1.1 GB).
huggingface-cli download Qwen/Qwen3-1.7B-Instruct-GGUF \
    qwen3-1.7b-instruct-q4_k_m.gguf --local-dir ./models
```

**2. Launch the server.**

```bash
./build/bin/llama-server \
    -m ./models/qwen3-1.7b-instruct-q4_k_m.gguf \
    -c 4096 \           # context length (judge prompts are <2k tokens)
    -ngl 99 \           # offload all layers to GPU
    --parallel 4 \      # 4 concurrent decoding slots
    --host 0.0.0.0 \
    --port 8080
```

On an RTX 2050 (4 GB): Q4_K_M weights ~1.1 GB + KV cache for 4 parallel slots ~1.5 GB + framework overhead → comfortable. Throughput ~30–60 rows/min depending on prompt length.

**3. Run the judge against the server.**

```bash
python judge.py \
    --in italian-sft-curated.jsonl \
    --out italian-sft-judged.jsonl \
    --endpoint http://localhost:8080/v1 \
    --concurrent 4 \           # MATCH the server's --parallel
    --min-overall 3 \
    --keep-top-n 100000
```

Outputs:

- `italian-sft-judged.jsonl` — final filtered dataset (apply this to your SFT trainer).
- `italian-sft-judged.scores.jsonl` — every row's score, written incrementally. **This file IS the resume mechanism**: if the run crashes after 80k rows, re-running the same command picks up where you left off (only the missing scores are computed).
- `italian-sft-judged.report.json` — score histograms, wallclock, config used.

**Concurrency tip.** Set `--concurrent` to match the server's `--parallel`. If the client sends 8 concurrent requests but the server only has 4 slots, the extra 4 queue server-side — no harm, just no benefit. Setting client concurrency *lower* than server slots wastes throughput.

**Reasoning ("thinking") models.** If you serve a reasoning model (Qwen3 / Qwen3.5, DeepSeek-R1, etc.), the judge **disables the `<think>` block by default** — it sends `chat_template_kwargs={"enable_thinking": false}`. This is not optional cosmetics: with thinking on, the model spends the entire `--max-tokens` budget reasoning and returns an *empty* answer, which parses as malformed and silently drops every row. You want a terse JSON verdict here, not a chain of thought. If you point `--endpoint` at a hosted API that rejects the `chat_template_kwargs` field (plain OpenAI), pass `--enable-thinking` to omit it. The judge also reads the `reasoning_content` channel as a fallback if `content` ever comes back empty.

#### Other backends

- **vLLM's OpenAI server**: works identically. Run `vllm serve <model>` and point `--endpoint` at it.
- **Ollama**: same. Endpoint is usually `http://localhost:11434/v1`.
- **API providers (Anthropic, OpenAI, OpenRouter, Together)**: set `--endpoint` to the provider's `/v1` URL, `--api-key` to your token, `--model` to a real model name (e.g. `claude-haiku-4-5-20251001`, `gpt-4o-mini`).
- **In-process transformers**: import `LocalJudge` from `judge.py` and pass it to `score_rows_resumable` directly. Use when you don't want a separate server process (less ergonomic — eats Python GPU memory you may want elsewhere).

#### Reasoning about threshold vs top-N

| Strategy | When to use |
|---|---|
| Threshold only (`--min-overall 4`) | When you trust score calibration and want a quality floor. Final count is whatever the data gives you. |
| Top-N only (`--keep-top-n 100000` with `--min-overall 0`) | When you want a fixed dataset size regardless of quality distribution. Useful for reproducibility. |
| Both (recommended) | Threshold establishes a floor; top-N picks the cream from above it. The most defensible default. |

The `Judge` is a Protocol — to plug in a custom backend, implement `__call__(row) -> JudgeScore` and pass your instance to `score_rows_resumable`. The methodology generalizes far beyond Italian and far beyond SFT.

## The held-out eval set

`eval_probes.jsonl` ships with this module — 150 candidate Italian prompts across 15 task categories (factual QA, summarization, translation, creative writing, reasoning, code, roleplay, Italian-cultural questions, register variation, refusal, clarification, …). Each prompt has a `category` tag and `notes` for the curator.

**The probe set is candidate-quality, not final-quality.** It was generated by an LLM needs revision before being used as an eval. Curate down to 50–100 entries you trust, fix any awkward phrasings, and ship the result as part of your final pipeline.

**Why a held-out eval set is non-negotiable for Italian SFT.** English benchmarks (MMLU, IFEval, …) don't tell you whether your Italian model speaks fluent Italian — they tell you whether it still knows the English answers. The only honest signal is: ask Italian questions, read the Italian answers. The probe set is what you ask. The grading is what you read.

## Running the pipeline

Install dependencies (note: not in the top-level `requirements.txt` — this module has its own):

```bash
cd part-1-data/03a-curating-sft-data
pip install -r requirements.txt
```

A small smoke run, no network, no heavy deps:

```bash
python prepare.py --local-jsonl=/path/to/tiny_fixture.jsonl \
    --skip-langid --skip-dedup --skip-diversity \
    --out=/tmp/smoke.jsonl
```

The full thing on OpenItalianData, end-to-end:

```bash
python prepare.py \
    --dataset DeepMount00/OpenItalianData \
    --target-size 150000 \
    --minhash-threshold 0.85 \
    --workers 8 \
    --out italian-sft-curated.jsonl
```

Outputs:

- `italian-sft-curated.jsonl` — the curated rows.
- `italian-sft-curated.report.json` — the drop-reason histogram and exact configs used.

### Crash-resume

A full curation run on 2.14M rows is long enough (~10–30 minutes wallclock) that a laptop reboot, a WSL crash, or an OOM kill mid-pipeline becomes a real concern. The pipeline writes checkpoints so a crash doesn't throw away work — both **between** stages and, for the slow MinHash pass, **within** stage 4a.

What gets written, and when:

| When | Files |
|---|---|
| After stage 1-3 (per-row filters) completes | `<out>.stage_1_3.jsonl` (kept rows) + `<out>.stage_1_3.report.json` (drop counts) |
| *During* stage 4a (every `--dedup-checkpoint-every` rows) | `<out>.stage_4a.jsonl` (kept rows, **streamed**) + `<out>.stage_4a.progress.json` (marker: `{processed, kept, dropped}`) |
| After stage 4a (MinHash dedup) completes | `<out>.stage_4a.report.json` (and the progress marker is deleted) |
| After stage 4b + final write | `<out>` itself + `<out>.report.json` |

**Why stage 4a needs intra-stage checkpointing.** Stage 4a is a single-core sequential pass over the ~1.6M survivors of stages 1-3 — minutes of work, and the most likely place to be interrupted. The other stages either parallelize (1-3) or are short relative to it. So 4a streams its kept rows to disk as it goes and drops a small progress marker every `--dedup-checkpoint-every` input rows (default 50k); the rows file is `fsync`'d before each marker is written, and the marker is replaced atomically. A crash resumes from the **last marker**, not the start of the stage: the rows file is truncated back to the rows the marker promised (discarding any half-written tail), the LSH index is rebuilt by re-hashing those kept rows — LSH queries are order-independent, so this reproduces the crash-free result exactly — and iteration continues from where it stopped.

On re-launch with the same `--out` path:

- If `<out>.stage_4a.report.json` exists → stage 4a finished; skip stages 1-3 AND 4a, load from the 4a checkpoint, run only 4b.
- Else if `<out>.stage_4a.progress.json` exists → stage 4a was interrupted mid-pass; load the 1-3 checkpoint as input and **resume 4a from the marker**.
- Else if `<out>.stage_1_3.jsonl` exists → skip stage 1-3, load from the 1-3 checkpoint, run 4a + 4b.
- Else → fresh run.

The `resumed_from` field in `<out>.report.json` records which path was taken (`scratch` / `stage_1_3` / `stage_4a`).

Successful runs **delete the intermediate checkpoints** by default — they're large (potentially hundreds of MB), and the final output is what you want. Pass `--keep-checkpoints` to retain them (useful when iterating on stage 4b thresholds — you can re-run 4b from the 4a checkpoint without redoing 1-3 or 4a).

Force a fresh run with `--no-resume`. Use this when you've changed stage 1-3 thresholds, the calque blocklist, or the langid model — the checkpoint reflects the OLD config and re-using it would silently skip your changes.

### Scaling across cores

Stage 1-3 is the long pole — 2.14M rows × ~7 filters × langid is single-core CPU-bound by default and runs in ~15 minutes. Multiprocessing via `--workers N` brings it down to a few minutes. Each worker lazy-loads its own `lid.176.ftz` (one-time ~100ms cost); the filter functions themselves are pure, so parallelism is essentially free.

| Machine | Suggested |
|---|---|
| Laptop (4-8 cores) | `--workers 4` or `-1` (auto = `cpu_count - 1`) |
| 16-core workstation | `--workers 15` |
| 64-core server | `--workers 63 --chunk-size 5000` |
| 128-core fat box | `--workers 127 --chunk-size 10000` |

Knobs:

- `--workers N` — process-pool size (default: `-1`, meaning `cpu_count() - 1`). Pass `1` for single-threaded (useful for apples-to-apples timing comparisons or debugging).
- `--chunk-size N` — rows dispatched per worker call (default: 1000). Bigger = less IPC overhead, worse load balance. For 64+ cores bump to ≥5000.

**Don't over-shoot.** At very high worker counts the single-threaded HF dataset iteration + chunk dispatch becomes the bottleneck. If CPU utilization plateaus at, say, 30 cores even with `--workers 64`, that's the dispatcher saturating, not the filter compute. Increasing `--chunk-size` further can claw back some headroom by reducing IPC overhead.

Stages 4 (MinHash dedup) and 5 (LLM-judge) stay single-process — MinHash LSH insert order matters, and the embedding model + judge already use multi-thread internally via PyTorch.

## What to do with the report

`report.json` is the artifact you should actually look at. It records:

- Total rows seen.
- Rows kept.
- Drop counts per reason (`assistant_too_short`, `english_artifact:us_dollar_amount`, `calque:fa_senso`, …).
- Stage 4 effects (`minhash_near_duplicate`, `diversity_downsample`).
- Full config (every threshold, every seed).

**What surprised you about this histogram?**

- If `assistant_too_short` is 40% of the dataset, the source is dominated by truncated responses — you may want a lower threshold, or you may want to *use a different source*.
- If `english_artifact:us_dollar_amount` is 10k+ rows, the upstream is more US-centric than you thought — flag this in the dataset card.
- If `calque:*` is small but you can read 50 rows and spot 10 calques, your blocklist is too short — go extend it.

That's the meta-skill: the drop report is data, and you debug your filter by reading it.

## Publishing to HuggingFace

After a satisfactory run, push the dataset:

```bash
# 1. Log in once.
huggingface-cli login

# 2. Create the repo and upload.
huggingface-cli repo create italian-sft-curated --type dataset
huggingface-cli upload <your-username>/italian-sft-curated \
    italian-sft-curated.jsonl --repo-type=dataset
huggingface-cli upload <your-username>/italian-sft-curated \
    dataset_card.md README.md --repo-type=dataset

# 3. Pin a revision so consumers can reproduce.
huggingface-cli repo tag <your-username>/italian-sft-curated v1.0
```

The `dataset_card.md` in this directory is the template — fill in the placeholders (`<N>`, `<date>`, `<your-username>`) before pushing. The card documents provenance, the curation pipeline, limitations, and licensing — all required for a reusable artifact.

## Module layout

```
03a-curating-sft-data/
├── README.md            # this file
├── requirements.txt     # module-specific deps (fasttext, datasketch, sentence-transformers, sklearn)
├── filters.py           # stages 1-4 filter implementations
├── prepare.py           # end-to-end pipeline orchestrator (CLI)
├── judge.py             # stage 5 LLM-as-judge (Protocol + LocalJudge + DummyJudge)
├── eval_probes.jsonl    # 150 candidate Italian probes for your held-out eval
├── dataset_card.md      # HF dataset card template
├── notebook.ipynb       # walkthrough: each stage on a tiny fixture, with drop reports
└── tests/
    └── test_filters.py  # 28 offline correctness tests (no GPU, no network)
```

## Stretch goals

Things worth doing if you want to push further:

- **Extend the calque blocklist.** Sample 500 survivors, read, add to `ITALIAN_CALQUES` in `filters.py`. Run again. Repeat until you can read 200 random rows without flagging any calques.
- **Replace `lid.176`** with a finer-grained Italian-vs-other classifier — `lid.176` is great at "is it Italian" but bad at "is this Romanian/Spanish that looks like Italian". A small fine-tuned model on Italian Wikipedia helps.
- **Reverse-translate sanity check.** For 1k random rows, translate the Italian response back to English with a strong MT model and compare against the original English source. Rows where the round-trip drifts heavily are likely poorly translated.
- **Cross-encoder reranking** instead of LLM-judge — a `cross-encoder/ms-marco-MiniLM-L-6-v2`-style model fine-tuned on Italian prompt-response quality. Cheaper than LLM-judge, harder to set up.
- **Multi-turn handling.** This module assumes single-turn pairs (OpenItalianData's shape). For multi-turn datasets the filter signal on later turns is weaker — extend the data shape and re-apply.

## Reading list

- **[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)** (Zhou et al. 2023) — the original "1k high-quality SFT examples beat 50k mediocre ones" result.
- **[Tülu 3](https://arxiv.org/abs/2411.15124)** (Lambert et al. 2024) — open recipe for SFT + post-training, with extensive data ablations showing what filtering choices actually move the needle.
- **[The Fineweb-Edu data classifier paper](https://huggingface.co/blog/fineweb-v1)** — the same methodology (small classifier scoring) applied to pretraining data; the SFT analogue is `judge.py`.
- **[Physics of LM 3.1 / 3.3](https://arxiv.org/abs/2309.14316)** (Allen-Zhu & Li, 2024) — why *synthetic paraphrase* augmentation matters for knowledge acquisition, with implications for how you should think about the response side of SFT pairs.
- **[Gekhman et al. 2024](https://arxiv.org/abs/2405.05904)** — SFT on facts the model doesn't know is actively training hallucination. Implication: response *correctness* in your SFT set is a real concern, which is one of the four axes the LLM judge scores.

## Next

This module is independent of the rest of the course. If you came from Part 1 and want to keep going on the main path, head to [Module 04 — Attention](../../part-2-architecture/04-attention/). If you came here because you're starting Part 4 and want a better SFT dataset than `no_robots`, push your curated artifact to HF and feed it into [Module 15 — SFT](../../part-4-post-training/15-sft/) via a config change.
