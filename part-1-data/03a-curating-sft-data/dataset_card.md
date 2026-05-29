---
language:
  - it
license: cc-by-4.0
size_categories:
  - 10K<n<1M
task_categories:
  - text-generation
  - question-answering
  - summarization
  - translation
task_ids:
  - language-modeling
  - instruction-tuning
pretty_name: Italian SFT (curated)
tags:
  - italian
  - sft
  - instruction-tuning
  - curated
source_datasets:
  - DeepMount00/OpenItalianData
---

# Italian SFT — Curated Subset

A high-quality Italian instruction-following dataset, derived from
[DeepMount00/OpenItalianData](https://huggingface.co/datasets/DeepMount00/OpenItalianData)
via a 5-stage filter cascade designed to remove machine-translation artifacts,
non-Italian content, and low-quality pairs.

> This dataset is part of the **llm-lab course** ([repo](https://github.com/AntonioGr7/llm-lab)),
> module *03a — Curating SFT Data*. The full filter pipeline that produced it
> lives at `part-1-data/03a-curating-sft-data/`; see the module README for the
> methodology in detail.

## TL;DR

- **Source:** `DeepMount00/OpenItalianData` (~2.14M rows of machine-translated
  English SFT data, primarily from instruction-tuning aggregations).
- **Final size:** ~**`<N>`k rows** after curation (fill in after a real run).
- **Language:** Italian only (verified via fastText langid + Italian function-word ratio).
- **Format:** `messages` field, list of `{role, content}` dicts. Chat-template-ready.
- **License:** CC-BY-4.0 (matching the upstream license).

## Schema

Each row has a single field, `messages`, holding a list of turns:

```json
{
  "messages": [
    {"role": "user", "content": "Riassumi in due frasi il seguente testo: ..."},
    {"role": "assistant", "content": "Il testo descrive ..."}
  ]
}
```

This matches the HuggingFace chat-template convention used by `tokenizer.apply_chat_template`,
so you can feed it directly into any standard SFT trainer (TRL, Axolotl, etc.).

## Curation pipeline

Five stages, cheapest first. Each stage's drop counts are in `report.json`
in the source repo (link below).

1. **Structural** — length bounds (min 10 words assistant, max 2000), repetition
   detection, response-vs-prompt copy-bug detection.
2. **Language fidelity** — fastText `lid.176` langid check + Italian
   function-word ratio. Catches partially-untranslated rows where English
   chunks slipped through.
3. **Translation-artifact patterns** — regex catches for untranslated English
   filler (`I'm sorry`, `as an AI`), US-only entities (`$25`, ZIP codes,
   Fahrenheit), and AI-brand mentions (`ChatGPT`, `OpenAI`).
4. **Dedup & diversity** — MinHash + LSH near-duplicate removal on prompts,
   followed by embedding-cluster downsampling to prevent over-represented
   task families (e.g. data-to-text) from dominating.
5. **LLM-as-judge (optional)** — small Italian-capable model rates each
   surviving pair on fluency, relevance, correctness, and overall quality;
   top-N pairs are kept.

Reproducibility: every step is deterministic given the random seed reported in
`report.json`. To re-derive this dataset from scratch, run:

```bash
git clone https://github.com/AntonioGr7/llm-lab
cd llm-lab/part-1-data/03a-curating-sft-data
pip install -r requirements.txt
python prepare.py \
    --dataset DeepMount00/OpenItalianData \
    --target-size <N> \
    --out italian-sft-curated.jsonl
```

## Limitations & known issues

- **Upstream is machine-translated.** OpenItalianData is itself a translated
  aggregation of English instruction datasets. Even after filtering, some
  cultural mismatches survive (UK/US place names, US holidays, etc.). The
  filter is precision-oriented but not exhaustive.
- **Calque coverage is partial.** The Italian calque blocklist used in
  Stage 3 is a small curated list (`fa senso`, `come modello linguistico`,
  etc.). Real Italian speakers will spot calques the filter misses; if you do,
  please open an issue or PR on the source repo.
- **No safety filtering done here.** This is a *quality* curation, not a
  *safety* curation. Use a separate moderation pass if you're training a
  model for production deployment.
- **Single-turn assumption.** Filters are designed around single (prompt,
  response) pairs. Multi-turn conversations from the upstream are still
  handled but the filtering signal on later turns is weaker.

## Held-out eval set

A separate `eval_probes.jsonl` of ~150 hand-curated Italian prompts ships
with the module (not included in this dataset). It covers 15 task categories
(factual QA, summarization, translation, creative writing, reasoning, code,
roleplay, etc.) and is intended as the evaluation set for any SFT run that
uses this data. See the source repo's `eval_probes.jsonl` for the full list.

## Citation

If you use this dataset, please cite:

- The upstream source: [`DeepMount00/OpenItalianData`](https://huggingface.co/datasets/DeepMount00/OpenItalianData).
- The llm-lab course it was curated in (link in the TL;DR above).

```bibtex
@misc{italian-sft-curated,
  author       = {Antonio Grimaldi},
  title        = {Italian SFT — Curated Subset},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/<your-username>/italian-sft-curated}}
}
```

## Changelog

- **v1.0** (`<date>`) — Initial curated release from OpenItalianData.

---

*This dataset card was generated from the template at `dataset_card.md` in the
[llm-lab](https://github.com/AntonioGr7/llm-lab) source module. Fill in the
`<N>`, `<date>`, and `<your-username>` placeholders before pushing to HF.*
