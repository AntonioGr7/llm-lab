# Module 00 — What We're Actually Building

> Part of [Part 0 — Mental Model](../). Reading time: ~20 minutes. Compute cost: $0.

## The 30-second version

A modern LLM is built in stages. **Pretraining** produces a model that predicts the next token. **Post-training** turns that token predictor into something a human would actually want to use. **Evaluation** tells you whether you succeeded. Each stage uses different data, different objectives, and surprisingly different engineering.

This course walks you through every stage end-to-end on a small enough model to run on a single A100 — but using the same techniques the frontier labs use in 2026.

## The pipeline in one picture

```
                  ┌──────────────────────────────────────┐
                  │ RAW WEB / BOOKS / CODE / SYNTHETIC   │
                  │ ~Terabytes, mostly garbage           │
                  └──────────────────┬───────────────────┘
                                     │   data pipeline
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ FILTERED + DEDUPED CORPUS            │
                  │ "FineWeb-Edu" style, billions of toks│
                  └──────────────────┬───────────────────┘
                                     │   tokenization
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ TOKEN STREAM                         │
                  │ Integer IDs the model actually sees  │
                  └──────────────────┬───────────────────┘
                                     │   PRETRAINING  ← Part 3
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ BASE MODEL                           │
                  │ Predicts next token. Not yet useful. │
                  └──────────────────┬───────────────────┘
                                     │   SFT  ← Module 15
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ INSTRUCT MODEL                       │
                  │ Follows instructions. Often bland.   │
                  └──────────────────┬───────────────────┘
                                     │   DPO  ← Module 16
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ ALIGNED MODEL                        │
                  │ Preferred outputs over rejected ones │
                  └──────────────────┬───────────────────┘
                                     │   GRPO  ← Module 17
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ REASONING MODEL                      │
                  │ Thinks before answering              │
                  └──────────────────┬───────────────────┘
                                     │   evaluation ← Part 5
                                     ▼
                               deployment
```

This is the whole course in a diagram. Every module is either a box, an arrow, or a tool for telling you whether one of them worked.

## What each stage actually buys you

| Stage | Input | Output | What you gain | What you pay |
|---|---|---|---|---|
| Data filtering | Web crawl | Clean corpus | Everything downstream depends on it | The expensive, unglamorous part |
| Tokenization | Text | Integer IDs | Compression + a unit of computation | Decisions you live with for the model's life |
| **Pretraining** | Token stream | Base model | World knowledge, grammar, reasoning seeds | The bulk of FLOPs |
| SFT | Instruction pairs | Instruct model | Knows how to respond, not just continue | Cheap if you have good data |
| DPO | Preference pairs | Aligned model | Tone, safety, helpfulness | Sensitive to base model quality |
| GRPO | Verifiable tasks | Reasoning model | Step-by-step thinking, self-correction | Compute-heavy (many samples per step) |
| Evaluation | The model | Numbers + judgment | Knowing if anything worked | The skill nobody teaches |

The lopsided thing about modern LLM training is that **pretraining is ~99% of the compute and maybe 50% of what makes the final model good**. The other 50% comes from the post-training steps, which together use a fraction of the compute.

This is why the field shifted. Naively throwing more pretraining at problems hit diminishing returns around 2023. Post-training is where the active research has been ever since.

## Where we are vs where the labs are

We're going to train something tiny. Here's the honest comparison.

| | This course | DeepSeek-V3 |
|---|---|---|
| Parameters | ~100M–1B | 671B total, 37B active per token |
| Training tokens | ~1B | 14.8T |
| GPUs | 1× A100 | 2,048× H800 |
| Wall-clock | ~hours | ~2 months |
| Cost | ~$50 | ~$6M |

The ratio is roughly 1:100,000. **But the code is the same shape.** The training loop you write in Part 3 is, structurally, the same loop DeepSeek wrote. The architectural choices (RoPE, SwiGLU, RMSNorm, GQA, MoE) are the same. The data hygiene principles are the same. The post-training stack is the same.

What changes at scale is engineering — multi-GPU communication, FP8 numerical stability, fault tolerance, checkpointing under hardware failure — not concepts. By Module 21 you'll be able to point at exactly what changes and what doesn't.

## The course philosophy

> **Deep enough to debug. Practical enough to ship.**

Every concept is taught through its consequences and the decisions it informs. We skip math derivations when they don't change a decision. We keep them when they do.

Four rules we'll follow throughout:

1. **Motivation before mechanism.** No technique is introduced without first showing the problem it solves. If you can't say what problem it solves, you don't understand it.
2. **Show the broken version too.** Loss curves of healthy runs and broken runs side by side. You can't recognize a problem you've never seen.
3. **Always answer "how do I know it worked?"** Most courses skip evaluation. We make it a first-class citizen — that's why Part 5 exists.
4. **Distributed-ready from day one.** Every training and post-training script is launched with `torchrun` and uses PyTorch's distributed primitives (FSDP for sharding), even when you only have one GPU. Single-GPU runs work transparently; multi-GPU is a flag away. This is how researchers actually write code, and it's what makes the "$50 course prepares you for the $50M run" claim non-empty.

## How to read this course

Every module has the same shape:

| File | Purpose | When to use |
|---|---|---|
| `README.md` | The lecture: motivation, concept, decision it informs | Always. Read first. |
| `notebook.ipynb` | Narrative + experiments | When you have GPU time, or to inspect outputs |
| `*.py` | Clean, reusable code | When you want production-quality reference |
| `results/` | Pre-run logs and checkpoints | To skip expensive runs or compare against |

You don't have to run everything. Three valid paths through the course:

| If you have... | Do this | Value captured |
|---|---|---|
| No GPU budget | Read every README. Run notebook cells locally on CPU with tiny tensors. Inspect provided `results/`. | ~60% |
| ~$50 of GPU | Run the small experiments yourself. Use provided checkpoints for the expensive ones (pretraining demo, GRPO). | ~90% |
| ~$200+ of GPU | Run everything. Vary hyperparameters. Compare your runs to the provided ones. | ~100% |

## The compute budget philosophy

The course is built around one constraint: a student should be able to finish it for ~$50 of GPU credits. That constraint shapes everything.

Two rules that make $50 enough:

**Develop offline, run online.** Every line of code is debugged on CPU with tiny tensors before it ever touches a GPU. The GPU exists to run the experiment, not to iterate on it. This is how researchers actually work. Module 01 will show you the exact workflow.

**Pre-run checkpoints exist.** The expensive runs (the pretraining demo in Module 11, the GRPO run in Module 17) have checkpoints committed to the repo. If you want to run them yourself, great. If you'd rather spend your budget on smaller experiments, also great — you can pick up from any stage.

The dollar value isn't really the point. The point is that running an experiment intentionally — not by reflex — is a skill that scales. The engineer who knows what experiments to run with a $10M budget is the same engineer who knew what to run with $50.

## What you'll be able to do

After **Part 0** (you are here): you'll know the shape of the field and how to use this repo.

After **Part 2**: you'll be able to read a transformer paper and identify what's novel vs standard. Most papers will start feeling smaller.

After **Part 3**: you'll have run an actual pretraining job. You'll know what a healthy loss curve looks like and how to recognize a broken one.

After **Part 4**: you'll have turned a base model into an instruction-following, preference-aligned, reasoning model — all on top of a real open-source base (Qwen 3.6 1.7B).

After **Part 6**: you'll be able to read the latest releases (the next DeepSeek, the next Qwen, whatever comes after) and tell what's actually new vs what's marketing.

## Next

[Module 01 — Tools and Environment Setup](../01-tools-and-environment/). Before any training, get your dev loop right. It's the difference between a $50 course and a $500 one.
