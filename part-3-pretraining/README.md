# Part 3 — Pretraining

The engine room.

This is where the course becomes a working pretraining framework. Modules 08–12 collectively build a small but real training stack: a model, a loop, a learning-rate schedule, distributed sharding, a data pipeline, and the orchestration that ties them together. The artifact is a directory you could **lift out of this repo and use to pretrain a SOTA model at any scale** — change the launch flags, not the code. Module 13 then puts that stack to a second use: **continual pretraining** — teaching a *finished* base model genuinely new, private knowledge it was never pretrained on.

A few decisions worth stating up front:

- **The model defaults to the Qwen3 architecture** via Hugging Face `transformers`. It's a well-tested implementation of the same components Part 2 built (RMSNorm, SwiGLU, RoPE, GQA), and using it lets the framework stay focused on the *training* side. Swapping in a different model — Llama, Mistral, DeepSeek, or your own Part-2 `TransformerLM` — is a single function. The convention is documented in Module 08.
- **Every script is multi-GPU from day one.** The entrypoint is always `torchrun`, even for `--nproc_per_node=1`. Sharding uses PyTorch's **FSDP2** (`fully_shard`), the 2026 default. The single-GPU case isn't a different code path; it's the same code with a smaller cluster.
- **The framework is `pretrain/`-shaped.** Modules 08, 09, 10 each contribute components (loop, schedule, sharding); Module 11 orchestrates them and runs the actual demo on FineWeb-Edu. The complete codebase is small — maybe 600 lines total — and self-contained.

## Modules

- **[08 — The Training Loop](08-training-loop/)** — Every line of a canonical training loop, **one concept per file**. The model builder (Qwen3 default), AdamW with proper param groups, BF16 mixed precision (and why not FP16), gradient accumulation, gradient clipping, FSDP2 wrapping, DCP checkpointing. CPU-runnable notebook that exercises `train_step` end-to-end on a synthetic dataset. No `train.py` here — the runnable entrypoint lives in Module 11.
- **[09 — The Learning Rate](09-learning-rate/)** — The most important hyperparameter. Warmup, cosine decay, minimum LR. muP transfer from small to large in concrete numbers. Diagnosing LR problems from a loss curve.
- **[10 — Scaling and Efficiency](10-scaling-and-efficiency/)** — Per-rank memory math, ZeRO stages mapped to FSDP2 sharding choices, activation checkpointing (one line, ~33% FLOPs for most of activation memory), tensor and pipeline parallelism conceptually, DeepSpeed sidebar, DeepSeek's multi-token prediction, and Chinchilla scaling laws as a budgeting tool.
- **[11 — Pretraining in Practice](11-pretraining-in-practice/)** — The complete, self-contained framework. FineWeb-Edu streaming pipeline, the composed `train.py` (pulls 08's loop + 09's schedule + 10's checkpointing), the demo config (~150M Qwen3, ~3B tokens, Chinchilla-optimal, ~$15–25 on a single A100), `eval.py` for perplexity + generation sanity check + lm-evaluation-harness pointer. The directory you lift out into a new repo.
- **[12 — Production Data Pipelines](12-production-data-pipelines/)** — The two things Module 11's streaming data loader can't do — checkpoint its position (so resume replays from sample 0) and tokenize only once — and the fix every frontier lab ships: a pre-tokenized binary corpus (`.bin` + index), an explicit permutation shuffle, a memory-mapped map-style dataset with O(1) random access, and a resumable distributed sampler that checkpoints the data position into **one integer** for **O(1) bit-exact resume**. CPU-runnable notebook + a resume-correctness test; drops into Module 11's `train.py` in three edits.
- **[13 — Continual Pretraining](13-continual-pretraining/)** — The problem every enterprise hits: you have a large *private* corpus (say 3B tokens) that no public model has seen, and SFT can't internalize it — knowledge is acquired in *pretraining*, and fine-tuning on facts the model doesn't know just trains it to hallucinate. So you **continue pretraining** the finished base model. The four levers that make it work without erasing the model's general ability: a **replay** mix of general data, a **re-warm then re-decay** learning-rate schedule, **synthetic augmentation** (paraphrase + QA) so the knowledge is *extractable* and not just stored, and bounded **data repetition**. Demo: inject a fictional company's universe into `Qwen3-0.6B-Base` and prove — with a held-out QA probe before and after — that the base knew nothing and the continually-pretrained model does, while measuring how much general capability you paid for it (catastrophic forgetting). Reuses Module 12's indexed corpus.

## What you'll be able to do at the end of this Part

- Write a pretraining loop from scratch that you'd trust at scale.
- Launch `torchrun train.py` on 1 GPU or 64 GPUs from the same code.
- Diagnose a training run from its loss curve and grad norm.
- Make budget tradeoffs between model size, token count, and compute.
- Take this directory to another repo and use it as the starting point for production pretraining.
- Continue-pretrain a finished base model on private data, and prove with evidence that it learned the new knowledge without forgetting the old.

## Time and cost

- Reading + coding: ~15 hours.
- Compute cost: ~$15–25, almost all of it in Module 11. Pre-run checkpoints are committed so you can skip the expensive run if you want. Module 12 is **$0** — its lab is CPU-only; building the *full* indexed corpus is optional and costs only CPU time + ~40 GB of disk. Module 13's notebook is CPU-runnable; the real `Qwen3-0.6B-Base` continual-pretraining demo is ~$1–3 on a single A100 (the corpus is tiny by pretraining standards), with a pre-run checkpoint committed.

## Swapping the model

Module 08's [`model.py`](08-training-loop/model.py) exposes a single `build_model(cfg)` function. The default is Qwen3:

```python
from transformers import Qwen3Config, Qwen3ForCausalLM
config = Qwen3Config(hidden_size=cfg.d_model, num_hidden_layers=cfg.n_layers, ...)
return Qwen3ForCausalLM(config)
```

To swap in another HF model, replace the imports and the config class. To swap in your own (e.g. Part 2's `TransformerLM`), make `build_model` instantiate that instead. Either way, the contract is: takes `cfg`, returns an `nn.Module` whose forward accepts `input_ids=...` and returns an object with `.loss` (when `labels=...` is passed) and `.logits`. That's it.
