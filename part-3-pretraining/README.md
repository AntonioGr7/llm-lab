# Part 3 — Pretraining

The engine room.

This is where the course becomes a working pretraining framework. Modules 08–11 collectively build a small but real training stack: a model, a loop, a learning-rate schedule, distributed sharding, a data pipeline, and the orchestration that ties them together. The artifact is a directory you could **lift out of this repo and use to pretrain a SOTA model at any scale** — change the launch flags, not the code.

A few decisions worth stating up front:

- **The model defaults to the Qwen3 architecture** via Hugging Face `transformers`. It's a well-tested implementation of the same components Part 2 built (RMSNorm, SwiGLU, RoPE, GQA), and using it lets the framework stay focused on the *training* side. Swapping in a different model — Llama, Mistral, DeepSeek, or your own Part-2 `TransformerLM` — is a single function. The convention is documented in Module 08.
- **Every script is multi-GPU from day one.** The entrypoint is always `torchrun`, even for `--nproc_per_node=1`. Sharding uses PyTorch's **FSDP2** (`fully_shard`), the 2026 default. The single-GPU case isn't a different code path; it's the same code with a smaller cluster.
- **The framework is `pretrain/`-shaped.** Modules 08, 09, 10 each contribute components (loop, schedule, sharding); Module 11 orchestrates them and runs the actual demo on FineWeb-Edu. The complete codebase is small — maybe 600 lines total — and self-contained.

## Modules

- **[08 — The Training Loop](08-training-loop/)** — Every line of a canonical training loop, **one concept per file**. The model builder (Qwen3 default), AdamW with proper param groups, BF16 mixed precision (and why not FP16), gradient accumulation, gradient clipping, FSDP2 wrapping, DCP checkpointing. CPU-runnable notebook that exercises `train_step` end-to-end on a synthetic dataset. No `train.py` here — the runnable entrypoint lives in Module 11.
- **[09 — The Learning Rate](09-learning-rate/)** — The most important hyperparameter. Warmup, cosine decay, minimum LR. muP transfer from small to large in concrete numbers. Diagnosing LR problems from a loss curve.
- **[10 — Scaling and Efficiency](10-scaling-and-efficiency/)** — Per-rank memory math, ZeRO stages mapped to FSDP2 sharding choices, activation checkpointing (one line, ~33% FLOPs for most of activation memory), tensor and pipeline parallelism conceptually, DeepSpeed sidebar, DeepSeek's multi-token prediction, and Chinchilla scaling laws as a budgeting tool.
- **[11 — Pretraining in Practice](11-pretraining-in-practice/)** — The complete, self-contained framework. FineWeb-Edu streaming pipeline, the composed `train.py` (pulls 08's loop + 09's schedule + 10's checkpointing), the demo config (~150M Qwen3, ~3B tokens, Chinchilla-optimal, ~$15–25 on a single A100), `eval.py` for perplexity + generation sanity check + lm-evaluation-harness pointer. The directory you lift out into a new repo.

## What you'll be able to do at the end of this Part

- Write a pretraining loop from scratch that you'd trust at scale.
- Launch `torchrun train.py` on 1 GPU or 64 GPUs from the same code.
- Diagnose a training run from its loss curve and grad norm.
- Make budget tradeoffs between model size, token count, and compute.
- Take this directory to another repo and use it as the starting point for production pretraining.

## Time and cost

- Reading + coding: ~10 hours.
- Compute cost: ~$15–25, almost all of it in Module 11. Pre-run checkpoints are committed so you can skip the expensive run if you want.

## Swapping the model

Module 08's [`model.py`](08-training-loop/model.py) exposes a single `build_model(cfg)` function. The default is Qwen3:

```python
from transformers import Qwen3Config, Qwen3ForCausalLM
config = Qwen3Config(hidden_size=cfg.d_model, num_hidden_layers=cfg.n_layers, ...)
return Qwen3ForCausalLM(config)
```

To swap in another HF model, replace the imports and the config class. To swap in your own (e.g. Part 2's `TransformerLM`), make `build_model` instantiate that instead. Either way, the contract is: takes `cfg`, returns an `nn.Module` whose forward accepts `input_ids=...` and returns an object with `.loss` (when `labels=...` is passed) and `.logits`. That's it.
