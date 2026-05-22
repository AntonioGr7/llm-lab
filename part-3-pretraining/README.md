# Part 3 — Pretraining

The engine room.

We take the data pipeline from Part 1 and the model from Part 2 and run an actual training run on an actual GPU. The model that comes out is small and undertrained — but it's a real base model, produced by a real loop, and the techniques are the same ones DeepSeek and Meta use at 1,000× the scale.

This is the most compute-expensive Part. We've structured it so the conceptual modules (08, 09, 10) cost almost nothing, and only Module 11 spends real money.

## Modules

- **[08 — The Training Loop](08-training-loop/)** — Every line of a canonical training loop. Gradient accumulation. BF16 mixed precision (and why not FP16). AdamW and what its hyperparameters actually do. Gradient clipping. The `torchrun` + DDP/FSDP entrypoint pattern that makes every script in this course multi-GPU ready from day one — even when you only have one GPU.
- **[09 — The Learning Rate](09-learning-rate/)** — The most important hyperparameter. Warmup, cosine decay, minimum LR. muP for transferring LR across scales. Diagnosing LR problems from a loss curve.
- **[10 — Scaling and Efficiency](10-scaling-and-efficiency/)** — ZeRO stages: the memory math and what each stage buys you. PyTorch FSDP (our default — native, clean) vs DeepSpeed (the older heavier ecosystem you'll meet in the wild). Gradient checkpointing. Tensor parallelism conceptually. DeepSeek's multi-token prediction. The Chinchilla scaling laws as a budgeting tool.
- **[11 — Pretraining in Practice](11-pretraining-in-practice/)** — The actual demo run on FineWeb-Edu, launched with `torchrun`. Logging, checkpointing, debugging loss spikes. Evaluating a base model and why perplexity alone lies.

## What you'll be able to do at the end of this Part

- Write a pretraining loop from scratch that you'd trust at scale.
- Diagnose a training run from its loss curve and grad norm.
- Make budget tradeoffs between model size, token count, and compute.

## Time and cost

- Reading + coding: ~10 hours.
- Compute cost: ~$15–25, almost all of it in Module 11. Pre-run checkpoints are committed so you can skip the expensive run if you want.
