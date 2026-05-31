# Part 4 — Post-Training

A base model predicts tokens. It does not follow instructions, it does not refuse harmful requests, it does not reason step by step. Post-training is how it learns to do all three.

This is where the modern field has shifted. Pretraining costs are eye-watering but the techniques have stabilized; post-training is where the active research is, and where small teams with modest budgets can still produce meaningful improvements on top of an open base model.

**Important pedagogical choice:** we switch base models here. Up to now, you've been working with the small model you pretrained yourself. For post-training, we move to Qwen 3.6 1.7B — properly pretrained on trillions of tokens — so that the post-training effects are visible and meaningful. We'll also run SFT on your own model first as a proof of concept, so you can compare.

## Modules

- **[14 — The Post-Training Landscape](14-post-training-landscape/)** — Why a base model is unusable. The alignment tax. The full post-training stack as practiced at labs.
- **[15 — Supervised Fine-Tuning (SFT)](15-sft/)** — What SFT does to the weight space. Chat templates. Assistant-only loss masking. Full fine-tuning only (the parameter-efficient alternative is the very next module). Demo: full-FT Qwen3-1.7B-Base.
- **[16 — Parameter-Efficient Fine-Tuning (LoRA / QLoRA)](16-parameter-efficient-finetuning/)** — The same SFT objective as Module 15, but updating a low-rank adapter instead of all the weights. Why it works (the intrinsic-dimension hypothesis), what it costs you vs full FT, and QLoRA's 4-bit base that fits a 1.7B fine-tune on a single consumer GPU. Demo: LoRA SFT of Qwen3-1.7B, merged and compared head-to-head against Module 15's full fine-tune.
- **[17 — Preference Optimization](17-preference-optimization/)** — Why SFT alone is not enough. RLHF in one slide. DPO as the practical replacement. IPO and KTO variants. Demo: DPO on Qwen3-1.7B.
- **[18 — Reasoning and GRPO](18-reasoning-and-grpo/)** — Chain-of-thought as a training target. GRPO (the DeepSeek-R1 technique). Reward design for reasoning. Demo: GRPO on GSM8K subset, watching reasoning emerge.
- **[19 — Distillation](19-distillation/)** — Transferring capability through token-level supervision. Three flavors, each fixing a limitation of the previous:
  - **Offline distillation** (R1-Distill style) — SFT on teacher samples. Simple, powerful, but off-policy: the student is trained on a distribution it won't see at inference.
  - **On-policy distillation** — student generates rollouts, teacher provides token-level supervision on the student's own distribution. Fixes the off-policy problem, but you still need a separate (larger) teacher.
  - **Self-Distillation Fine-Tuning (SDFT)** — the new technique from [Shenfeld et al., 2026](https://arxiv.org/abs/2601.19897). The *same* model is its own teacher when conditioned on demonstrations, giving you on-policy learning from demonstrations without needing rewards or a separate teacher. Substantially reduces catastrophic forgetting and enables continual learning. We'll demo it on a small skill-acquisition task and show how it stacks against plain SFT.

## What you'll be able to do at the end of this Part

- Take any open base model and turn it into an instruction-following, preference-aligned, reasoning model.
- Choose between full fine-tuning and LoRA/QLoRA for a given budget, and merge an adapter back into a single deployable model.
- Choose between SFT, DPO, GRPO, and the three distillation flavors based on what you're trying to fix.
- Build a small preference dataset that produces a measurable behavior change.
- Run SDFT on demonstrations and show measurably less forgetting than plain SFT.

## Time and cost

- Reading + coding: ~17 hours.
- Compute cost: ~$20–30, with most of it in Modules 18 (GRPO is sample-hungry) and 19 (on-policy methods need student rollouts at every step). LoRA (Module 16) is the cheapest run in the Part — under $1.
