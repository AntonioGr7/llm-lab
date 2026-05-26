# Module 14 — Supervised Fine-Tuning (SFT)

You finished Part 3 with a base model. You finished [Module 13](../13-post-training-landscape/) understanding *why* a base model is unusable as a product. This is where you do the first thing about it.

**SFT is the canonical post-training operation:** take a pretrained base model, fine-tune *all* its weights on curated `(prompt, response)` demonstrations, end up with an instruction-following model. Every frontier lab does this as their first post-training step. It's the same cross-entropy loss you used in pretraining, the same optimizer, the same FSDP wrap — but applied to chat-formatted data and masked so only the assistant's response counts toward the loss.

The two new pedagogical ideas in this module:

1. **Chat templates** — the strict text format that turns a flat next-token-predictor into a turn-taking assistant. The model sees `<|user|>... <|assistant|>...` with delimiters that mark whose turn it is.
2. **Assistant-only loss masking** — we apply cross-entropy *only* on the assistant's tokens. The user's tokens are inputs the model conditions on, not targets it predicts. Getting this mask right is the single most common silent bug in homemade SFT pipelines.

This module is **full fine-tuning only** — every weight in the model updates. LoRA / QLoRA / DoRA and other parameter-efficient methods are their own module later in Part 4, because the techniques deserve a focused treatment rather than being a "config flag" on top of the full-FT pipeline. Full FT first, parameter-efficient variants once you understand what they're being efficient *about*.

Like Module 11, this is a **framework directory** — lift it out, point it at your own data and base model, you have a working SFT codebase. Modules 15-17 (preference optimization, GRPO, distillation) will all build on this same pipeline rather than starting fresh.

## The thesis

Post-training is **data-bound, not compute-bound**. The SFT *algorithm* is identical to pretraining — `loss = CE(model(input_ids).logits, labels)`, same AdamW, same FSDP. What differs is everything around the algorithm:

- The data is no longer raw web text; it's curated `(prompt, response)` pairs that demonstrate the behavior you want.
- The loss is no longer applied to every token; it's masked to *only* the assistant's response tokens.
- The base model is no longer random init; it's a fully-pretrained checkpoint, and you want to *change it as little as possible* while still acquiring chat behavior. This is where the "alignment tax" from Module 13 lives, and it's why your SFT learning rate is ~30-100× smaller than what you used in pretraining.

If you internalize those three differences, you understand SFT. Everything else is engineering.

## What you'll be able to do at the end

1. Take any HuggingFace causal LM and full-FT it on chat data with under 200 lines of training-loop code.
2. Explain (and debug) the assistant-only loss mask — including why "flat-line loss" on step 0 is the signature of a broken mask.
3. Set an SFT learning rate from first principles (and understand why the right answer is "much lower than you think").
4. Read a paper's SFT recipe (Tülu 3, Llama 3, Qwen3 technical reports) and translate every config knob into something this codebase has.

## 1. Directory layout

```
14-sft/
├── README.md              you are here
├── config.py              TrainConfig — model, data, optimizer, schedule, training
├── model.py               build_model() — loads HF causal LM by name
├── data.py                ChatDataset — applies chat template + computes assistant-only loss mask
├── loop.py                forward_loss with masking, train_step (FSDP-aware)
├── train.py               torchrun entrypoint — FSDP + SFT loop
├── eval.py                generation samples + held-out perplexity on instruction data
├── fsdp_setup.py          copied from Module 11 — FSDP2 wrap helpers
├── optim.py               copied from Module 11 — AdamW with decay/no-decay groups
├── schedule.py            copied from Module 11 — warmup-cosine / WSD
├── checkpoint.py          copied from Module 11 — DCP save/load
├── efficiency.py          copied from Module 11 — activation checkpointing, memory math
├── configs/
│   ├── sft_demo.yaml      use your Module 11 150M checkpoint as base — $0-1 dev
│   └── sft_qwen3_1.7b.yaml  the real run on Qwen3-1.7B-Base — ~$3-5 on A100
├── tests/                 shape + masking sanity tests
└── results/               pre-run loss curves + checkpoint for the canonical run
```

Every component is either NEW (introduced in this module) or COPIED from Module 11. The "framework directory" promise from Module 11 holds: you can `cp -r 14-sft/ ~/my-sft-repo/` and have a complete SFT codebase.

## 2. Chat templates

A pretraining dataset is raw text:

```
The capital of France is Paris.
```

An SFT dataset is a conversation, but the model still only knows how to predict tokens. So the conversation gets *serialized* into a single string with delimiters marking whose turn it is. For Qwen3, the chat template looks like:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

Different model families use different delimiters (Llama uses `[INST]...[/INST]`, ChatML uses `<|im_start|>`, Mistral uses `<s>[INST]`, etc.) but the principle is identical: a fixed text format the model learns to recognize.

The tokenizer has the template baked in — `tokenizer.apply_chat_template(messages)` does the serialization. Use it. Do not invent your own delimiters. Mismatch between the template you train on and the template at inference time is the second most common silent bug in homemade SFT pipelines (after the loss mask).

## 3. The loss mask

If you trained on the rendered chat string above with **no masking**, you'd be teaching the model two things:

1. ✓ Given a user's question, predict the assistant's answer (what you want).
2. ✗ Given some preceding text, predict the user's question (a problem — the model will start asking *itself* questions during generation, because it's been trained to do exactly that).

So the loss must be applied **only to assistant tokens**:

```python
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
# where labels[position] = -100 anywhere outside an assistant turn.
```

Computing the mask correctly is the entire game. Two practical approaches:

**Approach A — naive: regex on the rendered string.** Find `<|im_start|>assistant\n` and `<|im_end|>` boundaries, mask everything outside. Brittle: breaks the moment you change template, and the boundary tokens themselves are ambiguous (do you train the model to *predict* `<|im_start|>assistant`? You probably want to, but the answer depends on the template).

**Approach B — diff-based: render twice, diff.** This is what [data.py](data.py) uses, and what HuggingFace's TRL SFTTrainer does internally. Render the conversation with `apply_chat_template(messages)` to get the full sequence. Then for each assistant turn, render `apply_chat_template(messages[:assistant_turn])` (everything up to but not including this turn) — the difference between the two lengths is exactly the span where this turn's tokens live. Mask everything else.

```python
def build_loss_mask(messages, tokenizer):
    full = tokenizer.apply_chat_template(messages, return_tensors=None)
    mask = [False] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True)
        end = tokenizer.apply_chat_template(messages[:i+1])
        # Tokens at positions [len(prefix), len(end)) are this assistant turn.
        for p in range(len(prefix), len(end)):
            mask[p] = True
    return mask
```

The diff approach is template-agnostic. It works for Qwen3, Llama 3, Mistral, Gemma, anything HF supports — because it only relies on `apply_chat_template` returning longer strings for longer message lists, which every template does by construction.

**The flat-line loss signature.** If your mask is broken (everything `True`), step 0 will show loss ≈ 0.1-0.5 instead of the 2-4 you'd expect. Why? Because most of the "predictions" are predicting `<|im_start|>` token after `<|im_end|>` — trivial after one update — and you're averaging that over the whole sequence. If you see step 0 loss < 1.0, your mask is wrong. The notebook has a unit test that catches this.

## 4. Full fine-tuning vs frozen-base alternatives

In full FT, every parameter updates. For Qwen3-1.7B with AdamW + BF16 + activation checkpointing:

| Component | Bytes per param | 1.7B params |
|---|---|---|
| Master weights (FP32) | 4 | 6.8 GB |
| Compute weights (BF16) | 2 | 3.4 GB |
| Gradients (BF16) | 2 | 3.4 GB |
| AdamW state (m, v in FP32) | 8 | 13.6 GB |
| **Subtotal** | | **27.2 GB** |
| Activations (with AC, seq=2048, bs=4) | | ~12 GB |
| **Total per rank** | | **~39 GB** |

Fits on a single A100-80GB with ~40 GB of headroom for batch scaling and PyTorch's caching allocator. Without activation checkpointing the activations balloon to ~100 GB and you OOM — this is why our default config has `activation_checkpointing: true`. The same 33% FLOP tax we discussed in Module 10 § 3, applied here.

**Why not just use LoRA and dodge the memory math?** A few reasons that matter pedagogically:

1. **Full FT is what frontier labs do.** Read any major lab's technical report (Llama 3, Qwen3, DeepSeek-V3) — their SFT stage is full FT. LoRA is a *resource-constrained* substitute, not the canonical operation. Teaching LoRA first inverts that and gives a misleading mental model of the field.
2. **Full FT exposes the alignment tax.** When you watch all 1.7B params shift, you can measure how far the model drifts from its base capability — that's the alignment-tax measurement we set up in Module 13. With LoRA, you can't see this because the base weights never move.
3. **LoRA has its own pedagogical content** worth a dedicated module: the low-rank assumption, the merge-back step, QLoRA's NF4 quantization, DoRA, etc. Trying to cover it as a config flag on top of this module would compress it into a footnote.

If you're cost-constrained and want to run SFT on a 7B+ model, jump ahead to the LoRA module and come back. The current module is the conceptual foundation; LoRA is the efficiency layer on top.

## 5. The training configuration

The canonical demo: full-FT Qwen3-1.7B-Base on [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) — 10k high-quality instruction-following examples, hand-curated by HuggingFace, no license drama.

```yaml
model:
  name: Qwen/Qwen3-1.7B-Base
  max_seq: 2048

data:
  source: HuggingFaceH4/no_robots
  seq_len: 2048
  batch_size_per_device: 4

optimizer:
  type: adamw
  lr: 1.0e-5              # ~30× lower than pretraining — see § 6
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0       # full FT + small dataset — decay hurts more than helps

schedule:
  type: cosine
  warmup_steps: 100
  min_lr_ratio: 0.1

training:
  total_steps: 600        # ~3 epochs over 10k examples at effective batch 64
  grad_accum: 16          # bs=4 × accum=16 × world=1 = 64 effective; ÷2 for 2 GPUs
  grad_clip: 1.0
  dtype: bf16
  activation_checkpointing: true
```

Expected wallclock on a single A100-80GB: ~1.5-2 hours. Cost: $3-5 on RunPod.

## 6. Why the SFT learning rate is so low

In pretraining (Module 11) we used a peak LR of 3e-4. Here we're at 1e-5 — about 30× lower. Why?

**Pretraining** moves the model from random init toward whatever the data implies. Every update is a *large* directional change because the model has nothing to lose; the gradient signal is the only information about where in weight space "good" lives.

**SFT** starts from a model that's already at a good point in weight space. The job is to nudge it toward a *specific* nearby point (the instruction-following one) without destroying the existing capability. Large LRs in this regime are catastrophic: they overshoot the local optimum, and you end up with a model that has good chat format but has forgotten how to do math, code, or recall facts. The literature calls this **catastrophic forgetting**; Module 13 framed it as the **alignment tax**.

The right SFT LR is roughly **"as small as it can be while still moving the loss"**. Common values:

| Model size | Typical SFT LR | Why |
|---|---|---|
| 150M-1B | 5e-5 to 2e-4 | Smaller models can tolerate larger relative updates |
| 1.7B-8B | 5e-6 to 2e-5 | Frontier labs' default range |
| 70B+ | 1e-6 to 5e-6 | Larger models are more sensitive — bigger absolute updates per param |

Our configs use 1e-5 for the 1.7B run and 5e-5 for the 150M demo. If you see the loss diverge after step ~50 (after warmup completes), halve the LR. If the loss is flat after 200 steps, double it.

## 7. Running it

```bash
# Recommended: single A100-80GB on RunPod / Lambda Labs
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/sft_qwen3_1.7b.yaml

# 8-GPU node (adjust grad_accum to keep effective batch constant)
torchrun --standalone --nproc_per_node=8 train.py \
    --config=configs/sft_qwen3_1.7b.yaml \
    --training.grad_accum=2

# Dev / smoke test on your Module 11 150M checkpoint — runs in minutes on any GPU
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/sft_demo.yaml
```

## 8. What you should see

Loss starts around 2.5-3.5 nats/token (the base model's perplexity on chat-formatted instruction data) and descends to 1.0-1.5 by step 600. Compare to Module 11 pretraining where end-of-run loss was ~2.8 on *raw text* — the SFT loss can go lower because the assistant-only positions are constrained to a much narrower distribution (formatted responses).

**The qualitative test** — and the only one that matters — is generation. After training:

```bash
python eval.py --checkpoint=results/checkpoints/step_00000600 \
    --prompts="Write a haiku about Python" \
              "Explain RNNs to a 10-year-old"
```

Before SFT, Qwen3-1.7B-Base continues the prompt as if it were web text ("...is a programming language. It was created in..."). After SFT, it should produce an actual haiku and an actual explanation, formatted as a turn-taking response with the right delimiters.

## 9. Stretch goals

- **Packed sequences.** Pack multiple short `(prompt, response)` pairs into one `seq_len`-length sequence with a per-sample attention mask. Doubles or triples training throughput on datasets with short responses. The data pipeline supports it via `--data.pack=true`.
- **Tülu 3 mix.** Scale up from `no_robots` (10k) to the Tülu 3 SFT mix (~939k). Better final model, ~5-8 hours wallclock, ~$15-25 cost. Same code; different config.
- **Liger fused linear+CE.** The same trick from [Module 11's H100 config](../../part-3-pretraining/11-pretraining-in-practice/configs/demo_h100.yaml). Lets you push micro-batch up at the same vocab.
- **FP8 with Transformer Engine.** On H100, ~40% wallclock savings. Requires `te.Linear` swapping in `model.py` — see Module 11 README §11.

## 10. Reading list

In order of "read this first":

- **[HuggingFace H4 — no_robots dataset card](https://huggingface.co/datasets/HuggingFaceH4/no_robots).** Read this if you're going to train on it (you are). License, provenance, and curation methodology matter.
- **[Allen Institute (2024) — Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124).** The reference for serious open-source SFT in 2024-2025. Section 4 (data curation) and Appendix C (training hyperparameters) are gold.
- **[Qwen3 Technical Report (2025)](https://qwenlm.github.io/blog/qwen3/).** Section on post-training pipeline gives you the recipe a frontier lab actually used for the model you're fine-tuning.
- **[Wang et al. (2022) — Self-Instruct](https://arxiv.org/abs/2212.10560).** Where modern instruction-following datasets come from. Skim for the methodology — the bootstrapping idea drives most synthetic SFT data today.
- **[Chung et al. (2022) — Scaling Instruction-Finetuned Language Models (FLAN)](https://arxiv.org/abs/2210.11416).** The paper that established instruction tuning as a distinct stage. Read for context on why this works.

## 11. What's next

[Module 15 — Preference Optimization](../15-preference-optimization/) — SFT teaches the model *what to say*; preference optimization teaches it *how to choose between things to say*. We layer DPO on top of the SFT checkpoint you produce here.
