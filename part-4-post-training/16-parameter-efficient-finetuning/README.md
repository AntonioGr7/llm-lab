# Module 16 — Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

In Module 15 you fine-tuned **every weight** in a 1.7B model. It worked, and it
cost ~$3–5 and ~40 GB of GPU memory (most of it the optimizer: AdamW keeps two
extra full-size copies of every parameter). Now do the same thing for ~$1 and
under 24 GB — and end up with a model that's within a point or two of the full
fine-tune.

That's LoRA. It is the single most important "make it affordable" technique in
post-training, and the reason a hobbyist with one consumer GPU can fine-tune
models that used to need a node of A100s.

## The thesis

> Fine-tuning doesn't need to move every weight. It needs to move a *small,
> low-rank* correction — and you can train just that correction.

A pretrained linear layer holds a weight matrix `W`. Full fine-tuning learns a
dense update `W ← W + ΔW`, where `ΔW` is the same size as `W` — millions of
numbers per layer, each with optimizer state. LoRA (Hu et al. 2021) makes one
bet: **`ΔW` is low-rank**, so you can write it as a product of two skinny
matrices `ΔW = B·A` and train only those. For a 2048×2048 projection at rank
`r=8`, that's `8·(2048+2048) = 32,768` trainable numbers instead of
`4,194,304` — a **128× reduction** in trainable parameters and optimizer state,
*for that layer*.

The bet pays off because of *intrinsic dimensionality* (Aghajanyan et al. 2020):
adapting a big pretrained model to a downstream task lives in a surprisingly
small subspace. You're not teaching the model new facts (that's pretraining,
Module 13) — you're nudging its behavior, and that nudge is low-rank.

## What you'll be able to do at the end

- Implement LoRA from scratch — the `LoRALinear` wrapper, injection, and merge —
  in ~150 lines of `torch`, no `peft` magic.
- Choose `r`, `alpha`, and which modules to adapt for a given budget.
- Run **QLoRA**: a 4-bit frozen base + bf16 adapters that fits a 1.7B fine-tune
  in ~6 GB (a free Colab T4).
- Merge an adapter into its base for zero-overhead deployment — or keep it
  separate to hot-swap ten fine-tunes off one base in memory.
- Say precisely what LoRA costs you versus full FT, and when that trade is wrong.

## 1. Directory layout

```
16-parameter-efficient-finetuning/
├── README.md            ← you are here
├── lora.py              ← THE new code: LoRALinear, inject, merge, state-dict
├── config.py            ← Module 15's config + a LoRAConfig section
├── model.py             ← load base (bf16 or 4-bit) + inject adapters
├── train.py             ← torchrun entrypoint (Module 15's, one swap)
├── merge.py             ← fold adapter into base → a deployable model
├── eval.py              ← before/after generation + held-out perplexity
├── data.py  loop.py  optim.py  schedule.py            ← Module 15, verbatim
├── fsdp_setup.py  efficiency.py  checkpoint.py        ← Module 11/15 infra
├── configs/
│   ├── lora_qwen3_1.7b.yaml     ← canonical (~$1, 24GB)
│   ├── qlora_qwen3_1.7b.yaml    ← 4-bit base (~6GB, Colab-able)
│   └── lora_demo.yaml           ← $0-1 smoke on Qwen3-0.6B
├── tests/   notebook.ipynb   results/
```

The point of how few files are *new* here: LoRA is not a different training
procedure. The data pipeline, the loss mask, the loop, the scheduler — all
identical to full-FT SFT. The only thing that changes is *which parameters the
optimizer is allowed to touch*. `lora.py` is where that change lives.

## 2. LoRA in one matrix equation

A wrapped layer computes:

```
y = W·x  +  (alpha/r) · B·(A·x)
    └ base ┘  └──── low-rank adapter ────┘
       frozen        A: r×in   B: out×r   (the only trainable params)
```

`lora.py`'s `LoRALinear` is exactly this: it keeps the original `nn.Linear` as
a frozen submodule and adds the `A`/`B` branch. Two initialization details are
load-bearing:

- **`B` initializes to zero**, `A` to small Kaiming noise. So at step 0,
  `B·A = 0` and the wrapped layer is *exactly* the base layer. You start
  training from the pretrained model, not a randomly perturbed one. (Test 1 in
  `tests/test_lora.py` pins this.)
- **`scaling = alpha/r`.** This decouples the update magnitude from the rank.
  If you double `r` (more capacity), `scaling` halves, so the effective step
  size stays put and you don't have to re-tune the LR. The convention is
  `alpha = 2·r` (our default: `r=8`, `alpha=16`).

## 3. The knobs: rank, alpha, and which modules

| Knob | What it does | Default | When to change |
|---|---|---|---|
| `r` | rank of `ΔW` = capacity = trainable count | 8 | ↑ to 16–64 for hard tasks approaching full-FT quality; ↓ to 4 for max savings |
| `alpha` | update scale (`alpha/r`) | 16 (= 2r) | keep at `2r` and forget about it; it exists so LR transfers across `r` |
| `dropout` | dropout on the adapter input | 0.0 | 0.05–0.1 on small (<50k) SFT sets |
| `target_modules` | which Linears to adapt | all 7 | the QLoRA recipe: adapt *more* modules at small `r`, not q/v at big `r` |

**Which modules?** The original LoRA paper adapted only the attention `q` and
`v` projections. QLoRA (Dettmers et al. 2023) found it's better to adapt **all**
linear layers — attention `q/k/v/o` *and* the MLP `gate/up/down` — at a smaller
rank. We default to all seven. They're matched by leaf attribute name, so the
same spec works on Qwen3, Llama, Mistral, and Gemma without edits.

## 4. Where the memory actually goes

For a 1.7B model, here's the rough budget (bf16 compute, AdamW):

| | Full FT (M15) | LoRA (this module) | QLoRA |
|---|---|---|---|
| Base weights | 6.8 GB (fp32 master) | **3.4 GB** (bf16, frozen) | **~1.0 GB** (4-bit NF4) |
| Gradients | 6.8 GB (all params) | ~tiny (adapters only) | ~tiny |
| Optimizer (AdamW = 2× params) | 13.6 GB | ~tiny (adapters only) | ~tiny |
| Activations | a few GB | a few GB | a few GB |
| **Fits on** | **A100-80GB (w/ AC)** | **a 24GB card** | **an 8GB card** |

The headline isn't the frozen base (you still pay to *store* it) — it's that
**gradients and optimizer state collapse to the adapter size**. Full FT's
dominant cost is the 20.4 GB of grads + Adam moments over *all* parameters;
LoRA pays that only for the ~0.1% of parameters that are trainable. That's why
LoRA's memory win is so lopsided, and why activation checkpointing is **off** by
default here — you don't need it (§6).

## 5. Why the LoRA learning rate is ~20× higher than full-FT SFT

Module 15 used `lr=1e-5`. This module defaults to `lr=2e-4`. That's not a typo.

Full-FT logic — "the base model is already at a good point in weight space;
nudge it gently" — does **not** apply to the adapter. The adapter starts at
**zero** (`B=0`). It has to travel a real distance to encode the task, and a
1e-5 LR barely moves it in a few hundred steps. The community-standard LoRA LR
is **1e-4 to 3e-4**, one to two orders of magnitude above full FT. Weight decay
stays 0: the low-rank bottleneck is its own regularizer.

## 6. Activation checkpointing is off (and why)

In Module 15, AC was mandatory — full-FT of 1.7B doesn't fit on 80 GB without
it. Here it's off by default. AC trades compute (a second forward pass) for
memory, and LoRA already solved the memory problem by shrinking the optimizer
state. Leaving AC off keeps the run fast. Flip it on (`--training.activation_checkpointing=true`)
only if you push sequence length or micro-batch size hard enough to run out of
room for activations.

## 7. Running it

```bash
# Canonical LoRA: single GPU, ~30-60 min, ~$1. Fits on a 24GB card.
torchrun --standalone --nproc_per_node=1 train.py --config=configs/lora_qwen3_1.7b.yaml

# QLoRA: 4-bit base, ~6GB. Single GPU only (bitsandbytes ≠ FSDP). Needs:
#   pip install bitsandbytes
torchrun --standalone --nproc_per_node=1 train.py --config=configs/qlora_qwen3_1.7b.yaml

# Multi-GPU standard LoRA — drop grad_accum to hold the effective batch
torchrun --standalone --nproc_per_node=8 train.py \
    --config=configs/lora_qwen3_1.7b.yaml --training.grad_accum=2

# $0-1 smoke on a small public base (downloads in seconds)
torchrun --standalone --nproc_per_node=1 train.py --config=configs/lora_demo.yaml
```

Then **eval** (before/after on the same prompts) and **merge** for deployment:

```bash
# Before: the raw base rambles
python eval.py --base --prompts "Write a haiku about Python"
# After: base + your adapter, kept live
python eval.py --checkpoint=results/checkpoints/step_00000600 \
    --prompts "Write a haiku about Python"

# Merge the adapter into the base → one standard model, zero inference overhead.
# Also dump the standalone adapter (a few MB) for hot-swap serving.
python merge.py --config=configs/lora_qwen3_1.7b.yaml \
    --checkpoint=results/checkpoints/step_00000600 \
    --out=results/merged --adapter-out=results/adapter.pt
```

## 8. Deploying: merge, or don't

A LoRA "model" is really **(base checkpoint, adapter file)**. The adapter is a
few megabytes. You have two deployment modes:

- **Merge** (`merge.py`): fold `ΔW` into `W` once, save a plain HF model. Zero
  inference overhead — the served model is indistinguishable from a fully
  fine-tuned one. This is what you ship when you serve a single fine-tune at
  scale. `merge_lora_weights` in `lora.py` is a four-line in-place fold.
- **Keep separate**: add the adapter at every forward (small latency cost), but
  now you can hot-swap adapters — serve ten fine-tunes off **one** copy of the
  base in memory. This is how multi-tenant LoRA serving (S-LoRA, vLLM's LoRA
  support) works.

**Checkpoints vs. the adapter file.** The DCP checkpoints `train.py` writes
contain the full state (frozen base + adapters) so *resume* is correct on any
cluster shape. The thing you *ship* is the tiny adapter — `merge.py` extracts
it (`--adapter-out`) or folds it in (`--out`).

## 9. LoRA vs full FT: when the trade is wrong

LoRA is the right default for instruction tuning and most preference work. It
is **not** free, and three cases favor full FT:

- **Teaching genuinely new knowledge.** "LoRA Learns Less and Forgets Less"
  (Biderman et al. 2024): on tasks far from the base distribution (a new
  programming language, a new domain's facts), LoRA underperforms full FT and
  the gap grows with how much there is to learn. Knowledge acquisition is a
  *pretraining* job (Module 13), and there full FT wins.
- **Squeezing the last point of quality.** At fixed budget LoRA gets within
  ~1–2 points; if you need the absolute best and can afford it, full FT still
  edges it out.
- **Very high rank.** As `r → min(in,out)`, LoRA *is* full FT with extra steps —
  you've reintroduced the cost you were avoiding.

The flip side — "Forgets Less" — is a feature: because the update is low-rank
and the base is frozen, LoRA damages the model's prior capabilities less than
full FT (a smaller alignment tax, Module 14 §3). For continual / sequential
fine-tuning that matters.

## 10. Gotchas

- **`alpha`/`r` confusion.** People report "LoRA didn't learn" when they set a
  tiny `alpha/r`. Keep `alpha = 2r` unless you know why you're deviating.
- **Forgetting to adapt the MLP.** Adapting only `q_proj`/`v_proj` (the 2021
  default) leaves most of the model's capacity untouched. Adapt all seven.
- **LR too low.** Copying Module 15's `1e-5` into a LoRA run is the most common
  silent failure — the adapter barely moves. Use `2e-4`.
- **QLoRA + FSDP.** bitsandbytes 4-bit tensors don't shard cleanly under FSDP.
  Run QLoRA on a single GPU; for multi-GPU PEFT use standard (bf16) LoRA, or the
  Answer.AI FSDP-QLoRA integration (out of scope here).
- **Merging into a 4-bit base.** You can't fold a bf16 adapter into a 4-bit
  weight. `merge.py` always loads a full-precision base to merge into — correct,
  and the reason it works for QLoRA checkpoints too (it only reads the adapters).
- **Saving the wrong thing.** Don't ship the multi-GB training checkpoint. Ship
  the merged model, or the few-MB adapter.

## 11. Using `peft` in production

We implement LoRA from scratch here because you should understand exactly what
`B·A` is doing — but in production most teams use Hugging Face
[`peft`](https://github.com/huggingface/peft): `get_peft_model(model,
LoraConfig(r=8, ...))` does what `inject_lora_adapters` does, plus DoRA, AdaLoRA,
IA³, prompt tuning, and battle-tested QLoRA+FSDP. The mental model you build in
`lora.py` maps one-to-one onto `peft`'s API; reach for the library once you've
seen the wrapper.

## 12. Stretch goals

- **rsLoRA** (`scaling = alpha/√r`): a more stable scaling at high rank.
- **DoRA** (Liu et al. 2024): decompose weights into magnitude + direction,
  LoRA the direction. A few points better at the same parameter budget.
- **Rank sweep**: train `r ∈ {4, 8, 16, 64}`, plot quality vs trainable params —
  the LoRA "scaling law" for your task.
- **Hot-swap serving**: load one base + two adapters, switch at inference.
- **Apply LoRA to DPO (Module 17)**: the adapter machinery composes with the
  preference loss — `inject_lora_adapters` then run the DPO loop.

## 13. Reading list

- Hu et al. (2021), [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685). The paper. Short and readable.
- Dettmers et al. (2023), [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314). 4-bit NF4, double quant, paged optimizers — and the "adapt all linear layers" finding.
- Aghajanyan et al. (2020), [*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*](https://arxiv.org/abs/2012.13255). Why low-rank works at all.
- Biderman et al. (2024), [*LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673). The honest accounting of where LoRA underperforms full FT — read before you default to LoRA for everything.
- Liu et al. (2024), [*DoRA: Weight-Decomposed Low-Rank Adaptation*](https://arxiv.org/abs/2402.09353). The current strong upgrade.
- The [`peft` docs](https://huggingface.co/docs/peft). The production library.

## 14. What's next

[Module 17 — Preference Optimization](../17-preference-optimization/) — you now
have two ways to produce an instruction-following model (full FT in Module 15,
LoRA here). Either one is the *starting point* for preference optimization: DPO
layers on top of an SFT checkpoint to teach the model which of two responses is
better. The adapter machinery from this module composes with it — you can run
DPO on a LoRA adapter too.
