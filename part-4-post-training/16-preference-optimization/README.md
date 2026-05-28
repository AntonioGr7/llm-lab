# Module 16 — Preference Optimization (DPO / IPO)

You finished [Module 15](../15-sft/) with an SFT model. It follows the chat template and answers questions. It also happily writes worse answers than it could, refuses things it shouldn't refuse, and has no sense of *which* of two plausible completions a human would prefer. SFT can teach the model to produce a valid completion. It can't teach the model to choose between two of them.

This is what preference optimization fixes. You give the model `(prompt, chosen, rejected)` triples — the prompt, a preferred completion, and a worse one — and train it to score the chosen one higher than the rejected one *relative to its current (frozen) self*.

The two new pedagogical ideas in this module:

1. **The DPO loss** — a closed-form way to do RLHF without a reward model, a value head, or PPO. The Bradley-Terry preference model `P(y_w > y_l | x) = σ(r_w - r_l)` plus the KL-constrained reward maximization solution lets you replace `r` with a log-probability ratio of the policy against a frozen reference. The whole RLHF pipeline collapses into a single supervised loss with a sigmoid.
2. **The reference model and β** — DPO holds a *frozen copy* of the policy at step 0 and explicitly penalizes drift away from it. The strength of that anchor (β) controls how far the model can roam from its SFT origin. Choosing β well is the difference between "aligned model" and "mode-collapsed paperweight".

This module is **DPO and IPO** (a 1-line loss-type switch). **KTO** — which uses a different data shape (single completions with binary labels, not pairs) — gets a concept-only section (§ 11) and a pointer to TRL's implementation, rather than a parallel runnable. Same call as Module 15's "full FT only, LoRA deferred": one pedagogical thread per module beats one config flag per topic.

Like Module 15, this is a **framework directory** — lift it out, point it at your own SFT checkpoint and preference data, you have a working DPO codebase. Modules 17-18 (GRPO, distillation) build on this same dual-model machinery.

## The thesis

Preference optimization is **a re-weighting of an SFT'd model**, not a re-training. The math makes this explicit: the loss only depends on log-probability *ratios* between policy and reference, so a position the policy and reference both find equally probable contributes zero gradient. DPO's only job is to widen the gap on `(chosen, rejected)` pairs while penalizing widening on everything else (that's the KL anchor). If you internalize that, you understand the failure modes:

- LR too high → ratios grow everywhere → KL anchor pulled tight → mode collapse.
- β too low → KL anchor weak → ratios diverge → degenerate short outputs.
- Reference is the BASE model (not the SFT) → enormous initial KL → loss optimizes around it → SFT capability degrades.

## What you'll be able to do at the end

1. Derive the DPO loss from KL-constrained reward maximization in under 30 lines of math (§ 4).
2. Implement DPO from scratch in ~50 lines of PyTorch (the actual [loop.py](loop.py)).
3. Pick β, learning rate, and training duration for your model + dataset from first principles (§ 5, § 6).
4. Read TRL's `DPOTrainer` source and recognize every line.
5. Decide when to reach for DPO vs IPO vs KTO (§ 10, § 11).

## 1. Directory layout

```
16-preference-optimization/
├── README.md              you are here
├── notebook.ipynb         CPU-only tour: the DPO loss math, β sweep, IPO vs DPO
├── config.py              TrainConfig — model (policy + reference), data, preference, optimizer
├── model.py               build_policy + build_reference (FP32 vs BF16-frozen)
├── data.py                PreferenceDataset — chosen/rejected with assistant-only masks
├── loop.py                gather_response_logps, compute_dpo_loss, forward_dpo, train_step
├── train.py               torchrun entrypoint — FSDP policy + FSDP-sharded frozen reference
├── eval.py                generation A/B + held-out preference accuracy
├── fsdp_setup.py          copied from Module 15
├── optim.py               copied from Module 15
├── schedule.py            copied from Module 15
├── checkpoint.py          copied from Module 15
├── efficiency.py          copied from Module 15
├── configs/
│   ├── dpo_demo.yaml      Qwen3-0.6B + tiny slice — minutes, any GPU
│   └── dpo_qwen3_1.7b.yaml  Qwen3-1.7B (SFT'd) — ~$4-6 on A100
├── tests/                 loss math + data pipeline correctness
└── results/               pre-run loss curves + checkpoint for the canonical run
```

NEW components: `model.py` (two models), `data.py` (preference pairs, not chat), `loop.py` (DPO loss). Everything else is copied from Module 15.

## 2. From RLHF to DPO in one page

The original RLHF pipeline (Ouyang et al., 2022) does three things:

1. **SFT** — fine-tune a base model on demonstrations.
2. **Train a reward model** `r_φ(x, y)` on pairwise preferences using the Bradley-Terry loss: `P(y_w > y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))`.
3. **PPO-finetune the policy** to maximize `E[r_φ(x, y)] - β · KL(π || π_ref)`. The KL term anchors the policy near the SFT model so the reward model isn't gamed.

This works. It's also operationally brutal: you need a separate reward-model training stage, a value head, GAE, PPO with KL clipping, distributed actor-critic rollouts, and a babysitter to catch reward hacking.

**DPO's observation** (Rafailov et al., 2023). The optimal policy under the KL-constrained reward maximization in step 3 has a *closed form*:

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x, y) / β)
```

Solving for the reward:

```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

The partition function `Z(x)` depends only on the prompt. Plug this into the Bradley-Terry loss for the preference dataset — `Z(x)` is the same on chosen and rejected and **cancels**:

```
P(y_w > y_l | x) = σ( β · [log π*(y_w|x)/π_ref(y_w|x)
                         - log π*(y_l|x)/π_ref(y_l|x)] )
```

This is a function of the policy alone (the optimal one), with the reference appearing as a frozen log-density. Training maximizes the log-likelihood of this preference probability — that's a standard supervised loss:

```
L_DPO = -log σ( β · [ Δ_logratio(y_w) - Δ_logratio(y_l) ] )

   where  Δ_logratio(y) = log π(y|x) - log π_ref(y|x)
```

That's it. No reward model. No value head. No PPO. Just two log-probability differences inside a log-sigmoid. The reward-model training stage was conceptually unnecessary all along — the policy *is* the reward model. The KL anchor is implicit in the logratio form: when the policy hasn't moved from the reference, all log-ratios are 0 and so is the gradient.

## 3. The loss in code

```python
def compute_dpo_loss(
    policy_chosen_logps, policy_rejected_logps,
    ref_chosen_logps, ref_rejected_logps,
    beta,
):
    chosen_logratios   = policy_chosen_logps   - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    margin = chosen_logratios - rejected_logratios
    loss = -F.logsigmoid(beta * margin).mean()
    return loss
```

The whole module hangs on those four log-probabilities per pair. Each is the *sum* of per-token log-probs over the assistant response — exactly the same response-only mask Module 15's SFT used, applied via [gather_response_logps](loop.py). The data pipeline ([data.py](data.py)) is doing nothing fancier than rendering `(prompt, chosen)` and `(prompt, rejected)` through the chat template, computing the assistant-only mask on each, and stacking them so the model sees a single 2B-row forward per micro-batch.

## 4. The IPO variant

IPO (Azar et al., 2024) keeps the same logratio difference but swaps `-log σ(·)` for a squared loss with an explicit target:

```
L_IPO = ( margin - 1/(2β) )^2
```

Two consequences:

- **DPO** is unbounded above when the margin is very negative (`-log σ(β·m)` → `-β·m` linearly), and it asks the model to push the margin to `+∞`. On unanimous-preference pairs (every annotator agrees), DPO's gradient never stops growing — leading to over-confident, often degenerate completions.
- **IPO** asks for a SPECIFIC margin (`1/(2β)`, ≈ 5 for β=0.1). Past that, the loss increases again. The model isn't allowed to push the logratio arbitrarily far on any one pair, which is more robust to label noise and to high-confidence-but-not-actually-good pairs.

Switching is one line in the config — `preference.loss_type: ipo`. The data pipeline is identical. Whether to use IPO vs DPO is empirical and dataset-dependent; § 10 of the [IPO paper](https://arxiv.org/abs/2310.12036) and the Hugging Face DPO trainer docs both have head-to-head comparisons.

We also expose **label_smoothing** (`α ∈ [0, 0.5)`) for cDPO ("conservative DPO"). With smoothing α the loss becomes a convex combination treating the chosen-preferred event as `(1-α)` probable and the rejected-preferred event as `α` — useful when your preference labels themselves are 10-20% noisy (human annotation, weak LLM judges). The math drops out of the same DPO derivation by replacing the deterministic preference with a smoothed one.

## 5. The β parameter

β is the KL anchor strength. Mechanically, it scales the logratio difference inside the sigmoid:

```
L_DPO = -log σ(β · margin)
```

Low β → flat sigmoid → small gradient even for big margin → the policy is allowed to drift far from the reference. High β → sharp sigmoid → small differences in logratio matter → policy is pinned to the reference.

Symmetric framing in IPO: low β → target margin `1/(2β)` is HUGE → IPO asks the policy to make extreme logratio differences → drifts far. High β → target margin is small → tightly anchored.

| β | Behavior | When to use |
|---|---|---|
| 0.01-0.05 | Very loose anchor. Policy can move far. | Your SFT is weak; you need lots of behavior change. |
| **0.1** | **Standard default.** Zephyr, Tülu, most published recipes. | First thing to try. Almost always the answer. |
| 0.3-1.0 | Tight anchor. Policy stays close to reference. | Your SFT is strong; you only want a polish pass. |
| > 1.0 | Effectively no learning. | (don't) |

Symptoms of a wrong β:

- Loss going down fast but generation degrades (short, repetitive, off-topic) → β too low. Drifted past the SFT manifold.
- Loss barely moves over 500 steps → β too high. Policy is anchored too tightly to learn.
- Eval `accuracy` hits 90%+ but generation is worse than reference → length bias / over-fit to the preference distribution. Try IPO or shorter training.

## 6. The training configuration

The canonical demo: DPO on `HuggingFaceH4/ultrafeedback_binarized` — 62k cleaned preference pairs from UltraFeedback (Cui et al., 2024), the dataset Zephyr was trained on.

```yaml
model:
  name: Qwen/Qwen3-1.7B           # an SFT'd model
  ref_name: ""                     # "" -> policy and reference both start from `name`
  max_seq: 1024

data:
  source: HuggingFaceH4/ultrafeedback_binarized
  split: train_prefs
  seq_len: 1024
  batch_size_per_device: 2         # effective per-rank-batch = 2 × 2 = 4 sequences

preference:
  loss_type: dpo
  beta: 0.1

optimizer:
  type: adamw
  lr: 5.0e-7                       # 20× lower than SFT, 600× lower than pretraining
  betas: [0.9, 0.999]
  weight_decay: 0.0

schedule:
  type: cosine
  warmup_steps: 50

training:
  total_steps: 1000                # ~1 epoch over 60k pairs at effective batch 32
  grad_accum: 16
  grad_clip: 1.0
  dtype: bf16
  activation_checkpointing: true
```

Expected wallclock on a single A100-80GB: ~2-3 hours. Cost: $4-6 on RunPod.

**Why the LR is even lower than SFT.** SFT moves the model from "base" to "instruction-following". DPO refines an already-aligned model along a narrow direction (the preference signal). At SFT-scale learning rates, DPO will overshoot the local optimum and the SFT'd capability erodes. The Zephyr / Tülu / TRL defaults all land between 1e-7 and 5e-7 for 1B+ models; ours is at 5e-7. If your loss is flat after 100 steps, double it; if generation quality drops, halve it.

## 7. Memory: holding two models

DPO loads TWO copies of the model. For Qwen3-1.7B with the standard mixed-precision setup:

| Component | Per param | 1.7B params | Notes |
|---|---|---|---|
| Policy: master weights (FP32) | 4 | 6.8 GB | Optimizer updates these |
| Policy: compute weights (BF16) | 2 | 3.4 GB | FSDP MixedPrecisionPolicy casts |
| Policy: gradients (BF16) | 2 | 3.4 GB | Reduced in FP32 |
| Policy: AdamW state (FP32 m+v) | 8 | 13.6 GB | |
| Reference: BF16 only, no grad | 2 | 3.4 GB | No optimizer, no FP32, no grad |
| **Subtotal** | | **30.6 GB** | |
| Activations (with AC, seq=1024, bs=2×2) | | ~8 GB | Effective bs=4 (chosen+rejected) |
| **Total per rank** | | **~38-40 GB** | |

Fits on a single A100-80GB. Without AC, activations balloon and the run OOMs — `activation_checkpointing: true` is mandatory, not optional, in the default config.

On multi-GPU FSDP shards both models 1/world across ranks, so per-rank VRAM falls roughly linearly. The reference is FSDP-wrapped purely for sharding parity (no gradients, no optimizer); the `apply_fsdp` call in [train.py](train.py) handles this uniformly.

**Stretch goal — reference-logprob caching.** Compute `(log π_ref(chosen), log π_ref(rejected))` once over the whole dataset, save to disk, then train with only the policy in VRAM (saving ~3.4 GB at 1.7B). TRL's `DPOTrainer` does this by default. We don't, for two pedagogical reasons: (a) the cached path looks like SFT with weird labels and obscures what the reference IS, and (b) you can change `model.ref_name` and re-run without re-preprocessing. If you're memory-pinched, see § 13.

## 8. Running it

```bash
# Recommended: single A100-80GB on RunPod / Lambda Labs
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/dpo_qwen3_1.7b.yaml

# 8-GPU node — keep effective batch constant by dropping grad_accum
torchrun --standalone --nproc_per_node=8 train.py \
    --config=configs/dpo_qwen3_1.7b.yaml \
    --training.grad_accum=2

# Try IPO instead with a one-line override
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/dpo_qwen3_1.7b.yaml \
    --preference.loss_type=ipo

# Dev smoke test (~5 min, any GPU) — uses Qwen3-0.6B as both policy and reference
torchrun --standalone --nproc_per_node=1 train.py --config=configs/dpo_demo.yaml
```

> **`dpo_demo.yaml` uses `Qwen/Qwen3-0.6B`** — Alibaba's already-SFT'd 0.6B instruct model. The whole codepath is exercised in ~5 minutes on any GPU; this is for sanity, not for producing a usable model. The canonical demo is `dpo_qwen3_1.7b.yaml`. If you ran Module 15 yourself, swap the canonical config's `model.name` to your exported SFT checkpoint directory (see Module 15's `export_hf.py`).

## 9. What you should see

Training metrics, in order of importance:

- **`margin`** — the chosen-rejected logratio difference. Starts at 0 (policy == reference at step 0); grows over training. At the end of a healthy run you should see margins of 2-5 nats/pair on average.
- **`accuracy`** — fraction of pairs where the policy ranks chosen > rejected. Starts at ~50% (random) and climbs. On ultrafeedback_binarized you should reach 65-80% after 1 epoch.
- **`chosen_rewards`** and **`rejected_rewards`** — β times the per-side logratios. Both should move from 0: chosen UP, rejected DOWN. If only one moves (usually only rejected falls), your reference is misaligned with your data or your LR is wrong.
- **`loss`** — should decline from ~0.69 (= log 2, the random-margin baseline) toward 0.3-0.5. Not informative as a single number; the metrics above tell you what's happening.

**Generation A/B**:

```bash
python eval.py --checkpoint=results/checkpoints/step_00001000 \
    --prompts="Write a haiku about Python" "Explain RNNs to a 10-year-old" \
    --also-reference
```

Before DPO, the SFT'd model answers competently but generically. After DPO on ultrafeedback_binarized, you should see longer, more structured, more "helpful-assistant" responses. The shift is subtle — DPO doesn't add new capabilities, it shifts the *distribution over completions*. If you can't tell the difference qualitatively, run `--accuracy` to see the quantitative shift.

**Preference accuracy**:

```bash
python eval.py --checkpoint=results/checkpoints/step_00001000 --accuracy
```

Reports the policy's accuracy on a held-out slice of `test_prefs`, alongside the reference's accuracy (the baseline). A healthy DPO run should beat the reference by 10-20 percentage points.

## 10. Gotchas

- **Length bias.** DPO sums log-probs over the entire response — *longer responses receive more gradient signal*. The model learns "longer = preferred" as a side effect. This is well-documented (Singhal et al., 2024). Mitigations: (a) length-normalized DPO (divide each logratio by response length — one line in [gather_response_logps](loop.py)); (b) SimPO (Meng et al., 2024) which does exactly that plus a margin term; (c) filter your preference dataset to length-balanced pairs.
- **Train/inference template mismatch.** Same risk as SFT (Module 15 § 2). [data.py](data.py) and [eval.py](eval.py) both render with `enable_thinking=False`; if you change one, change the other.
- **Reference drift via `ref_name`.** Setting the reference to the *base* model instead of the SFT model can be a useful trick (a stronger anchor away from arbitrary drift), but it interacts with β in non-obvious ways: the initial KL is huge, the loss gradient is dominated by it, and the SFT capability erodes faster. The textbook setup is `ref_name == policy.name`; deviate with a measurement.
- **Loss going to 0 fast.** Means the model is finding a degenerate fixed point — usually because β is too low or LR too high. Check `chosen_rewards` and `rejected_rewards`: if only `rejected_rewards` is moving (down), the model is just learning to suppress the rejected response without raising the chosen one. The fix is more β or less LR.
- **The reference must match the policy's tokenizer.** Both forwards use the same `input_ids`. If you accidentally point `ref_name` at a different model family (e.g. Llama-3 reference for a Qwen3 policy), the tokenizer mismatch silently breaks everything — token IDs map to different concepts. The tokenizer is loaded from `model.name` only; we use it for both.

## 11. KTO — the third major variant (concept-only)

KTO (Kahneman-Tversky Optimization, Ethayarajh et al., 2024) drops the pair requirement. Instead of `(prompt, chosen, rejected)` triples, KTO works on `(prompt, completion, label ∈ {desirable, undesirable})` rows — a single completion with a binary label.

Why this matters operationally: **preference pairs are expensive**. You need two completions on the same prompt with a human (or LLM) ranking. Binary labels are cheap — every "thumbs up / thumbs down" in your product is a free KTO datapoint. Anthropic-style HH-RLHF data, helpfulness scores, even "did the user re-prompt?" signals can drive KTO.

The KTO loss is derived from prospect theory: humans weight losses ~2× more than gains. The loss has the same logratio difference vs reference, but instead of a pair-margin form it uses an *individual* loss per (completion, label) that anchors against the average reference KL across the batch.

We don't ship a runnable KTO path because it needs a different data shape (single-completion + binary label), which would require a parallel `data.py`. The pedagogically cleanest separation is: this module = pair methods (DPO + IPO), KTO = its own follow-up if you need to teach it.

**To run KTO in practice today**, use TRL's `KTOTrainer`. The setup is otherwise identical to our DPO pipeline: same SFT model, same chat template, same FSDP wrap; only `data.py` and `loop.py` change.

## 12. Stretch goals

- **Length-normalized DPO.** Divide each `gather_response_logps` output by `(labels != IGNORE_INDEX).sum()`. One line. Reduces length bias dramatically; converges to a slightly different optimum.
- **Iterative DPO.** Run DPO, then use the new policy to generate preference pairs (judged by the *previous* policy as reward model), then DPO again. The trick used by Llama-3 and Tülu 3. Add a `--ref_name=results/checkpoints/step_00001000` override and re-run on a new generation-derived dataset.
- **SimPO.** Length-normalize the logratio + add an explicit reward margin term + drop the reference model entirely. One paper-sized change to [loop.py](loop.py); often outperforms DPO in head-to-head.
- **Reference-logprob caching.** Pre-compute `log π_ref(chosen|x), log π_ref(rejected|x)` for the whole dataset once, save to disk, train without the reference in VRAM. TRL's `precompute_ref_log_probs=True`. Halves peak memory.
- **DPO on synthetic pairs.** Use a frontier model (GPT-4o, Claude) to generate `(chosen, rejected)` from your prompts. Constitutional AI / RLAIF in a config flag.

## 13. Reading list

In order of "read this first":

- **[Rafailov et al. (2023) — Direct Preference Optimization](https://arxiv.org/abs/2305.18290).** The DPO paper. § 4 (derivation) is canonical; § 6 (analysis) explains *why* it works. Read it cover-to-cover before reading anything else.
- **[Ouyang et al. (2022) — Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155).** The pre-DPO RLHF pipeline. You need to understand what DPO replaced.
- **[Tunstall et al. (2023) — Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944).** The first major DPO-trained model. Their recipe is what `dpo_qwen3_1.7b.yaml` mirrors — β=0.1, lr=5e-7, ultrafeedback_binarized.
- **[Azar et al. (2024) — A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036).** The IPO paper. § 5.1 has the closed-form derivation of IPO from preference learning; § 6 has the head-to-head with DPO.
- **[Cui et al. (2024) — UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://arxiv.org/abs/2310.01377).** The dataset card for what you're training on. Important to understand how the labels were generated (GPT-4 judging) — that determines what behaviors you'll instill.
- **[Ethayarajh et al. (2024) — KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306).** The KTO paper. Read for the binary-label angle and the prospect-theory justification.
- **[Meng et al. (2024) — SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734).** The reference-free + length-normalized variant. Often the SOTA on benchmark leaderboards.
- **[Singhal et al. (2024) — A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716).** The length-bias paper. Read after running DPO so you have the empirical context to recognize the bias.

## 14. What's next

[Module 17 — Reasoning and GRPO](../17-reasoning-and-grpo/) — DPO teaches the model *which completion is better* among ones it can already write. GRPO teaches it to *generate completions it couldn't write before* by rewarding verified-correct outcomes (math, code). Same dual-model machinery (policy + reference + KL anchor); a new loss + actual on-policy rollouts.
