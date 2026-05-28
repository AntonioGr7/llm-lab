# Module 17 — Reasoning and GRPO

You finished [Module 16](../16-preference-optimization/) with a DPO'd model: an instruction-following assistant whose preferred completions are now preferred — more helpful, better formatted, polite. What it still can't do is *reason its way through a problem it hasn't seen*. Ask it `If 6 cats catch 6 mice in 6 minutes, how many cats catch 100 mice in 50 minutes?` and it confidently writes an answer that's wrong about half the time. The capability has to be *trained as a behavior*, not picked up from style.

This is what GRPO (DeepSeek-R1's algorithm) fixes. You define a *verifiable reward* — a rule-based function that checks whether the model's final answer is correct — and let the model figure out by trial and error which response shapes get higher reward. The shocking result from R1 was that under this regime, **chain-of-thought emerges from raw policy gradient**: response length grows, `<think>` blocks appear unprompted, the model starts revising its own intermediate steps. No demonstrations, no SFT trace, just outcome reward.

The new pedagogical ideas in this module:

1. **On-policy rollouts.** Unlike SFT (fixed demonstrations) or DPO (pre-collected pairs), GRPO consumes data the model *generates itself*, at training time. Each step samples G completions per prompt, scores them, and learns from the relative differences. This is what makes the loop expensive — most of each step is generation, not gradient — and what makes the policy improvable beyond its initial distribution.
2. **Group-relative advantages.** PPO learns a value head to estimate the per-state baseline reward. GRPO replaces that with a within-prompt mean: "compared to your other G−1 attempts on this prompt, was THIS one better?". No critic network, no GAE, no learned baseline. The `GR` in GRPO is the whole point.
3. **Verifiable reward.** Instead of training a neural reward model from human preferences, write a function that checks correctness. For math: parse the answer, compare to ground truth. For code: run the tests. This sidesteps the entire reward-model-stage of classical RLHF and makes the reward provably non-gameable on its target distribution (a 7 is a 7 — you can't hack that).

Like Modules 15 and 16, this is a **framework directory** — lift it out, point it at your own SFT/DPO checkpoint and a verifier function for your domain, you have a working GRPO codebase. Module 18 (distillation) builds on the same `policy + reference + per-token loss` machinery.

## The thesis

GRPO is a **structured kind of policy gradient** — same `E[A · ∇ log π(a|s)]` shape you'd see in any RL textbook — with three engineering decisions that make it work in the modern LLM regime:

- The *baseline* (the thing you subtract from raw reward to reduce variance) is the within-group mean reward, not a learned value head. Cheaper and more stable.
- The *KL anchor* is per-token (k3 estimator) against a frozen reference, not sequence-level. Keeps the model from drifting into degenerate-but-high-reward strategies.
- The *reward* is rule-based and verifiable, not modeled. Avoids reward hacking on the target distribution; trades off coverage (only works where you have a verifier).

If you internalize that, you understand the failure modes:

- LR too high → the policy moves faster than the KL anchor can hold → mode collapse on a tiny subset of completions.
- Verifier too lenient → the model finds a degenerate shortcut (e.g. just always emit "42") that satisfies the verifier without actually reasoning.
- G too small → group-mean baseline is high-variance → advantages noisy → updates inconsistent.
- max_new_tokens too short → policy can't complete the reasoning chain → all rollouts score 0 → no signal.

## What you'll be able to do at the end

1. Write the GRPO loss from scratch given (current, old, ref) per-token logps and group-normalized advantages (§ 3).
2. Design a verifiable reward function for a new domain in ~20 lines (§ 4).
3. Pick group size, KL β, and clip ratio from first principles (§ 5).
4. Read the DeepSeekMath paper and recognize every line.
5. Decide when GRPO is the right tool vs DPO / SFT / distillation (§ 9).

## 1. Directory layout

```
17-reasoning-and-grpo/
├── README.md              you are here
├── notebook.ipynb         CPU-only tour: GRPO loss math + group advantages + reward verifier
├── config.py              TrainConfig — model, data, reward, rl (GRPO knobs), optimizer
├── model.py               build_policy + build_reference + generation_mode helper
├── data.py                GSM8K loader — prompt-only chat-template + ground-truth parse
├── rewards.py             extract_answer, format_reward, accuracy_reward, compute_rewards
├── rollout.py             generate G completions per prompt, score, build training tensors
├── loop.py                compute_grpo_loss + train_step
├── train.py               torchrun entrypoint — rollout + K gradient steps per RL step
├── eval.py                GSM8K test-set accuracy + generation A/B
├── fsdp_setup.py          copied from Module 16
├── optim.py               copied from Module 16
├── schedule.py            copied from Module 16
├── checkpoint.py          copied from Module 16
├── efficiency.py          copied from Module 16
├── configs/
│   ├── grpo_demo.yaml         Qwen3-0.6B, 20 steps, ~10-15 minutes on any GPU
│   └── grpo_qwen3_1.7b.yaml   Qwen3-1.7B, 300 steps, ~4-8h on A100, ~$8-15
├── tests/
│   ├── test_rewards.py        the verifier
│   └── test_grpo_loss.py      the loss math + geometry
└── results/                   pre-run loss curves + checkpoint
```

NEW components vs Module 16: `rewards.py` (the verifier), `rollout.py` (on-policy generation + advantage), the RL-specific `train_step` in `loop.py`. Everything else is the same shape.

## 2. From PPO to GRPO in one page

Classical RLHF with PPO (Schulman et al., 2017) needs four moving pieces:

1. **Policy** — the LM being trained.
2. **Value head** — a separate scalar output trained to predict the expected future reward at each state. Used as the *baseline* in the advantage estimate.
3. **Reward model** — a separate neural net trained on human preferences (Module 16's frozen DPO predecessor).
4. **PPO update** — clipped surrogate `E[min(ρ·A, clip(ρ)·A)]` + KL penalty against the reference.

GRPO (Shao et al., 2024 — DeepSeekMath) drops piece (2) and replaces piece (3) with a hand-written verifier. The remaining objective:

```
For each prompt q:
    Sample G completions o_1, ..., o_G from π_θ(·|q)
    Score each:  r_i = verifier(o_i, ground_truth(q))
    Standardize within the group:
        A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)

Loss (per token t in completion o_i):
    L_t = -min( ρ_t · A_i,  clip(ρ_t, 1-ε_clip, 1+ε_clip) · A_i )  +  β · KL_t

where:
    ρ_t   = exp( log π_θ(o_t | q, o_<t) - log π_θ_old(o_t | q, o_<t) )
    KL_t  = exp(δ_t) - δ_t - 1     # k3 estimator (Schulman 2020)
    δ_t   = log π_ref(o_t | ...) - log π_θ(o_t | ...)
```

The pieces this kills:

- **Value head** → replaced by within-group standardization. The mean reward over G samples on the same prompt is a perfectly good baseline estimator; no training needed. This is the single most important simplification of GRPO.
- **Reward model** → replaced by `verifier(o, gt)`. For math, code, anything with a checkable answer, this is a pure function — no separate training stage, no reward hacking on out-of-distribution responses (the verifier doesn't generalize, which is the *point*).
- **GAE / advantage estimation across timesteps** → replaced by the constant per-completion advantage A_i broadcast across all tokens in that completion. R1 calls this "outcome-supervised" advantage and notes it's slightly worse than process-supervised for short tasks but scales much further.

What survives from PPO: the importance ratio + clip (so you can re-use a rollout for multiple gradient steps — `mu_epochs = K`), and the KL anchor against a frozen reference (the same KL anchor DPO uses, here written as a per-token penalty instead of a sequence-level log-sigmoid).

## 3. The loss in code

The whole thing is ~30 lines of PyTorch:

```python
def compute_grpo_loss(current_logps, old_logps, ref_logps,
                     advantages, response_mask, rl_cfg):
    mask_f = response_mask.float()
    n_tokens = mask_f.sum().clamp(min=1.0)

    # 1. Importance ratio
    log_ratio = current_logps - old_logps
    ratio = torch.exp(log_ratio)

    # 2. Clipped surrogate
    A = advantages.unsqueeze(1)
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - rl_cfg.clip_ratio, 1.0 + rl_cfg.clip_ratio) * A
    policy_loss = (-torch.min(surr1, surr2) * mask_f).sum() / n_tokens

    # 3. Per-token KL (k3 estimator, non-negative)
    delta = ref_logps - current_logps
    kl_per_token = torch.exp(delta) - delta - 1.0
    kl_per_token = torch.clamp(kl_per_token, min=0.0)
    kl_loss = rl_cfg.kl_beta * (kl_per_token * mask_f).sum() / n_tokens

    return policy_loss + kl_loss
```

The math hangs on four per-token logps: `current` and `old` (under the policy at different moments), and `ref` (under the frozen reference). The data pipeline is doing nothing fancier than generating G completions per prompt and computing those four numbers per token of each completion's response span.

## 4. Reward design — the verifier

The whole GRPO setup hinges on `compute_rewards(completion_text, ground_truth) -> float`. For GSM8K our verifier is two components summed:

**Format reward** (`w_format = 0.1`): 1.0 if the response matches `<think>...</think><answer>...</answer>`, else 0. This is a "training-wheels" term. Early in training, before the policy can solve any problem, accuracy is 0 across all G samples and the group-mean baseline collapses; the format bonus provides a non-zero signal that pulls the policy toward the expected response shape. By mid-training, accuracy dominates.

**Accuracy reward** (`w_accuracy = 1.0`): 1.0 if the integer parsed from the last `<answer>...</answer>` block equals the ground truth, else 0. This is the actual task signal. R1 famously achieved emergent reasoning with *only* this term + a format bonus — no reward shaping on the reasoning trace itself.

Three principles for verifiers, learned the hard way by every team that's tried this:

1. **Make the verifier strict.** If you accept "approximately right" answers (e.g. floats within ε), the policy will learn to game the tolerance. Better to mark borderline answers wrong and accept slower convergence.
2. **Verify the OUTCOME, not the TRACE.** Rewarding "looks like good reasoning" lets the policy learn to *produce text that looks like reasoning* without actually reasoning. R1 verifies only the final answer; the reasoning is incidental.
3. **Reward sparsity is OK; reward gaming is fatal.** It's fine if 80% of your rollouts get reward 0. It's not fine if the model finds a degenerate response (e.g. emit `<answer>42</answer>` and walk away) that satisfies the verifier without solving the task. The fix is always to make the verifier stricter, not to soften the reward.

The verifier is in [rewards.py](rewards.py). Swap the GSM8K integer-parse for `subprocess.run(['python', 'test.py'])` and you have a code-RL setup; swap for unit-test pass count and you have agent-RL; swap for "an LLM judge scores this between 0-1" and you have RLAIF.

## 5. Hyperparameters

Three knobs matter, in order of impact:

**`rl.group_size` (G).** The number of completions per prompt. DeepSeekMath uses G=64; we default G=8 for a 1.7B model on a single A100 (memory and time scale linearly with G). Smaller G means a noisier baseline (within-group mean has higher variance), but every doubling of G doubles your rollout cost. The cheap-but-still-good range is 8-16; below 4 the baseline collapses too often.

**`rl.kl_beta` (β).** The per-token KL anchor strength. DeepSeekMath ships β=0.04. Two effects of increasing β: the policy stays closer to the reference (tighter anchor, less drift), AND the KL loss adds to the gradient norm so the effective LR drops slightly. Symptoms of mis-tuning:

| Symptom | Diagnosis | Fix |
|---|---|---|
| Accuracy climbs then collapses fast | β too low — policy drifted to a degenerate mode | β ×2 |
| Accuracy doesn't move in 50+ steps | β too high — policy can't move | β ÷2 |
| KL grows unboundedly past ~0.5 | Anchor is broken (e.g. wrong ref_name) | check `model.ref_name` |

**`rl.clip_ratio` (ε).** PPO's ratio clip. At `mu_epochs=1` (one gradient step per rollout) the ratio is exactly 1 and ε is a no-op — leave at the default 0.2. For `mu_epochs > 1`, ε caps how much the policy can move per rollout reuse; smaller ε means more on-policy at the cost of using less of each rollout's gradient signal.

**Sampling temperature (`rl.temperature`).** Affects rollout *diversity*, not the gradient directly. Higher T → more diverse G samples → more spread in rewards → bigger group-relative advantage when one sample is much better than others. R1 uses T=1.0; we default 0.9 to converge a bit faster on the modest 1.7B budget. If you see all G samples within a group giving the same response, raise T.

## 6. The training configuration

The canonical demo: GRPO on `openai/gsm8k` — the standard math word-problem benchmark.

```yaml
model:
  name: Qwen/Qwen3-1.7B           # an already SFT'd model
  ref_name: ""                     # "" -> use `name` as the frozen reference
  max_seq: 2048

data:
  source: openai/gsm8k
  subset: main
  split: train
  seq_len: 1024
  prompts_per_step: 4              # P=4 prompts per RL step

reward:
  w_format: 0.1                    # small schema bonus (training wheels)
  w_accuracy: 1.0                  # accuracy is what we actually optimize

rl:
  group_size: 8                    # G=8 completions per prompt
  temperature: 0.9
  top_p: 0.95
  max_new_tokens: 512              # GSM8K CoT rarely needs more
  kl_beta: 0.04                    # DeepSeekMath default
  clip_ratio: 0.2
  mu_epochs: 1                     # K=1 (one gradient step per rollout)

optimizer:
  type: adamw
  lr: 1.0e-6                       # ~2× DPO's; see § 5
  betas: [0.9, 0.999]
  weight_decay: 0.0

schedule:
  type: constant                   # GRPO standard — short runs, no decay
  warmup_steps: 20

training:
  total_steps: 300                 # ~30 GSM8K passes at P=4, G=8
  grad_accum: 1                    # rollouts already aggregate the G·P signal
  grad_clip: 1.0
  dtype: bf16
  activation_checkpointing: true
```

Expected wallclock on a single A100-80GB: ~4-8 hours. Cost: $8-15 on RunPod/Lambda Labs.

**Why lower LR than SFT but higher than DPO.** SFT (1e-5) moves the model a long way from base to instruct on dense demonstration signal. DPO (5e-7) refines an already-aligned model on dense preference signal. GRPO (1e-6) refines an already-aligned model on *sparse* reward signal — most groups will have all-zero or near-degenerate rewards (and contribute no policy gradient). To compensate for that sparsity without destabilizing, we lift the LR ~2× over DPO while keeping the KL anchor tight (β=0.04, lower than DPO's β=0.1, BUT per-token rather than per-sequence).

## 7. Memory: two models plus rollouts

GRPO is the most memory-hungry stage of the post-training pipeline. For Qwen3-1.7B:

| Component | Per param | 1.7B params | Notes |
|---|---|---|---|
| Policy: master weights (FP32) | 4 | 6.8 GB | Optimizer updates these |
| Policy: compute weights (BF16) | 2 | 3.4 GB | FSDP MixedPrecision casts |
| Policy: gradients (BF16) | 2 | 3.4 GB | Reduced in FP32 |
| Policy: AdamW state (FP32 m+v) | 8 | 13.6 GB | |
| Reference: BF16 only, no grad | 2 | 3.4 GB | No optimizer, no FP32, no grad |
| **Static subtotal** | | **30.6 GB** | |
| Rollout KV cache (G·max_new × cache_per_token) | | ~4-8 GB | G=8, max_new=512 |
| Activations (AC on, seq=1024+512, batch=2) | | ~6-10 GB | |
| **Total per rank** | | **~45-50 GB** | |

Fits on a single A100-80GB with comfortable headroom. Without activation checkpointing it OOMs — `activation_checkpointing: true` is mandatory in the default config.

On multi-GPU FSDP, both models shard 1/world across ranks. Each rank still holds its own KV cache during rollouts (KV cache lives on a single GPU per sequence; we generate one prompt at a time per rank).

**Stretch goal — vLLM for rollouts.** Production GRPO setups (Tülu 3, OpenRLHF) use vLLM for the rollout phase — 5-10× faster than HF `generate` thanks to continuous batching and paged-attention KV cache. We don't, for two reasons: (a) you'd ship two model copies anyway (one for training, one for vLLM inference), increasing memory; (b) the vLLM integration adds a major dep that distracts from the pedagogy. If you're scaling past 7B, see [verl](https://github.com/volcengine/verl) or [openrlhf](https://github.com/OpenRLHF/OpenRLHF).

## 8. Running it

```bash
# Recommended: single A100-80GB
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/grpo_qwen3_1.7b.yaml

# 8-GPU node — drop prompts_per_step to keep the effective rollout size constant
torchrun --standalone --nproc_per_node=8 train.py \
    --config=configs/grpo_qwen3_1.7b.yaml \
    --data.prompts_per_step=1

# Lower KL anchor — let the policy drift more (riskier, but faster early gains)
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/grpo_qwen3_1.7b.yaml \
    --rl.kl_beta=0.01

# Dev smoke (~10-15 min, any GPU) — Qwen3-0.6B, tiny GSM8K slice
torchrun --standalone --nproc_per_node=1 train.py --config=configs/grpo_demo.yaml
```

> **`grpo_demo.yaml` uses `Qwen/Qwen3-0.6B`** for cheap end-to-end exercise; this doesn't produce a useful reasoner (too small, too few steps). The canonical demo is `grpo_qwen3_1.7b.yaml`. If you ran Modules 15 (SFT) or 16 (DPO) yourself, swap the canonical config's `model.name` to your exported checkpoint directory.

## 9. What you should see

Training metrics, in order of importance:

- **`acc`** — fraction of rollouts whose final answer equals ground truth. THE headline curve. Should climb from ~10-30% (base) to 50-70% over the run.
- **`fmt`** — fraction of rollouts with the schema. Should climb to ~95% in the first 50-100 steps (the format bonus + sparse accuracy reward make this the easiest signal to learn) and then plateau.
- **`len`** — average tokens per completion. Should GROW as reasoning emerges. R1 reports response length tripling over a long run; for our 1.7B / 300-step demo, expect length to grow from ~100 to ~250-400 tokens.
- **`reward`** — total weighted reward. Combines fmt and acc; useful as a single-number summary.
- **`kl`** — mean per-token KL against the reference. Should rise from ~0 to ~0.1-0.3 nat/token over training. If it shoots past 0.5, β is too low.
- **`clipfrac`** — fraction of tokens where the PPO clip kicked in. At K=1 (`mu_epochs=1`) this is exactly 0 — the ratio is 1.0 by construction. If you raise mu_epochs, watch this — values past 0.3 mean you're throwing away too much gradient signal and should reduce mu_epochs.

**Qualitative A/B**:

```bash
python eval.py --config=configs/grpo_qwen3_1.7b.yaml \
    --checkpoint=results/checkpoints/step_00000300 \
    --prompts "If 6 cats catch 6 mice in 6 minutes, how many cats catch 100 mice in 50 minutes?" \
    --also-reference
```

The GRPO'd policy should produce a structured `<think>...</think><answer>...</answer>` response with multi-step arithmetic. The reference (untrained) typically produces either no schema or a one-shot answer with no reasoning.

**Quantitative GSM8K accuracy**:

```bash
python eval.py --config=configs/grpo_qwen3_1.7b.yaml \
    --checkpoint=results/checkpoints/step_00000300 --gsm8k --n=500
```

Reports overall accuracy on GSM8K test. The Qwen3-1.7B baseline (the un-GRPO'd policy used as reference) lands around 35-45% on GSM8K test with greedy decoding; after 300 steps of GRPO you should see 55-70% on the same set.

## 10. Gotchas

- **All-zero advantages.** If every completion in a group gets reward 0 (very common early in training), `std=0`, and the within-group normalization sets all advantages to 0 → no policy gradient on that group. The KL term is the only contributor that step. This is correct behavior, not a bug; if it persists past ~50 steps, the format reward is failing to bootstrap and you should check the schema regex.
- **Rollout is the bottleneck.** Generation time dominates each step. A single rollout step on Qwen3-1.7B at G=8, max_new=512 is ~20-30 seconds on A100; the gradient step takes ~1-2. Don't optimize the gradient code; if you want speed, parallelize generation (more GPUs, vLLM, or batched generation across prompts).
- **Reward hacking.** Watch for the model emitting `<answer>42</answer>` (or some fixed integer) for every prompt — if your dataset has 42 as the answer to many problems and the format bonus is too large, this is the local optimum the policy will find first. Mitigations: lower `w_format` once the format is learned, or filter your training set for ground-truth diversity.
- **Train/inference template mismatch.** Same risk as SFT/DPO. The rollout uses `enable_thinking=False` so the chat template doesn't pre-inject a `<think>` block (we want the *policy* to learn to emit it). `eval.py` matches.
- **EOS handling.** The model can emit EOS before reaching `max_new_tokens`; rollout.py truncates the completion at the FIRST EOS so post-EOS pad tokens don't contribute to the loss. If you see `len` saturating at exactly `max_new_tokens`, the model is over-generating without converging on an answer — increase `max_new_tokens` or check the EOS token in your tokenizer.
- **Numerical instability of the importance ratio.** `exp(log_ratio)` can explode if the policy moves dramatically. Grad clip (default 1.0) and the PPO clip (default ε=0.2) both help. If you see `grad_norm` spikes past 100, lower LR or raise grad_clip's denominator.

## 11. R1-Zero — can we skip SFT?

DeepSeek-R1 famously trained TWO variants:

- **R1-Zero**: GRPO applied directly to the base model (no SFT). Reasoning emerges, but the model's outputs are weird — mixing languages, switching tone mid-trace, sometimes refusing to use the schema.
- **R1**: SFT first (on a curated reasoning trace dataset), then GRPO. Cleaner outputs, same final accuracy.

R1-Zero scaled at the DeepSeek-V3-Base level (671B). At Qwen3-1.7B scale, you can run a zero-style experiment by setting `model.name: Qwen/Qwen3-1.7B-Base` (the un-SFT'd base), but expect:
- Format reward takes much longer to bootstrap (50-200 steps vs ~20 for the SFT'd model).
- Accuracy plateau is lower (~25-35% vs ~55-70% with SFT-then-GRPO).
- Output quality is rough — the schema appears but the reasoning traces are short and unfocused.

The pedagogical lesson is intact at any scale: **reasoning is a behavior that emerges from outcome-supervised RL, not a capability the model needs to be pre-taught**. It just emerges *faster and cleaner* when the SFT base has the basic conversational machinery in place. The canonical config takes the SFT route.

## 12. Stretch goals

- **Iterative GRPO.** After this round finishes, set the new policy as the new reference and run another 300 steps. R1 used 3-4 rounds; each round picks up another 5-15 percentage points on GSM8K. One config flag (`model.ref_name=<previous checkpoint>`).
- **vLLM rollouts.** Replace `model.generate` with a vLLM async client. 5-10× faster rollouts at the cost of an extra deployment surface; reaches 7B-13B model scale on a single GPU.
- **Process reward.** Instead of (or in addition to) the outcome reward, ask a verifier to score each STEP of the reasoning trace. DeepSeekMath shows this helps small models. Needs a stronger verifier (often an LLM judge); the loss code stays the same but advantages now have per-step structure.
- **Multi-task rewards.** Train on a mix of math + code + logic, each with its own verifier. `compute_rewards` becomes a dispatcher; the rest of the pipeline is unchanged.
- **DAPO / GSPO.** Recent GRPO refinements that smooth the per-token vs sequence-level loss aggregation. Often outperforms GRPO on harder benchmarks; one paper-sized change to `compute_grpo_loss`.

## 13. Reading list

In order of "read this first":

- **[Shao et al. (2024) — DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300).** The original GRPO paper. § 5 (the algorithm) is canonical; § 6 (analysis) explains why outcome reward beats process reward at scale.
- **[DeepSeek-AI (2025) — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).** The R1 paper. Read in full: the "emergence" story, the R1-Zero ablation, the rule-based reward design.
- **[Schulman et al. (2017) — Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).** The PPO paper. § 6 is the clipped surrogate that GRPO inherits. Read for the importance-ratio + clip intuition.
- **[Schulman (2020) — Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).** Short blog post that derives the k1, k2, k3 KL estimators. We use k3 in `compute_grpo_loss`; this post tells you why.
- **[Lambert et al. (2024) — Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124).** The most-documented open RL recipe. § 6 (RLVR — RL with verifiable rewards) is the modern GRPO recipe applied at scale.
- **[Singhal et al. (2024) — A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716).** Length bias is real for GRPO too; this paper diagnoses it for RLHF generally.
- **[Cobbe et al. (2021) — Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).** The GSM8K paper. Read for the dataset card — what the problems look like, how the verifiers are intended to work.

## 14. What's next

[Module 18 — Distillation](../18-distillation/). GRPO grows a capability that wasn't there before. Distillation goes the other direction: take a model that already has the capability (R1, GPT-4o, your GRPO'd Qwen3) and transfer it to a SMALLER student. Three flavors: offline (R1-Distill), on-policy (student rollouts + teacher tokens), and SDFT (the same model is its own teacher). Module 18 walks through all three on the same `policy + reference + per-token loss` machinery this module ends with.
