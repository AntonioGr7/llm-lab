# Module 18 — Distillation

You have a frontier model, or a teacher you trust, or just a model that works *and* a small set of demonstrations of a new behavior. You want a smaller, cheaper, or differently-skilled student. Distillation is how you move capability through token-level supervision instead of through outcome reward (Module 17) or pairwise preference (Module 16).

This module covers the three flavors that companies actually use in 2026:

1. **Offline distillation** (R1-Distill / Alpaca / OpenHermes style) — teacher generates `(prompt, completion)` data once, student SFTs on it. Simple, powerful, mature. Off-policy: the student is trained on a distribution it won't produce at inference.
2. **On-policy distillation** (GKD — Agarwal et al., 2024) — student generates rollouts on its OWN distribution; teacher provides per-token logits as supervision; student matches via forward KL. Fixes the off-policy mismatch; needs a separate (larger) teacher.
3. **SDFT** (Self-Distillation Fine-Tuning — Shenfeld et al., 2026) — the headline new technique. The SAME MODEL acts as its own teacher, conditioned on K demonstrations in-context. No separate teacher in memory. The crucial empirical result: **dramatically less catastrophic forgetting**, enabling continual skill acquisition without a replay corpus.

The framework directory ships **runnable code for the on-policy and SDFT flavors** (the loss is the same — per-token forward KL — only the data source differs). Offline distillation is covered conceptually in §2 and routes to Module 15's SFT code; the loss is identical to SFT, only the dataset changes.

The new pedagogical ideas in this module:

1. **Forward KL as a distillation loss.** Why per-token KL beats cross-entropy on hard labels: the teacher's full distribution carries *uncertainty* information that hard labels destroy. This is the same insight behind label smoothing, generalized to "the teacher's distribution is the smoothing".
2. **On-policy vs off-policy data.** When the student is trained on data it doesn't produce at inference, errors compound (the "exposure bias" problem). On-policy distillation puts the supervision on the student's own trajectories — small errors don't snowball.
3. **In-context conditioning as a teacher signal.** SDFT's central observation: if the model can do the task with K demonstrations in context, then *the model with demonstrations* is a perfectly good teacher for *the model without demonstrations*. No second model, no separate training stage, no reward — and the gradient updates stay small (the teacher target is already-reachable), which is what protects prior skills from being overwritten.

## The thesis (one paragraph)

Distillation is **moving a distribution, not a set of labels**. Whatever produced the teacher's distribution — a bigger model, the same model with hints, a frontier API — the loss is the same: at every response token, push the student's distribution toward the teacher's. The three flavors differ only in **where the teacher's distribution comes from** and **whose completions we score** (the teacher's own, or the student's). Once you internalize that, the loss code is ~30 lines (`loop.py`), and the rollout code is the only thing that distinguishes the three flavors (`rollout.py`).

## What you'll be able to do at the end

1. Write the forward-KL distillation loss from scratch (§ 3).
2. Decide between offline, on-policy, and SDFT based on what you have available (teacher? demonstrations? compute budget?) — see § 4.
3. Run SDFT on a small company-specific demonstration corpus and verify (numerically) that prior skills are preserved — the "have your cake and eat it" demo (§ 6).
4. Read the SDFT paper (Shenfeld et al., 2026) and the GKD paper (Agarwal et al., 2024) and recognize every line.

## 1. Directory layout

```
18-distillation/
├── README.md              you are here
├── notebook.ipynb         CPU-only tour: forward KL + the SDFT teacher/student split
├── config.py              TrainConfig — student, optional teacher, distill mode + KL knobs
├── model.py               build_student + build_teacher + generation_mode toggle
├── data.py                tool-use corpus loader + GSM8K prior-skill eval reuse
├── rollout.py             teacher-side generation + per-token logp extraction
├── loop.py                compute_distill_loss (forward / reverse KL) + train_step
├── train.py               torchrun entrypoint — mode-dispatches sdft / on_policy
├── eval.py                two-axis: tool-use new-skill + GSM8K prior-skill
├── prepare_tooluse_corpus.py REAL data loader — Hermes/Glaive function-calling subset
├── make_tooluse_corpus.py    synthetic fallback (offline, $0, for tests + the notebook)
├── fsdp_setup.py          copied from Module 17
├── optim.py               copied from Module 17
├── schedule.py            copied from Module 17
├── checkpoint.py          copied from Module 17
├── efficiency.py          copied from Module 17
├── configs/
│   ├── sdft_qwen3_1.7b.yaml   canonical SDFT — the main story
│   ├── on_policy_distill.yaml Qwen3-1.7B → Qwen3-0.6B teacher→student
│   └── sdft_demo.yaml         cheap smoke
├── tests/
│   ├── test_tooluse_corpus.py the verifier + corpus generator
│   └── test_distill_loss.py   the KL loss + rollout geometry
└── results/                   pre-run two-axis curves + checkpoint
```

NEW components vs Module 17: `prepare_tooluse_corpus.py` (real-data loader), `make_tooluse_corpus.py` (synthetic fallback), the forward-KL `loop.py`, and a much-simplified `rollout.py` (one rollout per prompt; no group sampling). Infra (`optim.py`, `schedule.py`, `fsdp_setup.py`, `checkpoint.py`, `efficiency.py`) is identical to M17.

### Where the data comes from

Two paths, same `Example(user, tool, args)` on-disk schema so downstream code doesn't care which generator wrote it:

- **`prepare_tooluse_corpus.py` (default for the canonical run)** — loads `NousResearch/hermes-function-calling-v1` (config `glaive_func_calling`, Apache 2.0 via Glaive), filters to single-tool-call rows with ≤4 available tools, formats `user` as `"Available tools: {schemas}\n\nRequest: {query}"` so the tool definitions arrive *in-context per example* (the production tool-calling UX). ~3,300 usable single-call rows after filtering; we sample 8 demos + 2000 train + 200 eval by default. Ungated, ~50 MB download.
- **`make_tooluse_corpus.py` (used by tests + the CPU notebook + `sdft_demo.yaml`)** — procedural synthetic generator: 5 hand-templated tools with deterministic slot-filling. $0 cost, no network, fully reproducible from seed. The simpler surface is fine for offline CI and the notebook tour; the demo config uses it so the dev loop stays offline.

Both paths produce identical-format jsonl files (`demos.jsonl`, `train.jsonl`, `eval.jsonl`) so swapping is one line in your config (`data.corpus_dir`).

**Scaling up from the canonical run.** Two levers, in order of bang-for-buck:

1. `python prepare_tooluse_corpus.py --out=./data_big --n-train=10000` + `--data.corpus_dir=./data_big` — bigger pool, more diverse coverage, ~$15-25.
2. `--model.name=Qwen/Qwen3-7B` — SDFT paper-style effect size (the paper reports the gap over plain SFT GROWS with model scale), ~$30-60.

## 2. Offline distillation (R1-Distill style) — concept-only, points at Module 15

The simplest distillation: a stronger teacher generates demonstrations of the target behavior; the student SFTs on them with standard cross-entropy.

```python
# Conceptually:
for prompt in prompt_pool:
    completion = teacher.generate(prompt)
    sft_dataset.append({"prompt": prompt, "completion": completion})

# Then run Module 15's SFT on sft_dataset.
```

Examples in the wild:

- **R1-Distill** (DeepSeek, 2025): R1 generated millions of reasoning traces; Qwen 2.5 / Llama 3 students were SFT'd on those.
- **Alpaca / Vicuna** (Stanford, 2023): GPT-3.5/4 generated instruction-following data; LLaMA students were SFT'd on it.
- **OpenHermes / OpenOrca / Dolphin** — entire chat-model pipelines built on stronger-teacher generations.

**Why this works**: a frontier-quality demonstration is, empirically, a very strong gradient signal. You're not asking the student to learn *from scratch*; you're asking it to imitate a specific function.

**Why this has limits**:

1. **Off-policy**: the student is trained on completions IT wouldn't have produced at inference. Small errors at inference time compound — this is the classic "exposure bias" problem (Ranzato et al., 2016).
2. **Hard labels lose information**: the teacher's *uncertainty* — the cases where it gave 60-40 between two completions — is collapsed to a single token. The student trains on the one chosen token, not the distribution.
3. **No defense against forgetting**: SFT on the teacher's narrow distribution can destroy capabilities the student had before. Classical fix is to mix in pretraining data; SDFT (below) addresses this fundamentally.

**We don't ship a separate runnable for offline distillation here** because the loss is identical to SFT (Module 15). To run it in practice:

```bash
# 1. Generate teacher samples — e.g. with vLLM or an API
python -c "
import json
from openai import OpenAI  # or any teacher
client = OpenAI()
with open('teacher_data.jsonl', 'w') as f:
    for prompt in load_prompts('my_pool.txt'):
        out = client.chat.completions.create(model='gpt-4o', messages=[
            {'role': 'user', 'content': prompt},
        ])
        f.write(json.dumps({
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': out.choices[0].message.content},
            ],
        }) + chr(10))
"

# 2. Point Module 15's SFT at this data
cd ../15-sft/
torchrun --standalone --nproc_per_node=1 train.py \
    --config=sft_qwen3_1.7b.yaml \
    --data.source=local_jsonl --data.path=/path/to/teacher_data.jsonl
```

(`data.source=local_jsonl` is a small extension to M15's loader. The point of this section is the *concept*, not the code path.)

## 3. On-policy distillation (GKD-style)

Off-policy → on-policy. The student GENERATES the completions; the (separate, larger) teacher SCORES them with per-token logits; the student matches via forward KL.

```python
for prompt in prompt_pool:
    completion = student.generate(prompt)               # on-policy: student's own distribution
    teacher_logp = teacher(prompt + completion).logp    # teacher's distribution over those tokens
    student_logp = student(prompt + completion).logp    # student's distribution (with grad)
    loss = forward_kl(teacher_logp, student_logp)       # match teacher
    loss.backward()
```

Two key advantages over offline:

1. **The student trains on its own distribution.** Inference looks like training. The exposure-bias error-compounding goes away.
2. **The teacher's full distribution is the target.** Where the teacher is confident, the student is pushed hard. Where the teacher is uncertain (multiple plausible next tokens), the student is allowed to match that uncertainty.

The cost: **two models in memory**. The teacher is BF16 + no_grad (same memory shape as M16/M17's reference model), so the overhead is ~3.4 GB at 1.7B teacher — manageable on A100.

**The loss code (`compute_distill_loss` in [loop.py](loop.py)):**

```python
# student_log_probs: [B, C, V]  — student's log-softmax at C completion positions, WITH grad
# teacher_log_probs: [B, C, V]  — teacher's log-softmax, detached
# response_mask:     [B, C]      — True at valid completion tokens (not pad, not past EOS)

teacher_p = teacher_log_probs.exp()
per_token_kl = (teacher_p * (teacher_log_probs - student_log_probs)).sum(dim=-1)
loss = (per_token_kl * response_mask.float()).sum() / response_mask.sum().clamp(min=1.0)
```

That's it. Forward KL is the cross-entropy between the teacher's distribution (as label weights) and the student's distribution (as the predictions), summed over the vocabulary. The student gets gradient on EVERY vocab position where the teacher places mass — that's the "soft labels carry uncertainty" effect.

Reverse KL is a one-line swap (mode-seeking, sharper student); we expose it via `distill.kl_direction=reverse`.

**Why on-policy isn't always preferred over offline**: it costs more. You're running the student's generation + the teacher's forward + the student's forward on every step, vs. offline distillation which is just a forward+backward on pre-computed data. For very small data, offline wins on cost; for any scale where exposure bias matters, on-policy wins on quality.

**[See `configs/on_policy_distill.yaml`](configs/on_policy_distill.yaml)** for the canonical Qwen3-1.7B → Qwen3-0.6B setup.

## 4. SDFT — Self-Distillation Fine-Tuning

[Shenfeld, Damani, Hübotter, Agrawal, MIT/ETH, 2026 — *Self-Distillation Enables Continual Learning*](https://arxiv.org/abs/2601.19897) ([project page](https://self-distillation.github.io/SDFT)).

The pivotal observation: **if the model can do a task when shown K demonstrations in context, then "the model with demonstrations" is a usable teacher for "the model without demonstrations."** No second model, no offline data generation step.

```python
# Same model, two ROLES:
#   teacher = model conditioned on K demonstrations (no_grad)
#   student = model NOT conditioned (grad)
for prompt in prompt_pool:
    teacher_input = [system, *K_demonstrations, user(prompt)]
    student_input = [system, user(prompt)]

    # Generate from the teacher (= model with demos)
    completion = model.generate(teacher_input)

    # Score teacher's distribution on (teacher_input + completion)
    with torch.no_grad():
        teacher_logp = model(teacher_input + completion).logp_at_completion

    # Score student's distribution on (student_input + completion), with grad
    student_logp = model(student_input + completion).logp_at_completion

    loss = forward_kl(teacher_logp, student_logp)
    loss.backward()
```

Three properties that fall out of this construction:

1. **No teacher model in memory.** ONE FSDP-wrapped student. The "teacher" is just the same weights queried in a different context. Best memory profile of any distillation flavor.
2. **The teacher target is *reachable*.** Unlike on-policy distillation against a 10× larger teacher (where the student's gradient has to bridge a real capability gap), the SDFT teacher target is what the SAME model can do given hints. The required update is small. Small updates mean less collateral damage to prior skills — the empirical finding that motivated the paper.
3. **Demonstrations replace rewards.** Module 17's GRPO needed verifiable rewards (math correctness, code tests). SDFT needs only K worked examples of the target behavior. For domains where verification is hard (writing in a brand voice, following a domain-specific format, learned-from-PDF Q&A), this is a substantial unlock.

### The SDFT step, in this code

```python
# In rollout.py (one prompt, expanded for clarity):

# 1. Build the two message lists.
teacher_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    # K demonstration turns (user, assistant) — same demos for every prompt
    *[{"role": "user", "content": d.user},
      {"role": "assistant", "content": d.assistant}
      for d in demonstrations],
    {"role": "user", "content": prompt},
]
student_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]

# 2. Teacher generates (= student model under no_grad with demos in context)
with torch.no_grad(), generation_mode(model):
    completion_ids = model.generate(teacher_messages)

# 3. Score teacher's log-probs on the completion (still no_grad)
with torch.no_grad():
    teacher_log_probs = log_softmax(model(teacher_messages + completion).logits)

# 4. Score student's log-probs on the completion (with grad — this is the gradient forward)
student_log_probs = log_softmax(model(student_messages + completion).logits)

# 5. Forward KL
loss = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1)
```

### Why this fights catastrophic forgetting

Plain SFT on K demonstrations updates the model to produce tokens identical to those demonstrations. Gradient flows through the cross-entropy of `(model_output, demo_token)` for every token in every demo — and because the demos are small and out-of-distribution, the update has to make a big move to match them. Big move = far from initial weights = damage to prior skills.

SDFT replaces "match the demonstration token" with "match the same-model-with-demos distribution at every token". The teacher's distribution is *already* close to the student's (it's the same model). The required gradient update is small — the student just has to learn the implicit pattern the demonstrations expressed, not the surface form. Small update = small drift = preserved prior skills.

This is the empirical finding: on the SDFT paper's continual-learning benchmarks (7B / 14B Qwen2.5), SDFT preserves prior-task accuracy where plain SFT loses 10-30 points.

### Hyperparameters that matter

| Knob | What it does | Course default |
|---|---|---|
| `distill.n_demonstrations` (K) | How many demos the teacher sees in-context | 8 (paper's value) |
| `distill.temperature` | Softens both distributions before KL | 1.0 |
| `distill.kl_direction` | forward (mode-covering) or reverse (mode-seeking) | forward |
| `distill.sampling_temperature` | Teacher rollout diversity | 0.9 |
| `optimizer.lr` | Learning rate. SDFT can take higher LR than DPO/GRPO because the target is reachable. | 2e-5 |

### Limits of SDFT

- **Needs in-context learning ability**. SDFT is bottlenecked by the model's ability to follow demonstrations zero-shot. The paper reports the effect SIZE growing with model scale — at 1.7B it works but the gap over plain SFT is smaller than at 7B+. For very small models (<1B), prefer offline distillation from a larger teacher.
- **Demonstrations must actually demonstrate.** If your K demos don't unambiguously specify the target behavior, the teacher's in-context generations will be inconsistent — and you'll distill noise into the student. Demo quality dominates.
- **Doesn't extend the model's capability frontier.** SDFT moves what the model can do with demos into what it can do without. If the model can't do the task with demos, SDFT can't teach it. For genuine new capabilities (not new formats), you need offline distillation from a stronger teacher or GRPO.

## 5. The training configuration

The canonical SDFT demo: teach Qwen3-1.7B to produce a structured tool-call format from 8 demonstrations, while preserving GSM8K math reasoning.

```yaml
model:
  name: Qwen/Qwen3-1.7B           # the student. In SDFT, also the teacher.
  teacher_name: ""                 # "" -> SDFT (use `name`)
  max_seq: 4096                    # demos + prompt + completion can be long

data:
  corpus_dir: ./data               # produced by make_tooluse_corpus.py
  prompts_per_step: 4
  seq_len: 4096

distill:
  mode: sdft
  n_demonstrations: 8
  kl_direction: forward            # standard imitation
  temperature: 1.0
  sampling_temperature: 0.9
  max_new_tokens: 128              # tool calls are short

optimizer:
  type: adamw
  lr: 2.0e-5

schedule:
  type: cosine
  warmup_steps: 20

training:
  total_steps: 200
  grad_accum: 1
  grad_clip: 1.0
  dtype: bf16
  activation_checkpointing: true
  eval_every: 50                   # two-axis eval (tool-use + GSM8K) every 50 steps
```

Expected wallclock on a single A100-80GB: ~1.5-3 hours. Cost: $2-6 on RunPod/Lambda Labs.

## 6. Running it

```bash
# 0. Prepare the tool-use corpus.
#    Canonical path — REAL function-calling data from Hermes/Glaive (~30s, ~50 MB download)
python prepare_tooluse_corpus.py --out=./data
#    OR offline / $0 — procedural synthetic (5 tools, hand-templated)
#    python make_tooluse_corpus.py --out=./data

# 1. SDFT canonical run — single A100-80GB
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/sdft_qwen3_1.7b.yaml

# 2. On-policy distillation (Qwen3-1.7B teacher → Qwen3-0.6B student)
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/on_policy_distill.yaml

# 3. Demo (~10 min on any GPU) — Qwen3-0.6B, tiny corpus
torchrun --standalone --nproc_per_node=1 train.py --config=configs/sdft_demo.yaml

# Two-axis eval — full diagnostic at the end
python eval.py --config=configs/sdft_qwen3_1.7b.yaml \
    --checkpoint=results/checkpoints/step_00000200 --full

# Baseline (no fine-tune) — what the model knows without SDFT
python eval.py --config=configs/sdft_qwen3_1.7b.yaml --base --full
```

## 7. What you should see — the two-axis curve

Two trajectories matter; they answer different questions.

**Tool-use accuracy** (new-skill axis, measured on 50 held-out tool-use prompts):

- Base (no fine-tune): typically ~5-15% (the model emits the tool-call format incidentally but doesn't follow the schema strictly).
- After SDFT: should climb to 60-90% schema_ok and 40-75% strictly-correct (schema + tool + args all match).
- After plain SFT on the same 8 demos: similar new-skill accuracy, BUT see the next axis.

**GSM8K accuracy** (prior-skill axis, measured on 100-500 GSM8K test problems):

- Base: typically 35-45% on Qwen3-1.7B with our schema-strict eval.
- After SDFT: should stay within ±2-3 percentage points of the base. The whole point.
- After plain SFT on the same 8 demos: typically drops by 10-25 points. This is the catastrophic-forgetting cost SDFT is designed to avoid.

The "have your cake and eat it" comparison plot — new skill gained vs prior skill preserved, both axes shown side by side — is what makes the SDFT story land. The notebook draws this plot from mock data; running the real configs gives you real numbers.

## 8. Memory: SDFT vs on-policy

Qwen3-1.7B, BF16 mixed precision, single A100-80GB.

| Component | SDFT | On-policy (1.7B teacher) |
|---|---|---|
| Student: master weights (FP32) | 6.8 GB | 6.8 GB |
| Student: compute weights (BF16) | 3.4 GB | 3.4 GB |
| Student: gradients (BF16) | 3.4 GB | 3.4 GB |
| Student: AdamW state (FP32) | 13.6 GB | 13.6 GB |
| Teacher (BF16, no grad) | — | 3.4 GB |
| KV cache during generation | ~2 GB | ~2 GB |
| Activations (AC on) | ~6 GB | ~6 GB |
| **Total per rank** | **~35 GB** | **~38 GB** |

SDFT saves the 3.4 GB of teacher weights by reusing the student. The activation cost during the SDFT step is HIGHER (because the teacher-side input has K demos prepended, lengthening the sequence) — but activation checkpointing keeps that bounded.

The headline: SDFT has the best memory profile of any distillation flavor. On-policy is close; both fit comfortably on A100-80GB.

## 9. Decision framework — which flavor when

| You have... | You want... | Reach for |
|---|---|---|
| A frontier API + cash | Generic capabilities (chat, code, math) at much smaller scale | **Offline** (R1-Distill style). The library of pre-generated traces is the moat. |
| A 7-70B model you trust + GPU budget | A smaller deployable student that matches its behavior on your distribution | **On-policy** distillation. Less off-policy drift, sharper student. |
| Just K demonstrations of a new behavior, AND prior skills to preserve | A model that can do the new thing without forgetting the old | **SDFT**. The "company-real" use case — domain Q&A, brand voice, internal tool format. |
| A reward function (math correctness, test pass rate) | A model that gets better at outcomes you can verify | Not distillation. **Use [Module 17](../17-reasoning-and-grpo/) GRPO.** |

The boundaries are fuzzy. Distillation + GRPO are common combinations (offline-distill from R1 first, then GRPO to extend). SDFT + replay is sometimes used when prior-skill preservation is critical and you don't trust SDFT's anchor alone. We cover the building blocks; the recipe space combinatorially explodes from there.

## 10. Gotchas

- **In-context learning matters.** SDFT only works when the model can perform the task given K demos in context, zero-shot. Test that first with `python eval.py --base --full` — if base tool-use accuracy is ~0%, the model can't follow the demos and SDFT will produce noise. Mitigation: lower the difficulty (clearer demos, simpler schema) or use offline distillation from a stronger teacher.
- **Numerical instability under temperature scaling.** The KL formula with non-unit temperature requires recomputing log-probs in FP32 — see `loop.train_step`. Don't skip the `logsumexp` re-normalization or the KL will be off by a constant.
- **Demonstration quality dominates.** Eight clean demos produce a sharper SDFT teacher than eighty noisy ones. Curate ruthlessly.
- **The teacher target is small but not zero.** If the K demos perfectly reflect what the student already does, the KL is ~0 and nothing learns. The demos must demonstrate behavior the un-conditioned student doesn't reliably produce.
- **Vocab size mismatch.** Tokenizer reports `vocab_size` (base) but chat-template special tokens live above that. Use `len(tokenizer)` or just trust the model's `config.vocab_size`. (We caught this in our integration smoke.)
- **Train/inference template mismatch.** Same risk as SFT/DPO/GRPO. `enable_thinking=False` in every chat-template render; if you change one, change them all.

## 11. Stretch goals

- **Top-K KL.** The full-vocab KL stores `[B, C, V]` teacher log-probs — 150k vocab at C=128, B=4 is ~300 MB. `distill.top_k_kl > 0` projects to the top-K teacher positions. Memory drops linearly; quality is essentially unchanged for K=512+. (`top_k_kl > 0` is wired in but ships disabled in the canonical config.)
- **SDFT + replay.** Mix a few % of pretraining or prior-task tokens into each batch. Belt-and-braces against forgetting. Easy to add as a second data source in `data.py`.
- **Iterative SDFT.** After this round, regenerate the K demonstrations using the just-SDFT'd model (now slightly better at the task), then run SDFT again with the improved demos. Each round bootstraps the teacher signal.
- **Distillation + GRPO.** Run R1-Distill style SFT first (Module 15 with teacher data), then GRPO (Module 17) on a reward signal. The DeepSeek-R1-Distill series uses exactly this recipe.
- **Multiple skills via sequential SDFT.** Train through TASK_A, then TASK_A→TASK_B with TASK_A demos in context, then TASK_A→TASK_B→TASK_C, etc. The SDFT paper shows this scales further than naive sequential SFT.

## 12. Reading list

In order of "read this first":

- **[Shenfeld et al. (2026) — Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897).** The SDFT paper. § 3 (method) is the core; § 5 (continual-learning results) is the empirical motivation. [Project page](https://self-distillation.github.io/SDFT) with code at [github.com/idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation).
- **[Agarwal et al. (2024) — On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) (GKD).** The on-policy distillation paper SDFT inherits its loss from. § 3 (the GKD loss) is canonical; § 5 ablates forward vs reverse KL.
- **[Hinton et al. (2015) — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).** The original distillation paper. Read it for the "soft labels carry uncertainty" intuition. (Pre-LLM but the principle is identical.)
- **[DeepSeek-AI (2025) — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948).** § 4.3 (R1-Distill) is the offline-distillation playbook at scale.
- **[Gekhman et al. (2024) — Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904).** The "SFT on facts the model doesn't know teaches hallucination" finding. Useful context for why SDFT-style "soft" training is gentler.
- **[Ranzato et al. (2016) — Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/abs/1511.06732).** The exposure-bias paper. The original argument for on-policy training over teacher-forced training.
- **[Mukherjee et al. (2023) — Orca: Progressive Learning from Complex Explanation Traces](https://arxiv.org/abs/2306.02707).** Frontier-teacher-driven offline distillation done thoroughly. Useful as a contrast to SDFT (lots of teacher data, separate larger model, replay).

## 13. What's next

This is the final module of Part 4 — the post-training stack you've been climbing through (SFT → DPO → GRPO → distillation) is now complete. [Part 5](../../part-5-evaluation/) takes a step back and asks the harder question: **how do you actually know any of this worked?** Benchmarks lie, leaderboards drift, "vibes" don't scale. Part 5 is about evaluation that survives contact with reality.
