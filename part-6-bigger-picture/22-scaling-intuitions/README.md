# Module 22 — Scaling Intuitions (and What Comes Next)

This is the last module. You've built a tokenizer, an attention block, a full transformer, a pretraining loop, a production data pipeline, continual pretraining, the entire post-training stack, and an evaluation harness. Everything you ran was small — a 150M model on a few billion tokens, a 1.7B model fine-tuned on a single GPU.

The question this module answers: **does any of that prepare you for the runs that cost $50 million?**

The answer is yes, and the reason is the single most important idea in modern machine learning. The behavior of these systems is *predictable across scale*. The loss curve of a $50M run is not a surprise to the team that launched it — they computed it, on a whiteboard, from runs that cost a few thousand dollars. The arithmetic you'll do in the notebook is the same arithmetic a frontier lab does before it commissions a cluster. The constants change. The shape does not.

This module is half scaling laws, half career. The scaling laws are why your small runs generalize. The career half is what to do with that.

## What you'll walk away with

1. **The 6ND rule** — the one equation that converts model size and data into compute, and lets you estimate the cost of *any* run on the back of an envelope.
2. **Compute-optimal allocation** — given a fixed budget, how to split it between a bigger model and more data. Why the answer is "~20 tokens per parameter," and why labs deliberately violate it.
3. **Loss prediction** — how labs know the final loss of a run before they start it, and why "predictable" is the whole game.
4. **What changes and what doesn't at scale** — the things your small runs taught you that stay true, and the things that only appear when the cluster is big enough.
5. **What labs actually hire for**, and how to choose a personal project that signals you can operate here.

The notebook is a working scaling-law calculator. Put in a budget, get out a model size, a token count, a predicted loss, a wall-clock time, and a dollar figure. Then push the budget up by six orders of magnitude and watch your course demo turn into a frontier model on the same two lines of arithmetic.

## 1. The one equation: C ≈ 6ND

A forward pass through a dense transformer costs about `2N` FLOPs per token (one multiply and one add for each of the `N` parameters). The backward pass costs about twice that. So a full training step — forward plus backward — is about `6N` FLOPs per token, and a whole run over `D` tokens is:

```
C ≈ 6 · N · D
```

`N` is the (non-embedding) parameter count, `D` is the number of training tokens, `C` is total training FLOPs. That's it. This is the most useful equation in the field and it fits on a sticky note.

It is astonishingly robust. It doesn't care about your architecture — GQA, SwiGLU, RoPE, the choices from Part 2 don't change it. For a Mixture-of-Experts model (Module 06) you use the *active* parameter count, and it still holds. It's the bridge between the abstract ("how big, how much data") and the concrete ("how many GPU-hours, how many dollars").

Worked example: Chinchilla was 70B parameters trained on 1.4T tokens.

```
C ≈ 6 · 70e9 · 1.4e12 ≈ 5.9e23 FLOPs
```

Hold onto that number — `5.9e23` — it's the reference point for "a serious 2022 run." A frontier 2024 run is closer to `1e25`–`1e26`: a hundred to a thousand times more.

## 2. Compute-optimal allocation: the Chinchilla rule

You have a compute budget `C`. You can spend it on a **bigger model** (more `N`) or **more data** (more `D`). `C = 6ND` is fixed, so it's a trade: doubling `N` halves `D`. Which split gives the lowest loss?

For years (the Kaplan et al. 2020 era) the field believed the answer was "mostly bigger models." GPT-3 was the monument to that belief: 175B parameters, but only ~300B tokens — fewer than 2 tokens per parameter. It was *under-trained*. Most of those parameters never saw enough data to be worth their cost.

Hoffmann et al. 2022 (the **Chinchilla** paper) re-ran the experiment carefully and found the opposite: model size and data should scale **roughly in lockstep**. At the optimum you want about **20 tokens per parameter**. Their proof was a 70B model (Chinchilla) trained on 1.4T tokens that beat the 280B Gopher — *a quarter the size, trained on the same compute, and better.*

The arithmetic, with `D = 20N` and `C = 6ND = 120N²`:

```
N_opt = √(C / 120)        D_opt = 20 · N_opt
```

That's the whole rule. `chinchilla_optimal(C)` in [scaling.py](scaling.py) is those two lines. Feed it Chinchilla's compute and it returns 70B / 1.4T, as it should.

**Why labs deliberately break this rule.** Chinchilla optimality minimizes *training* compute for a target loss. But a deployed model is run *for inference* billions of times, and inference cost scales with `N`, not `C`. So if you're going to serve a model at scale, it pays to train a **smaller** model **past** its compute-optimal point — spend more training compute now to get a cheaper-to-serve model forever. This is exactly what Llama 3 8B did: ~15T tokens, **~1,900 tokens per parameter**, ~100× past "optimal." The training bill was larger; the inference bill, amortized over hundreds of millions of users, made it the deal of the decade. The notebook's `overtrain_savings` quantifies this trade.

The lesson is not "20 is the magic number." The lesson is: **the optimal allocation depends on what you're optimizing for**, and once you can do the arithmetic you can answer that for your own situation.

## 3. Predicting loss before you spend a dollar

Here's the part that feels like magic the first time. Hoffmann et al. fit a single smooth surface to loss as a function of size and data:

```
L(N, D) = E + A/N^α + B/D^β
```

- `E` is the **irreducible loss** — the entropy of language itself, the floor no model beats. (~1.69 nats/token in their fit.)
- `A/N^α` is the penalty for a **finite model**: not enough parameters to represent everything.
- `B/D^β` is the penalty for **finite data**: not enough tokens to learn from.

Both exponents are small (`α≈0.34`, `β≈0.28`), which is *why scaling is exhausting*: loss falls as a slow power of compute. To halve the gap to the irreducible loss you need not 2× but ~10–100× the compute. Diminishing returns is not a bug in the technique; it's the mathematical shape of the problem.

But — and this is the point — **the curve is smooth and it extrapolates.** A lab fits this surface on a sweep of small, cheap runs (a few thousand dollars of GPU time), reads off the constants, and then *predicts the loss of the run that costs ten thousand times more.* GPT-4's technical report shows exactly this: they predicted the final model's loss from runs using `1/1000`–`1/10000` of the compute, and hit it. The big run held no surprises because the small runs had already drawn the curve.

This is why your $50 of experiments matter. **Predictability across scale is the entire reason a lab is willing to spend $50M on one run.** Nobody bets a year's budget on a hunch; they bet it on a power law fit to runs that look like yours.

### A note for a course about reading the literature

Use the absolute constants above with suspicion. Besiroglu et al. (2024) attempted to replicate Chinchilla's parametric fit and found the published constants internally inconsistent with the paper's own other estimates. The *headline* (≈20 tokens/param, lockstep scaling) survived scrutiny; the third-decimal constants did not. This is a perfect Module 21 case study: a hugely influential result, mostly right, with a quietly wrong table that took two years and a replication to surface. Trust the shape. Verify the digits. `scaling.py` uses the published constants for relative predictions (loss *ratios*, which are robust) and leans on the 20:1 rule for allocation (which is self-consistent).

## 4. From your $50 run to a $50M run

The notebook builds a ladder. The same two functions — `chinchilla_optimal` and `cost_and_time` — take you across **twelve orders of magnitude** of compute without changing a line:

| Run | Params | Tokens | Compute (FLOPs) | ~Cost |
|-----|--------|--------|-----------------|-------|
| Your Part 3 demo | 150M | 3B | ~3e18 | a few $ |
| Chinchilla (2022) | 70B | 1.4T | ~6e23 | ~$1–2M |
| Llama 3 70B (2024) | 70B | 15T | ~6e24 | ~$10–20M |
| Frontier (~2024 est.) | ~1T (active) | ~15T | ~1e26 | ~$100M+ |

Multiply your demo's compute by a million and spend it compute-optimally, and the calculator hands you a ~150B-parameter model on ~3T tokens. That is a frontier-class run, derived from your toy run with a square root. The cluster is bigger, the team is bigger, the failure modes are nastier — but the arithmetic on the whiteboard is the arithmetic you already know.

What the dollar figure hides: the headline FLOPs cost assumes a realized **MFU** (Model FLOPs Utilization) of 30–50%. The gap between peak GPU FLOP/s and what you actually achieve is where Part 3's lessons (FSDP, gradient checkpointing, the efficiency knobs of Module 10) cash out. At $50M scale, raising MFU from 35% to 45% is millions of dollars. The same optimization you did to fit a model on one GPU is, at scale, a line item.

## 5. What changes and what doesn't at scale

**What stays exactly the same** (and is why this course transfers):

- The math. Attention, the transformer block, the loss, the optimizer — identical from 150M to 1T. You debugged it at a scale you could see.
- The training dynamics in miniature. Warmup, the cosine schedule, gradient clipping, loss spikes — a 1T run has all the same pathologies as your toy run, just more expensive to hit.
- The data principles. Quality over quantity, dedup, the curation cascade from Part 1 — *more* important at scale, not less.
- Post-training. SFT → preference → RL is the same recipe whether the base is 1.7B or 1T.

**What only appears when the cluster is big:**

- **Systems failures dominate.** At 10,000 GPUs, something fails every few hours — a node dies, a NIC flakes, an optimizer state corrupts. Frontier training is as much distributed-systems reliability engineering as it is ML. Checkpointing (which you built) stops being a convenience and becomes survival.
- **Communication becomes the bottleneck.** At small scale, GPUs compute; at large scale, they spend much of their time *talking*. The topology of your interconnect, the overlap of communication with computation, 3D/4D parallelism — these are real only past a certain size.
- **Loss spikes and instability.** Big runs hit instabilities (loss divergence, gradient explosions) that small runs rarely see. The mitigations (z-loss, careful init, learning-rate babysitting, embedding norms) are a craft. You won't meet most of them at 150M.
- **Data runs out.** At frontier scale you exhaust the high-quality public internet. The frontier problem is now *data*, not compute — synthetic data, multi-epoch training, and licensing deals. This is the live research frontier as of 2026.

The honest summary: **you have learned the model and the algorithm completely, and the systems engineering partially.** The thing you can't practice at $50 is the thing that breaks at $50M. That's fine — labs know this, and it's what the team around you is for.

## 6. What labs actually hire for

Frontier labs hire, roughly, into three lanes. This course touched all three; here's where to push.

**Research scientist / research engineer.** Designs and runs experiments, reads and writes papers, owns a piece of the model (a new attention variant, a post-training method, an eval). What signals readiness: you can read a paper and implement it from scratch (Module 21), you have an opinion on a tradeoff backed by an experiment you actually ran, you've reproduced a published result and found where it's fragile. A PhD helps but is not required; a public reproduction of a recent paper is worth more than most coursework.

**ML / infra / performance engineer.** Makes the runs fast and reliable — the MFU, the parallelism, the data pipeline, the checkpointing, the failure recovery. This course's Part 3 (Module 10's efficiency, Module 12's pipeline) is the seed. What signals readiness: you can profile a training step and say where the time goes, you've made something measurably faster, you understand the memory and communication accounting cold. This lane is in *enormous* demand and underrated by newcomers who chase the research lane.

**Applied / product (fine-tuning, evals, deployment).** Takes a base model and makes it do a specific job — the entire post-training stack (Part 4) plus evaluation (Part 5) plus serving. What signals readiness: you've taken a model from base to genuinely-useful on a real task, you can build an eval that catches regressions, you know the cost/latency/quality tradeoffs of deployment. This is where most LLM work in industry actually lives, and the course is closely aligned to it.

A blunt truth about all three: **the bottleneck is rarely knowledge, it's evidence.** Everyone has read the papers. Few have a repository that shows they ran the experiment, hit a wall, debugged it, and wrote up what they learned. That repository is the hire.

## 7. Choosing your next project

The course is over; the field is not. Pick a next project that *signals real understanding* — meaning it has a result, a surprise, and a write-up. Some templates, roughly in order of effort:

- **Reproduce a recent paper at small scale.** Take something from 2025–2026, implement it on Qwen3-0.6B, and report where it works and where it doesn't. The "where it doesn't" is the valuable part — it proves you ran it, not just read it. (This is the Module 21 skill turned into an artifact.)
- **Run the real version of a course module.** You skipped the GPU runs. Pick one — the SFT, the DPO, the GRPO, the LoRA head-to-head — rent an A100 for an evening, fill in the `results/` table, and write up what surprised you. ~$5–15 and a weekend.
- **An ablation nobody bothered to do.** Take a design choice this course made (the LoRA rank, the KL coefficient in GRPO, the replay ratio in continual pretraining) and sweep it. A clean ablation with error bars (Module 20!) is a real contribution.
- **A domain adaptation end-to-end.** Continual-pretrain (Module 13) on a corpus you care about, post-train it (Part 4), evaluate it honestly (Part 5), and ship it. This exercises the *whole* course and produces something useful.
- **A from-scratch reimplementation of one frontier component** — a real Flash Attention kernel, an MoE router with load balancing, an FSDP-from-scratch. Deep, narrow, and a strong infra-lane signal.

Whatever you pick: **scope it so it finishes, measure it so it's honest, and write it up so it's legible.** A small finished project with error bars beats a grand unfinished one every time.

## 8. Where to keep learning

The field moves weekly; Module 21 is your method for keeping up. A few durable anchors:

- **The labs' own technical reports.** Qwen, Llama, DeepSeek, OLMo/Tülu (AI2), the model cards. They are increasingly the best textbooks, and they're free.
- **Replication-minded research.** EleutherAI, AI2/OLMo, Epoch AI (for scaling/compute trends). Groups that publish what *actually* reproduces.
- **The systems frontier.** TorchTitan, Megatron-LM, the FSDP/DTensor docs — read the code, not just the blog posts.
- **One eval you trust.** Pick a benchmark you understand deeply (you built the harness in Module 20) and use it as your personal sniff test for new model claims.

## Reading list

**Scaling laws (read in this order):**
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020). The origin of `C ≈ 6ND` and the power laws. Read it knowing the allocation conclusion was later corrected.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022). The correction. The ~20-tokens-per-parameter result. The single most important scaling paper.
- Besiroglu et al., *Chinchilla Scaling: A replication attempt* (2024). Why you trust the shape and verify the constants — and a model of careful, adversarial reading.
- OpenAI, *GPT-4 Technical Report* (2023), the scaling-prediction section. Loss predicted from runs 1000–10000× smaller. The payoff of predictability.

**Scale, systems, and the data wall:**
- *The Llama 3 Herd of Models* (Meta, 2024). The over-training decision, the systems engineering, the honest scale.
- Sardana et al., *Beyond Chinchilla-Optimal* (2023). Formalizing the train-more-to-serve-cheaper trade.
- Villalobos et al., *Will we run out of data?* (Epoch AI, 2024). The data wall, quantified.
- The TorchTitan / Megatron-LM repositories — what "at scale" looks like in code.

**Career and craft:**
- Chip Huyen, *Designing Machine Learning Systems* — the applied/product lane, written down.
- The "engineering" sections of any frontier model card — what the job actually is.

## Notebook

[notebook.ipynb](notebook.ipynb) is the calculator, hands-on:

1. The `6ND` rule on real models — and a check that it reproduces published compute figures.
2. Compute-optimal allocation across budgets — the `√(C/120)` curve, and where real models sit relative to it (GPT-3 under, Llama over).
3. The loss surface — predict loss for any (N, D), and plot loss vs. compute (the slow power-law decline made visible).
4. The `$50 → $50M` ladder — your demo extrapolated twelve orders of magnitude on two lines of arithmetic.
5. The over-training trade — train-smaller-longer to serve-cheaper-forever, quantified.
6. Cost and wall-clock — FLOPs converted to GPU-hours and dollars at realistic MFU.

It runs in milliseconds on CPU. No model loads, no network — just the arithmetic that runs the field.

---

That's the course. You started not knowing what a token was; you've now built, trained, aligned, and evaluated a language model, and you can do the arithmetic that governs the ones a thousand times bigger. The gap between you and a frontier engineer is now mostly *evidence* — runs you've done, results you've measured, things you've shipped. Go make some.
