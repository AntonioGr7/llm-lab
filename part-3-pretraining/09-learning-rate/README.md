# Module 09 — The Learning Rate

> Part of [Part 3 — Pretraining](../). Reading time: ~60 minutes. Compute cost: ~$0 (CPU walkthrough).

## The thesis

There is exactly one hyperparameter that, set wrong, kills your training run silently: the learning rate. Set it 3× too high and the loss diverges in the first 100 steps with no clear error. Set it 3× too low and the loss decreases — slowly, smoothly, and to a final perplexity that's 10–30% worse than it should be. The training looks fine. The model is just bad.

Every other hyperparameter in pretraining has reasonable defaults that work across two orders of magnitude in model size. LR doesn't. It changes with width, with batch size, with sequence length, with the optimizer, and with how warm the residual stream is. The good news is that the dependencies are *understood* — modern recipes give you a recipe for picking LR that works in 2026 without guessing.

This module covers four things:

1. **The canonical schedule** — linear warmup, then cosine decay to a small minimum. The shape every frontier model uses.
2. **WSD** — DeepSeek-V3's warmup-stable-decay alternative. Better for runs whose length you don't know in advance.
3. **muP transfer in practice** — sweep LR at a small proxy width, transfer to the big one. With concrete numbers.
4. **Diagnostics** — what too-high and too-low LR look like in a loss curve, so you don't waste a training run finding out.

The framework piece from this module is one file: [`schedule.py`](schedule.py). Module 08's `train.py` becomes scheduler-aware with two added lines.

## What you'll be able to do at the end

- Pick a sensible LR + schedule for a new model size without re-running a sweep from scratch.
- Implement linear warmup + cosine decay and DeepSeek's WSD from primitives.
- Apply muP LR transfer: tune at one width, train at any wider width.
- Read a loss curve and identify "LR too high," "LR too low," "schedule too aggressive," "warmup too short."
- Tune LR for your own model in ~3–5 small-scale runs (not 30).

## 1. The canonical shape

The 2026-canonical LR schedule:

```
LR
 │         ╭─────────────────╮
 │        ╱                   ╲
 │       ╱                     ╲
 │      ╱                       ╲___________
 │     ╱                                    ╲___________
 │    ╱                                                 ╲_____
 │   ╱
 │  ╱            ←  warmup  →    ←        cosine decay        →    min_lr
 │ ╱
 └─────────────────────────────────────────────────────────────  steps
   0          warmup_steps                              total_steps
```

Three regions:

1. **Warmup** (steps 0 to `warmup_steps`): LR climbs linearly from 0 to the **peak LR** $\eta_{\max}$.
2. **Cosine decay** (steps `warmup_steps` to `total_steps`): LR follows a cosine from $\eta_{\max}$ down to **min LR** $= r \cdot \eta_{\max}$, where $r$ is `min_lr_ratio` (typically 0.1).
3. **(Optional)** Beyond `total_steps` the LR stays at min — but you should have stopped training by then.

The math:

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{\text{warmup}} & t < \text{warmup} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\!\left(\pi \cdot \frac{t - \text{warmup}}{\text{total} - \text{warmup}}\right)\right) & t \geq \text{warmup} \end{cases}$$

`warmup_steps` is typically **1% to 5% of `total_steps`**. Frontier-scale runs often use a fixed number like 2000 steps regardless of total length.

## 2. Why warmup

At step 0, every weight is randomly initialized. The model's representations are random. The first gradients are noisy — they reflect the loss landscape *at the init point*, not the structure of the language.

If you start at full LR, the first few updates can move weights in a direction that breaks the residual stream's RMS calibration before the norms have learned to compensate. In the worst case the loss spikes and never recovers. Less dramatically, you waste compute getting back to a stable training regime.

Linear warmup gives the model a few hundred to a few thousand steps to:

- Let RMSNorm `gamma`s adjust to the actual residual stream magnitudes.
- Let AdamW's running second-moment estimates ($\hat{v}_t$) build up — at step 0, $\hat{v}_0 = 0$, so the first update is *huge* if you don't either warmup or have a tiny initial LR.
- Let the loss start to track the right direction in the very high-dimensional weight space.

**Practical defaults:**

- 5B-token training: 1000–2000 warmup steps.
- 50B-token training: 2000–5000 warmup steps.
- 500B+ training: still 2000–5000 steps. Warmup is *not* proportional to total length at scale — it's a fixed cost.

DeepSeek-V3 uses 2000 warmup steps over a 14.8T-token training run. The warmup is 0.005% of total training.

## 3. Why cosine

After warmup, the model is at its productive LR regime. Now you decay. Why decay at all?

The intuition is that early training has a lot of high-loss gradient signal — the model has obvious mistakes to fix. Late training is fine-tuning the same model against subtler distinctions, where large updates would overshoot.

Cosine specifically (vs linear, exponential, polynomial) became dominant after empirical sweeps in 2018–2020 showed it consistently edges out other shapes for transformer LMs. The reason is partly that cosine spends *more time near the high LR* than linear or exponential — useful, since the model learns most early — and partly *more time near the min LR* — useful for the final convergence. Other shapes either rush the high-LR phase or never get to a properly small final LR.

The width of the cosine "tail" is set by `min_lr_ratio`. Common choices:

- **0.1** (10% of peak) — Llama 2, Llama 3, most modern recipes. The model is still learning at min LR; you just slowed it down.
- **0.0** (decay all the way to zero) — older recipe. Stop-loss happens at the boundary and the final update is trivial.

**Decision rule for 2026: use 0.1.** Decaying to zero leaves a learnable gap at the end of training that you're not exploiting.

## 4. WSD — DeepSeek-V3's alternative

The cosine schedule has one annoying property: **you have to know `total_steps` in advance**. The shape of the decay depends on it. If you want to train for longer halfway through, you can't — extending the schedule changes the LR trajectory you've already committed to.

**Warmup-Stable-Decay (WSD)** was introduced in DeepSeek's training papers (and is used in V3 and V3.2). Three regions:

```
LR
 │         ╭───────────────────────────────────╮
 │        ╱                                     ╲
 │       ╱                                       ╲___
 │      ╱                                            ╲____
 │     ╱                                                  ╲_____
 │    ╱
 │   ╱        warmup       stable                decay         min_lr
 │  ╱
 └─────────────────────────────────────────────────────────────  steps
   0                                          decay_start    total_steps
```

1. **Warmup** — same linear climb as cosine.
2. **Stable** — LR held constant at $\eta_{\max}$ for the bulk of training. *Most of training happens here.*
3. **Decay** — over the last 10–20% of training, LR decays linearly (or with a short cosine) to min LR.

The advantage: you can extend training mid-flight. Just keep the model in the stable region for longer; you only commit to decay when you decide to finish. Cosine would have already started decaying by then. This matters at frontier scale where training runs can be paused, evaluated, and resumed weeks later.

There's also a small empirical win: WSD's final loss is slightly lower than cosine's at the same compute in DeepSeek's ablations. Not large (~0.2% on validation perplexity), but real.

**Decision rule.** If you know the total step count and won't change it: cosine. If you're not sure or might extend: WSD. For the course's pretraining demo (fixed budget), cosine is fine — but the framework supports both.

## 5. muP LR transfer — the practical recipe

Module 07 introduced muP at the init level (the std=1/√m scaling for hidden weights and the LR groups). Here's what you actually do with it:

### The transfer recipe

Suppose your target is a 1B-param model at $d_{\text{model}} = 2048$, and you want to know the optimal LR.

**Step 1** — Build a small *proxy* model at a base width. Common choices:

- `d_base = 256` or `d_base = 512`. The proxy should be *small enough to sweep cheaply* but *not so small that the dynamics differ qualitatively*. Below ~128 the regime is too different.

**Step 2** — Run an LR sweep at the proxy. Train 5–10 models at LRs spanning `[1e-5, 1e-2]` (geometric grid). Each runs for a fraction of full training (say 1k–10k steps). Pick the LR that gives the lowest validation loss.

**Step 3** — Apply the transfer. At the target width $d_{\text{target}}$, set the **hidden LR** to:

$$\eta_{\text{target}}^{\text{hidden}} = \eta^*_{\text{proxy}} \cdot \frac{d_{\text{base}}}{d_{\text{target}}}.$$

The embedding LR and output LR stay at $\eta^*_{\text{proxy}}$ (unchanged across widths under muP).

That's it. No re-sweep at the target. The mu-transfer property is what makes this work.

### Concrete numbers

If $\eta^*_{\text{proxy}} = 3 \times 10^{-3}$ at $d_{\text{base}}=256$, then at:

| Target width | Width multiplier $m$ | Hidden LR | Embed/Output LR |
|---|---|---|---|
| 256 (proxy) | 1 | $3 \times 10^{-3}$ | $3 \times 10^{-3}$ |
| 512 | 2 | $1.5 \times 10^{-3}$ | $3 \times 10^{-3}$ |
| 1024 | 4 | $7.5 \times 10^{-4}$ | $3 \times 10^{-3}$ |
| 2048 | 8 | $3.75 \times 10^{-4}$ | $3 \times 10^{-3}$ |
| 4096 | 16 | $1.875 \times 10^{-4}$ | $3 \times 10^{-3}$ |
| 8192 | 32 | $9.375 \times 10^{-5}$ | $3 \times 10^{-3}$ |

The hidden LR halves with every doubling of width; the embedding and output LR stay put.

The peak LR you'd use in the scheduler is the *base* per-group LR; the scheduler's cosine just multiplies all groups by a time-varying factor (PyTorch's `LRScheduler` preserves per-group `initial_lr` automatically), so muP's per-group ratios stay correct across warmup and decay.

### How to plug this in

`init.py` from Module 07 provides `param_groups_mup(model, d_model, base_d, base_lr)`. Pass its output to `build_optimizer` instead of letting it construct the default split:

```python
from init import param_groups_mup
groups = param_groups_mup(model, d_model=cfg.model.d_model, base_d=256, base_lr=cfg.optimizer.lr)
optimizer = build_optimizer(model, cfg.optimizer, param_groups=groups)
scheduler = build_scheduler(optimizer, cfg.schedule)  # cosine or WSD, doesn't care about muP
```

The scheduler is muP-agnostic — it just sees three groups with three different initial LRs and treats them uniformly. The per-group ratios are preserved through warmup and decay.

## 6. Reading a loss curve to diagnose LR

The cheapest LR diagnosis tool is just looking at a loss curve. After 500–2000 steps you can usually tell what's wrong. Four signatures:

### "LR too high"

```
loss
 │  ╱╲
 │ ╱  ╲╱╲
 │╱      ╲╱╲╱╲       ← oscillation or rebound
 │           ╲   ╲╱╲
 │            ╲___╲___
                    
       steps →
```

Loss decreases at first, then **oscillates with large amplitude**, possibly rebounds back up. Grad norm is large (>10). Eventual final loss is *worse* than at the LR's productive peak. **Fix: halve the LR.**

In the worst case: loss diverges to NaN in the first 100 steps. Even more dramatic — the model overflowed in BF16, training is irrecoverable, restart with a lower LR.

### "LR too low"

```
loss
 │  ╲
 │   ╲___
 │       ╲___
 │            ╲___          ← slow, smooth, no oscillation
 │                ╲___
 │                    ╲___
                    
       steps →
```

Loss decreases slowly, smoothly, no oscillation. Grad norm is tiny (~0.1). The training looks fine but converges to a *worse* final loss than it should — and burns through compute getting there. **Fix: 2–3× the LR. Watch for the high-LR signatures.**

This is the dangerous one. Nothing in your monitoring will scream. You just get a worse model.

### "Warmup too short"

```
loss
 │   ╱╲
 │  ╱  ╲___                 ← loss spike near step warmup_steps
 │ ╱       ╲___              
 │╱            ╲___
 │                 ╲___
       
       steps →
```

Loss is fine during warmup, then **spikes when peak LR hits**, then recovers. Visible bump at exactly `warmup_steps`. **Fix: 2× warmup_steps, or reduce peak LR.**

### "Schedule too aggressive at the end"

```
loss
 │  ╲
 │   ╲___
 │       ╲___
 │            ╲___          ← flatlines well before total_steps
 │                ╲___
 │                    ╲___╲___╲___
                    
       steps →
```

Loss decreases nicely, then **flatlines for the last 30–50% of training**. The cosine pushed LR too low too fast; the model still had learning left to do. **Fix: extend training, or use WSD with a shorter decay region.**

## 7. Concrete numbers from frontier models

The "what LR does $X$ use" table:

| Model | Peak LR | Schedule | Warmup | min_lr_ratio | Notes |
|---|---|---|---|---|---|
| GPT-3 175B | 6e-5 | Cosine | 375M tokens | 0.1 | One of the first published configs |
| Llama 2 7B | 3e-4 | Cosine | 2000 steps | 0.1 | The "modern default" |
| Llama 2 70B | 1.5e-4 | Cosine | 2000 steps | 0.1 | Smaller for larger model |
| Llama 3 8B | 3e-4 | Cosine | 8000 steps | 0.0 | Decays to zero (unusual in 2026) |
| **DeepSeek-V3** | 4.2e-4 | **WSD** | 2000 steps | 0.1 | Peak is high; balanced by WSD's late decay |
| Qwen 3 (varies) | 1–3e-4 | Cosine | 2000–5000 steps | 0.1 | Standard |

**The rough relationship**: peak LR ∝ 1/$\sqrt{d_{\text{model}}}$. A doubling of width halves the LR — which is exactly what muP predicts.

For the course's pretraining demo (a 150M-param Qwen3-shaped model trained at $d=768$):

- Without muP: peak LR ≈ 5e-4, warmup 1000, cosine to 5e-5, total 50k–100k steps.
- With muP (tuned at $d_{\text{base}}=256$, transferred to $d=768$): peak LR_hidden ≈ 1e-3 × (256/768) ≈ 3.3e-4, peak LR_embed/output = 1e-3. Module 11 ships the exact config.

## 8. What we implement

| Component | Status |
|---|---|
| Linear warmup + cosine decay | Full implementation ([`schedule.py`](schedule.py)) |
| WSD (warmup–stable–decay) | Full implementation |
| Constant LR (warmup then hold) | Full implementation, for ablations |
| muP LR groups | Provided by Module 07's `init.param_groups_mup`; muP transfer is built on these + cosine/WSD |
| LR finder (Smith 2017) | Described; not implemented (the muP transfer recipe makes it unnecessary) |

The scheduler interface follows PyTorch's `LRScheduler` so it composes with any optimizer including FSDP-wrapped ones. Step it once per optimizer step, *after* `optimizer.step()`.

## 9. Wiring it into the training loop

Adding the scheduler to Module 08's `train.py` is two lines: one for setup, one for the per-step call.

```python
from schedule import build_scheduler           # new import

# After build_optimizer(...)
scheduler = build_scheduler(optimizer, cfg.schedule)   # new line — see schedule.py

# Inside the training loop, after optimizer.step():
optimizer.step()
scheduler.step()                                       # new line
```

The notebook in this directory shows this integrated into a small run. Module 11's final `train.py` ships the full version.

## 10. Reading list

- **muP transfer (the practical paper)**: Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2021), Section 6 (the transfer recipe with empirical evidence).
- **The cosine-vs-linear ablation that won the field**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017). Older but the empirical comparison that established cosine for transformer training.
- **WSD**: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024), Appendix B (the schedule description). Also referenced in V3's report.
- **What a real LR sweep looks like**: any of the 2024 frontier model papers (Llama 3, DeepSeek-V3, Qwen 3) include an "LR ablation" figure. Worth reading one to calibrate intuition.

## Next

[Module 10 — Scaling and Efficiency](../10-scaling-and-efficiency/). FSDP2 sharding stages, gradient checkpointing, multi-token prediction, Chinchilla scaling laws. Where the framework becomes throughput-aware.
