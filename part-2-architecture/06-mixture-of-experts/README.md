# Module 06 — Mixture of Experts

> Part of [Part 2 — Architecture](../). Reading time: ~70 minutes. Compute cost: ~$0 (CPU-friendly toy experiments).

## The thesis

So far every parameter in our model is touched by every token. Every linear, every attention head, every FFN — *the entire weight set runs on every forward pass*. That's the assumption a vanilla transformer makes, and at some scale it becomes the wrong assumption.

**Mixture of Experts (MoE)** breaks the tie between parameter count and per-token compute.

Conceptually: replace the FFN with $N$ parallel "expert" FFNs and a small router. For each token, the router picks the top $k$ experts (typically $k = 1$ or $k = 2$) out of $N$ (typically 8–256), and only those $k$ experts run. Per-token compute scales with $k$, not $N$.

The win is concrete: DeepSeek-V3 has **671B total parameters** but only **37B active per token**. The model has the knowledge capacity of a 671B dense model and the per-token inference cost of a 37B dense model. That ratio — 18× capacity per FLOP — is the dominant reason every frontier lab in 2026 ships MoE.

The cost is concrete too: routing creates a load-balancing problem (some experts get overused, others get starved), and the distributed-training story is more complicated (expert parallelism + all-to-all communication). Both have known solutions. This module covers them.

## What you'll be able to do at the end

- Explain why MoE decouples parameter count from per-token compute, and when that matters.
- Implement a top-$k$ MoE FFN with a learned router that produces correct shapes and routes tokens to the right experts.
- Diagnose and fix the load-balancing problem — both the classical auxiliary-loss approach (Switch / GShard) and DeepSeek-V3's auxiliary-loss-free balancing.
- Pick reasonable settings ($N$, $k$, expert size, shared expert presence) for a given compute and memory budget.
- Read a frontier-model MoE config and reproduce the design pressures behind it.

## 1. The basic mechanism

Take a transformer block. Replace its single SwiGLU FFN with this:

```
                ┌── Expert 1 ──┐
                │              │
x  ──→ Router ──┼── Expert 2 ──┼──→ weighted sum  ──→ output
                │   ...        │
                └── Expert N ──┘
```

The router is a small linear layer $W_g \in \mathbb{R}^{d_{\text{model}} \times N}$ followed by a softmax. For each token $x_t$:

$$g_t = \text{softmax}(W_g x_t) \in \mathbb{R}^N.$$

$g_t$ is a probability distribution over the $N$ experts. Pick the top $k$ entries:

$$\mathcal{T}_t = \text{topk-indices}(g_t), \quad w_t^{(i)} = \text{normalize}(g_t^{(i)} \text{ for } i \in \mathcal{T}_t).$$

Run only those $k$ experts on $x_t$ and combine:

$$y_t = \sum_{i \in \mathcal{T}_t} w_t^{(i)} \cdot \text{Expert}_i(x_t).$$

That's it. Each expert is a small FFN (typically a SwiGLU), exactly like the one in Module 05 but narrower. The router is a single linear layer; routing overhead is tiny.

### Two interpretations of MoE

**As a sparse weight matrix**: the full FFN of an MoE layer has $N \cdot 3 \cdot d_{\text{model}} \cdot d_{\text{ffn}}$ parameters, but the per-token effective matrix is the weighted combination of $k$ rows. You get a token-conditional FFN.

**As parameter-count-decoupled compute**: each forward pass uses $k \cdot 3 \cdot d_{\text{model}} \cdot d_{\text{ffn}}$ FLOPs. The router adds $d_{\text{model}} \cdot N$ FLOPs — negligible. Total params live in expert space; active compute lives in $k$-expert space.

Both interpretations are useful. The second is the one driving the scaling story.

## 2. Why MoE wins at scale

The Chinchilla story is "FLOPs are the budget; parameter count and tokens trade off optimally at roughly $1{:}1$." For dense models that's a tight ceiling. For MoE it's *much* looser, because:

- **The router lets the model store knowledge in experts that don't all need to run.** Expert specialization is real: probing studies on Mixtral and Switch show experts cluster around language, domain, syntax patterns. Specialization is what makes MoE pay for itself.
- **At inference, you pay $k$-expert compute and load $k$-expert weights** (with cleverness; see expert parallelism below). The memory footprint is harder to manage than dense, but the FLOPs win directly translates to throughput and latency.

The empirical ratio: MoE models match the *quality* of a dense model with **roughly 2–4× their active parameter count**, and use 4–8× less compute than the equivalent-quality dense model. Both numbers come from the original Switch Transformer paper and have replicated across every serious MoE training run since.

At frontier scale (2025–2026):

| Model | Total params | Active per token | Sparsity |
|---|---|---|---|
| Mixtral-8x7B | 47B | 13B | ~3.5× |
| Mixtral-8x22B | 141B | 39B | ~3.6× |
| DeepSeek-V2 | 236B | 21B | ~11× |
| **DeepSeek-V3** | **671B** | **37B** | **~18×** |
| Qwen 3.5-MoE (varies) | — | — | ~6–15× |

Notice DeepSeek's sparsity ratio is much higher than Mixtral's. That's because Mixtral has 8 large experts and DeepSeek-V3 has 256 small experts. The "fine-grained expert" design choice is one of the few real MoE-architecture battlegrounds, and we discuss it in Section 6.

## 3. The load-balancing problem

The whole MoE story has one structural failure mode: **the router likes to collapse**.

Imagine the router at initialization. The logits $W_g x$ are tiny random numbers; softmax over them is near-uniform; top-$k$ is essentially random. So far so fine.

Now training starts. Suppose by chance expert 3 produces slightly more useful gradients on the first few batches. The router learns to send more tokens to expert 3. Expert 3 gets more tokens, learns faster, produces even more useful gradients. The router doubles down. By the end of the first epoch:

- Expert 3 sees 80% of tokens. Trained on a flood of data.
- Experts 1, 2, 4 see ~20% combined. Mediocre.
- Experts 5–N see ~0%. **They never trained at all.** They are dead weight, taking up parameter count for no quality contribution.

This is **expert collapse**, the central failure mode of MoE training. It happens by default. Every successful MoE training recipe is, in part, a strategy for preventing it.

### What "balanced" means

We want, over a batch, each expert to receive approximately $1/N$ of the tokens. Define the **expert utilization** at step $t$:

$$f_i^{(t)} = \frac{1}{B \cdot T} \sum_{b,\tau} \mathbb{1}\left[i \in \text{topk}(g_{b,\tau})\right].$$

(Fraction of tokens in the batch that picked expert $i$ in their top-$k$.) A well-balanced router has $f_i \approx k/N$ for all $i$. A collapsed router has $f_i$ concentrated on a few experts.

The question is how to *make* the router behave well-balanced without sacrificing the quality it would produce if left alone.

## 4. Classical solution: auxiliary loss

The Switch Transformer (Fedus et al., 2021) and GShard (Lepikhin et al., 2020) both add an **auxiliary loss term** to the main training objective:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i,$$

where:

- $f_i$ is the fraction of tokens routed to expert $i$ (the discrete utilization).
- $P_i = \frac{1}{B \cdot T} \sum_{b,\tau} g_{b,\tau}^{(i)}$ is the *average router probability* for expert $i$ (the continuous score, differentiable through softmax).
- $\alpha$ is a small coefficient (typically $0.01$).

Why this works: the product $f_i \cdot P_i$ is minimized when both are uniform — $f_i = P_i = 1/N$. The gradient through $P_i$ pushes the router away from concentrating probability mass. The $f_i$ factor (non-differentiable) acts as a re-weighting that makes the gradient stronger for over-used experts.

**Why nobody loves it.** The auxiliary loss is in tension with the main loss. The model wants to specialize experts (which means concentrating routing decisions on the *right* expert for each token); the auxiliary loss wants every expert to see uniform traffic. Tuning $\alpha$ is fiddly: too small and the router collapses; too large and the model can't specialize. Frontier labs report spending real effort getting this right.

The auxiliary loss also has a subtle problem: **it pushes the router away from confident decisions**. A well-trained router *should* be very confident that token "the" goes to the syntax expert and token "differential equations" goes to the math expert. The auxiliary loss penalizes that confidence at the population level. The training dynamics fight each other.

## 5. DeepSeek-V3's improvement — auxiliary-loss-free balancing

DeepSeek-V3 (Liu et al., December 2024) shipped a beautifully simple alternative: **don't add a loss term. Add a bias to the router logits and adjust the bias to make utilization uniform.**

The mechanism, in full:

1. Maintain a per-expert bias $b_i$ alongside the router weights. **Crucially, $b_i$ is not learned by SGD — it's updated by a rule.**
2. At each forward pass, the router computes logits $s_i = W_g x + b_i$. The top-$k$ selection uses $s_i$. The *routing weights* used to combine experts still use the gateless logits $W_g x$ (i.e. the bias affects only the *which expert*, not the *how much*).
3. After each step, measure $f_i$ (fraction of tokens that went to each expert). Update the biases:
   - If expert $i$ was *over*-used ($f_i > k/N$), decrease $b_i$ by a small amount $u$.
   - If expert $i$ was *under*-used ($f_i < k/N$), increase $b_i$ by $u$.

That's the whole mechanism. The bias is a steering knob outside the gradient graph. It nudges the router toward selecting under-used experts without touching the gradient signal that drives specialization.

```python
# Pseudocode for one router step (training mode)
scores = self.W_g(x)                            # (B*T, N)
biased_scores = scores + self.bias              # bias is a per-expert nn.Buffer
top_indices = biased_scores.topk(k, dim=-1).indices
weights = scores.softmax(-1).gather(-1, top_indices)  # weights use UN-biased scores

# After the optimizer step, update biases (this runs in no_grad)
fi = compute_expert_utilization(top_indices)    # (N,) — fraction routed to each
error = fi - k / N                              # +ve = over-used, -ve = under-used
self.bias -= self.update_rate * torch.sign(error)
```

Why this is better than auxiliary loss:

1. **No tension with the main loss.** The gradient on $W_g$ only sees the routing signal; load balancing is enforced through a separate channel.
2. **The router can specialize sharply.** Confident routing for "the" → syntax expert is fine, because the bias mechanism doesn't fight it; it just nudges traffic when the cumulative distribution gets too unbalanced.
3. **No hyperparameter to tune across the entire training.** The update rate $u$ is robust over a wide range; DeepSeek reports $u \in [10^{-4}, 10^{-3}]$ works.

The empirical evidence in the DeepSeek-V3 paper is striking: at 671B parameters, the model trains with effectively perfect load balancing (max-to-min expert utilization ratio of ~1.05) and converges faster than the auxiliary-loss baseline. The mechanism has been adopted in derivative MoE papers since.

Our implementation in [`experts.py`](experts.py) uses this approach. We expose the auxiliary-loss path as a comparison but recommend aux-loss-free as the default.

## 6. Fine-grained experts and the shared expert

DeepSeek-V2 introduced two design choices that are now central to its lineage:

### Fine-grained experts

The original MoE designs (Switch, GShard, Mixtral) use a small number of large experts. Mixtral has 8 experts, each with $d_{\text{ffn}}$ = 14336 (huge). DeepSeek-V2/V3 went the other way:

- **DeepSeek-V3**: 256 routed experts + 1 shared expert. Each routed expert is small — $d_{\text{ffn}}$ = 2048. Top-$k$ = 8.

So instead of "1 big expert per token" you have "8 small experts per token, combined." The total active FFN width is similar (8 × 2048 = 16384, close to Mixtral's 14336). But the *combinatorial space* of expert selections is much richer: choosing 8 experts out of 256 gives $\binom{256}{8} \approx 4 \times 10^{14}$ possible combinations, vs $\binom{8}{2} = 28$ for Mixtral-style.

The argument: fine-grained experts allow more specialization, because each expert can be tuned to a narrower slice of the input distribution. The combination of multiple small experts gives the model the same effective FFN capacity as a few large experts, but with the ability to mix specializations finely per token.

### The shared expert

DeepSeek's shared expert is *always active* for every token, regardless of routing. It runs in parallel with the top-$k$ routed experts:

$$y_t = \text{SharedExpert}(x_t) + \sum_{i \in \mathcal{T}_t} w_t^{(i)} \cdot \text{RoutedExpert}_i(x_t).$$

The motivation: there's a lot of generic computation every token needs — basic syntax processing, common-word handling, that sort of thing. If you don't have a shared expert, every routed expert has to learn that baseline too, eating into their specialization budget. The shared expert absorbs the generic work; the routed experts get to specialize harder.

The cost: the shared expert is always on, so it adds to per-token compute. DeepSeek-V3 sizes it modestly (similar to one routed expert's width) so the overhead is ~10-15%. The quality lift is reported as larger than that.

**Used by**: DeepSeek-V2, V3, V3.2. Adopted in some Qwen variants. Mixtral does *not* use a shared expert.

## 7. Top-1 vs top-2 vs top-k

The choice of $k$ — how many experts per token — has three regimes:

- **Top-1** (Switch Transformer): one expert per token. Simplest. Lowest per-token compute. Quality is OK but distinctly worse than top-2+ at the same parameter count. **Used by**: Switch, Google's GLaM in some configs.
- **Top-2** (GShard, Mixtral): two experts per token, weighted-summed. The "default" MoE setting from 2020–2023. Good quality-compute trade-off. **Used by**: most pre-2024 MoE models, all Mixtral variants.
- **Top-k with $k \geq 4$** (DeepSeek-V3 with $k=8$): more experts per token, smaller experts. Goes hand-in-hand with the fine-grained expert design.

The general trend in 2024–2026 has been **smaller, more numerous experts, with higher top-$k$**. The reasoning: finer-grained routing decisions, more combinatorial expressivity, and better load balancing (with $k$ large, $f_i$ values are smoother).

| Year | Canonical recipe | $N$ | $k$ | Expert size |
|---|---|---|---|---|
| 2020 | Switch | 64–2048 | 1 | Standard FFN width |
| 2022 | GShard | 8–512 | 2 | Standard FFN width |
| 2023 | Mixtral | 8 | 2 | Large (~$4d_{\text{model}}$) |
| 2024 | DeepSeek-V2 | 160 | 6 | Small (~$d_{\text{model}}/2$) |
| 2024 | **DeepSeek-V3** | **256+1** | **8** | **Small (~$d_{\text{model}}/2$)** |

The 2026-canonical recipe is closer to DeepSeek-V3 than to Mixtral.

## 8. Expert parallelism (distributed training)

MoE creates a new kind of distributed-training problem that dense models don't have: **the experts are large, sparsely accessed, and need to be distributed across devices**.

The strategy that works is **expert parallelism (EP)**:

- Partition the $N$ experts across $D$ devices. Each device holds $N/D$ experts.
- During the forward pass, the router on each device computes which experts each of its local tokens needs.
- **All-to-all communication**: tokens are shuffled across devices so each device receives the tokens that need its local experts.
- Each device runs its local experts on its received tokens.
- A second all-to-all sends the expert outputs back to the original token positions.

This is conceptually simple but mechanically heavy: two all-to-all collectives per MoE layer per forward pass. The all-to-all is the bottleneck for MoE scaling — it's why MoE training requires high-bandwidth interconnect (NVLink, InfiniBand) and why scaling beyond ~512 GPUs on commodity Ethernet is impractical.

For this course, we describe EP and run all MoE training/inference on a single device. Module 10 covers the distributed-training side in more detail; in Module 11 the pretraining-in-practice demo uses a *dense* DeepSeek-V3-shaped block (not MoE) to stay within the A100-single-node budget.

## 9. Multi-Token Prediction (MTP) — DeepSeek-V3's other trick

Not strictly an MoE feature, but it ships together with DeepSeek-V3 and is worth knowing about: instead of predicting the next token (one cross-entropy loss per position), MTP predicts the next *several* tokens — typically 2 — with separate "MTP heads" attached to the main backbone.

**Why this matters for MoE training specifically**: the auxiliary objective forces the model to encode more future-relevant information in the residual stream, which makes routing more informative. The MTP heads are small and discarded at inference. DeepSeek-V3 reports a meaningful pretraining quality lift (~0.5% on most benchmarks) from MTP, at no inference cost.

We mention it here and revisit in Module 11. Implementing the MTP head is a small addition to the main forward pass — not module-sized.

## 10. When MoE is worth it

| You're... | MoE? | Why |
|---|---|---|
| Pretraining a small model on a tight budget | **No** | The router and EP overhead don't pay off below ~3B effective params |
| Pretraining a frontier-scale model | **Yes** | Strictly better quality per FLOP at scale; every 2024+ frontier model uses it |
| Serving an existing dense model | **No** | Conversion is research-grade; not production-ready |
| Long-context inference where memory is the bottleneck | **It depends** | MoE *increases* total parameter memory; combine with MLA (Module 04) for the right balance |
| Constrained latency at given throughput | **Yes** | Active-param count is what determines latency |
| Constrained GPU memory at given quality | **No** | Total params still need to fit in memory; MoE makes that *harder*, not easier |

The headline: **MoE is a frontier-scale tool**. Below the ~10B-active-param scale, dense is usually a better choice because the overhead (routing, EP, load balancing) eats more than the win. Above ~30B effective params, MoE starts to dominate.

## 11. Scaling from this implementation to a real MoE training run

Suppose you've decided MoE is the right call for your scale. Here's what changes between the code in [`experts.py`](experts.py) and a production training run. **None of this is necessary for the course's pretraining demo (Module 11), which is dense.** But the threshold where MoE starts to win — roughly multi-A100, ~3B+ active params — is a real frontier, and you'll cross it the moment you have the cluster.

**The code change is trivial.** In a transformer block, swap the dense FFN for the MoE FFN:

```python
# Dense (Module 05, what Module 11's pretraining uses)
self.ffn = SwiGLU(d_model, d_ffn=swiglu_hidden_dim(d_model))

# MoE (this module)
self.ffn = MoEFFN(d_model, n_experts=256, top_k=8, d_ffn_expert=d_model // 2,
                  n_shared_experts=1, balancing="aux_loss_free")
```

In the training step, capture the router's output and apply the bias update:

```python
out, router_info = block(x, freqs_cis)        # router_info instead of just out
# ... loss, backward, optimizer.step() ...
block.moe.router.update_bias(router_info.utilization)  # AFTER opt.step()
```

That's the entire code-level diff. The hard parts are everywhere else:

**Expert parallelism (EP).** Pure FSDP doesn't fit MoE well — each expert is too big to shard usefully and too sparsely accessed to amortize the all-gather. The shipping strategy is **EP combined with FSDP**: experts are partitioned across devices (each device holds $N/D$ experts) and the dense parts (attention, norms, embeddings) are FSDP-sharded as usual. Each MoE forward becomes: dispatch (all-to-all to send tokens to their expert's device) → run local experts → combine (all-to-all to send outputs back). PyTorch's native EP support is still maturing; in 2026 the canonical libraries are **Megatron-Core**, **DeepSpeed-MoE**, and **MegaBlocks** (the last one ships fused grouped-GEMM kernels so the expert forward is one matmul instead of N).

**Interconnect.** Two all-to-alls per layer per forward pass means MoE is communication-bound. **NVLink within a node is mandatory**; cross-node needs InfiniBand or equivalent (200+ Gbps). Commodity Ethernet won't scale MoE past ~8 GPUs in practice. This is the single biggest infrastructure jump from a dense training run.

**Memory.** Total parameters are 10–30× the active count. For DeepSeek-V3 (671B total / 37B active) the full model only fits across ~80+ H100s in BF16. You don't get out of holding the whole model in aggregate device memory; MoE saves *compute and KV cache*, not parameter memory.

**Hyperparameters.** A few things shift:
- **Batch size**: larger, to amortize routing + all-to-all overhead.
- **Learning rate**: typically the same as the dense-equivalent active-param model, since the gradient signal per parameter is on a similar scale.
- **Bias update rate $u$**: $u \in [10^{-4}, 10^{-3}]$ is robust. Larger $u$ converges faster but can oscillate; smaller is safer.
- **Capacity factor**: if you're using a non-aux-loss-free implementation, you'll see a "capacity factor" hyperparameter that bounds how many tokens any expert can take per batch (overflow tokens get dropped). With aux-loss-free this is unnecessary — the bias keeps utilization within bounds by construction.

**What to monitor.** Three things that don't appear in dense training:
- **Per-expert utilization** every step. Our `RouterOutput.utilization` is the source. Plot the max/min ratio across experts; anything past 3× is a problem.
- **The bias trajectory.** If biases keep growing without bound, something is broken in the routing signal (usually a degenerate input distribution).
- **Token-drop rate** (with classical aux-loss balancing only). High drop = capacity factor too tight or balancing not converging.

**When to upcycle.** If you've already trained a dense base model and want a faster route to MoE, **sparse upcycling** initializes each routed expert from a copy of the dense FFN weights. Komatsuzaki et al. 2022 showed this beats from-scratch MoE training at the same *additional* compute, but loses to from-scratch MoE at the same *total* compute. It's a budget play: you spend compute on the dense first, then accelerate the MoE phase. Frontier labs train MoE from scratch; mid-tier labs sometimes upcycle.

The bottom line: the architecture sketched in this module is the right shape, and `MoEFFN` is a real building block. The gap between "works on a single A100" and "scales to a 256-GPU MoE training run" is mostly infrastructure (EP, interconnect, fused kernels), not algorithm.

## 12. What we implement

| Component | Status |
|---|---|
| Expert (SwiGLU FFN, configurable width) | Full implementation ([`experts.py`](experts.py)) |
| Router with top-$k$ selection | Full implementation |
| **Aux-loss-free load balancing** | **Full implementation, default mode** |
| Auxiliary-loss balancing (Switch / GShard) | Implemented as a comparison; available via flag |
| MoE FFN (routed + optional shared expert) | Full implementation |
| Expert parallelism (distributed) | Described conceptually; single-device implementation only |
| Multi-Token Prediction | Described; defer implementation to Module 11 (pretraining) |
| Capacity factor / token dropping | Described; not implemented (the aux-loss-free path makes it unnecessary in practice) |

The implementation composes cleanly with the Module 05 attention/normalization stack: an MoE-augmented transformer block is just `RMSNorm → Attention → +x → RMSNorm → MoEFFN → +x`. Module 07 will assemble this into a full model.

## 13. Frontier-model MoE configurations

| Model | $N$ | $k$ | Expert $d_{\text{ffn}}$ | Shared expert | Balancing |
|---|---|---|---|---|---|
| Switch Transformer | 64–2048 | 1 | Standard | No | Aux loss |
| Mixtral-8x7B / 8x22B | 8 | 2 | 14336 / 16384 | No | Aux loss |
| DeepSeek-V2 | 160 | 6 | 1408 | Yes (1) | Aux loss + bias |
| **DeepSeek-V3** | **256** | **8** | **2048** | **Yes (1)** | **Aux-loss-free + sequence balance** |
| Qwen 3.5-MoE | varies | 4–8 | small | varies | Hybrid |

DeepSeek's "sequence balance loss" is a small term they keep even with aux-loss-free balancing, to prevent extreme imbalance within a single sequence. Section 7 of the DeepSeek-V3 paper has the exact formula; it's a tiny coefficient and we don't include it in the default `experts.py`.

## 14. Reading list

- **Auxiliary-loss-free balancing**: DeepSeek-AI, "DeepSeek-V3 Technical Report" (Dec 2024), Section 2.1.2 specifically. The bias-update rule.
- **MoE foundations**: Fedus et al., "Switch Transformer" (2021). The first MoE paper to scale convincingly, and the source of the auxiliary loss.
- **Fine-grained experts**: Dai et al., "DeepSeekMoE" (2024). Argues for many small experts + shared expert.
- **Routing analysis**: Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022). The clearest treatment of what goes wrong in MoE training and how to fix it. The router-z-loss they introduce is sometimes used in addition to the bias trick.

## Next

[Module 07 — Assembling the Full Model](../07-full-model/). Embeddings, weight tying, muP initialization, shape tests. Where attention, blocks, and experts come together as a single `nn.Module` you can run forward on a batch.
