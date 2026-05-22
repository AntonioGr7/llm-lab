# Module 04 — Attention

> Part of [Part 2 — Architecture](../). Reading time: ~90 minutes. Compute cost: ~$0–2 (a few minutes on a GPU for the FlashAttention timing; everything else is CPU or tiny-tensor).

## The thesis

Attention is the operation. The transformer block is glue around it.

Every architectural innovation in language models since 2017 — multi-query, grouped-query, FlashAttention, sliding-window, linear-attention hybrids, Multi-head Latent Attention, sparse routing — has been driven by *one* tension: keeping the expressive power of softmax attention while reducing its $O(n^2)$ cost, either in compute, in memory, in inference-time KV cache, or in all three at once.

This module walks the full arrow from Vaswani 2017's scaled dot-product attention to DeepSeek V3.2's Sparse Attention and V4's Compressed Sparse Attention. At every step we ask the same questions:

1. **What does it compute?** The math.
2. **What does it cost?** Compute, memory, KV cache, network for distributed inference.
3. **What does it give up?** Quality, generality, hardware compatibility.
4. **When do you reach for it?** The decision.

By the end you will implement three of these variants from scratch (MHA, GQA, MLA) and read the rest fluently enough to follow any frontier paper.

## What you'll be able to do at the end

- Derive scaled dot-product attention and explain every term (including why √d_k).
- Implement Multi-Head Attention, Grouped-Query Attention, and Multi-Head Latent Attention end-to-end, with shape-checked forward passes.
- Use FlashAttention transparently through PyTorch's `scaled_dot_product_attention`.
- Compute KV-cache size for a given architecture and explain why MLA wins at long context.
- Articulate when to reach for sliding-window, linear-attention hybrids (Qwen 3.5 style), and sparse-routing attention (DeepSeek DSA/CSA).
- Read a frontier-model paper's "attention" section and identify which design pressures their choices reflect.

## 1. Scaled dot-product attention

The thing every variant in this module is a refinement of:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Three matrices: queries $Q$, keys $K$, values $V$. Read this as:

1. For each query (each row of $Q$), compute its dot product with every key (each row of $K$). Result: a matrix of pairwise affinities, shape $(n_q, n_k)$.
2. Scale by $\sqrt{d_k}$ and softmax across the key dimension. Each query row becomes a probability distribution over keys.
3. Use those probabilities to weight-sum the values $V$. Each query gets back a $d_v$-dimensional vector — its "summary" of $V$, weighted by relevance.

**Why softmax?** It enforces the weights are non-negative and sum to 1 across keys, so each query's output is a *convex combination* of values. This is the "selection" intuition: the query asks a question, the keys decide who answers, the values are the answers.

**Why √d_k?** Without it, the dot product $Q K^\top$ grows with $d_k$. For large $d_k$ the entries get large in magnitude, softmax saturates to one-hot, and gradients vanish. Dividing by $\sqrt{d_k}$ keeps the dot-product variance roughly constant. (Vaswani 2017 derived this from the variance of a sum of $d_k$ products of zero-mean unit-variance vectors.)

**Causality.** For autoregressive language models, queries at position $t$ may only attend to keys at positions $\le t$. Enforced by adding $-\infty$ to the affinities above the diagonal before softmax (equivalently, multiplying weights below the diagonal by 1 and above by 0 after a normalized softmax — but the additive-mask version is standard because it composes cleanly with the softmax).

## 2. Multi-head attention (MHA)

One attention head is a single perspective. A real model needs many.

**The mechanics.** Split the model dimension $d_{\text{model}}$ into $H$ heads, each with $d_{\text{head}} = d_{\text{model}} / H$ dimensions. Run attention in parallel on each head's slice. Concatenate the outputs along the feature dimension. One final linear projection $W_O$ mixes information back across heads.

```python
def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads):
    B, T, D = x.shape
    H = n_heads
    d_head = D // H

    q = (x @ W_q).view(B, T, H, d_head).transpose(1, 2)  # (B, H, T, d_head)
    k = (x @ W_k).view(B, T, H, d_head).transpose(1, 2)
    v = (x @ W_v).view(B, T, H, d_head).transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    out = weights @ v                                     # (B, H, T, d_head)

    return (out.transpose(1, 2).reshape(B, T, D)) @ W_o
```

That's the canonical implementation. Every variant below mutates exactly one of these lines.

**Why multiple heads matter.** Different heads end up specializing — some attend to syntactic neighbors, some to long-range coreferences, some to specific token types. Probing studies show this consistently. You won't outperform 4 heads at $d_{\text{model}}$=512 with 1 head at $d_{\text{head}}$=512; the parallel diversity is the point.

## 3. The complexity wall

Attention is $O(n^2)$ in sequence length. Two costs to track separately:

- **Compute** — the $QK^\top$ matmul is $n^2 \cdot d_{\text{head}}$ FLOPs per head. At $n=8192$ this is the dominant kernel in a transformer block. At $n=128\text{k}$, attention compute exceeds the FFN cost.
- **Memory** — the attention matrix is $n \times n$. At $n=64\text{k}$, that's a 4 GB tensor per head per layer in BF16, before FlashAttention. This is the part that breaks first.

There are also two *very different* attention regimes:

- **Training / prefill** — you have the full sequence at once; $Q, K, V$ all have $n$ rows; attention is one big $n^2$ matmul (which is what FlashAttention optimizes).
- **Decode** (inference, one token at a time) — at step $t$ you have one query row, but you need all $t$ previous $K, V$ rows. The "compute" cost per step is $O(n \cdot d_{\text{head}})$, but you must store all previous $K, V$. This is the **KV cache** problem, and it grows linearly with both the context length and the model size.

A 70B model at 32k context can use **~40 GB of KV cache per inference request**. That's the cost everyone is trying to attack.

The next sections each pick at one of these pressure points:

| Innovation | Attacks |
|---|---|
| GQA / MQA | KV cache size (share K/V across heads) |
| FlashAttention | Attention-matrix memory (don't materialize it) |
| Sliding window | Both, via locality (compute fewer pairs) |
| MLA | KV cache size (cache a compressed latent, not K/V) |
| Hybrid linear-attention | Compute cost (replace softmax in most layers) |
| DSA / CSA | Compute cost (sparse routing) and KV cache (compression) |

## 4. Grouped-Query Attention (MQA → GQA)

The easiest KV-cache win, and the one every modern decoder uses.

**Observation:** the queries learn different things across heads, but the keys and values do a lot of redundant work. So **share K and V across groups of queries**.

- **MHA**: $H$ query heads, $H$ key heads, $H$ value heads. Maximum capacity, maximum cache.
- **MQA** (Shazeer 2019): $H$ query heads, **1** key head, **1** value head. Minimum cache (1/H the size), some quality loss.
- **GQA** (Ainslie 2023): $H$ query heads, $G$ key/value head *groups* with $G < H$. Each group of $H/G$ query heads shares one K and one V head. Tunable middle ground.

```python
# Same as MHA, but k and v have fewer heads
def gqa(x, W_q, W_kv, W_o, n_q_heads, n_kv_heads):
    q = (x @ W_q).view(B, T, n_q_heads,  d_head).transpose(1, 2)
    kv = (x @ W_kv).view(B, T, n_kv_heads, 2 * d_head).transpose(1, 2)
    k, v = kv.chunk(2, dim=-1)
    # Repeat K and V across query-head groups before attention
    k = k.repeat_interleave(n_q_heads // n_kv_heads, dim=1)
    v = v.repeat_interleave(n_q_heads // n_kv_heads, dim=1)
    # ... rest is standard attention
```

**Cache reduction** = $H / G$. Llama 2 70B uses $H=64, G=8$ → 8× smaller KV cache than MHA. Most papers report this saves 90%+ of inference memory at minimal quality cost (~0.1–0.3% on benchmarks).

**Tradeoff.** Smaller $G$ means more cache savings but less expressive K/V. Empirical sweet spot for ~70B models: $G$ around 8. For very small models it matters less; for very long context it matters more.

In our implementation ([`attention.py`](attention.py)), `GroupedQueryAttention(n_heads=H, n_kv_heads=G)` is the canonical class. Setting $G = H$ recovers MHA; setting $G = 1$ recovers MQA.

## 5. FlashAttention — same math, different I/O

The Big Lie about attention is that it's compute-bound. It's not. **It's memory-bound.** The bottleneck on modern GPUs is high-bandwidth-memory (HBM) traffic — moving the $n \times n$ attention matrix in and out of HBM for the softmax. The matmul FLOPs are easy; the memory I/O is what's slow.

**FlashAttention** (Dao 2022) is an *algorithm-equivalent* rewrite that:

1. Splits Q, K, V into tiles that fit in SRAM (the small, fast on-chip memory).
2. Computes attention tile-by-tile, using the **online softmax** trick (Milakov & Gimelshein 2018) to compute softmax incrementally without ever materializing the full $n \times n$ matrix.
3. Never writes the attention matrix to HBM.

It is **bit-exact with standard attention** at the numerical level (modulo BF16/FP16 nondeterminism). It is up to 7× faster at $n=8192$ and trivially handles much longer sequences than naive attention.

Versions:

| Version | Year | What changed |
|---|---|---|
| FA1 | 2022 | The tiling + online softmax algorithm |
| FA2 | 2023 | Better parallelization across heads and batches; ~2× over FA1 |
| FA3 | 2024 | Hopper-specific (H100/H200): async memory copies, FP8 path, ~1.5–2× over FA2 |

**The practical takeaway**: use PyTorch's `torch.nn.functional.scaled_dot_product_attention`. It dispatches transparently to FA2 on Ampere (A100), FA3 on Hopper (H100), or a fallback CPU implementation if no GPU is present. The same code runs everywhere.

```python
# All you have to write:
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

That single call is what every implementation in this course uses for the attention kernel itself. The complexity is hidden behind a stable API.

**When you'd write your own FlashAttention.** Rare. The reasons people do: custom masking patterns that SDPA doesn't support (block-sparse, document-boundary), exotic data types (FP8 with custom scaling), or research on alternatives. For pretraining, fine-tuning, and most inference, the bundled SDPA backend is the right call.

## 6. Sliding Window Attention

The simplest sparsity pattern. Each token attends only to the previous $W$ tokens; entries past $W$ are masked out.

- **Compute** drops from $O(n^2)$ to $O(n \cdot W)$. Linear in $n$ when $W$ is fixed.
- **KV cache** drops from $O(n)$ to $O(W)$ per layer — once you're past position $W$, you can drop old entries.
- **Quality cost**: distant dependencies must hop through layers. With $L$ layers and window $W$, the effective receptive field is $L \cdot W$. Mistral 7B with $W=4096$ and 32 layers covers a 128k effective receptive field, even though each individual layer only sees 4k.

**Used by:** Mistral 7B, parts of Gemma's design. Pure sliding-window architectures have largely been replaced by hybrid approaches (next section), but the technique is still common in conjunction with full attention every $k$ layers.

**Tradeoff:** you give up *direct* long-range attention. For some tasks (long-range factual recall, in-context learning over long documents) this hurts. For most local-context tasks it doesn't.

## 7. The KV cache wall — and why it forces compression

Quick math. For a model with $L$ layers, $H$ heads, $d_{\text{head}}$ per head, BF16 (2 bytes per number), at context length $n$, the KV cache size is:

$$\text{KV cache} = 2 \cdot L \cdot H \cdot d_{\text{head}} \cdot n \cdot 2 \text{ bytes}$$

For Llama 3 70B ($L=80, H=64, d_{\text{head}}=128$) at $n=64\text{k}$, that's

$$2 \cdot 80 \cdot 64 \cdot 128 \cdot 65{,}536 \cdot 2 \approx 170 \text{ GB}.$$

Llama 3 actually uses GQA with 8 KV heads, so divide by 8: ~21 GB per request. Still a lot. For batched inference (multiple users at once), this scales linearly with batch size and is *the* dominant memory cost.

Two ways out:

1. **Cache *less*** — share K/V across heads (GQA), or compress K/V into a smaller latent (MLA).
2. **Cache *cheaper*** — quantize the cache (INT8/INT4), use page-able KV memory (vLLM's PagedAttention), or offload to slower storage.

MLA, next, is the most aggressive answer to (1).

## 8. Multi-Head Latent Attention (MLA) — DeepSeek V2/V3

The biggest single architectural change in 2024 frontier models, and a real piece of design cleverness.

### The core idea

K and V at each position $t$ are functions of the residual stream $x_t$ that produced them. They're not arbitrary; they're projections of a lower-dimensional thing. **So cache the lower-dimensional thing.**

Concretely: instead of caching $K \in \mathbb{R}^{n \times H \cdot d_{\text{head}}}$ and $V \in \mathbb{R}^{n \times H \cdot d_{\text{head}}}$, cache a *latent* $c_{KV} \in \mathbb{R}^{n \times d_c}$ where $d_c \ll H \cdot d_{\text{head}}$. At attention time, decompress on-the-fly:

$$K = c_{KV} \, W_{K}^\text{up}, \quad V = c_{KV} \, W_{V}^\text{up}.$$

### Why this isn't trivially worse

You might think "compressing K and V to a smaller space must hurt quality." It doesn't — much. The intuition: $K, V$ for a single head are already low-rank in practice (the model doesn't fill the full $d_{\text{head}}$-dimensional space with information). Sharing a single learned latent across all heads lets the model *choose* what to compress and what to keep, instead of redundantly storing similar K/V across heads.

DeepSeek V3 uses $d_c = 512$ for $H = 128$ heads at $d_{\text{head}} = 128$. KV cache shrinks from $2 \cdot 128 \cdot 128 = 32{,}768$ values per token (MHA) to $512$ (MLA) — a **64× reduction**.

### The RoPE problem, and DeepSeek's fix

There's a subtlety. Rotary position embeddings (RoPE, see [Module 05](../05-transformer-block/)) are applied to Q and K *element-wise based on absolute position*. RoPE doesn't commute with the linear projection $W_K^\text{up}$: you can't apply RoPE to the cached latent and then decompress, because the rotation depends on position in a way that gets scrambled by the linear projection.

DeepSeek's fix is elegant: **decoupled RoPE**. Split the key into two parts:

- A "content" part of dimension $d_{\text{head}}$, computed from the latent. **No RoPE applied.**
- A "RoPE" part of dimension $d_r$ (e.g. 64), computed directly from $x_t$ via a small projection and **shared across heads**. RoPE applied here.

The query mirrors this split: a content query of dim $d_{\text{head}}$ and a RoPE query of dim $d_r$ per head. The two parts are concatenated and attention runs as usual.

What gets cached per token:

1. The latent $c_{KV} \in \mathbb{R}^{d_c}$.
2. The shared RoPE key $k_r \in \mathbb{R}^{d_r}$ (one for all heads, not per-head).

Total cached state per token: $d_c + d_r$. For DeepSeek-V3: $512 + 64 = 576$ values. Compare:

| Variant | Cache per token | vs MHA |
|---|---|---|
| MHA ($H=128, d_h=128$) | 32,768 | 1× |
| GQA (groups=8) | 2,048 | 16× smaller |
| **MLA** | **576** | **57× smaller** |

That's the headline. For long-context inference, MLA is the design choice that makes 128k+ context economically tractable.

### Implementation

Our [`attention.py`](attention.py) has a `MultiHeadLatentAttention` class. The forward pass is:

1. Compute the query: a content projection $q_c$ and a RoPE projection $q_r$ (both per-head). Apply RoPE to $q_r$.
2. Compute the KV latent $c_{KV} = x W_{KV}^\text{down}$ and the shared RoPE key $k_r$ (rotated).
3. Decompress K and V from $c_{KV}$.
4. Concatenate per-head: $q = [q_c \, | \, q_r]$, $k = [k_c \, | \, k_r \text{ (broadcast across heads)}]$. $v = v_c$ (no RoPE on V).
5. Standard scaled-dot-product attention via SDPA.

At inference time you cache $c_{KV}$ and $k_r$; the decompression and attention happen as part of the forward pass at each new step.

**Training cost**: roughly the same as MHA. **Inference memory**: the order-of-magnitude smaller cache is the entire point.

## 9. Hybrid attention — Qwen 3.5 and the linear-attention comeback

Two observations are now load-bearing:

1. Softmax attention is expensive ($O(n^2)$) but very expressive.
2. **Most layers don't need that expressivity.** Probing experiments find that quality is dominated by a small fraction of attention layers; the rest do bookkeeping that a cheaper operator could handle.

**Linear attention** and **state-space models** (S4, Mamba, RWKV, RetNet) compute a related operation in $O(n)$. At training they look like attention with a kernel feature map; at inference they have a recurrent form with a fixed-size hidden state, like an RNN. They're much faster at long context but historically worse on language modeling.

**Hybrid architectures** alternate softmax-attention layers with linear-attention/SSM layers in a fixed ratio. The "interesting" reasoning happens in the softmax layers; the linear layers handle the rest at a fraction of the cost.

Examples in 2025–2026:

- **Jamba** (AI21, 2024): alternating Transformer blocks and Mamba blocks. 7:1 ratio of Mamba:Transformer for some configurations.
- **Qwen 3.5** (2025): **3:1 ratio of gated DeltaNet to softmax attention**. DeltaNet is a particular linear-attention variant from the recurrent-attention research line (Schlag et al., 2021; Yang et al., 2024); "gated" adds a forget gate to control how aggressively old context is overwritten. The 3:1 ratio means 3 out of every 4 layers are the cheap variant.
- **Various open hybrids**: RWKV, RetNet's hybrid configs, Mamba-2-Hybrid.

**What you get.** Roughly 2–4× faster inference at long context, with ~equal quality on most benchmarks. The exact tradeoff depends on the linear-attention choice and the ratio; in 2026 the field is converging on "alternate, but keep enough softmax layers to handle the genuinely hard cases."

**What we don't implement.** Hybrid attention requires picking and implementing a specific linear-attention variant, and the choice space is still moving. We describe the design at the conceptual level here. If you build a hybrid model later, the canonical references to start with are [Mamba](https://arxiv.org/abs/2312.00752) (state-space), [DeltaNet](https://arxiv.org/abs/2406.06484) (linear-attention with delta-rule updates), and Jamba's architecture paper.

## 10. DeepSeek Sparse Attention (DSA) — V3.2

> **Note**: my recall here is good at the conceptual level but I'd want to double-check the specific hyperparameters against the V3.2 paper. Treat the high-level mechanism as reliable; treat the exact dimensions and training schedule as "approximate, verify."

The next pressure point after MLA: **even with MLA's compression, every query still attends to every key.** Compute is still $O(n \cdot k_{\text{seq}})$ at decode (where $k_{\text{seq}}$ is the context length), and prefill is still $O(n^2)$.

**DSA's idea.** Don't attend to every key. Learn a lightweight *indexer* that, for each query, scores which keys are worth attending to and selects the top-$k$.

The architecture:

1. **Lightning indexer** — a small auxiliary scoring head, much cheaper than full attention, that produces a score for each (query, key) pair. Specifically: a low-rank score $s_{ij} = u_i^\top v_j$ where $u, v$ are projections cheaper than full attention.
2. **Top-k selection** — for each query, keep only the top-$k$ scored keys ($k \ll n$). The attention is computed only over those $k$ keys.
3. **End-to-end training** — the indexer is trained jointly with the model, typically with a distillation loss against the full-attention version so the selection learns to keep the keys that *would* have gotten attention weight.

**What it costs.** Prefill compute drops from $O(n^2)$ to $O(n \cdot k)$ where $k$ is the selection size (e.g. 256–2048 depending on context length). KV cache size is unchanged from MLA (MLA + DSA stack), but the *attention compute* is now sublinear in some configurations.

**The not-quite-free part.** Top-$k$ selection breaks the dense matrix-multiply primitive that GPUs love. DSA needs custom kernels to be fast in practice — the gather operations are unfriendly to TensorCore-style fused matmuls. This is why open implementations of sparse attention have been slow to ship; the kernels are the work.

**Used by:** DeepSeek-V3.2 (production-shipped, late 2025). Mechanism appears in derivative papers since.

**Why this isn't the same as "just use a longer context with linear attention".** Linear attention's $O(n)$ comes from approximating softmax with a kernel; you lose the sharp peakedness that lets attention pick out one specific match in a long context. DSA keeps softmax (over the selected $k$), so you keep the picking-power, you just don't pay for all the keys you'd have ignored anyway.

## 11. Compressed Sparse Attention (CSA) — DeepSeek V4

> **Honest caveat**: My training data is light on V4-specific details. The high-level framing here ("combine MLA compression with DSA sparsity") is my best reconstruction; the specific architectural changes vs V3.2 are at the edge of what I can verify. If you can share the V4 paper or model card, I'll refine this section against canonical text.

The conceptual direction is clear even if the specifics aren't: V4 fuses the **compression** lesson from MLA with the **sparsity** lesson from DSA into a single operator.

The design pressures motivating it:

- MLA shrunk the cache; the attention compute is still proportional to context length.
- DSA shrunk the compute; the cache is still set by MLA.
- Combine them and you have *small cache* + *sublinear compute* + *full-softmax expressivity on the kept keys*.

The mechanism (high-confidence parts):

- KV is still compressed into a latent (MLA-style).
- A lightning-indexer-style routing selects the top-$k$ keys per query.
- The selected keys are decompressed from the latent on demand, attention runs over them.

The mechanism (less-confident parts — flagging for source-based refinement):

- Whether V4 changes the indexer architecture (e.g. multiple routing heads).
- Whether the selection is per-head or per-layer.
- Specific dimensions and training-recipe differences from V3.2.
- Whether there's an additional "compression-aware" routing signal.

**What's robust regardless of specifics.** The trajectory MLA → DSA → CSA is a clean illustration of how frontier attention research moves: each generation identifies the *remaining* cost (cache, then compute, then both jointly) and engineers it down without giving up the mathematical core. If you understand MLA and DSA conceptually, the V4 paper will read as engineering deltas, not new fundamentals.

## 12. The decision table

What to use, and when, in 2026:

| You're... | Reach for | Why |
|---|---|---|
| Pretraining a small model from scratch on a budget | **GQA + FlashAttention (SDPA)** | Best quality/cost ratio. ~95% of frontier model behavior at a fraction of the engineering. |
| Building a research demo of DeepSeek-V3 architecture | **MLA + FlashAttention** | The defining feature of V3; we use it in [Module 11](../../part-3-pretraining/11-pretraining-in-practice/) |
| Serving long-context inference on a small budget | **MLA**, possibly + KV quantization | Cache size dominates inference cost at long context |
| Optimizing inference throughput at fixed quality | **Hybrid (3:1 linear:softmax)** | Most layers don't need softmax expressivity |
| At the frontier, pushing context past 256k | **MLA + DSA**, or CSA when V4-style kernels become open | Both cache and compute scale better than alternatives |
| Doing research on novel attention | Start from SDPA, replace one term at a time | The math is stable; the cost structure is what you're moving |

## 13. What we implement, what we describe

| Variant | Status in this course |
|---|---|
| Scaled dot-product attention | Implemented from scratch ([`attention.py`](attention.py)) — also via PyTorch SDPA |
| Multi-Head Attention | Full implementation |
| Grouped-Query Attention | Full implementation (parameterized; MHA = GQA(groups=H), MQA = GQA(groups=1)) |
| FlashAttention | Used transparently via PyTorch SDPA — no separate implementation needed |
| Sliding-window attention | Described; trivial to enable via SDPA's `attn_mask` argument |
| **Multi-Head Latent Attention** | **Full implementation, including decoupled RoPE** |
| Hybrid attention (Qwen 3.5 style) | Described conceptually; implementation deferred (would require picking a linear-attention variant) |
| DSA (DeepSeek V3.2) | Described; implementation deferred (custom kernels required for speed) |
| CSA (DeepSeek V4) | Described conceptually; specifics to be refined against canonical sources |

For the rest of the course (Module 11's pretraining run, Part 4's post-training), we use **MLA + SDPA** as the default attention. That's the most aggressive choice that's still entirely standard PyTorch and trains in reasonable time on an A100.

## 14. What "knowing attention" means

By the end of this module you should be able to do four things on the back of an envelope:

1. **Forward pass, by hand**, on a 2-token sequence with 1-head attention. Just to prove the dimensions and the softmax to yourself.
2. **KV cache size**, for an arbitrary architecture, given $L, H, G, d_{\text{head}}, n$. Including the MLA case where it's a latent.
3. **The bottleneck**, for a given workload: training vs prefill vs decode; compute-bound vs memory-bound; per-request vs per-batch. The right architecture depends on which of these is binding.
4. **Read a paper.** Open a frontier paper's "attention" section and identify: what's the variant, what pressure is it attacking, what does it cost, and what does it give up.

If you can do those four things, you can debug, scale, or replace any attention design in production.

## Next

[Module 05 — Transformer Block](../05-transformer-block/). RMSNorm, SwiGLU, RoPE, pre-norm vs post-norm — the rest of the transformer block, and how attention plugs into it.
