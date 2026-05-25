# Module 05 — The Transformer Block

> Part of [Part 2 — Architecture](../). Reading time: ~80 minutes. Compute cost: ~$0 (everything here is CPU-friendly).

## The thesis

Attention got a 90-minute treatment because every variant in it is an engineering response to its $O(n^2)$ cost. The rest of the transformer block is the opposite: a handful of small components, each one a clean win, that have *stopped* mutating. Modern frontier models — Llama 3.x, DeepSeek V3, Qwen 3.5 — disagree on a lot, but they all use **pre-norm with RMSNorm + SwiGLU FFN + RoPE for positions**. This is the canonical block in 2026.

This module explains why that combination won, and what the residual stream — the single tensor that flows through every block — actually represents. It's also where RoPE gets the full treatment that Module 04 deferred.

## What you'll be able to do at the end

- Explain what the residual stream is and why pre-norm is the only sane choice for deep models.
- Implement RMSNorm and explain when it's safe to drop LayerNorm's mean centering (always, in practice).
- Implement a SwiGLU FFN at the right hidden width, and explain the "8/3" rule.
- Implement RoPE end-to-end, derive why the inner product after rotation depends only on relative position, and pick a base frequency for a given target context length.
- Apply frequency interpolation (linear / NTK / YaRN) to extend a model's context window without retraining.
- Compose all of these into a working transformer block whose forward pass shape-checks against MHA, GQA, and MLA from Module 04.

## 1. The residual stream

The single most useful mental model for a transformer is this:

> The transformer maintains a **residual stream** — a tensor of shape `(batch, seq_len, d_model)` — that flows through every block. Each block *reads* from the stream, *computes* something, and *writes* back to the stream by addition.

```
x_0 → [Block 1] → x_1 → [Block 2] → x_2 → ... → [Block L] → x_L → logits
                  ↑                  ↑
            x_1 = x_0 + Δ_1     x_2 = x_1 + Δ_2
```

Each $\Delta_\ell$ is the output of an attention sub-layer or an FFN sub-layer (a pre-norm block has two of each per "layer"). The block doesn't *replace* the residual stream; it *adds to it*.

Three consequences fall out of this framing, and they explain almost every architectural choice in this module:

1. **The residual path is the central bus.** Information has to live somewhere across many layers; the residual stream is where. Each block contributes a small update. Empirically, the residual stream has very wide-tailed dynamics: a few dimensions carry most of the energy, and they grow in magnitude with depth. Normalization is what keeps this manageable.
2. **Gradients flow through addition cleanly.** $\partial x_L / \partial x_0$ contains an identity term plus a sum of block Jacobians. Identity = no vanishing gradient on the residual path itself. This is *the* reason transformers can go to 100+ layers.
3. **Sub-layers are perturbations.** If the FFN or attention output is much larger than the residual, training is unstable. If it's much smaller, the layer is dead weight. Normalization keeps each sub-layer's contribution at roughly the right scale.

Anthropic's interpretability work made this framing famous (they read the residual stream as a "communication channel" between heads across layers). It's not a metaphor — it's literally how the model is wired.

## 2. Pre-norm vs post-norm

The original Transformer (Vaswani 2017) was **post-norm**:

```
x_out = LayerNorm(x_in + SubLayer(x_in))
```

Modern transformers are **pre-norm**:

```
x_out = x_in + SubLayer(LayerNorm(x_in))
```

The difference looks tiny. It is not.

**Post-norm** puts the normalization *on the residual path*. Every layer's output passes through a LayerNorm before being read by the next layer. This means the gradient flowing backward through the residual stream gets multiplied by the LayerNorm Jacobian at every step. With $L$ layers, you accumulate $L$ such factors. Without aggressive learning-rate warmup, training diverges past ~12 layers.

**Pre-norm** puts the normalization *inside the sub-layer*, before the attention/FFN computation. The residual path itself is just $x + f(\text{LN}(x))$: pure addition, no transformations on the bus. Gradients on the residual path are clean (identity). This is what made it possible to train 70B+ models without bespoke warmup schedules.

The cost: the *output* of a pre-norm stack is not normalized. You need one final LayerNorm/RMSNorm at the very top, before the unembedding, to keep logits well-scaled. Every modern model has this "final norm" — look for `model.norm` or `final_layer_norm` in any open weights.

**Other options exist and you should know they exist but not pick them:**

- **Sandwich norm** (norm before *and* after the sub-layer). Used in a few early-2022 papers. Strictly worse than pre-norm in practice.
- **DeepNorm** (Microsoft, 2022). A post-norm variant with a careful weight-initialization scaling that lets post-norm work at depth. The motivation was theoretical; the conclusion of the field was "or you could just use pre-norm." DeepNorm is in no recent frontier model.

**Decision:** pre-norm. Always. Nothing in this course will use anything else.

## 3. RMSNorm — LayerNorm without the mean

Plain LayerNorm:

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \frac{1}{d}\sum_i x_i, \; \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2.$$

RMSNorm (Zhang & Sennrich, 2019):

$$\text{RMS}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}.$$

The differences:

1. No mean subtraction. Just rescaling by the root-mean-square of the features.
2. No bias term $\beta$. (Could be added; in practice no modern model does.)

**Why this matters in practice:**

- **Quality**: identical, within noise. Every paper that's ablated it (PaLM, Llama, Qwen) finds no measurable degradation.
- **Compute**: ~7% faster per call, because you skip one reduction (the mean) and one elementwise subtraction.
- **Distributed training**: the mean reduction in LayerNorm couples across the feature dimension. If you tensor-parallel-shard the feature dim, you need an extra all-reduce to compute $\mu$. RMSNorm just needs the local $\sum x_i^2$ followed by one all-reduce. This is small but cumulative — Llama-3-scale training reports >1% throughput from the change.

The bias removal generalizes: **modern transformers use almost no bias terms anywhere**. Not in linear layers, not in norms, not in attention. The reasons:

1. Biases interact awkwardly with weight decay (no one wants to decay them, but excluding them from the decay group adds bookkeeping).
2. They cost an extra parameter per output dimension for no measurable quality benefit at scale.
3. Removing them simplifies tensor parallelism (a bias would need to be added on a specific rank, breaking symmetry).

Llama, Qwen, DeepSeek, Mistral — all bias-free. We follow that.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.gamma
```

Note the `.float()` casts: the squared mean can underflow in BF16 for small magnitudes. Promote to FP32 for the normalization, cast back to the input dtype, then apply the (FP32) learned scale. This is how Llama and PyTorch reference implementations do it; deviating costs you accuracy.

## 4. The FFN — from MLP to SwiGLU

Every block has two sub-layers: attention and an FFN. The FFN is per-token (no mixing across positions); it's the model's "compute budget" for transforming features in place.

### The original FFN

Vaswani 2017's FFN was:

$$\text{FFN}(x) = W_2 \, \text{ReLU}(W_1 x).$$

Two linear layers with a non-linearity in between. The hidden width is typically $4 \cdot d_{\text{model}}$. The hidden width is where most of the parameter count of a transformer lives: an L-layer model has $L$ FFNs, each carrying about $8 \cdot d^2$ parameters (vs ~$4 \cdot d^2$ for attention). At big enough scale the FFNs are >2/3 of the total weight count.

### Gated Linear Units

The first useful refinement: **gate the hidden representation.**

$$\text{GLU}(x) = (W_1 x) \odot \sigma(W_2 x).$$

Two parallel projections of $x$. One gets passed through (the "value" branch); the other goes through a sigmoid (the "gate" branch). The product is the gated output. Then a third projection $W_3$ projects back to $d_{\text{model}}$.

This was floating around in the literature for years; the breakthrough was Shazeer's 2020 paper "GLU Variants Improve Transformer" which showed that **replacing ReLU with sigmoid-gated activations gives a clean perplexity win at fixed FLOPs**. The gate lets the model learn to selectively pass certain features through; in a non-gated MLP the activation function decides this rigidly.

### SwiGLU

Shazeer's variant that won: replace the sigmoid gate with Swish (a.k.a. SiLU):

$$\text{SiLU}(x) = x \cdot \sigma(x).$$

So:

$$\text{SwiGLU}(x) = (W_v x) \odot \text{SiLU}(W_g x), \qquad \text{FFN}(x) = W_o \, \text{SwiGLU}(x).$$

Three matrices instead of two: a gate projection $W_g$, a value projection $W_v$, both of width $d_{\text{ffn}}$; and an output projection $W_o$ that goes back to $d_{\text{model}}$.

**Why SiLU over sigmoid:** the gate is non-saturating (Swish is roughly linear for large $x$). Sigmoid gates clamp at $\pm 1$, which kills gradient. Empirically SwiGLU gives the cleanest quality wins of any GLU variant tested.

### The 8/3 rule

A SwiGLU FFN has *three* matrices, where a vanilla FFN has *two*. To keep parameter count comparable to a vanilla 4× FFN, modern models use:

$$d_{\text{ffn}} = \frac{8}{3} \cdot d_{\text{model}} \approx 2.67 \cdot d_{\text{model}},$$

usually rounded up to a multiple of 64 or 128 for hardware alignment. (Llama 3's 70B uses $d_{\text{model}}=8192$, $d_{\text{ffn}}=28672 = 28672/8192 = 3.5$ — they round up slightly past 8/3.)

Param count check:

| FFN variant | Matrices | Params |
|---|---|---|
| Vanilla, $d_{\text{ffn}}=4d$ | $W_1: d \to 4d$, $W_2: 4d \to d$ | $8d^2$ |
| SwiGLU, $d_{\text{ffn}}=\frac{8}{3}d$ | $W_g, W_v: d \to \frac{8}{3}d$, $W_o: \frac{8}{3}d \to d$ | $3 \cdot \frac{8}{3} d^2 = 8d^2$ |

Same parameter count. SwiGLU gives a consistent quality bump per param across every scale anyone's reported. **Universal in 2026 frontier models.**

## 5. RoPE — Rotary Position Embedding

Module 04 used RoPE without explaining it, deferring to here. Now we do it properly. This is the longest section in the module because RoPE is the most subtle component of the block, and because frontier-model "long context" papers from 2024 onward live inside the RoPE design space.

### 5.1 The position problem

Self-attention has no notion of position. The expression

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

is *permutation-equivariant* in its sequence dimension. Shuffle the rows of $Q$, $K$, and $V$ by the same permutation — you get the rows of the output shuffled by the same permutation. Attention by itself cannot tell "the cat sat on the mat" from "mat the on sat cat the." Position information has to be added in somewhere.

There are essentially three families of solution:

1. **Absolute position embeddings.** Add a fixed or learned vector $p_t$ to the token embedding at position $t$. Used by the original Transformer (sinusoidal) and BERT (learned). Cheap. Doesn't extrapolate past the trained length — at position $t = L_{\text{train}} + 1$ the model has never seen $p_t$, so the embedding is essentially random.
2. **Relative position embeddings.** Modify the attention computation so the score depends on $i - j$, not on $i$ and $j$ individually. T5 and Transformer-XL did this with learned biases on the attention logits.
3. **Rotary position embedding (RoPE).** Rotate $Q$ and $K$ in a 2D-plane-by-2D-plane fashion such that the dot product naturally depends only on the relative offset. This is the modern winner — used by Llama, Qwen, DeepSeek, Mistral, Gemma, and essentially every model trained since 2022.

### 5.2 The RoPE construction

Split the per-head feature dimension into adjacent pairs:

$$x = [x_0, x_1, x_2, x_3, \ldots, x_{d-2}, x_{d-1}] = [(x_0, x_1), (x_2, x_3), \ldots, (x_{d-2}, x_{d-1})].$$

So a $d$-dimensional vector becomes $d/2$ 2D points. Pick a frequency $\theta_k$ for the $k$-th pair, defined as

$$\theta_k = b^{-2k/d}, \qquad k = 0, 1, \ldots, d/2 - 1,$$

where $b$ is the **RoPE base** (originally 10,000; modern long-context models use 500,000 or 1,000,000). The frequencies are geometrically spaced from $1$ (slowest, when $k=0$) down to $b^{-(d-2)/d}$ (fastest).

At position $m$ in the sequence, rotate the $k$-th pair by angle $m \cdot \theta_k$:

$$R_m^{(k)} = \begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix}.$$

That's it. Apply the rotation to every pair of every $Q$ row and every $K$ row, with the rotation angle determined by the row's position in the sequence.

### 5.3 Why this gives you relative position

The point of doing the rotation pair-by-pair is that the dot product of a rotated $q_m$ and a rotated $k_n$ depends only on $m - n$. Consider one 2D pair:

$$q_m^{(k)} = R_m^{(k)} q^{(k)}, \quad k_n^{(k)} = R_n^{(k)} k^{(k)}.$$

Then

$$\langle q_m^{(k)}, k_n^{(k)} \rangle = (R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m}^{(k)} k.$$

The last step uses $R_m^\top R_n = R_{n-m}$ (rotation matrices: $R_\alpha^\top = R_{-\alpha}$, and rotations compose). The inner product on each 2D plane is a function of relative position only. Sum over all pairs: the full attention score is also a function of relative position only.

This is exactly the inductive bias you want from a position encoding. The model doesn't have to learn "position 14 is similar to position 15"; geometry gives it for free.

### 5.4 The complex-number view

The same thing is cleaner in complex notation. View each pair $(x_{2k}, x_{2k+1})$ as the complex number $x_{2k} + i\, x_{2k+1}$. A 2D rotation by angle $m \theta_k$ is multiplication by $e^{i m \theta_k}$. RoPE becomes:

$$\text{RoPE}(x, m)_k = x_k \cdot e^{i m \theta_k}.$$

And the inner product becomes:

$$\sum_k \text{Re}\!\left[ \overline{(\text{RoPE}(q, m)_k)} \cdot \text{RoPE}(k, n)_k \right] = \sum_k \text{Re}\!\left[ \overline{q_k} \, k_k \, e^{i (n-m) \theta_k} \right].$$

This is why the implementation precomputes a `freqs_cis` tensor of complex exponentials — multiplying by $e^{i m \theta_k}$ is one elementwise complex multiply per position, much cleaner than building rotation matrices.

The code in [`rope.py`](rope.py) does exactly this:

```python
def precompute_freqs_cis(dim, max_seq_len, base=10_000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (d/2,)
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)                                    # (T, d/2)
    return torch.polar(torch.ones_like(freqs), freqs)                # complex64

def apply_rotary_emb(x, freqs_cis):
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_c * freqs_cis).flatten(-2).type_as(x)
```

### 5.5 The base frequency

The base $b$ controls how fast the slowest pair rotates. At pair $k = d/2 - 1$ (the fastest pair), the angle is $\theta_{d/2-1} = b^{-(d-2)/d}$ — nearly 1 radian per position. At pair $k = 0$ (the slowest), the angle is $1$ per position — wait, that's the fastest. Let me restate. The frequency $\theta_k = b^{-2k/d}$ *decreases* with $k$: pair 0 rotates fastest ($\theta_0 = 1$ radian/position), pair $d/2-1$ rotates slowest ($\theta = b^{-(d-2)/d}$).

The slowest pair completes a full $2\pi$ rotation every $\frac{2\pi}{\theta_{d/2-1}} \approx 2\pi \cdot b^{(d-2)/d}$ positions. With $b = 10{,}000$ and $d = 128$, that's about $2\pi \cdot 10000^{126/128} \approx 60{,}000$ positions for one full cycle.

This matters because **once the slowest pair has rotated by more than $2\pi$, positions start aliasing**. Position 70k and position 10k look identical to that pair. The faster pairs alias *much* sooner; the slowest pair is the one that has to carry "I am at absolute position $t$" information all the way across the context.

Pre-2023 models used $b = 10{,}000$ and trained on 2k–4k contexts, so aliasing didn't bite. Long-context models (LongChat, Llama 2 32k, Yi-200k) ran into it immediately. The community fix was twofold:

1. **Increase $b$.** Llama 3 uses $b = 500{,}000$. DeepSeek V3 uses $b = 10{,}000$ for the main pretraining (8k context) and switches to larger bases via frequency interpolation for long-context fine-tuning. Qwen 3 long-context variants push to $b = 1{,}000{,}000$.
2. **Apply frequency interpolation** at long-context fine-tuning time. This is the next subsection.

A higher base $b$ at pretraining time costs nothing — it's still one number — but it widens the range of positions the model can reliably distinguish.

### 5.6 Context extension: PI, NTK, YaRN

Suppose you trained on 8k context and want to deploy at 32k. Naïvely, the model will be terrible: positions 8k–32k were never seen at training, and the slow pairs are now rotating into angles the model has never had to handle. Three techniques, in order of sophistication:

**Position Interpolation (PI)** [Chen et al., 2023]. Scale positions down by $s = L_{\text{target}} / L_{\text{train}}$. At inference, instead of rotating by $m \theta_k$, rotate by $m \theta_k / s$. Every position now lives in the angle range the model trained on. **Cost:** the model has lost angular resolution by a factor $s$ — two adjacent positions look closer together. Usually requires 1k–5k steps of fine-tuning at the new length to recover quality. Cheap and effective up to ~4× extension.

**NTK-aware scaling** [bloc97, 2023, blog post that became canonical]. Observation: PI loses *high-frequency* resolution (the fast pairs that distinguish adjacent positions). What you actually want is to slow down only the *low-frequency* pairs that were going to alias, not the high-frequency ones that were doing fine. The fix is to scale the *base* $b$ instead of the positions:

$$b' = b \cdot s^{d / (d-2)}.$$

This stretches the low-frequency end (long-period pairs now fit more positions in one cycle) without touching the high-frequency end (short-period pairs are still rotating at near the same rate). Better quality than PI at zero fine-tuning steps, but the math is hand-wavy: there's no derivation, just an empirical fit.

**YaRN** [Peng et al., 2023, "YaRN: Efficient Context Window Extension"]. Combines NTK's spirit with a careful frequency-dependent interpolation schedule. The key ideas:

1. Define a per-pair scaling factor $f_k$ that depends on the pair's *wavelength* $\lambda_k = 2\pi / \theta_k$ relative to the training context length.
2. Pairs with $\lambda_k$ much shorter than $L_{\text{train}}$ (the fast pairs) are left untouched: $f_k = 1$.
3. Pairs with $\lambda_k$ much longer than $L_{\text{train}}$ (the slow pairs that were never seeing a full cycle anyway) are scaled by $1/s$, fully interpolated.
4. Pairs in between are smoothly ramped between the two regimes — typically a linear ramp on $\log \lambda_k$ with `low` and `high` cutoff parameters.
5. A small *temperature* correction to the attention scores compensates for the dynamics shift. YaRN multiplies the attention logits by a factor like $1 + 0.1 \log s$ to keep softmax sharpness consistent.

YaRN is the technique most production long-context models use as of 2026. Llama 3 long-context fine-tunes, Qwen 3 long-context variants, and DeepSeek V3's long-context phase all use YaRN or close variants. Typical configuration: extend from 8k to 128k with a few thousand steps of YaRN-scaled fine-tuning on long documents.

We include `precompute_freqs_cis_yarn` in [`rope.py`](rope.py) so you can see the ramp explicitly. For ordinary pretraining at the trained context length, you don't need it.

### 5.7 The decoupled RoPE problem (revisited from Module 04)

Now that you know what RoPE is, Module 04's "decoupled RoPE" trick makes mechanical sense.

In MLA, K is decompressed from a latent at attention time: $K = c_{KV} W_K^\text{up}$. RoPE rotates per-position: $K_\text{rotated}^{(m)} = \text{Rotate}_m(c_{KV} W_K^\text{up})$.

You'd like to cache $c_{KV}$ (small) and apply RoPE to it. But rotation doesn't commute with the linear projection:

$$\text{Rotate}_m(c_{KV} W_K^\text{up}) \neq \text{Rotate}_m(c_{KV}) \cdot W_K^\text{up}.$$

The rotation operates pair-wise on the *final* feature dimension; the projection mixes features across that dimension. If you rotated first and then projected, you'd be applying linear combinations of rotations of different pair indices, which is meaningless.

DeepSeek's fix: don't try. Route the position info through a *separate* small path (the shared RoPE key $k_r$) that's computed directly from $x_t$ and rotated. The "content" K is computed from the latent and never rotated. Concatenate at attention time.

Now you cache: latent $c_{KV}$ (no RoPE applied) + shared $k_r$ (RoPE applied once, shared across heads). Both small; both correct.

This is why Module 04's MLA section had two distinct K paths and why the per-token cache was $d_{\text{kv\_latent}} + d_{\text{rope}}$.

## 6. Putting it all together — the canonical block

The block, in pseudocode:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ffn):
        self.norm_attn = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm_attn(x), freqs_cis)   # attention sub-layer
        x = x + self.ffn(self.norm_ffn(x))                # FFN sub-layer
        return x
```

That's a modern transformer block. Two residual additions per block, two RMSNorms per block, one attention call, one SwiGLU call. The `freqs_cis` tensor is precomputed once for the full model (not per block) and threaded through.

A full model is `L` of these blocks stacked, with one final RMSNorm before the unembedding linear:

```python
class TransformerLM(nn.Module):
    def __init__(self, n_layers, ...):
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.unemb = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        x = self.tok_emb(ids)
        freqs_cis = self.freqs_cis[:ids.size(1)]
        for block in self.blocks:
            x = block(x, freqs_cis)
        x = self.final_norm(x)
        return self.unemb(x)
```

This is the skeleton Module 07 will fill in with weight tying, muP init, and shape tests. The block itself is done in this module.

## 7. What frontier models use

| Component | Llama 3.x | DeepSeek V3 | Qwen 3.x | Notes |
|---|---|---|---|---|
| Norm | RMSNorm | RMSNorm | RMSNorm | Universal |
| Norm placement | Pre-norm | Pre-norm | Pre-norm | Universal |
| FFN | SwiGLU | SwiGLU | SwiGLU | Universal |
| $d_{\text{ffn}}$ multiplier | ~3.5× | ~2.67× (8/3) | ~2.67× (8/3) | Llama rounds up |
| Position encoding | RoPE base 500k | RoPE base 10k → YaRN | RoPE base 1M (long-ctx) | All RoPE; differ on base + extension |
| Attention | GQA (8 KV groups) | MLA + decoupled RoPE | GQA in dense layers; linear attention in hybrid layers | Module 04 |
| Biases | None | None | None | Universal |
| Final norm | RMSNorm | RMSNorm | RMSNorm | Universal |

**Read:** the *block* has converged across frontier labs. The remaining axis of variation in 2026 is the attention design (Module 04) and the FFN structure when MoE is involved (Module 06). Everything else is settled.

## 8. What we implement

| Component | Status |
|---|---|
| RMSNorm | Full implementation ([`block.py`](block.py)) |
| SwiGLU FFN | Full implementation, with the 8/3 rule |
| RoPE | Full implementation with PI / NTK / YaRN scaling ([`rope.py`](rope.py)) |
| Pre-norm transformer block | Full implementation, pluggable attention from Module 04 |
| Post-norm, DeepNorm, sandwich norm | Described; not implemented |

The implementation in [`block.py`](block.py) accepts any of Module 04's attention classes (MHA, GQA, MLA), so you can swap attention designs without changing the block. Module 11's pretraining will use this exact block with MLA.

## 9. Reading list

- **RoPE original**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). The paper that introduced RoPE.
- **YaRN**: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023). Cleanest treatment of frequency interpolation; reads in ~30 min.
- **GLU Variants**: Shazeer, "GLU Variants Improve Transformer" (2020). One-page paper showing SwiGLU > ReLU; reads in 5 min.
- **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019). The original.
- **Interpretability framing of the residual stream**: Anthropic, "A Mathematical Framework for Transformer Circuits" (2021). The first half of this paper establishes the residual-stream view we used in Section 1.

## Next

[Module 06 — Mixture of Experts](../06-mixture-of-experts/). Where the FFN gets sparser, the parameter count gets larger, and the load-balancing problem becomes a first-class design constraint.
