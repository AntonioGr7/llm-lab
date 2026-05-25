# Module 07 — Assembling the Full Model

> Part of [Part 2 — Architecture](../). Reading time: ~60 minutes. Compute cost: ~$0 (CPU; we forward a tiny model).

## The thesis

Everything in Part 2 has been a component: attention (Module 04), the transformer block (Module 05), MoE (Module 06). This module wires them into a single `nn.Module` you can run forward on a batch of token IDs and get logits out. The result is a real autoregressive language model — small, untrained, but structurally identical to the one Module 11 will pretrain at the course's compute budget.

Three things are new here:

1. **The bookkeeping at the edges** — token embeddings, the precomputed `freqs_cis` table, the final RMSNorm, the unembedding/LM head. These are easy to forget and easy to get wrong.
2. **Weight tying** — sharing the embedding matrix between input and output. Worth knowing because it saves parameters and is universal at small scale.
3. **Initialization** — the standard recipe, plus **muP** (Maximal Update Parameterization), the modern technique that lets you tune hyperparameters on a tiny model and transfer them to a much larger one without retuning.

By the end you'll be able to instantiate a model, count its parameters, sanity-check a forward pass, and read a state dict from any modern open-weights checkpoint without confusion about what's what.

## What you'll be able to do at the end

- Diagram a modern transformer LM end-to-end — every tensor in the graph, every parameter group.
- Implement `TransformerLM` from token IDs to logits, with weight tying and a clean forward.
- Explain what muP fixes and apply it as a 50-line init function.
- Compute the parameter count of a given configuration from first principles (and verify it matches your model).
- Run shape tests that would catch the most common assembly bugs before any training run.

## 1. Model anatomy

A modern decoder-only LM has exactly six kinds of components:

```
input_ids                       (B, T)
    │
    ▼
Token Embedding  ──→ x          (B, T, d_model)
    │
    │     ┌──────────────────────────────┐
    │     │  TransformerBlock × n_layers │   ←── identical blocks; pre-norm
    │     │   norm → attention → +res    │       attention from Module 04
    │     │   norm → ffn       → +res    │       block + ffn from Module 05
    │     └──────────────────────────────┘
    ▼
Final RMSNorm  ──→ x            (B, T, d_model)
    │
    ▼
Unembedding  ──→ logits         (B, T, vocab_size)
    │
    ▼
(Loss with cross-entropy against shifted input_ids)
```

That's the whole graph. Six components:

1. **Token embedding** — `nn.Embedding(vocab_size, d_model)`. Maps integers to vectors.
2. **Position info** — `freqs_cis` precomputed once per model (not per block). Applied inside attention via RoPE.
3. **Transformer blocks** — `nn.ModuleList` of identical blocks. Width and depth are the model's main capacity dials.
4. **Final RMSNorm** — required after pre-norm blocks (Section 2 of Module 05). Without it, logit magnitudes drift.
5. **Unembedding / LM head** — `nn.Linear(d_model, vocab_size, bias=False)`. Projects to vocabulary logits.
6. **Loss** — cross-entropy between logits and shifted targets. Lives outside the model, but the model's forward needs to return logits in the right shape for it.

Everything from Modules 04–06 plugs into component 3. Components 1, 2, 4, 5 are what this module owns.

## 2. Token embeddings

```python
self.tok_emb = nn.Embedding(vocab_size, d_model)
```

Each row of the embedding matrix is the model's representation of one vocabulary token. The matrix is large — at a 128k vocabulary and $d_{\text{model}}$=4096, that's ~525M parameters. For a small model, the embedding can be the majority of the parameter count.

**Initialization.** The modern recipe is **`std=0.02` truncated normal** for the embedding — *the same scale as the hidden linears*. GPT-2, Llama 1/2/3, Qwen 3 all use this. The shared scale is what keeps the model's logits-at-init close to $\ln(V)$ (Section 8), and it pairs cleanly with weight tying (Section 3) because the embedding *is* the LM head when tied.

What you should **not** do:

- Initialize embeddings at std=1.0. The original Transformer paper scaled them by $\sqrt{d}$ at the input to compensate, which is fiddly and modern recipes have dropped it. Untreated, std=1 embeddings give logits magnitudes of order $\sqrt{d}$, blowing past the well-behaved logits-at-init range.
- Scale the embedding by $1/\sqrt{d}$ at init. Some tutorials still do this; Llama, Qwen, DeepSeek do not.

**Position embeddings — don't add them here.** We use RoPE (Module 05), which lives inside attention. Token embeddings carry no position information; positions are injected when Q and K are rotated inside the attention call. This is the cleanest factoring: the embedding is a pure lookup; the position-mixing is in attention.

## 3. Weight tying

Modern LMs almost always **tie the unembedding to the embedding**:

```python
self.tok_emb = nn.Embedding(vocab_size, d_model)
self.unembed = nn.Linear(d_model, vocab_size, bias=False)
self.unembed.weight = self.tok_emb.weight   # the actual sharing
```

After this assignment, the same parameter tensor is used to embed input tokens *and* project hidden states to logits. Two consequences:

1. **Parameter count drops by $d_{\text{model}} \times \text{vocab\_size}$.** For our 128k-vocab × 4096-d_model example, that's ~525M params saved. For small models (say, 1B params total) this is meaningful; for frontier models (70B+) it's a rounding error.
2. **A geometric tie is enforced.** The vector that "means" a token and the vector that "predicts" a token are the same. This is interpretable (the embedding space *is* the prediction space) and empirically helps at small scale; at very large scale the benefit shrinks.

**Who ties:** Llama 1/2 small variants, GPT-2, Qwen 3 small (≤3B), most ≤7B models.  
**Who doesn't:** Llama 1/2/3 at 70B+, GPT-3+, DeepSeek-V3 (separate unembedding is reported to help at frontier scale).

**Decision rule for the course.** Default to tied for our pretraining demo (small model, embedding is most of the params). Module 11's pretraining will use weight tying. For frontier-scale work, untie.

## 4. Initialization — the standard recipe and muP

A modern transformer is initialized with three kinds of weights, each with its own rule:

1. **Embeddings** (input projections from token IDs): `std=0.02` normal init (same as hidden, in the standard recipe).
2. **Hidden weights** (all the linear layers inside blocks): `std=0.02` truncated normal, or `std=1/√fan_in` (He-like).
3. **Output weights** (the unembedding / LM head): for the standard recipe, `std=0.02` like other hiddens. For muP, the output weights are scaled DOWN by a factor proportional to the width multiplier.

This is the **standard recipe**. It's what GPT-2 used, what most open implementations default to, and what works fine if you tune your learning rate at every model size you care about.

### What muP fixes

The problem: if you tune the optimal learning rate $\eta^*$ at $d_{\text{model}}=512$ and then train at $d_{\text{model}}=4096$, your old $\eta^*$ is wrong. The optimal LR shifts with width, and for the standard parameterization the shift is *width-dependent and unstable*. Frontier labs end up burning compute on small-scale ablations just to predict where the optimum sits at scale. That's a lot of wasted GPU time.

**muP (Yang & Hu, 2021, "Tensor Programs V")** is a re-parameterization of init scales and per-group learning rates such that **the optimal learning rate becomes width-invariant**. Tune at a small "proxy" width, transfer to any larger width, no retuning. This is the property called "$\mu$-transfer" (mu-transfer).

The mechanics, simplified:

Let $m = d_{\text{model}} / d_{\text{base}}$ be the *width multiplier* relative to a base width $d_{\text{base}}$ where you did your tuning sweep.

| Parameter group | Init std (vs base) | LR multiplier |
|---|---|---|
| Input (embedding, scalar inputs) | unchanged | $1$ (unchanged) |
| Hidden (linears inside blocks) | $1/\sqrt{m}$ × base init | $1/m$ |
| Output (LM head, when untied) | $1/m$ × base init | $1$ (unchanged) |

The intuition: as you widen the network, hidden activations don't blow up (the $1/\sqrt{m}$ init keeps them at $O(1)$), gradients on hidden weights are scaled to match the slower-learning regime that wide networks naturally want, and the output weights are kept small so logit magnitudes don't drift. The combination is what makes $\eta^*$ stop moving with width.

**For our course**: we apply muP init in [`init.py`](init.py) as the default. The corresponding LR groups are set up in Module 08 (the training loop). For students who just want to run the demo, the defaults are pre-configured. For students who want to scale up: tune $\eta^*$ at $d_{\text{base}}$=512, then transfer to whatever width you actually train at.

**The caveat.** muP isn't free magic. The transfer property requires:

- Standard SwiGLU / RMSNorm / pre-norm transformer (the kind we built).
- Adam-family optimizer (AdamW, Lion). It mostly holds.
- Reasonable batch-size scaling (proportional to width, roughly).
- No exotic regularization (heavy dropout, label smoothing) — these can break the scaling.

Frontier-scale models (DeepSeek-V3, Llama 3) report using muP-style scaling for the hyperparameter sweep, then small final adjustments at the target scale. The mu-transfer property holds for the optimal $\eta^*$ within a factor of ~1.3× across two orders of magnitude in width — good enough that nobody's tuning from scratch anymore.

## 5. The forward pass, end to end

```python
def forward(self, input_ids):
    # input_ids: (B, T) int64 token IDs
    x = self.tok_emb(input_ids)                           # (B, T, d_model)
    freqs_cis = self.freqs_cis[:input_ids.size(1)]        # slice to current seq len

    for block in self.blocks:                             # n_layers identical blocks
        x = block(x, freqs_cis)                           # (B, T, d_model)

    x = self.final_norm(x)                                # (B, T, d_model)
    logits = self.unembed(x)                              # (B, T, vocab_size)
    return logits
```

That's the whole forward. Three subtleties worth flagging:

1. **`freqs_cis` is sliced at the start.** The model precomputes a `freqs_cis` table at `max_seq_len`; the forward slices it to the current batch's actual length. This avoids recomputing rotations every forward.
2. **No causal mask is applied here.** It's applied inside attention by `is_causal=True` in the SDPA call (Module 04). The model doesn't need to know about it.
3. **The model returns logits, not loss.** Loss computation lives in the training loop (Module 08). This factoring lets the same model serve training, evaluation, and inference with one forward.

For MoE training, the forward also needs to collect router outputs from each MoE block so the training loop can apply the bias update. Our `model.py` exposes an optional `return_router_info=True` flag for this case.

## 6. Generation

A forward pass gives you logits. *Sampling* the next token from those logits — and looping — is generation. The naive form:

```python
def generate(self, input_ids, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        logits = self.forward(input_ids)[:, -1, :]        # last position only
        probs = (logits / temperature).softmax(-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids
```

This is the *correct* but *wasteful* form. Each iteration runs the full forward over the entire growing prefix; per-token compute is $O(n^2)$ where $n$ is the cumulative length. For research demos at small $n$ this is fine. For real inference you'd use KV caching: cache the K and V tensors from each block at every layer, append the new token's K/V at each step, and only recompute attention against the cache. KV-cached generation is what Module 04's MLA design is built for.

Our [`model.py`](model.py) ships the naive generate for clarity. Real inference would use the `transformers` library's KV-cached path or a server like vLLM. We don't need our own production-grade decoder.

## 7. Parameter count from first principles

For a configuration $(L, d, H, V)$ — $L$ layers, $d$ model width, $H$ heads, $V$ vocab — and ignoring norms (they're a rounding error), the parameter count is approximately:

$$P \approx \underbrace{2 \cdot V \cdot d}_{\text{embed + unembed}} + L \cdot \underbrace{\left(4 d^2 + 3 d \cdot d_{\text{ffn}}\right)}_{\text{block: attn + ffn}}$$

If you weight-tie the embedding and unembedding, the $2 V d$ becomes $V d$.

For SwiGLU FFN with $d_{\text{ffn}} = 8d/3$, the per-block params are:

- Attention (MHA): $4 d^2$ (Q, K, V, O projections each $d \times d$)
- FFN (SwiGLU): $3 \cdot d \cdot (8d/3) = 8 d^2$
- **Total per block: $12 d^2$**

So:

$$P \approx V d + 12 L d^2 \quad \text{(weight-tied, MHA)}.$$

Check this against frontier models:

| Model | $L$ | $d$ | $V$ | Reported params | Formula prediction |
|---|---|---|---|---|---|
| Llama 3 8B | 32 | 4096 | 128256 | 8.0B | $128k \cdot 4k + 12 \cdot 32 \cdot 16M = 6.7B$* |
| Llama 3 70B | 80 | 8192 | 128256 | 70.0B | $128k \cdot 8k + 12 \cdot 80 \cdot 67M = 65.4B$* |

*The formula undershoots a bit because Llama 3 uses $d_{\text{ffn}} \approx 3.5d$ (not $8d/3$) and untied embeddings, so the real numbers are about 5-10% higher than the simple formula predicts. Close enough for an envelope check; exact counts need the actual config.

[`model.py`](model.py)'s `count_params()` method returns the exact number, broken out by group (embedding, attention, FFN, norms). Useful for ablations.

## 8. Shape tests — the cheapest insurance you'll ever buy

Before any training run, do these four checks on the assembled model. Each takes a second; together they catch ~80% of "the loss is NaN immediately" bugs.

1. **Forward shape.** Build the model, forward a random `(B=2, T=16)` int tensor, assert the output is `(2, 16, vocab_size)`. Anything else means a transpose or reshape went wrong somewhere.
2. **Parameter count.** Compare `sum(p.numel() for p in model.parameters())` against your envelope prediction (Section 7). Discrepancies bigger than 5% mean a layer was duplicated or skipped.
3. **Loss at init.** Build the model, forward random data, compute cross-entropy against random targets. At init the model is uniform random; the expected loss is $\ln(\text{vocab\_size})$. For 128k vocab that's ~11.76. If your loss at init isn't in $[\ln V - 1, \ln V + 1]$, something is very wrong (probably a misaligned weight tying or unembedding scale).
4. **Backward and step.** Forward, compute loss, call `loss.backward()`, then check that `param.grad is not None` for every parameter. If any grad is None, that parameter isn't in the graph. Then take one optimizer step and verify the loss decreases on a second forward of the same input. If it doesn't, your LR is wrong or there's a parameter group that isn't being updated.

These checks live in [`model.py`](model.py)'s `if __name__ == "__main__"` and in the notebook.

## 9. What we implement

| Component | Status |
|---|---|
| `TransformerLM` end-to-end | Full implementation ([`model.py`](model.py)) |
| Token embedding | Built-in `nn.Embedding` |
| Weight tying | On by default; togglable |
| Final RMSNorm + unembedding | Standard pre-norm assembly |
| muP init | Full implementation ([`init.py`](init.py)); standard init also available |
| MoE assembly | Supported via flag — swap dense block for `MoETransformerBlock` |
| KV-cached generation | Not implemented; naive generation only |
| Parameter count breakdown | `model.count_params()` |
| Shape/sanity tests | Notebook + smoke test |

The model is configured to **default to the dense DeepSeek-V3-shaped architecture** that Module 11 will pretrain: MLA attention, RMSNorm, SwiGLU, RoPE base 10,000. Switching to GQA, MHA, or an MoE FFN is a flag.

## 10. Reading list

- **muP**: Yang & Hu, "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2021). The paper. The math is heavy; the [maximal-update-parameterization-explained](https://www.eleuther.ai/papers-blog/muP) summary by EleutherAI is the readable version.
- **Weight tying**: Press & Wolf, "Using the Output Embedding to Improve Language Models" (2017). The original argument and ablation.
- **Modern recipe survey**: any frontier model's tech report, particularly DeepSeek-V3's Section 2 ("Architecture"). The component-by-component decisions are clearly stated.
- **Init at scale**: Le Scao et al., "What Language Model to Train if You Have One Million GPU Hours?" (2022). Empirical comparisons of init schemes at scale; the paper that convinced many labs to standardize on small-std truncated-normal.

## Next

[Part 3 — Pretraining](../../part-3-pretraining/). The model is built. Now we train it. Module 08 starts with the training loop (BF16/FP8, AdamW, gradient clipping, the boring infrastructure that all real training depends on).
