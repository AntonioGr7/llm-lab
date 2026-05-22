# Part 2 — Architecture

Build every component from scratch, then assemble.

The transformer is not a monolith. It's a stack of well-understood pieces — attention, normalization, an MLP, position encoding — and the modern variants (Llama, DeepSeek, Qwen) differ in small, specific choices at each layer. Once you can name those choices and explain what each one buys, you can read any new architecture paper in 30 minutes.

This Part is the longest in the course and the most code-heavy. By the end, you'll have a working model that's structurally indistinguishable from a frontier-lab base model — just smaller.

## Modules

- **[04 — Attention](04-attention/)** — Self-attention from scratch, then the variants. Multi-head, Flash Attention, GQA. MLA (DeepSeek-V2's contribution) covered conceptually with a code reference.
- **[05 — Transformer Block](05-transformer-block/)** — RMSNorm vs LayerNorm. SwiGLU. RoPE. Pre-norm vs post-norm. The residual stream as the central organizing concept.
- **[06 — Mixture of Experts](06-mixture-of-experts/)** — Conditional computation. Top-k routing. DeepSeek-V3's auxiliary-loss-free load balancing. When MoE is worth it and when it isn't.
- **[07 — Assembling the Full Model](07-full-model/)** — Putting components together. Weight tying. muP for hyperparameter transfer. Initialization. Shape tests and forward-pass sanity checks before any training.

## What you'll be able to do at the end of this Part

- Write a small but real transformer in PyTorch, layer by layer, without copying.
- Read a new architecture paper and identify what's novel vs standard.
- Decide between GQA and MHA, dense and MoE, for a given budget and use case.

## Time and cost

- Reading + coding: ~10 hours.
- Compute cost: ~$0–5. Everything here is forward-passes and shape checks on tiny tensors. We don't train anything yet.
