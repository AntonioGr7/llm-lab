"""Model builder for Part 3 pretraining — defaults to Qwen3 architecture.

The framework's contract: `build_model(model_cfg) -> nn.Module` whose forward
accepts `input_ids=..., labels=...` and returns an object with `.loss` and
`.logits`. This matches every Hugging Face causal-LM class.

**To swap in a different architecture**: replace the function body. Examples:

```python
# Llama (HF transformers)
from transformers import LlamaConfig, LlamaForCausalLM
def build_model(cfg):
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        intermediate_size=cfg.d_ffn,
        max_position_embeddings=cfg.max_seq,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_weights,
        rms_norm_eps=cfg.norm_eps,
    ))

# Our Part-2 TransformerLM with an adapter for the (.loss, .logits) contract.
# See module README, Section 3.
```
"""
from __future__ import annotations

import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from config import ModelConfig


def build_model(cfg: ModelConfig) -> nn.Module:
    """Build a randomly-initialized Qwen3-architecture model for pretraining.

    The returned model:
    - Has `.forward(input_ids, labels=...)` returning a `CausalLMOutput`
      with `.loss` and `.logits`.
    - Is FSDP2-compatible (its layers live in `model.model.layers`).
    - Is sized by the fields of `cfg`. Defaults give a ~150M-param model.
    """
    config = Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        intermediate_size=cfg.d_ffn,
        max_position_embeddings=cfg.max_seq,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_weights,
        rms_norm_eps=cfg.norm_eps,
        # Disable HF's attention-mask quirks we don't need; we always do causal LM.
        use_cache=False,
    )
    return Qwen3ForCausalLM(config)


def count_params(model: nn.Module) -> dict[str, int]:
    """Param-count breakdown by group. Useful for sanity checks at startup."""
    counts = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "head": 0, "total": 0}
    seen_ids: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen_ids:           # weight tying — count once
            continue
        seen_ids.add(id(p))
        n = p.numel()
        counts["total"] += n
        # Heuristic name-matching for the breakdown.
        if "embed_tokens" in name:
            counts["embedding"] += n
        elif "lm_head" in name:
            counts["head"] += n
        elif "self_attn" in name:
            counts["attention"] += n
        elif "mlp" in name:
            counts["mlp"] += n
        elif "norm" in name:
            counts["norm"] += n
    return counts


if __name__ == "__main__":
    # Smoke test: build a tiny model, print parameter counts.
    cfg = ModelConfig(
        vocab_size=2048, d_model=128, n_layers=4,
        n_heads=4, n_kv_heads=2, d_ffn=256,
        max_seq=128, tie_weights=True,
    )
    model = build_model(cfg)
    print(f"Built Qwen3 with d={cfg.d_model}, L={cfg.n_layers}")
    counts = count_params(model)
    total = counts["total"]
    for k, v in counts.items():
        print(f"  {k:10s} {v/1e6:>7.3f}M  ({v/total*100:>5.1f}%)")
