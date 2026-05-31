"""Model builder for Module 16 — LoRA / QLoRA.

Two differences from Module 15's `build_model`:

1. **The frozen base loads in BF16, not FP32.** In full FT every weight is
   trainable, so you need an FP32 *master copy* for stable optimizer updates
   (Module 15 §4). In LoRA the base is frozen — the optimizer never touches
   it — so a master copy buys nothing. Loading it in BF16 halves the base
   footprint (3.4 GB vs 6.8 GB at 1.7B). The trainable adapters, created by
   `LoRALinear` in FP32, get the master-weight treatment instead. This is the
   whole memory story: optimizer state lives only on the adapters.

2. **Adapters are injected after load.** `inject_lora_adapters` swaps the
   targeted projections for `LoRALinear` and freezes everything else. From
   the loop's point of view it's still "a model whose `forward(input_ids,
   attention_mask)` returns `.logits`" — `loop.py` is unchanged from Module 15.

QLoRA (`lora.qlora=true`) loads the base in 4-bit NF4 via bitsandbytes. The
4-bit weights dequantize to `bnb_4bit_compute_dtype` on the fly for each
matmul; the BF16 adapters sit on top. This fits a 1.7B fine-tune in ~6 GB.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig, LoRAConfig
from lora import inject_lora_adapters, LoRASpec, trainable_summary


_BNB_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_model(
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig,
    fused_ce: bool = False,
) -> nn.Module:
    """Load a pretrained HF causal LM and attach LoRA adapters.

    Args:
        model_cfg: which base to load (`name`).
        lora_cfg: rank/alpha/dropout/targets and the QLoRA 4-bit knobs.
        fused_ce: optional Liger fused linear+CE on the (frozen) LM head — a
            memory optimization carried over from Module 15. LoRA already saves
            most of the memory, so this is rarely needed here; kept for parity.

    Returns:
        The model with `LoRALinear` adapters injected and only the adapter A/B
        matrices trainable. Decoder layers live in `model.model.layers`
        (Qwen3 / Llama / Mistral / Gemma), so `fsdp_setup.apply_fsdp` works.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if fused_ce:
        try:
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
        except ImportError as e:
            raise ImportError(
                "training.use_fused_ce=true requires `pip install liger-kernel`."
            ) from e
        model_type = AutoConfig.from_pretrained(model_cfg.name).model_type
        _apply_liger_kernel(
            model_type=model_type, rope=False, rms_norm=False, swiglu=False,
            cross_entropy=False, fused_linear_cross_entropy=True,
        )

    load_kwargs: dict = {}
    if lora_cfg.qlora:
        # 4-bit NF4 base. Needs bitsandbytes + a CUDA GPU.
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as _  # noqa: F401  (import check only)
        except ImportError as e:
            raise ImportError(
                "lora.qlora=true requires `pip install bitsandbytes`. QLoRA also "
                "needs a CUDA GPU — the 4-bit kernels have no CPU path."
            ) from e
        compute_dtype = _BNB_DTYPE[lora_cfg.bnb_4bit_compute_dtype]
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=lora_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=lora_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        # device_map pins the 4-bit weights to the current GPU at load time.
        load_kwargs["device_map"] = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    else:
        # Standard LoRA: frozen base in BF16 (no FP32 master needed — it never
        # updates). Adapters carry the FP32 master, created inside LoRALinear.
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **load_kwargs)
    model.config.use_cache = False

    spec = LoRASpec(
        r=lora_cfg.r, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout,
        target_modules=tuple(lora_cfg.target_modules),
    )
    n_adapted = inject_lora_adapters(model, spec)
    model._lora_n_adapted = n_adapted     # stashed for the startup banner

    return model


def count_params(model: nn.Module) -> dict[str, int]:
    """Trainable/total breakdown — the LoRA banner number. Wraps lora.trainable_summary
    into the same dict shape Module 15's count_params returned (`total` key kept)."""
    s = trainable_summary(model)
    return {
        "trainable": int(s["trainable"]),
        "total": int(s["total"]),
        "trainable_pct": s["trainable_pct"],
        "reduction_x": s["reduction_x"],
    }


if __name__ == "__main__":
    # Offline smoke: build a tiny Qwen3 from scratch (no download), inject LoRA.
    from transformers import Qwen3Config, Qwen3ForCausalLM

    m = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=2048, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=128, tie_word_embeddings=True, use_cache=False,
    ))
    spec = LoRASpec(r=8, alpha=16)
    n = inject_lora_adapters(m, spec)
    c = count_params(m)
    print(f"adapted {n} projections")
    print(f"trainable {c['trainable']:,} / {c['total']:,} "
          f"({c['trainable_pct']:.2f}%, {c['reduction_x']:.0f}× fewer)")
    print("To load a real base: build_model(ModelConfig('Qwen/Qwen3-1.7B-Base'), LoRAConfig())")
