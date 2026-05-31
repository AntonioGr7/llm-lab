"""Config tests for Module 16 — defaults, YAML load, and the LoRA-specific
tuple coercion in CLI overrides.

    cd 16-parameter-efficient-finetuning/
    python tests/test_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from config import TrainConfig, load_yaml, apply_dotted_overrides  # noqa: E402


def test_defaults():
    c = TrainConfig()
    c.sync()
    assert c.lora.r == 8 and c.lora.alpha == 16
    assert c.optimizer.lr == 2e-4, "LoRA default LR should be 20x full-FT's 1e-5"
    assert c.training.activation_checkpointing is False, "AC off by default for LoRA"
    assert len(c.lora.target_modules) == 7
    print("  defaults: r=8 alpha=16 lr=2e-4 AC=off 7 targets  ✓")


def test_yaml_load():
    cfgdir = _HERE.parent / "configs"
    for name, qlora in [("lora_qwen3_1.7b.yaml", False),
                        ("qlora_qwen3_1.7b.yaml", True),
                        ("lora_demo.yaml", False)]:
        c = load_yaml(str(cfgdir / name))
        c.sync()
        assert c.lora.qlora is qlora, name
        assert c.schedule.total_steps == c.training.total_steps  # sync propagated
    print("  yaml: all 3 configs load + sync  ✓")


def test_override_coercion():
    c = TrainConfig()
    apply_dotted_overrides(c, [
        "lora.r=16",
        "lora.alpha=32",
        "lora.qlora=true",
        "lora.target_modules=q_proj,v_proj",     # tuple-of-str coercion
        "optimizer.lr=1e-4",
    ])
    assert c.lora.r == 16 and c.lora.alpha == 32
    assert c.lora.qlora is True
    assert c.lora.target_modules == ("q_proj", "v_proj"), c.lora.target_modules
    assert c.optimizer.lr == 1e-4
    print("  overrides: int/bool/float/tuple-of-str all coerce  ✓")


if __name__ == "__main__":
    print("Running config tests...")
    test_defaults()
    test_yaml_load()
    test_override_coercion()
    print("\nAll config tests passed.")
