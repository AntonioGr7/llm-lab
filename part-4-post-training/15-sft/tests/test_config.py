"""Config-loader tests for Module 15.

Cheap guards that the YAML configs parse, cross-section invariants propagate,
and the dotted CLI overrides coerce types correctly. No network, no models.

Usage:
    cd 15-sft/
    python tests/test_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODDIR = _HERE.parent
sys.path.insert(0, str(_MODDIR))

from config import TrainConfig, load_yaml, apply_dotted_overrides


def test_configs_load() -> bool:
    ok = True
    for name in ("sft_qwen3_1.7b.yaml", "sft_demo.yaml"):
        cfg = load_yaml(str(_MODDIR / "configs" / name))
        # sync() (called inside load_yaml) must propagate total_steps.
        prop = cfg.schedule.total_steps == cfg.training.total_steps
        sane = cfg.optimizer.lr > 0 and cfg.data.seq_len <= cfg.model.max_seq
        betas_tuple = isinstance(cfg.optimizer.betas, tuple) and len(cfg.optimizer.betas) == 2
        good = prop and sane and betas_tuple
        ok = ok and good
        print(f"  [load] {name}: {'OK' if good else 'FAIL'}  "
              f"(steps_sync={prop}, lr={cfg.optimizer.lr}, betas_tuple={betas_tuple})")
    return ok


def test_seqlen_guard() -> bool:
    cfg = TrainConfig()
    cfg.model.max_seq = 1024
    cfg.data.seq_len = 2048
    raised = False
    try:
        cfg.sync()
    except ValueError:
        raised = True
    print(f"  [guard] seq_len > max_seq raises: {'OK' if raised else 'FAIL'}")
    return raised


def test_dotted_overrides() -> bool:
    cfg = TrainConfig()
    apply_dotted_overrides(cfg, [
        "training.total_steps=123",
        "optimizer.lr=2e-5",
        "training.activation_checkpointing=false",
        "model.name=Qwen/Qwen3-0.6B-Base",
    ])
    checks = [
        cfg.training.total_steps == 123 and isinstance(cfg.training.total_steps, int),
        abs(cfg.optimizer.lr - 2e-5) < 1e-12 and isinstance(cfg.optimizer.lr, float),
        cfg.training.activation_checkpointing is False,
        cfg.model.name == "Qwen/Qwen3-0.6B-Base",
    ]
    # Unknown field must raise.
    bad_raised = False
    try:
        apply_dotted_overrides(cfg, ["training.does_not_exist=1"])
    except ValueError:
        bad_raised = True
    checks.append(bad_raised)

    ok = all(checks)
    print(f"  [override] dotted overrides + coercion: {'OK' if ok else 'FAIL'}  "
          f"({sum(checks)}/{len(checks)})")
    return ok


def main() -> int:
    print("SFT config tests (offline)\n")
    results = [
        test_configs_load(),
        test_seqlen_guard(),
        test_dotted_overrides(),
    ]
    print()
    if all(results):
        print("  PASS — configs load, invariants hold, overrides coerce.")
        return 0
    print("  FAIL — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
