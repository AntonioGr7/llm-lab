"""The composed pretraining config.

Extends Module 08's TrainConfig with two additions:
  - `ScheduleConfig` from Module 09 — wired in so `train.py` builds a
    scheduler at startup.
  - `activation_checkpointing` flag from Module 10 — set to True for any
    realistic run on memory-constrained hardware.

Also adds FineWeb-Edu-specific knobs to `DataConfig` (tokenizer_name,
fineweb_subset, shuffle_buffer, pin_memory).

This file is the single source of truth for every hyperparameter the demo
run takes. `configs/demo.yaml` is the materialized form for the demo;
CLI overrides on top of that go through dotted paths (see `train.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    """Architecture knobs. Defaults to ~150M Qwen3 (the demo size)."""
    vocab_size: int = 151_936           # Qwen3 tokenizer vocab
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 2                 # GQA(6x) — Qwen3-0.6B's ratio
    d_ffn: int = 2048
    max_seq: int = 2048
    rope_theta: float = 10_000.0
    tie_weights: bool = True
    norm_eps: float = 1e-6


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw", "muon"]


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters. See Module 08 § 4 for the choice rationale."""
    type: OptimizerType = "adamw"
    # AdamW
    lr: float = 3e-4                    # peak LR (post-warmup, pre-decay)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    # Muon (used only when type='muon')
    muon_lr: float = 2e-2
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5


# =============================================================================
# LR schedule
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """LR schedule hyperparameters. See Module 09 for shape choices."""
    type: ScheduleType = "cosine"
    # The scheduler reads `total_steps` from this config (mirrored from training).
    # train.py keeps it in sync at startup.
    total_steps: int = 3_000
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    decay_steps: int = 300              # WSD-only — last 10% by default


# =============================================================================
# Data
# =============================================================================

DataSource = Literal["synthetic", "fineweb_edu"]


@dataclass
class DataConfig:
    """Where training data comes from."""
    source: DataSource = "fineweb_edu"
    seq_len: int = 2048
    batch_size_per_device: int = 8      # micro-batch per rank per accum step
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 0
    # Synthetic-only knob
    synthetic_samples: int = 10_000
    # FineWeb-Edu knobs
    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    fineweb_subset: str = "sample-10BT"  # 10B-token public subset
    shuffle_buffer: int = 10_000


# =============================================================================
# Training (the loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32", "fp8"]


@dataclass
class TrainingConfig:
    total_steps: int = 3_000
    grad_accum: int = 4
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    activation_checkpointing: bool = True   # Module 10 — on by default for the demo
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 0                     # 0 = disabled; set to e.g. 500 to run eval mid-training
    checkpoint_dir: str = "./results/checkpoints"
    resume_from: Optional[str] = None
    # Checkpoint rotation (Section "Managing disk" in the README).
    # `keep_last_n=3` keeps only the 3 most recent rolling checkpoints; older
    # ones are deleted after each save. `milestone_every=0` disables the
    # "permanent" milestone pattern; set to e.g. 1000 to keep step_1000,
    # step_2000, ... forever even when rolling deletes everything else.
    keep_last_n_checkpoints: int = 3
    milestone_every: int = 0
    # W&B logging (rank-0 only). See the README section "Logging to W&B" for
    # the auth + setup story. Empty `wandb_project` disables W&B entirely.
    wandb_project: str = ""
    wandb_entity: str = ""                  # your username / team slug; "" = your default entity
    wandb_run_name: str = ""                # "" = let W&B assign a random name
    wandb_run_id: str = ""                  # set to a previous run's id to resume that run on restart
    wandb_tags: tuple[str, ...] = ()        # free-form tags, e.g. ("demo", "fineweb-edu", "150M")


# =============================================================================
# The full config
# =============================================================================

@dataclass
class TrainConfig:
    """The complete config a `train.py` run consumes."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def sync(self) -> None:
        """Propagate cross-section invariants. Called by `train.py` after load.

        - `schedule.total_steps` must equal `training.total_steps` so the
          cosine decays over the right horizon.
        """
        self.schedule.total_steps = self.training.total_steps

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides
# =============================================================================

def load_yaml(path: str) -> TrainConfig:
    """Load a TrainConfig from a YAML file. Missing fields take defaults."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _from_dict(raw)
    cfg.sync()
    return cfg


def _from_dict(d: dict) -> TrainConfig:
    """Build TrainConfig from a (possibly partial) nested dict."""
    cfg = TrainConfig()
    for section_name, section_cfg in d.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"unknown config section: {section_name!r}")
        section = getattr(cfg, section_name)
        for k, v in (section_cfg or {}).items():
            if not hasattr(section, k):
                raise ValueError(f"unknown field: {section_name}.{k}")
            if section_name == "optimizer" and k == "betas" and isinstance(v, list):
                v = tuple(v)
            setattr(section, k, v)
    return cfg


def apply_dotted_overrides(cfg: TrainConfig, overrides: list[str]) -> None:
    """Apply CLI overrides like `--training.total_steps=100`.

    `overrides` is a list of "section.field=value" strings. Values are
    coerced to the field's declared type (via the type annotation on the
    dataclass).
    """
    import dataclasses
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be 'section.field=value', got {ov!r}")
        dotted, raw = ov.split("=", 1)
        if "." not in dotted:
            raise ValueError(f"override path must be dotted, got {dotted!r}")
        section_name, field_name = dotted.split(".", 1)
        if not hasattr(cfg, section_name):
            raise ValueError(f"unknown config section: {section_name!r}")
        section = getattr(cfg, section_name)
        if not hasattr(section, field_name):
            raise ValueError(f"unknown field: {section_name}.{field_name}")

        # Type-coerce based on the dataclass field annotation.
        ann = {f.name: f.type for f in dataclasses.fields(section)}
        t = ann.get(field_name)
        coerced = _coerce(raw, t)
        setattr(section, field_name, coerced)


def _coerce(raw: str, ann):
    """Coerce a string to the type implied by a dataclass annotation.

    Handles bool, int, float, str, Optional[X], and Literal[...]. Returns
    `raw` unchanged for anything we don't know how to coerce."""
    s = raw.strip()
    if isinstance(ann, str):  # postponed evaluation — best effort
        ann_str = ann
    else:
        ann_str = repr(ann)

    if "bool" in ann_str:
        if s.lower() in ("true", "1", "yes"):
            return True
        if s.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot coerce {raw!r} to bool")
    if "int" in ann_str:
        return int(float(s))   # allow "1e3"
    if "float" in ann_str:
        return float(s)
    return s


if __name__ == "__main__":
    cfg = TrainConfig()
    cfg.sync()
    print("--- TrainConfig defaults (Module 11 demo) ---")
    for section_name in ("model", "optimizer", "schedule", "data", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
