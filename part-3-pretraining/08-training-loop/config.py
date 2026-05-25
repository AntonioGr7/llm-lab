"""Training config schema for Part 3's pretraining framework.

One dataclass per concern (model, optimizer, data, training). The full
`TrainConfig` composes them. CLI args (in `train.py`) override fields by
dotted path, and a YAML loader is provided for repeatable runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


Dtype = Literal["bf16", "fp32", "fp8"]


@dataclass
class ModelConfig:
    """Architecture knobs. Defaults to a ~150M Qwen3-shaped model."""
    vocab_size: int = 32_000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 2
    d_ffn: int = 2048
    max_seq: int = 2048
    rope_theta: float = 10_000.0
    tie_weights: bool = True
    norm_eps: float = 1e-6


OptimizerType = Literal["adamw", "muon"]


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters.

    `type='adamw'` (default) uses standard AdamW everywhere. `type='muon'`
    uses Jordan's Muon update on 2D Linear weights and AdamW on the rest
    (embeddings, norms, biases). See Module 08 § 4 and `muon.py`.

    The AdamW fields below are used both for the pure-AdamW path and for
    the "AdamW arm" of the Muon hybrid (so embeddings and norms get the
    same treatment in both modes). The `muon_*` fields are ignored when
    `type='adamw'`.
    """
    type: OptimizerType = "adamw"
    # AdamW
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    # Muon (used only when type='muon')
    muon_lr: float = 2e-2
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5


@dataclass
class DataConfig:
    """Where training data comes from. Module 08 ships synthetic data only;
    Module 11 replaces this with the FineWeb-Edu loader."""
    source: str = "synthetic"          # "synthetic" or "fineweb_edu" (Module 11)
    seq_len: int = 2048
    batch_size_per_device: int = 4     # micro-batch per rank per accum step
    num_workers: int = 2
    seed: int = 0
    # Synthetic-only knob: number of fake samples to materialize per epoch.
    synthetic_samples: int = 10_000


@dataclass
class TrainingConfig:
    """Loop-level knobs."""
    total_steps: int = 1_000
    grad_accum: int = 1
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    log_every: int = 10
    save_every: int = 500
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


@dataclass
class TrainConfig:
    """The complete config a `train.py` run consumes."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_yaml(path: str) -> TrainConfig:
    """Load a TrainConfig from a YAML file. Missing fields take defaults."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(raw)


def _from_dict(d: dict) -> TrainConfig:
    """Build TrainConfig from a (possibly partial) nested dict."""
    cfg = TrainConfig()
    for section_name, section_cfg in d.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"unknown config section: {section_name}")
        section = getattr(cfg, section_name)
        for k, v in (section_cfg or {}).items():
            if not hasattr(section, k):
                raise ValueError(f"unknown field: {section_name}.{k}")
            setattr(section, k, v)
    return cfg
