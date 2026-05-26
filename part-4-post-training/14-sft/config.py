"""The composed config for Module 14 — Supervised Fine-Tuning.

Mirrors Module 11's `config.py` structure (single source of truth, YAML
loader, dotted CLI overrides) but with sections rewritten for SFT:

- `ModelConfig`: HF causal LM by name (e.g. `Qwen/Qwen3-1.7B-Base`), not
  the from-scratch shape knobs from pretraining. The architecture is
  whatever HF gives us; we just need to know the tokenizer and seq_len.
- `DataConfig`: a chat dataset name + chat-template handling knobs.
- `OptimizerConfig`: AdamW with SFT-appropriate defaults — much smaller
  LR than pretraining (see the README §6 LR discussion).
- `TrainingConfig`: shorter total_steps, no wandb-specific quirks
  carried over, activation checkpointing default ON (1.7B full FT on
  A100-80GB needs it to fit).

The schedule config and most loop-level knobs are identical to Module 11
— SFT and pretraining share the underlying training loop. We deliberately
do NOT include a LoraConfig here: LoRA / QLoRA / DoRA are out of scope
for this module and get their own focused treatment in a later Part 4
module (see README §4).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    """Which HF causal LM to fine-tune.

    Unlike Module 11 (which built a Qwen3 from-scratch via geometry knobs),
    SFT loads a fully-pretrained checkpoint. The only architectural knob is
    `max_seq` because longer sequences cost more memory and most chat data
    fits comfortably under 2048 tokens.

    `name` is either:
    - A HuggingFace Hub model ID, e.g. `"Qwen/Qwen3-1.7B-Base"`.
    - A local path to a pretrained checkpoint, e.g. the `results/checkpoints/
      step_00003000/` directory from Module 11. The latter lets you SFT your
      own 150M model as a proof-of-concept run.
    """
    name: str = "Qwen/Qwen3-1.7B-Base"
    max_seq: int = 2048


# =============================================================================
# Data
# =============================================================================

@dataclass
class DataConfig:
    """The chat dataset to SFT on.

    Defaults to `HuggingFaceH4/no_robots` (10k hand-curated instruction
    examples). The dataset is loaded via `datasets.load_dataset` and is
    expected to expose either:
    - a `messages` field with the standard `[{role, content}, ...]` shape, OR
    - a `prompt` + `response` pair we wrap into a 2-turn conversation.

    `chat_template_name`:
        - "auto": use the tokenizer's built-in chat template (recommended).
        - any other string: load `~/.cache/.../chat_templates/{name}.jinja`.

    `pack`:
        - false (default): one example per training sample. Padded to `seq_len`.
          Simpler; works for any dataset.
        - true: pack multiple short conversations into one `seq_len`-long
          sequence, with a per-sample attention mask so they don't attend
          across boundaries. 2-3x throughput on short-response datasets.

    `system_prompt`:
        - "": no system prompt (most datasets prefer this).
        - a string: prepended as a system turn before each conversation.
    """
    source: str = "HuggingFaceH4/no_robots"
    split: str = "train"
    seq_len: int = 2048
    batch_size_per_device: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 0
    # Chat template handling
    chat_template_name: str = "auto"
    system_prompt: str = ""
    # Optional packing optimization (stretch goal in README §9)
    pack: bool = False
    # Optional cap for smoke testing
    max_examples: Optional[int] = None


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw"]


@dataclass
class OptimizerConfig:
    """AdamW hyperparameters tuned for SFT.

    Key differences from pretraining defaults:

    - `lr: 1e-5` is ~30× smaller than the 3e-4 we used in Module 11.
      The base model is already at a good point in weight space; we're
      nudging it, not rebuilding it. Large LRs in this regime cause
      catastrophic forgetting (the "alignment tax" from Module 13).
      See README §6 for the full LR discussion.

    - `betas[1] = 0.999` instead of the 0.95 we used in pretraining.
      The standard AdamW default; 0.95 is a pretraining-specific tweak
      from the GPT-3 paper that lets the second-moment estimate adapt
      faster on noisy early-stage gradients. For SFT, gradients are
      cleaner and we want more averaging.

    - `weight_decay: 0.0`. On full FT of a pretrained model with a
      small dataset, weight decay pulls the weights toward zero faster
      than the data pulls them toward the SFT objective. The result is
      capability loss without offsetting regularization benefit. Some
      labs use 1e-2 for very large SFT datasets; for the 10k-example
      default here, 0 is the right call.
    """
    type: OptimizerType = "adamw"
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


# =============================================================================
# LR schedule
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """Warmup-cosine by default; same shape as Module 11's schedule.

    With `total_steps=600` and `warmup_steps=100`, the LR ramps from 0 to
    peak over the first ~17% of training, then cosine-decays to
    `min_lr_ratio · peak_lr` over the remaining 83%. The shape is well
    understood from Module 09; we just retune the magnitudes.
    """
    type: ScheduleType = "cosine"
    total_steps: int = 600
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    decay_steps: int = 60   # WSD-only — last 10% by default


# =============================================================================
# Training (loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32", "fp8"]


@dataclass
class TrainingConfig:
    """Loop-level knobs. Mostly carry over from Module 11 with SFT-tuned
    defaults: shorter run, AC on, smaller save frequency.
    """
    total_steps: int = 600
    grad_accum: int = 16
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    # Full FT of 1.7B on A100-80GB doesn't fit without AC — see README §4.
    activation_checkpointing: bool = True
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 0                     # 0 = disabled
    checkpoint_dir: str = "./results/checkpoints"
    resume_from: Optional[str] = None
    keep_last_n_checkpoints: int = 3
    milestone_every: int = 0
    # W&B (rank-0 only). Empty `wandb_project` disables W&B entirely.
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_run_id: str = ""
    wandb_tags: tuple[str, ...] = ()


# =============================================================================
# The full config
# =============================================================================

@dataclass
class TrainConfig:
    """Complete config consumed by `train.py`."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def sync(self) -> None:
        """Propagate cross-section invariants. Called by `train.py` after load."""
        self.schedule.total_steps = self.training.total_steps
        # Data's seq_len follows the model's max_seq unless explicitly set elsewhere.
        # (Both default to 2048; this just keeps them aligned if one is overridden.)
        if self.data.seq_len > self.model.max_seq:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must be <= "
                f"model.max_seq ({self.model.max_seq})"
            )

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides (carried verbatim from Module 11's pattern)
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
    """Apply CLI overrides like `--training.total_steps=100`."""
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

        ann = {f.name: f.type for f in dataclasses.fields(section)}
        t = ann.get(field_name)
        coerced = _coerce(raw, t)
        setattr(section, field_name, coerced)


def _coerce(raw: str, ann):
    """Coerce a string to the type implied by a dataclass annotation."""
    s = raw.strip()
    if isinstance(ann, str):
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
        return int(float(s))
    if "float" in ann_str:
        return float(s)
    return s


if __name__ == "__main__":
    cfg = TrainConfig()
    cfg.sync()
    print("--- TrainConfig defaults (Module 14 SFT) ---")
    for section_name in ("model", "data", "optimizer", "schedule", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
