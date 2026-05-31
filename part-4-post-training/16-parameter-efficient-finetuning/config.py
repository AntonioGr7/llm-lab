"""The composed config for Module 16 — Parameter-Efficient Fine-Tuning.

Same structure as Module 15 (single source of truth, YAML loader, dotted CLI
overrides), with two additions and one retuned default:

- NEW `LoRAConfig`: rank `r`, `alpha`, `dropout`, which projections to adapt,
  and the QLoRA 4-bit knobs. This is the only conceptually new section.
- `OptimizerConfig.lr` defaults to **2e-4**, ~20× higher than full-FT SFT's
  1e-5. LoRA adapters start at zero (ΔW=0) and have to travel; a tiny LR
  barely moves them. The community-standard LoRA LR is 1e-4–3e-4. See README §5.
- `TrainingConfig.activation_checkpointing` defaults to **False**. LoRA's
  memory win is that the optimizer state covers only the adapters, so a 1.7B
  LoRA run fits on a single 24GB GPU *without* AC. Flip it on only if you push
  batch size or sequence length hard.
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

    `name` is a HuggingFace Hub ID (e.g. `"Qwen/Qwen3-1.7B-Base"`) or a local
    HF-format directory. For an apples-to-apples comparison against Module 15's
    full fine-tune, point this at the *same* base and use the same dataset.
    """
    name: str = "Qwen/Qwen3-1.7B-Base"
    max_seq: int = 2048


# =============================================================================
# Data  (identical to Module 15 — chat templates + assistant-only masking)
# =============================================================================

@dataclass
class DataConfig:
    source: str = "HuggingFaceH4/no_robots"
    split: str = "train"
    seq_len: int = 2048
    batch_size_per_device: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 0
    chat_template_name: str = "auto"
    system_prompt: str = ""
    pack: bool = False
    max_examples: Optional[int] = None


# =============================================================================
# LoRA  (the new section)
# =============================================================================

@dataclass
class LoRAConfig:
    """Low-rank adaptation knobs.

    - `r`: the rank of the update ΔW = B@A. Bigger r = more capacity = more
      trainable params. r=8–16 is the sweet spot for instruction tuning; r=64+
      approaches full-FT quality (and cost) on harder tasks. Default 8.
    - `alpha`: scales the update by `alpha/r`. Setting `alpha = 2·r` is a
      common default that keeps the effective update magnitude stable as you
      sweep r, so you don't have to re-tune the LR. Default 16 (= 2·8).
    - `dropout`: dropout on the adapter input. 0.05–0.1 helps on small SFT
      datasets; 0.0 is fine for larger ones. Default 0.0.
    - `target_modules`: which Linear projections to adapt, matched by *leaf
      attribute name*. The QLoRA recipe (adapt all 7: attention q/k/v/o + MLP
      gate/up/down) beats adapting only q/v at higher rank.

    QLoRA (Dettmers et al. 2023) — load the frozen base in 4-bit NF4, attach
    bf16 adapters on top:
    - `qlora`: enable 4-bit base loading (needs `pip install bitsandbytes`).
    - `bnb_4bit_quant_type`: "nf4" (the information-theoretically optimal
      4-bit type from the paper) or "fp4".
    - `bnb_4bit_use_double_quant`: quantize the quantization constants too
      (~0.4 bits/param more savings).
    - `bnb_4bit_compute_dtype`: dtype the 4-bit weights dequantize to for the
      matmul. "bfloat16" on Ampere+.
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # QLoRA (4-bit base)
    qlora: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw"]


@dataclass
class OptimizerConfig:
    """AdamW. The headline change from Module 15 is `lr`: 2e-4, not 1e-5.

    LoRA adapters initialize to ΔW=0 and must travel a real distance to encode
    the task; the full-FT logic of "the base is already good, nudge it gently"
    does not apply to the adapter, which starts at *nothing*. Weight decay stays
    0.0 (adapters are small and regularize themselves via the low-rank bottleneck).
    """
    type: OptimizerType = "adamw"
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


# =============================================================================
# LR schedule
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    type: ScheduleType = "cosine"
    total_steps: int = 600
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    decay_steps: int = 60


# =============================================================================
# Training (loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32", "fp8"]


@dataclass
class TrainingConfig:
    total_steps: int = 600
    grad_accum: int = 16
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    # OFF by default for LoRA — the run fits on 24GB without it. See README §6.
    activation_checkpointing: bool = False
    use_fused_ce: bool = False
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 0
    checkpoint_dir: str = "./results/checkpoints"
    resume_from: Optional[str] = None
    keep_last_n_checkpoints: int = 3
    milestone_every: int = 0
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
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def sync(self) -> None:
        self.schedule.total_steps = self.training.total_steps
        if self.data.seq_len > self.model.max_seq:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must be <= "
                f"model.max_seq ({self.model.max_seq})"
            )
        if self.lora.r <= 0:
            raise ValueError(f"lora.r must be >= 1, got {self.lora.r}")

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides (carried verbatim from Module 15's pattern)
# =============================================================================

def load_yaml(path: str) -> TrainConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _from_dict(raw)
    cfg.sync()
    return cfg


def _from_dict(d: dict) -> TrainConfig:
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
            if section_name == "lora" and k == "target_modules" and isinstance(v, list):
                v = tuple(v)
            setattr(section, k, v)
    return cfg


def apply_dotted_overrides(cfg: TrainConfig, overrides: list[str]) -> None:
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
    s = raw.strip()
    ann_str = ann if isinstance(ann, str) else repr(ann)

    # tuple-of-str fields (e.g. lora.target_modules) — split on commas
    if "tuple" in ann_str and "str" in ann_str:
        return tuple(x.strip() for x in s.split(",") if x.strip())
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
    print("--- TrainConfig defaults (Module 16 LoRA/QLoRA) ---")
    for section_name in ("model", "data", "lora", "optimizer", "schedule", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
