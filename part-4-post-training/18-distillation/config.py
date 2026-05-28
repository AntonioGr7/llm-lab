"""The composed config for Module 18 — Distillation.

This module covers THREE flavors of distillation:

  - **offline**       — train on pre-generated (prompt, teacher_completion)
                        pairs with standard cross-entropy. This is exactly
                        SFT (Module 15) with a different data source; we
                        cover it in README §2 and don't ship a separate
                        train.py for it — just a config flag here.
  - **on_policy**     — student generates completions on its OWN
                        distribution; teacher provides per-token logits;
                        student matches via forward KL. Needs a separate
                        (larger) teacher.
  - **sdft**          — same loss as `on_policy`, but the "teacher" is
                        the SAME MODEL conditioned on K demonstrations
                        in-context. No separate teacher. (Shenfeld et
                        al., 2026)

Mirrors Modules 15-17's config pattern: dataclass-per-section, YAML
loader, dotted CLI overrides. The new pieces are `DistillationConfig`
(mode + teacher_name + n_demos + kl_direction + temperature) and a
slight rework of `DataConfig` to handle the three-mode corpus layout.

GRPO-specific knobs from Module 17 are gone — distillation doesn't need
group-relative advantages, PPO clip, or KL anchor against a frozen
reference (the teacher's distribution IS the anchor by construction).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    """Student and teacher.

    `name`:
        The STUDENT — what we're training. Loaded with FP32 master weights,
        every parameter trainable. FSDP MixedPrecision casts to BF16 for
        compute (same recipe as M15-17).

    `teacher_name`:
        The TEACHER — frozen, BF16, no grad. Empty string has different
        meanings depending on `distill.mode`:
          - `offline`:    teacher_name is IGNORED (the teacher is implicit
                          in the pre-generated dataset).
          - `on_policy`:  teacher_name MUST be set to a different (typically
                          larger) HF model id.
          - `sdft`:       teacher_name="" means "use `name`" — the same
                          model serves as both teacher and student. The
                          teacher path runs with demonstrations prepended
                          in-context; the student path doesn't. This is
                          the textbook SDFT setup.
    """
    name: str = "Qwen/Qwen3-1.7B"
    teacher_name: str = ""                # "" => SDFT (self) OR offline (ignored)
    max_seq: int = 4096                   # demos + prompt + completion can be long

    def resolved_teacher_name(self) -> str:
        return self.teacher_name or self.name


# =============================================================================
# Distillation mode + KL
# =============================================================================

DistillMode = Literal["offline", "on_policy", "sdft"]
KLDirection = Literal["forward", "reverse"]


@dataclass
class DistillationConfig:
    """The three-mode dispatch + the KL hyperparameters.

    `mode`:
        Which flavor (see module docstring).

    `n_demonstrations`:
        K — number of in-context demonstrations the SDFT teacher sees.
        The SDFT paper uses K=8 across their experiments; we default
        the same. Has no effect when `mode != "sdft"`.

    `kl_direction`:
        "forward" — KL(teacher || student). Student covers all modes the
        teacher uses. Standard imitation distillation default.
        "reverse" — KL(student || teacher). Mode-seeking; sharper student.
        GKD ablates both; we default forward (one-line swap in the loss).

    `temperature`:
        Softmax temperature applied to BOTH teacher and student logits
        before computing KL. T > 1 softens the distributions (more uniform
        signal across the vocab); T < 1 sharpens. Default 1.0.

    `top_k_kl`:
        If > 0, restrict the KL computation to the top-K vocabulary
        positions of the TEACHER. Saves memory (we don't need to project
        the full 150k vocab through softmax for every position) and is
        equivalent to assuming the bottom of the teacher's tail
        contributes negligibly. 256-1024 is the common range. Set 0 to
        disable (use full vocab).

    `sampling_temperature`:
        Temperature used for the teacher's GENERATIONS (rollout). This is
        a separate knob from `temperature` (which only affects the
        loss-time KL). Default 0.9.

    `top_p`, `top_k`, `max_new_tokens`:
        Sampling hyperparameters for the teacher's rollout.
    """
    mode: DistillMode = "sdft"
    n_demonstrations: int = 8
    kl_direction: KLDirection = "forward"
    temperature: float = 1.0
    top_k_kl: int = 0                     # 0 = use full vocab
    sampling_temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 128             # tool calls are short; raise for longer tasks


# =============================================================================
# Data
# =============================================================================

@dataclass
class DataConfig:
    """The tool-use corpus + the GSM8K prior-skill eval anchor.

    `corpus_dir`:
        Path to the tool-use corpus produced by `make_tooluse_corpus.py`.
        Expects `demos.jsonl`, `train.jsonl`, `eval.jsonl` under it.

    `prompts_per_step`:
        How many tool-use prompts feed ONE distillation step. Each prompt
        generates one teacher rollout in `on_policy`/`sdft` modes (or one
        SFT-style example in `offline` mode).

    `gsm8k_source`/`gsm8k_subset`/`gsm8k_split`:
        Where to pull the prior-skill eval from. Default: the same
        `openai/gsm8k` test set Module 17 uses, so prior-skill accuracy
        directly compares against our M17 numbers.

    `seq_len`:
        Padding budget for the (demos + prompt + completion) sequences.
        For SDFT this needs to fit K demonstrations + the training prompt
        + the teacher's generation. The 8 demos in our corpus average
        ~50 tokens each, so 4096 is generous.
    """
    corpus_dir: str = "./data"
    prompts_per_step: int = 4
    seq_len: int = 4096
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 0
    chat_template_name: str = "auto"
    system_prompt: str = (
        "You are a tool-calling assistant. Given a user request, respond "
        "with exactly one tool call in the format "
        "<tool>NAME</tool><args>JSON</args>."
    )
    # Cap for smoke testing
    max_train: Optional[int] = None
    max_eval: Optional[int] = None

    # GSM8K prior-skill eval (reuses M17's loader logic)
    gsm8k_source: str = "openai/gsm8k"
    gsm8k_subset: str = "main"
    gsm8k_split: str = "test"
    gsm8k_n_eval: int = 100                # how many GSM8K problems to score
    gsm8k_system_prompt: str = (
        "You are a math reasoning assistant. Solve the problem step by step. "
        "Put your reasoning inside <think>...</think> tags, then your final "
        "numerical answer (digits only) inside <answer>...</answer> tags."
    )


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw"]


@dataclass
class OptimizerConfig:
    """AdamW tuned for distillation.

    Distillation LR sits between SFT (1e-5) and GRPO (1e-6) — the
    gradient signal is denser than GRPO (per-token KL on every response
    token; no advantage-sparsity) but less dense than SFT (we score
    the teacher's distribution, which may have lots of high-entropy
    tokens contributing small but non-zero KL).

    SDFT paper uses lr=5e-5 on Qwen2.5-7B; we go 2e-5 as a course default
    on Qwen3-1.7B. The SDFT loss is harder to over-train than SFT (the
    teacher is the model itself with demos, so the target is reachable
    rather than 'a frontier-lab demonstration'), but pushing LR too high
    still degrades prior skills.
    """
    type: OptimizerType = "adamw"
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


# =============================================================================
# LR schedule
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """Warmup-cosine by default — matches the SFT recipe."""
    type: ScheduleType = "cosine"
    total_steps: int = 200
    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    decay_steps: int = 50


# =============================================================================
# Training (loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32"]


@dataclass
class TrainingConfig:
    """Loop-level knobs."""
    total_steps: int = 200
    grad_accum: int = 1
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    activation_checkpointing: bool = True
    log_every: int = 1
    save_every: int = 50
    eval_every: int = 50                   # periodic two-axis eval (toolcall + GSM8K)
    eval_n_gsm8k: int = 50                 # smaller during training; full at the end
    checkpoint_dir: str = "./results/checkpoints"
    resume_from: Optional[str] = None
    keep_last_n_checkpoints: int = 3
    milestone_every: int = 0
    # W&B (rank-0 only).
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
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def sync(self) -> None:
        """Propagate cross-section invariants. Called by `train.py` after load."""
        self.schedule.total_steps = self.training.total_steps
        if self.data.seq_len > self.model.max_seq:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must be <= "
                f"model.max_seq ({self.model.max_seq})"
            )
        if self.distill.n_demonstrations < 1 and self.distill.mode == "sdft":
            raise ValueError(
                f"sdft requires n_demonstrations >= 1, got "
                f"{self.distill.n_demonstrations}"
            )
        if self.distill.mode == "on_policy" and not self.model.teacher_name:
            raise ValueError(
                "distill.mode='on_policy' requires model.teacher_name to be "
                "set to a different (typically larger) model. For self-"
                "distillation use mode='sdft'."
            )
        if self.distill.temperature <= 0:
            raise ValueError(f"distill.temperature must be > 0, got "
                             f"{self.distill.temperature}")

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides (verbatim from Module 17's pattern)
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
    """Apply CLI overrides like `--distill.mode=on_policy`."""
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
    ann_str = ann if isinstance(ann, str) else repr(ann)
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
    print("--- TrainConfig defaults (Module 18 Distillation) ---")
    for section_name in ("model", "data", "distill", "optimizer", "schedule", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            if isinstance(v, str) and len(v) > 80:
                v = v[:80] + "..."
            print(f"  {k} = {v}")
