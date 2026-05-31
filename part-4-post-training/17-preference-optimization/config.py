"""The composed config for Module 17 — Preference Optimization (DPO / IPO).

Mirrors Module 15's `config.py` (single source of truth, YAML loader, dotted
CLI overrides) with two preference-specific additions:

- `ModelConfig.ref_name` — the reference model. DPO/IPO need a FROZEN copy of
  the policy at step 0; the loss penalizes drift away from it (the KL anchor).
  Conventionally `ref_name == name` (you start from the SFT checkpoint and the
  reference is that same checkpoint frozen). Sometimes labs set the reference
  to the BASE model (pre-SFT) for harsher alignment; we expose the knob.
- `PreferenceConfig` — DPO-specific hyperparameters. `beta` (the KL strength,
  default 0.1 from the DPO paper), `loss_type` (`"dpo"` or `"ipo"` — Azar et
  al.'s identity-PO variant, a 1-line swap), `label_smoothing` (cDPO, treats
  preference labels as noisy by smoothing them toward 0.5), and a
  `reference_free` debug knob that pins ref_logps to 0 (for ablations / sanity
  tests; never use in real runs).

The schedule + training sections are unchanged from Module 15 — DPO uses the
same LR shape, just at an even lower magnitude. See README §6.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    """Which HF causal LM is the policy, and which is the reference.

    DPO needs TWO copies of the model in memory at every step:
    - The POLICY is what we're training. Standard FSDP, optimizer state, the
      whole pretraining/SFT machinery.
    - The REFERENCE is frozen. No optimizer, no gradients, no FP32 master.
      We hold it in BF16 (and FSDP-shard it on multi-GPU runs for parity).

    `name`: the policy. Almost always your SFT checkpoint from Module 15 —
        a model that already speaks chat. DPO on a base model "works"
        but the loss is mostly fighting un-formatted output, which is
        what SFT was supposed to do.

    `ref_name`: the reference. Defaults to "" which means "use `name`". This
        is the textbook setup (policy and reference both start from the SFT
        checkpoint). Override to a different path/HF ID if you want to anchor
        to something else — e.g. the base model, for a stronger pull away
        from arbitrary drift, or a previous DPO round for iterative DPO.
    """
    name: str = "Qwen/Qwen3-1.7B"            # the SFT'd model from Module 15
    ref_name: str = ""                         # "" -> use `name`
    max_seq: int = 2048

    def resolved_ref_name(self) -> str:
        return self.ref_name or self.name


# =============================================================================
# Data
# =============================================================================

@dataclass
class DataConfig:
    """Preference dataset to optimize on.

    Defaults to `HuggingFaceH4/ultrafeedback_binarized` — 62k cleaned
    preference pairs derived from UltraFeedback. This is the dataset Zephyr
    was trained on and the de-facto baseline for DPO recipes.

    Expected row shape:
        {"prompt": str,
         "chosen":   [{role, content}, ...],   # ends with the better assistant turn
         "rejected": [{role, content}, ...]}   # ends with the worse  assistant turn

    `seq_len` here is the per-completion budget: we render `prompt + chosen`
    and `prompt + rejected` separately, each padded/truncated to `seq_len`.
    The micro-batch the model sees is therefore `2 * batch_size_per_device`
    sequences (chosen+rejected concatenated) — see `data.collate_preference`.

    `system_prompt` and template handling carry over from Module 15.
    """
    source: str = "HuggingFaceH4/ultrafeedback_binarized"
    split: str = "train_prefs"
    seq_len: int = 1024                      # DPO is OOM-prone (2 models) — start smaller than SFT
    batch_size_per_device: int = 2           # effective is 2× this (chosen+rejected)
    # 0 is the right default: PreferenceDataset pre-tokenizes into memory, so
    # workers add IPC cost with no payoff. See Module 15 data.py for the
    # forkserver / file_system gotcha if you ever push num_workers > 0.
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 0
    # Chat template handling (same as Module 15)
    chat_template_name: str = "auto"
    system_prompt: str = ""
    # Optional cap for smoke testing
    max_examples: Optional[int] = None


# =============================================================================
# Preference loss
# =============================================================================

LossType = Literal["dpo", "ipo"]


@dataclass
class PreferenceConfig:
    """DPO / IPO hyperparameters.

    `loss_type`:
        - "dpo" (default): the original Rafailov et al. (2023) loss.
          L = -log σ(β · (Δlogratios_chosen - Δlogratios_rejected))
          where Δlogratios = log π(y|x) - log π_ref(y|x). Sharp; can over-fit
          to high-confidence pairs; blows up if a pair is unanimous.
        - "ipo": Azar et al. (2024) identity-PO. Same logratio difference,
          but the squared-loss link function in place of log-sigmoid:
          L = (Δlogratios_chosen - Δlogratios_rejected - 1/(2β))²
          More robust to noisy / unanimous labels. Same data, same code path,
          one-line switch.

    `beta`:
        The KL strength. Low β (0.01-0.05) lets the policy drift far from the
        reference — useful if your SFT model is weak; risky because it
        produces "DPO mode collapse" (degenerate short outputs). High β (0.3-
        1.0) anchors tightly to the reference — useful if your SFT is strong
        and you only want a gentle nudge. The Zephyr / Tülu recipes settled
        on β=0.1 as the standard default. See README §5.

    `label_smoothing`:
        0.0 (default): trust the preference labels exactly.
        0.1: cDPO ("conservative DPO"). Treat labels as 10% noisy by
        symmetrizing the loss. Helps when preference data is human-noisy.

    `reference_free`:
        False (default): real DPO — use the reference model's logprobs.
        True: ablation / sanity-check mode. Sets ref_logps = 0 so the loss
        becomes pure logprob maximization on chosen, minimization on rejected.
        This collapses DPO to "RRHF without the rank" and is missing the
        KL anchor — the model WILL drift. Never use for a real run.
    """
    loss_type: LossType = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw"]


@dataclass
class OptimizerConfig:
    """AdamW tuned for DPO.

    DPO is even more drift-sensitive than SFT — the loss explicitly rewards
    moving away from a reference, which is the *opposite* of SFT's implicit
    "stay near init" regularization. Two consequences in the defaults:

    - `lr: 5e-7`. ~20× smaller than SFT's 1e-5 and ~600× smaller than
      pretraining. This is the standard Zephyr-style DPO LR for 1B+ models.
      Larger LRs (1e-6, 5e-6) produce visibly degraded models — try them
      and you'll see the symptoms in eval: shorter outputs, occasional
      gibberish, lost factuality. β does NOT save you from too-large LR.

    - `weight_decay: 0.0`. Same reasoning as SFT — full FT of a converged
      model with a small dataset; decay hurts more than helps.
    """
    type: OptimizerType = "adamw"
    lr: float = 5e-7
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


# =============================================================================
# LR schedule
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """Warmup-cosine by default — same shape as Module 15/11.

    DPO typically runs SHORTER than SFT: 1-3 epochs over a preference dataset
    is plenty; longer often degrades the model (over-optimizing the
    preference signal at the expense of capability — the classic DPO
    "alignment over-shoot"). Default total_steps=1000 with 50 warmup ≈ 1
    epoch over a 60k-pair dataset at effective batch 64.
    """
    type: ScheduleType = "cosine"
    total_steps: int = 1000
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    decay_steps: int = 100   # WSD-only


# =============================================================================
# Training (loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32", "fp8"]


@dataclass
class TrainingConfig:
    """Loop-level knobs. Carry over from Module 15 with DPO-tuned defaults.

    AC is ON by default for the same reason as SFT (1.7B full FT) PLUS the
    DPO-specific reason: we hold a second model (the reference) in memory.
    See README §4 for the memory accounting.
    """
    total_steps: int = 1000
    grad_accum: int = 16
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    activation_checkpointing: bool = True
    log_every: int = 10
    save_every: int = 250
    eval_every: int = 0                     # 0 = disabled
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
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
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
        if not (0.0 <= self.preference.label_smoothing < 0.5):
            raise ValueError(
                f"preference.label_smoothing must be in [0, 0.5), got "
                f"{self.preference.label_smoothing}"
            )
        if self.preference.beta <= 0:
            raise ValueError(f"preference.beta must be > 0, got {self.preference.beta}")

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides (verbatim from Module 15's pattern)
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
    """Apply CLI overrides like `--preference.beta=0.05`."""
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
    print("--- TrainConfig defaults (Module 17 Preference Optimization) ---")
    for section_name in ("model", "data", "preference", "optimizer", "schedule", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
