"""The composed config for Module 17 — Reasoning and GRPO.

Mirrors Module 16's `config.py` (single source of truth, YAML loader, dotted
CLI overrides) with three GRPO-specific additions:

- `ModelConfig.ref_name` — same dual-model machinery as DPO: a frozen reference
  model anchors the policy via KL. Defaults to "" which means "use `name`".

- `RewardConfig` — verifier weights. GRPO needs a *verifiable* reward signal
  (the whole point: skip the reward-model training stage by hand-crafting one).
  For GSM8K-style math the reward is `w_format · 1[has tags] + w_accuracy · 1[answer==gt]`.
  R1's recipe: heavy accuracy weight, light format weight as a "training-wheels"
  bonus that pushes the policy into the expected schema.

- `RLConfig` — GRPO hyperparameters. Group size, sampling temperature, max
  generation length, PPO clip ratio, and the KL coefficient β. These are
  *separate* from optimizer-level betas and dtype — RL adds its own knob layer.

`PreferenceConfig` (DPO) is GONE — the loss derives from on-policy rollouts,
not a pre-collected (chosen, rejected) dataset. The data pipeline therefore
only ships (prompt, ground_truth) pairs and never sees completions; the model
generates those at training time.
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

    Same two-model pattern as Module 16:
    - The POLICY is what we're training. Standard FSDP, optimizer state, the
      whole pretraining/SFT machinery.
    - The REFERENCE is frozen. No optimizer, no gradients, no FP32 master.
      We hold it in BF16. Provides the per-token KL anchor inside the loss.

    `name`: the policy. For the canonical R1-style demo this is an already
        SFT'd (or SFT+DPO'd) model — `Qwen/Qwen3-1.7B`. GRPO refines the
        reasoning capability already implicit in the SFT'd policy.

    `ref_name`: the reference. Defaults to "" which means "use `name`". The
        textbook GRPO setup is to anchor against the SFT model the policy
        STARTED from — so `ref_name == name` and both load the same
        checkpoint, one frozen, one trainable. Override to point at a
        different anchor (the base model, a previous GRPO round, etc.).
    """
    name: str = "Qwen/Qwen3-1.7B"
    ref_name: str = ""
    max_seq: int = 2048

    def resolved_ref_name(self) -> str:
        return self.ref_name or self.name


# =============================================================================
# Data
# =============================================================================

@dataclass
class DataConfig:
    """The (prompt, ground_truth_answer) dataset GRPO rolls out on.

    For the canonical demo this is `openai/gsm8k` (the math word-problem
    benchmark). GSM8K rows look like:
        {"question": "Janet's ducks lay 16 eggs per day...",
         "answer":   "Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18"}
    where `answer` is a free-form reasoning trace ending in `#### <int>`.
    The data pipeline strips the trace and keeps only the integer ground
    truth — what the *student* reasoning trace produces is up to GRPO.

    `system_prompt` defaults to the R1-style schema directive (asks for a
    `<think>...</think>` block followed by `<answer>...</answer>`).
    `format_reward` later checks the response against this same schema.

    `seq_len` here is the per-rollout budget: prompt + completion live in
    the same tensor. The completion length is bounded by `rl.max_new_tokens`;
    `seq_len` must be ≥ tokenized(prompt) + max_new_tokens, but is otherwise
    just a padding ceiling.
    """
    source: str = "openai/gsm8k"
    subset: str = "main"                     # gsm8k has "main" and "socratic"
    split: str = "train"
    seq_len: int = 1024
    prompts_per_step: int = 4                # how many prompts to roll out per RL step
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 0
    chat_template_name: str = "auto"
    system_prompt: str = (
        "You are a math reasoning assistant. Solve the problem step by step. "
        "Put your reasoning inside <think>...</think> tags, then your final "
        "numerical answer (digits only) inside <answer>...</answer> tags."
    )
    max_examples: Optional[int] = None       # cap for smoke testing


# =============================================================================
# Reward
# =============================================================================

@dataclass
class RewardConfig:
    """Verifier weights. R1's recipe: heavy accuracy, light format bonus.

    `w_format`: weight on the schema-correctness bonus. Set higher (~0.5)
        early in training if the policy hasn't learned the tag format yet;
        drop to 0.1 once accuracy starts dominating.
    `w_accuracy`: weight on the answer-correctness bonus. Should always be
        the dominant term — accuracy is what we're actually optimizing.
    `format_pattern`: the regex shape the format reward checks for. Default
        matches R1's `<think>...</think><answer>...</answer>`.
    """
    w_format: float = 0.1
    w_accuracy: float = 1.0
    # Regex string for the format check; see rewards.py for usage.
    format_pattern: str = r"<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>"


# =============================================================================
# RL (GRPO hyperparameters)
# =============================================================================

@dataclass
class RLConfig:
    """GRPO-specific knobs that don't fit the SFT-shape config.

    `group_size`: the G in GRPO. We sample G completions per prompt, score
        them, and normalize advantages WITHIN each group (this is the
        "group-relative" part — the baseline isn't a learned value head,
        it's just the within-prompt mean reward). DeepSeekMath uses G=64;
        smaller G (8-16) trades variance for memory. Default 8 for the
        1.7B demo.

    `temperature`, `top_p`, `top_k`: sampling knobs for `model.generate()`
        during rollout. R1 uses T=1.0 with no top-k for maximum exploration;
        for the demo we lean slightly conservative (T=0.9, top_p=0.95) so
        runs converge faster on a budget.

    `max_new_tokens`: per-completion generation budget. GSM8K answers are
        rarely longer than 256 tokens of CoT; 512 leaves headroom.

    `kl_beta`: the per-token KL anchor strength. DeepSeekMath ships β=0.04;
        substantially smaller than DPO's β=0.1 because the KL here is a
        PER-TOKEN penalty inside the policy gradient (it's summed over all
        response tokens), whereas DPO's β scales a single sequence-level
        log-sigmoid. Same role (anchor against drift), different scale.

    `clip_ratio`: the PPO ε. The standard 0.2 lets ratios move ±20% per
        update before clipping kicks in. For K=1 (one gradient step per
        rollout) the ratio is exactly 1 at the first step so the clip is
        a no-op — but the code is correct for K>1 (mu_epochs > 1).

    `mu_epochs`: K in the PPO/GRPO literature — how many gradient updates
        per rollout batch. K=1 is the cheapest and most stable; K>1 reuses
        the rollout for more gradient signal but needs the importance ratio
        + clip to stay on-policy. We default to 1 and note the extension.

    `adv_eps`: numerical stabilizer in the within-group advantage
        normalization. Prevents division by zero on groups where all
        rewards are identical (which DOES happen — e.g. when all G samples
        miss the format reward).
    """
    group_size: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0                           # 0 = disabled
    max_new_tokens: int = 512
    kl_beta: float = 0.04
    clip_ratio: float = 0.2
    mu_epochs: int = 1                       # K = 1 (standard for stability)
    adv_eps: float = 1e-6


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw"]


@dataclass
class OptimizerConfig:
    """AdamW tuned for GRPO.

    GRPO learning rate sits between SFT (1e-5) and DPO (5e-7). The signal
    is sparser than DPO (you only get gradient on the tokens the policy
    actually generates, weighted by advantages that can be all-zero on a
    group with all-miss rewards), so a slightly larger LR than DPO helps
    the policy move at all — but too large blows up the KL anchor budget
    and the policy collapses fast.

    DeepSeekMath ships 1e-6; R1 uses 3e-6. We default 1e-6 — Zephyr-DPO
    territory pushed up ~2× to account for advantage sparsity.
    """
    type: OptimizerType = "adamw"
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


# =============================================================================
# LR schedule
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """Constant-with-warmup is the GRPO standard.

    DeepSeekMath and Tülu 3's RL stage both use constant LR with a short
    warmup. Why no cosine: GRPO runs are typically short (hundreds, not
    thousands, of steps) and adding decay on top of the natural reward
    plateau just slows convergence near the end. We keep `cosine` as an
    option for ablations but default to `constant`.
    """
    type: ScheduleType = "constant"
    total_steps: int = 300
    warmup_steps: int = 20
    min_lr_ratio: float = 1.0                # constant => keep peak LR
    decay_steps: int = 50                    # WSD-only


# =============================================================================
# Training (loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32"]


@dataclass
class TrainingConfig:
    """Loop-level knobs. Carry over from Module 16 with GRPO-tuned defaults.

    AC is ON by default for the same reason as DPO (1.7B full FT + a
    second model in memory). Save_every is shorter than DPO because GRPO
    steps are EXPENSIVE (each step includes G·B generations) — losing a
    long run to a crash is more painful.
    """
    total_steps: int = 300
    grad_accum: int = 1                      # rollouts already aggregate G groups
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    activation_checkpointing: bool = True
    log_every: int = 1                       # log every step — RL steps are slow
    save_every: int = 50
    eval_every: int = 0                      # 0 = disabled
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
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
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
        if self.rl.group_size < 2:
            raise ValueError(
                f"rl.group_size must be >= 2 (need at least 2 samples to compute "
                f"group-relative advantage), got {self.rl.group_size}"
            )
        if self.rl.kl_beta < 0:
            raise ValueError(f"rl.kl_beta must be >= 0, got {self.rl.kl_beta}")
        if not (0 < self.rl.clip_ratio < 1):
            raise ValueError(f"rl.clip_ratio must be in (0, 1), got {self.rl.clip_ratio}")
        if self.rl.mu_epochs < 1:
            raise ValueError(f"rl.mu_epochs must be >= 1, got {self.rl.mu_epochs}")

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides (verbatim from Module 16's pattern)
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
    """Apply CLI overrides like `--rl.group_size=16`."""
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
    print("--- TrainConfig defaults (Module 17 Reasoning and GRPO) ---")
    for section_name in ("model", "data", "reward", "rl", "optimizer", "schedule", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
