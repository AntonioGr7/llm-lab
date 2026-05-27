"""The composed config for Module 13 — Continual Pretraining (CPT).

Same structure as Modules 11/15 (single source of truth, YAML loader, dotted
CLI overrides). The sections are retuned for CPT — continuing the pretraining of
a *finished* base model on a private corpus, without erasing what it knows:

- `ModelConfig`: the finished base checkpoint to continue (default
  `Qwen/Qwen3-0.6B-Base`). Loaded, not built — same as SFT's model.py.
- `DataConfig`: TWO indexed corpora — your **domain** corpus and a general
  **replay** corpus (both produced by `make_corpus.py`, or replay can point at
  Module 12's FineWeb-Edu corpus) — plus `replay_ratio`, the fraction of tokens
  drawn from replay. Replay is the single most important anti-forgetting lever
  (README lever 1).
- `OptimizerConfig`: AdamW. **`lr` is the RE-WARM PEAK** — deliberately set to
  ~10-30% of the model's *original* pretraining peak, not the from-scratch
  value. Too high re-warms into catastrophic forgetting; too low and the corpus
  never gets learned (README lever 2).
- `ScheduleConfig`: warmup (the "re-warm") then cosine/WSD decay (the
  "re-decay"). Same scheduler as Module 09; the CPT story is in the magnitudes.
- `TrainingConfig`: short run, BF16, periodic checkpoints.

There is deliberately no separate `CPTConfig`: "continual pretraining" is just
pretraining with (a) a loaded base, (b) a replay mix, and (c) a low re-warm
peak. Each of those lives in the section it belongs to.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    """Which finished base checkpoint to continue.

    `name` is a HuggingFace Hub ID or a local HF-format directory. The default
    `Qwen/Qwen3-0.6B-Base` is chosen because it is a *true base* checkpoint (CPT
    operates on base, not instruct), it is the same family as the SFT module
    (Module 15) so the course narrative is continuous, it is small enough to
    continue-pretrain on one consumer GPU, and — crucially — it scores clearly
    above chance on MMLU/GSM8K, so the *forgetting* half of the eval is
    measurable. A 135M toy model is at random on those, which makes the
    retention demo meaningless.
    """
    name: str = "Qwen/Qwen3-0.6B-Base"
    max_seq: int = 1024


# =============================================================================
# Data — the domain + replay mixture
# =============================================================================

@dataclass
class DataConfig:
    """The two indexed corpora to mix, and at what ratio.

    Both `domain_prefix` and `replay_prefix` point at a Module-12 indexed corpus
    (`<prefix>.bin` / `.idx.npy` / `.meta.json`). `make_corpus.py` writes both;
    for a real run you'd usually repoint `replay_prefix` at the FineWeb-Edu
    corpus you built in Module 12 so the replay is genuine general data.

    `replay_ratio`: the fraction of training tokens drawn from the replay
    corpus. Because the packed dataset uses fixed-length `seq_len+1` blocks,
    every sample carries the same token count — so a token ratio is exactly a
    sample ratio, which is what `data.py` interleaves on. `0.0` disables replay
    entirely (the "domain only" ablation that demonstrates forgetting); the
    course default `0.25` follows the small-corpus end of the literature's
    1:1–1:4 domain:general range.

    `qa_heldout` points at the held-out QA probe file `make_corpus.py` writes —
    used by `eval.py`, never read during training.
    """
    domain_prefix: str = "results/corpus/domain"
    replay_prefix: str = "results/corpus/replay"
    replay_ratio: float = 0.25
    seq_len: int = 1024
    batch_size_per_device: int = 8       # the per-rank micro-batch
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 0
    # Optional cached epoch-0 permutations (mmap'd) for very large corpora.
    domain_perm_path: str = ""
    replay_perm_path: str = ""
    # Held-out QA probe set (eval only).
    qa_heldout: str = "results/corpus/qa_heldout.jsonl"


# =============================================================================
# Optimizer
# =============================================================================

OptimizerType = Literal["adamw"]


@dataclass
class OptimizerConfig:
    """AdamW. `lr` is the RE-WARM PEAK — the most important CPT knob.

    - `lr: 5e-5` is a *re-warm* peak: roughly 10-30% of a small model's original
      pretraining peak (~2e-4–3e-4). This is the dial in README lever 2. Sweep it
      against the two-axis eval: higher = more acquisition AND more forgetting.
      It is much higher than SFT's 1e-5 (CPT actually moves weights to store
      knowledge) but well below a from-scratch peak (which would wipe the base).

    - `betas: (0.9, 0.95)` — the *pretraining* betas (GPT-3 / Module 11), not
      SFT's 0.999. CPT is pretraining; the faster second-moment adaptation suits
      the higher LR and larger gradient noise.

    - `weight_decay: 0.0` for this short, small-corpus demo — same reasoning as
      the SFT module: on a finished model, decay can pull weights toward zero
      faster than a small corpus pulls them toward the new knowledge. A
      large-scale CPT run (tens of B tokens) would use ~0.1, as in pretraining.
    """
    type: OptimizerType = "adamw"
    lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0


# =============================================================================
# LR schedule — re-warm + re-decay
# =============================================================================

ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """The re-warm/re-decay schedule (Module 09's scheduler, CPT magnitudes).

    "Re-warm" = the linear warmup from 0 to `optimizer.lr` (the low re-warm
    peak). "Re-decay" = the cosine (or WSD) decay back toward `min_lr_ratio *
    peak`. A short warmup (~5% of steps) is enough — the model is already
    trained; you're not stabilizing from random init, just easing the LR up so
    the first steps don't shock the weights.

    WSD's flat-then-decay shape pairs naturally with the mid-training/annealing
    idea (README §"Where to inject"): concentrate your highest-quality domain
    data in the final `decay_steps`, where the model is most plastic.
    """
    type: ScheduleType = "cosine"
    total_steps: int = 400
    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    decay_steps: int = 40   # WSD-only — last 10% by default


# =============================================================================
# Training (loop-level knobs)
# =============================================================================

Dtype = Literal["bf16", "fp32", "fp8"]


@dataclass
class TrainingConfig:
    """Loop-level knobs. Short run; AC off by default (0.6B fits comfortably).

    For a 0.6B model on an 80GB A100 you do not need activation checkpointing —
    leave it off for speed. Flip it on (and/or drop `batch_size_per_device`) only
    if you scale the base model up.
    """
    total_steps: int = 400
    grad_accum: int = 8
    grad_clip: float = 1.0
    dtype: Dtype = "bf16"
    activation_checkpointing: bool = False
    log_every: int = 10
    save_every: int = 200
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
        if self.data.seq_len > self.model.max_seq:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must be <= "
                f"model.max_seq ({self.model.max_seq})"
            )
        if not (0.0 <= self.data.replay_ratio < 1.0):
            raise ValueError(
                f"data.replay_ratio must be in [0, 1); got {self.data.replay_ratio}. "
                "1.0 would mean no domain data at all."
            )

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# YAML loader + CLI overrides (carried verbatim from Module 11/15's pattern)
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
    """Apply CLI overrides like `--optimizer.lr=3e-5` or `--data.replay_ratio=0`."""
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
        coerced = _coerce(raw, ann.get(field_name))
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
    print("--- TrainConfig defaults (Module 13 Continual Pretraining) ---")
    for section_name in ("model", "data", "optimizer", "schedule", "training"):
        section = getattr(cfg, section_name)
        print(f"\n[{section_name}]")
        for k, v in asdict(section).items():
            print(f"  {k} = {v}")
