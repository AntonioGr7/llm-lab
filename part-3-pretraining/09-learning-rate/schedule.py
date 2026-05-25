"""Learning-rate schedulers for Part 3 pretraining.

Three schedules:

- `WarmupCosineLR` — linear warmup, then cosine decay to `min_lr_ratio * peak`.
  The 2026-canonical schedule. Use when total_steps is fixed.
- `WSDLR` — warmup, stable, then short decay at the end. DeepSeek-V3's
  schedule. Use when you might extend training mid-flight.
- `ConstantWithWarmupLR` — linear warmup then hold. For ablations.

All three subclass `torch.optim.lr_scheduler.LRScheduler`, so they preserve
each param group's `initial_lr` and multiply by a single time-varying
factor. This means muP-scaled LR groups (from Module 07's `param_groups_mup`)
flow through correctly — the cosine multiplies all groups by the same
factor, leaving the per-group ratios intact.

Step the scheduler once per optimizer step, AFTER `optimizer.step()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


ScheduleType = Literal["cosine", "wsd", "constant"]


@dataclass
class ScheduleConfig:
    """LR schedule hyperparameters."""
    type: ScheduleType = "cosine"
    total_steps: int = 1_000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    # WSD-only knobs (ignored for cosine/constant). decay_steps is the
    # length of the final decay phase; everything between warmup and decay
    # is the stable region.
    decay_steps: int = 100


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine decay to min_lr_ratio * peak.

    Args:
        optimizer: the optimizer whose LRs to schedule.
        warmup_steps: number of steps to linearly warm up.
        total_steps: total training steps. Cosine decays from `warmup_steps`
            to `total_steps` and holds at min beyond that.
        min_lr_ratio: minimum LR as fraction of peak (e.g. 0.1 = 10% floor).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        assert 0 <= warmup_steps < total_steps, (
            f"need 0 <= warmup ({warmup_steps}) < total ({total_steps})"
        )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [base_lr * self._factor(self.last_epoch) for base_lr in self.base_lrs]

    def _factor(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup, 0 -> 1
            return step / max(1, self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr_ratio
        # Cosine decay from 1 to min_lr_ratio
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos


class WSDLR(LRScheduler):
    """Warmup-Stable-Decay (DeepSeek-V3 style).

    Three regions:
    - Steps [0, warmup_steps):                       linear warmup, 0 -> 1.
    - Steps [warmup_steps, total_steps - decay_steps): constant at peak.
    - Steps [total_steps - decay_steps, total_steps): linear decay, 1 -> min_lr_ratio.

    The advantage over cosine: the stable region's LR doesn't depend on
    total_steps, so you can extend training mid-flight by just increasing
    total_steps before the decay starts.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        assert decay_steps + warmup_steps < total_steps, (
            f"warmup ({warmup_steps}) + decay ({decay_steps}) must be less than total ({total_steps})"
        )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [base_lr * self._factor(self.last_epoch) for base_lr in self.base_lrs]

    def _factor(self, step: int) -> float:
        decay_start = self.total_steps - self.decay_steps
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        if step < decay_start:
            return 1.0
        if step >= self.total_steps:
            return self.min_lr_ratio
        # Linear decay 1 -> min_lr_ratio over decay_steps
        progress = (step - decay_start) / max(1, self.decay_steps)
        return 1.0 - (1.0 - self.min_lr_ratio) * progress


class ConstantWithWarmupLR(LRScheduler):
    """Linear warmup then constant. For ablations only — no decay phase
    is essentially never the right call for real pretraining."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            f = self.last_epoch / max(1, self.warmup_steps)
        else:
            f = 1.0
        return [base_lr * f for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: Optimizer, cfg: ScheduleConfig) -> LRScheduler:
    """Build the scheduler matching cfg.type."""
    if cfg.type == "cosine":
        return WarmupCosineLR(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
    if cfg.type == "wsd":
        return WSDLR(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
            decay_steps=cfg.decay_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
    if cfg.type == "constant":
        return ConstantWithWarmupLR(optimizer, warmup_steps=cfg.warmup_steps)
    raise ValueError(f"unknown schedule type: {cfg.type}")


if __name__ == "__main__":
    # Trace each schedule and dump the LR curve to verify shapes.
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = WarmupCosineLR(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    print("WarmupCosineLR (warmup=10, total=100, min=0.1):")
    for step in [0, 5, 10, 25, 50, 75, 99, 100]:
        # Replay: step the scheduler to that point.
        opt2 = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        s = WarmupCosineLR(opt2, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
        for _ in range(step):
            opt2.step(); s.step()
        print(f"  step {step:3d}: lr = {opt2.param_groups[0]['lr']:.4f}")

    print("\nWSDLR (warmup=10, total=100, decay=20, min=0.1):")
    for step in [0, 5, 10, 50, 79, 80, 90, 100]:
        opt2 = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        s = WSDLR(opt2, warmup_steps=10, total_steps=100, decay_steps=20, min_lr_ratio=0.1)
        for _ in range(step):
            opt2.step(); s.step()
        print(f"  step {step:3d}: lr = {opt2.param_groups[0]['lr']:.4f}")
