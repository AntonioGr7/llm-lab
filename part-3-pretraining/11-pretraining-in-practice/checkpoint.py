"""FSDP2-aware checkpointing using `torch.distributed.checkpoint` (DCP).

DCP is PyTorch's native distributed checkpoint format. It writes per-rank
shards as a directory, can be loaded into a differently-sharded model,
and is the only API recommended for FSDP2 production training.

Single-process / non-distributed runs fall back to plain `torch.save`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn


def save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    out_dir: str,
    scheduler: Optional["torch.optim.lr_scheduler.LRScheduler"] = None,
) -> str:
    """Save a checkpoint after `step` optimizer updates have been applied.

    `step` is the **number of updates completed**, not a loop index. The
    invariant: after `save(..., step=N, ...)`, a `load(...)` followed by
    `for s in range(N, total_steps): train_step(...)` produces a result
    bit-identical to an uninterrupted `for s in range(0, total_steps)`.

    Persisted state (everything needed for a correct resume):
      - `model.state_dict()`        — weights
      - `optimizer.state_dict()`    — Adam moments, step counter, etc.
      - `scheduler.state_dict()`    — LR schedule position (if scheduler passed)
      - `step`                      — number of updates completed
      - `rng_state_cpu`             — PyTorch CPU RNG
      - `rng_state_cuda`            — list of CUDA RNG states per device
                                      (Python `random` and NumPy aren't used in
                                      our hot path; add them here if you do.)

    NOT persisted (known limitation):
      - **Data loader position**. Streaming datasets restart from sample 0 on
        resume; the model replays the first few percent of the corpus. See the
        README "Resume correctness" subsection for the implications.

    With FSDP2, each rank writes its own shard via DCP. Without distributed,
    falls back to a single `.pt` file via `torch.save`.
    """
    ckpt_dir = Path(out_dir) / f"step_{step:08d}"
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else []
    )

    if dist.is_initialized():
        # Distributed save via DCP — every rank participates.
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": torch.tensor(step),
            "rng_state_cpu": rng_state_cpu,
            "rng_state_cuda": rng_state_cuda,
        }
        if scheduler is not None:
            # LRScheduler.state_dict returns a small Python dict; DCP handles it.
            state["scheduler"] = scheduler.state_dict()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dcp.save(state, checkpoint_id=str(ckpt_dir))
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "rng_state_cpu": rng_state_cpu,
            "rng_state_cuda": rng_state_cuda,
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, ckpt_dir / "state.pt")

    return str(ckpt_dir)


def load(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    scheduler: Optional["torch.optim.lr_scheduler.LRScheduler"] = None,
) -> int:
    """Load model + optimizer + (optionally) scheduler + RNG state.

    Returns the number of optimizer updates already completed (use this as
    the start index for the next loop iteration: `for step in range(returned_value, total_steps)`).

    The model's current sharding determines how DCP rehydrates the state —
    you can resume on a different number of GPUs than you trained on.
    """
    ckpt_path = Path(ckpt_dir)
    assert ckpt_path.exists(), f"checkpoint dir not found: {ckpt_dir}"

    if dist.is_initialized():
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": torch.tensor(0),
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_cuda": [
                torch.cuda.get_rng_state(i)
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        dcp.load(state, checkpoint_id=str(ckpt_path))
        step = int(state["step"].item())
        torch.set_rng_state(state["rng_state_cpu"])
        if torch.cuda.is_available() and state.get("rng_state_cuda"):
            for i, rng_state in enumerate(state["rng_state_cuda"]):
                torch.cuda.set_rng_state(rng_state, i)
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
    else:
        ckpt = torch.load(ckpt_path / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        if "rng_state_cpu" in ckpt:
            torch.set_rng_state(ckpt["rng_state_cpu"])
        if torch.cuda.is_available() and ckpt.get("rng_state_cuda"):
            for i, rng_state in enumerate(ckpt["rng_state_cuda"]):
                torch.cuda.set_rng_state(rng_state, i)
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    return step


def latest(out_dir: str) -> Optional[str]:
    """Find the latest checkpoint in `out_dir`, or None if there isn't one."""
    p = Path(out_dir)
    if not p.exists():
        return None
    ckpts = sorted(p.glob("step_*"))
    return str(ckpts[-1]) if ckpts else None


def cleanup_old(
    out_dir: str,
    keep_last: int = 3,
    milestone_every: int = 0,
) -> list[str]:
    """Trim the checkpoint directory. Two passes:

      1. **Keep the `keep_last` most recent.** The rolling-checkpoint window —
         useful for resume-after-crash.
      2. **Keep every checkpoint at a step that's a multiple of `milestone_every`.**
         These survive forever. Useful for "permanent snapshot every 5000 steps."
         Set `milestone_every=0` to disable and only keep the rolling window.

    Idempotent. Safe to call after every save.

    Args:
        out_dir: directory containing `step_XXXXXXXX/` subdirs.
        keep_last: number of most-recent checkpoints to always retain. Must be ≥ 1.
        milestone_every: keep every checkpoint whose step is a multiple of this.
            0 disables milestones.

    Returns:
        List of paths that were deleted.
    """
    assert keep_last >= 1, f"keep_last must be ≥ 1, got {keep_last}"
    p = Path(out_dir)
    if not p.exists():
        return []
    ckpts = sorted(p.glob("step_*"))
    if len(ckpts) <= keep_last:
        return []

    keep_paths = set(ckpts[-keep_last:])
    if milestone_every > 0:
        for ck in ckpts:
            try:
                step = int(ck.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if step > 0 and step % milestone_every == 0:
                keep_paths.add(ck)

    import shutil
    deleted: list[str] = []
    for ck in ckpts:
        if ck not in keep_paths:
            shutil.rmtree(ck, ignore_errors=True)
            deleted.append(str(ck))
    return deleted
