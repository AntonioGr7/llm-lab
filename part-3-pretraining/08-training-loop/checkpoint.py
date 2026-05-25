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
) -> str:
    """Save a checkpoint at the given step. Returns the checkpoint path.

    With FSDP2 each rank writes its own shard into the output directory.
    Without distributed, falls back to a single `.pt` file.
    """
    ckpt_dir = Path(out_dir) / f"step_{step:08d}"

    if dist.is_initialized():
        # Distributed save via DCP — every rank participates.
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": torch.tensor(step),
        }
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dcp.save(state, checkpoint_id=str(ckpt_dir))
    else:
        # Single-process: just torch.save it.
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            ckpt_dir / "state.pt",
        )

    return str(ckpt_dir)


def load(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
) -> int:
    """Load model + optimizer state from `ckpt_dir`. Returns the saved step.

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
        }
        dcp.load(state, checkpoint_id=str(ckpt_path))
        step = int(state["step"].item())
    else:
        ckpt = torch.load(ckpt_path / "state.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])

    return step


def latest(out_dir: str) -> Optional[str]:
    """Find the latest checkpoint in `out_dir`, or None if there isn't one."""
    p = Path(out_dir)
    if not p.exists():
        return None
    ckpts = sorted(p.glob("step_*"))
    return str(ckpts[-1]) if ckpts else None


def cleanup_old(out_dir: str, keep_last: int = 3) -> None:
    """Delete all but the `keep_last` most recent checkpoints. Idempotent."""
    p = Path(out_dir)
    if not p.exists():
        return
    ckpts = sorted(p.glob("step_*"))
    for old in ckpts[:-keep_last]:
        # Recursive delete for the DCP directory case.
        import shutil
        shutil.rmtree(old, ignore_errors=True)
