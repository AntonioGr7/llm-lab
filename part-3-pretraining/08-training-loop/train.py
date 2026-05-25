"""Entry point for Part 3 pretraining. Launch via `torchrun`.

```
# Single GPU (still goes through torchrun)
torchrun --standalone --nproc_per_node=1 train.py --total_steps=20

# Single node, 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --config=cfg.yaml

# Multi-node
torchrun --nnodes=4 --nproc_per_node=8 \\
         --rdzv_backend=c10d --rdzv_endpoint=$HEAD_IP:29500 \\
         train.py --config=cfg.yaml
```

The script:
1. Reads a YAML config (or defaults), with CLI overrides for top-level fields.
2. Initializes distributed (from torchrun env vars).
3. Builds model, applies FSDP, builds optimizer + dataloader.
4. Resumes from a checkpoint if `--resume_from` was passed (or the dir contains one).
5. Runs the training loop with logging and periodic checkpointing.
6. Cleans up.

Module 09 will plug in the LR scheduler after `build_optimizer`.
Module 10 will refine the FSDP setup. Module 11 will swap `SyntheticDataset`
for FineWeb-Edu and ship the actual demo config.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from config import TrainConfig, load_yaml
from model import build_model, count_params
from optim import build_optimizer
from data import make_dataloader, cycle
from loop import train_step
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed


def parse_args():
    p = argparse.ArgumentParser(description="Part 3 pretraining entrypoint")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    # Quick CLI overrides for the most common knobs (full configs go through YAML).
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--batch_size_per_device", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "muon"],
                   help="optimizer type — 'adamw' (default, production-canonical) "
                        "or 'muon' (Newton-Schulz orthogonalized momentum on 2D weights)")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    return p.parse_args()


def apply_overrides(cfg: TrainConfig, args) -> None:
    """In-place: copy non-None CLI args into the nested config."""
    overrides = {
        "training.total_steps": args.total_steps,
        "training.log_every": args.log_every,
        "training.save_every": args.save_every,
        "training.checkpoint_dir": args.checkpoint_dir,
        "training.resume_from": args.resume_from,
        "training.grad_accum": args.grad_accum,
        "model.vocab_size": args.vocab_size,
        "model.d_model": args.d_model,
        "model.n_layers": args.n_layers,
        "data.batch_size_per_device": args.batch_size_per_device,
        "optimizer.lr": args.lr,
        "optimizer.type": args.optimizer,
    }
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, field = dotted.split(".")
        setattr(getattr(cfg, section), field, value)


def main():
    args = parse_args()

    # 1. Load + override config.
    cfg = load_yaml(args.config) if args.config else TrainConfig()
    apply_overrides(cfg, args)

    # 2. Init distributed (no-op for non-torchrun launches).
    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if rinfo.is_main:
        print(f"[rank 0] world_size={rinfo.world_size}  device={device}  dtype={cfg.training.dtype}")
        print(f"[rank 0] model: d={cfg.model.d_model} L={cfg.model.n_layers} "
              f"H={cfg.model.n_heads} kv={cfg.model.n_kv_heads} vocab={cfg.model.vocab_size}")
        print(f"[rank 0] training: total_steps={cfg.training.total_steps} "
              f"per-device-bs={cfg.data.batch_size_per_device} accum={cfg.training.grad_accum}")

    # 3. Build model, wrap with FSDP.
    model = build_model(cfg.model).to(device)
    if rinfo.is_main:
        counts = count_params(model)
        print(f"[rank 0] params total: {counts['total']/1e6:.1f}M")

    model = apply_fsdp(model, dtype=cfg.training.dtype)

    # 4. Optimizer + dataloader.
    optimizer = build_optimizer(model, cfg.optimizer)
    loader = make_dataloader(cfg.data, vocab_size=cfg.model.vocab_size)
    batch_iter = cycle(loader)

    # 5. Resume if requested.
    start_step = 0
    resume = cfg.training.resume_from or latest_ckpt(cfg.training.checkpoint_dir)
    if resume:
        start_step = load_ckpt(model, optimizer, resume)
        if rinfo.is_main:
            print(f"[rank 0] resumed from {resume} at step {start_step}")

    # 6. Train.
    model.train()
    t_start = time.time()
    last_log_time = t_start
    tokens_per_step = (
        cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )

    for step in range(start_step, cfg.training.total_steps):
        loss, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            batch_iter=batch_iter,
            grad_accum=cfg.training.grad_accum,
            grad_clip=cfg.training.grad_clip,
            dtype=cfg.training.dtype,
            device=device,
        )

        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt = now - last_log_time
            tok_per_sec = (cfg.training.log_every * tokens_per_step) / max(dt, 1e-9)
            print(f"step {step:6d}  loss {loss:.4f}  grad_norm {grad_norm:.3f}  "
                  f"tok/s {tok_per_sec/1e3:.1f}k")
            last_log_time = now

        if step > 0 and step % cfg.training.save_every == 0:
            ckpt_path = save_ckpt(model, optimizer, step, cfg.training.checkpoint_dir)
            if rinfo.is_main:
                print(f"[rank 0] saved checkpoint -> {ckpt_path}")

    # Final checkpoint.
    if cfg.training.total_steps > 0:
        ckpt_path = save_ckpt(model, optimizer, cfg.training.total_steps, cfg.training.checkpoint_dir)
        if rinfo.is_main:
            print(f"[rank 0] saved final checkpoint -> {ckpt_path}")

    if rinfo.is_main:
        total_time = time.time() - t_start
        print(f"[rank 0] training finished in {total_time:.1f}s")

    cleanup_distributed()


if __name__ == "__main__":
    main()
