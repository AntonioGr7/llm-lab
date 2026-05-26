"""The Module 11 pretraining entrypoint. Launch via `torchrun`.

```
# Single A100 (still goes through torchrun)
torchrun --standalone --nproc_per_node=1 train.py --config=configs/demo_a100.yaml

# Single H100 — use the H100-tuned config + --gpu=H100 for accurate MFU
torchrun --standalone --nproc_per_node=1 train.py \\
    --config=configs/demo_h100.yaml --gpu=H100

# Single node, 8 GPUs — same config, adjust grad_accum to keep tokens/step constant
torchrun --standalone --nproc_per_node=8 train.py \\
    --config=configs/demo_a100.yaml --training.grad_accum=1

# Multi-node — 4 nodes x 8 GPUs
torchrun --nnodes=4 --nproc_per_node=8 \\
    --rdzv_backend=c10d --rdzv_endpoint=node-0:29500 \\
    train.py --config=configs/demo_a100.yaml --training.grad_accum=1
```

The composition:

  config.py:    load YAML, apply CLI overrides, sync cross-section invariants
  fsdp_setup:   init_distributed + apply_fsdp (Module 08)
  efficiency:   apply_activation_checkpointing BEFORE apply_fsdp (Module 10)
  model:        build_model (Qwen3 default, Module 08)
  optim:        build_optimizer (AdamW or Muon, Module 08)
  schedule:     build_scheduler (warmup-cosine default, Module 09)
  data:         make_dataloader (synthetic or fineweb-edu, this module)
  checkpoint:   DCP save/load (Module 08)
  loop:         train_step (Module 08, with no_sync + clipping)

CLI overrides use dotted paths: `--training.grad_accum=1`, `--model.d_model=512`.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_model, count_params
from optim import build_optimizer
from schedule import build_scheduler
from data import make_dataloader, cycle
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt, cleanup_old as cleanup_old_ckpts
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed
from efficiency import apply_activation_checkpointing, gpu_utilization_snapshot
from loop import train_step


# A100 BF16 peak FLOPS (TFLOPs). Used to compute MFU. Override via env if running
# on a different GPU (e.g. H100 = 990, V100 = 125 for FP16 only).
_GPU_PEAK_TFLOPS = {"A100": 312, "H100": 990, "V100": 125}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Module 11 pretraining entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/demo_a100.yaml\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/demo_h100.yaml --gpu=H100\n"
            "  torchrun ... train.py --config=configs/demo_a100.yaml "
            "--training.total_steps=100 --training.grad_accum=1\n"
        ),
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML config path (e.g. configs/demo_a100.yaml or configs/demo_h100.yaml)")
    p.add_argument("--gpu", type=str, default="A100",
                   choices=list(_GPU_PEAK_TFLOPS.keys()),
                   help="GPU class for MFU calculation (default: A100)")
    # Everything else: dotted overrides. We accept any `--section.field=value`.
    args, extra = p.parse_known_args(argv)
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
        else:
            raise SystemExit(
                f"unrecognized arg: {tok!r}. "
                "Use --section.field=value form, e.g. --training.total_steps=100"
            )
    return args, overrides


def _log_startup(rinfo, cfg: TrainConfig, model, device: str) -> None:
    """Rank-0 startup summary. Catches misconfiguration before the run starts."""
    if not rinfo.is_main:
        return
    counts = count_params(model)
    tokens_per_step = (
        cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )
    print("=" * 64)
    print("Module 11 — Pretraining in Practice")
    print("=" * 64)
    print(f"world_size: {rinfo.world_size}   device: {device}   dtype: {cfg.training.dtype}")
    print(f"model:      d={cfg.model.d_model} L={cfg.model.n_layers} "
          f"H={cfg.model.n_heads} kv={cfg.model.n_kv_heads} "
          f"vocab={cfg.model.vocab_size}   params: {counts['total']/1e6:.1f}M")
    print(f"optimizer:  {cfg.optimizer.type}  peak_lr={cfg.optimizer.lr}  "
          f"betas={cfg.optimizer.betas}  wd={cfg.optimizer.weight_decay}")
    print(f"schedule:   {cfg.schedule.type}  warmup={cfg.schedule.warmup_steps}  "
          f"min_lr_ratio={cfg.schedule.min_lr_ratio}")
    print(f"data:       source={cfg.data.source}  seq_len={cfg.data.seq_len}  "
          f"per-device-bs={cfg.data.batch_size_per_device}")
    print(f"training:   total_steps={cfg.training.total_steps}  "
          f"grad_accum={cfg.training.grad_accum}  "
          f"tokens/step={tokens_per_step:,}  "
          f"total_tokens≈{tokens_per_step*cfg.training.total_steps/1e9:.2f}B")
    print(f"            grad_clip={cfg.training.grad_clip}  "
          f"activation_checkpointing={cfg.training.activation_checkpointing}  "
          f"fused_ce={cfg.training.use_fused_ce}")
    print(f"checkpoints: {cfg.training.checkpoint_dir}  every {cfg.training.save_every}")
    print("=" * 64, flush=True)


def main(argv=None):
    args, overrides = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- 1. Load + override config ----------------------------------------
    cfg = load_yaml(args.config) if args.config else TrainConfig()
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()      # propagate training.total_steps -> schedule.total_steps

    # ---- 2. Distributed init (no-op for non-torchrun launches) -----------
    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 3. Build model, apply efficiency + sharding ---------------------
    # Order matters: activation_checkpointing wraps each decoder layer; FSDP
    # then shards the wrapped layer. Reverse order silently breaks FSDP.
    model = build_model(cfg.model, fused_ce=cfg.training.use_fused_ce).to(device)
    if cfg.training.activation_checkpointing:
        apply_activation_checkpointing(model)
    model = apply_fsdp(model, dtype=cfg.training.dtype)

    _log_startup(rinfo, cfg, model, device)

    # ---- 4. Optimizer + scheduler + dataloader ---------------------------
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.schedule)
    loader = make_dataloader(cfg.data, vocab_size=cfg.model.vocab_size)
    batch_iter = cycle(loader)

    # ---- 5. Optional W&B (rank-0 only) -----------------------------------
    # Setup: `pip install wandb` then `wandb login` once (or export
    # WANDB_API_KEY=...). Set `training.wandb_project` to enable.
    # On resume, set `training.wandb_run_id` to the previous run's id so
    # W&B appends to that run instead of starting a new one.
    wandb_run = None
    if rinfo.is_main and cfg.training.wandb_project:
        try:
            import wandb
            # console="off": don't let wandb redirect stdout/stderr. Under torchrun
            # (and in Lightning AI / Docker / other non-TTY environments) wandb's
            # default fd-level redirect deadlocks the training loop after init.
            init_kwargs = dict(
                project=cfg.training.wandb_project,
                name=cfg.training.wandb_run_name or None,
                config=cfg.to_dict(),
                tags=list(cfg.training.wandb_tags) or None,
                settings=wandb.Settings(console="off"),
            )
            if cfg.training.wandb_entity:
                init_kwargs["entity"] = cfg.training.wandb_entity
            if cfg.training.wandb_run_id:
                init_kwargs["id"] = cfg.training.wandb_run_id
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)
            print(f"[rank 0] wandb run: {wandb_run.url}", flush=True)
        except ImportError:
            print("[rank 0] wandb requested but not installed (pip install wandb); "
                  "logging to stdout only")
        except Exception as e:
            print(f"[rank 0] wandb.init() failed ({e}); logging to stdout only. "
                  "Did you run `wandb login` or export WANDB_API_KEY?")

    # ---- 6. Resume if requested ------------------------------------------
    # Semantics: `start_step` = number of optimizer updates already completed.
    # The next iteration applies update #`start_step+1` (0-indexed: step=start_step).
    # Scheduler state is loaded directly from the checkpoint (not replayed).
    start_step = 0
    resume = cfg.training.resume_from or latest_ckpt(cfg.training.checkpoint_dir)
    if resume:
        start_step = load_ckpt(model, optimizer, resume, scheduler=scheduler)
        if rinfo.is_main:
            print(f"[rank 0] resumed from {resume} at step {start_step}  "
                  f"(scheduler.last_epoch={scheduler.last_epoch}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e})", flush=True)

    # ---- 7. Train --------------------------------------------------------
    model.train()
    if rinfo.is_main:
        print("[rank 0] entering training loop; first batch may take 30-60s "
              "while DataLoader workers tokenize the initial pack", flush=True)
    t_start = time.time()
    last_log_time = t_start
    tokens_per_step = (
        cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )
    peak_tflops = _GPU_PEAK_TFLOPS[args.gpu]
    n_params = sum(p.numel() for p in model.parameters())   # rough; sharded params count once
    if dist_initialized():
        # Sum across ranks to get total model param count for FLOP math.
        n_params_t = torch.tensor(float(n_params), device=device)
        torch.distributed.all_reduce(n_params_t)
        n_params = int(n_params_t.item())

    for step in range(start_step, cfg.training.total_steps):
        loss, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            batch_iter=batch_iter,
            grad_accum=cfg.training.grad_accum,
            grad_clip=cfg.training.grad_clip,
            dtype=cfg.training.dtype,
            device=device,
            fused_ce=cfg.training.use_fused_ce,
        )
        scheduler.step()

        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt_window = now - last_log_time
            steps_window = cfg.training.log_every if step > start_step else 1
            tok_per_sec = (steps_window * tokens_per_step) / max(dt_window, 1e-9)
            # MFU: training FLOPs / wall FLOPs. 6N approximation.
            flops_per_step = 6.0 * n_params * tokens_per_step
            mfu = (flops_per_step * steps_window / max(dt_window, 1e-9)
                   / (peak_tflops * 1e12 * rinfo.world_size))
            lr_now = optimizer.param_groups[0]["lr"]
            # Live GPU utilization (rank 0's device). SM% is the headline:
            # if it's ~90%+ you're compute-bound and tuning batch/checkpointing
            # won't buy throughput. If it's <60% the GPU is starving — look
            # at the dataloader or kernel launch overhead, not at HBM headroom.
            gpu = gpu_utilization_snapshot(0)
            gpu_str = ""
            if gpu is not None:
                gpu_str = (f"  sm {gpu['sm_util']:.0f}%  "
                           f"mem {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.0f}GB")
            print(f"step {step:6d}  loss {loss:.4f}  grad_norm {grad_norm:.3f}  "
                  f"lr {lr_now:.2e}  tok/s {tok_per_sec/1e3:.1f}k  mfu {mfu*100:.1f}%"
                  f"{gpu_str}", flush=True)
            last_log_time = now

            if wandb_run is not None:
                log_payload = {
                    "loss": loss, "grad_norm": grad_norm, "lr": lr_now,
                    "tokens_per_second": tok_per_sec, "mfu": mfu, "step": step,
                }
                if gpu is not None:
                    log_payload.update({
                        "gpu/sm_util": gpu["sm_util"],
                        "gpu/mem_bw_util": gpu["mem_bw_util"],
                        "gpu/mem_used_gb": gpu["mem_used_gb"],
                        "gpu/mem_used_pct": gpu["mem_used_pct"],
                    })
                wandb_run.log(log_payload)

        # Save AFTER scheduler.step() has run. The "step" we persist is the
        # number of updates completed (step + 1), not the loop counter. That
        # way a resume with start_step=N correctly starts at update N+1 instead
        # of redoing update N. See checkpoint.save() docstring.
        steps_done = step + 1
        if steps_done % cfg.training.save_every == 0:
            ckpt_path = save_ckpt(model, optimizer, steps_done,
                                 cfg.training.checkpoint_dir, scheduler=scheduler)
            if rinfo.is_main:
                deleted = cleanup_old_ckpts(
                    cfg.training.checkpoint_dir,
                    keep_last=cfg.training.keep_last_n_checkpoints,
                    milestone_every=cfg.training.milestone_every,
                )
                msg = f"[rank 0] saved checkpoint -> {ckpt_path}"
                if deleted:
                    msg += f"  (pruned {len(deleted)} old)"
                print(msg, flush=True)

    # ---- 8. Final save + cleanup -----------------------------------------
    # If the last loop iteration didn't land on a save_every boundary, write
    # one final checkpoint reflecting all total_steps updates.
    if cfg.training.total_steps > 0:
        already_saved = cfg.training.total_steps % cfg.training.save_every == 0
        if not already_saved:
            final_path = save_ckpt(model, optimizer, cfg.training.total_steps,
                                  cfg.training.checkpoint_dir, scheduler=scheduler)
            if rinfo.is_main:
                cleanup_old_ckpts(
                    cfg.training.checkpoint_dir,
                    keep_last=cfg.training.keep_last_n_checkpoints,
                    milestone_every=cfg.training.milestone_every,
                )
                print(f"[rank 0] saved final checkpoint -> {final_path}", flush=True)

    if rinfo.is_main:
        elapsed = time.time() - t_start
        print(f"[rank 0] training finished in {elapsed:.1f}s "
              f"({elapsed / max(cfg.training.total_steps - start_step, 1):.2f}s/step avg)",
              flush=True)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed()


def dist_initialized() -> bool:
    try:
        return torch.distributed.is_initialized()
    except Exception:
        return False


if __name__ == "__main__":
    main()
