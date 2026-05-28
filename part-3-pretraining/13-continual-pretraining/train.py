"""The Module 13 continual-pretraining entrypoint. Launch via `torchrun`.

```
# Single A100 (still through torchrun) — the canonical demo
torchrun --standalone --nproc_per_node=1 train.py --config=configs/cpt_qwen3_0.6b.yaml

# 8-GPU node — drop grad_accum to keep the effective batch (and replay mix) constant
torchrun --standalone --nproc_per_node=8 train.py \\
    --config=configs/cpt_qwen3_0.6b.yaml --training.grad_accum=1

# The "domain only" ablation that demonstrates catastrophic forgetting
torchrun --standalone --nproc_per_node=1 train.py \\
    --config=configs/cpt_qwen3_0.6b.yaml --data.replay_ratio=0
```

The composition is deliberately identical to Modules 11/15 — continual
pretraining reuses the pretraining substrate. What is CPT-specific lives in two
places only:

  data:   `make_mixed_loader` — a domain+replay token-ratio mixture (this module)
  config: a low RE-WARM peak LR (optimizer.lr) + warmup/decay schedule

Everything else — the FSDP wrap, the AdamW build, the train step, the
checkpointing — is the same machinery you already built. Data resume is
*derived* from the optimizer step (`mixed.seek(step * grad_accum)`), so it is
bit-exact and independent of DataLoader prefetch (see `data.py`).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_model, count_params
from optim import build_optimizer
from schedule import build_scheduler
from data import make_mixed_loader
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt, cleanup_old as cleanup_old_ckpts
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed
from efficiency import apply_activation_checkpointing, gpu_utilization_snapshot
from loop import train_step


_GPU_PEAK_TFLOPS = {"A100": 312, "H100": 990, "V100": 125}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Module 13 continual-pretraining entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/cpt_qwen3_0.6b.yaml\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/cpt_qwen3_0.6b.yaml --data.replay_ratio=0  # forgetting ablation\n"
        ),
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu", type=str, default="A100", choices=list(_GPU_PEAK_TFLOPS.keys()),
                   help="GPU class for the MFU estimate (default: A100)")
    args, extra = p.parse_known_args(argv)
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
        else:
            raise SystemExit(
                f"unrecognized arg: {tok!r}. Use --section.field=value, "
                "e.g. --optimizer.lr=3e-5")
    return args, overrides


def _log_startup(rinfo, cfg, model, mixed, tokens_per_step, device):
    if not rinfo.is_main:
        return
    counts = count_params(model)
    replay_pct = cfg.data.replay_ratio * 100
    print("=" * 70)
    print("Module 13 — Continual Pretraining")
    print("=" * 70)
    print(f"world_size: {rinfo.world_size}   device: {device}   dtype: {cfg.training.dtype}")
    print(f"base model: {cfg.model.name}   params: {counts['total']/1e6:.1f}M "
          f"(rank-local under FSDP; ×world for the global total)")
    print(f"RE-WARM peak lr: {cfg.optimizer.lr:.2e}   betas={cfg.optimizer.betas}   "
          f"wd={cfg.optimizer.weight_decay}")
    print(f"schedule:   {cfg.schedule.type}  warmup={cfg.schedule.warmup_steps}  "
          f"min_lr_ratio={cfg.schedule.min_lr_ratio}")
    print(f"data mix:   domain={mixed.domain_samples:,} samples  "
          f"replay={mixed.replay_samples:,} samples  "
          f"replay_ratio={cfg.data.replay_ratio:.2f} ({replay_pct:.0f}% of tokens)")
    if not mixed.use_replay:
        print("            !! replay DISABLED — this is the catastrophic-forgetting ablation")
    print(f"training:   total_steps={cfg.training.total_steps}  "
          f"grad_accum={cfg.training.grad_accum}  seq_len={cfg.data.seq_len}  "
          f"tokens/step={tokens_per_step:,}")
    print(f"            activation_checkpointing={cfg.training.activation_checkpointing}  "
          f"fused_ce={cfg.training.use_fused_ce}")
    print(f"checkpoints: {cfg.training.checkpoint_dir}  every {cfg.training.save_every}")
    print("=" * 70, flush=True)


def main(argv=None):
    args, overrides = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- 1. Config -------------------------------------------------------
    cfg = load_yaml(args.config) if args.config else TrainConfig()
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()

    # ---- 2. Distributed --------------------------------------------------
    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 3. Data: the domain + replay mixture ----------------------------
    mixed = make_mixed_loader(cfg.data, num_replicas=rinfo.world_size, rank=rinfo.rank)

    # ---- 4. Model + efficiency + sharding --------------------------------
    model = build_model(cfg.model, fused_ce=cfg.training.use_fused_ce).to(device)
    if cfg.training.activation_checkpointing:
        apply_activation_checkpointing(model)
    model = apply_fsdp(model, dtype=cfg.training.dtype)

    tokens_per_step = (cfg.data.batch_size_per_device * cfg.data.seq_len
                       * cfg.training.grad_accum * rinfo.world_size)
    _log_startup(rinfo, cfg, model, mixed, tokens_per_step, device)

    # ---- 5. Optimizer + scheduler ----------------------------------------
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.schedule)

    # ---- 6. Optional W&B (rank-0 only) -----------------------------------
    wandb_run = None
    if rinfo.is_main and cfg.training.wandb_project:
        try:
            import wandb
            init_kwargs = dict(project=cfg.training.wandb_project,
                               name=cfg.training.wandb_run_name or None,
                               config=cfg.to_dict(),
                               tags=list(cfg.training.wandb_tags) or None,
                               settings=wandb.Settings(console="off"))
            if cfg.training.wandb_entity:
                init_kwargs["entity"] = cfg.training.wandb_entity
            if cfg.training.wandb_run_id:
                init_kwargs["id"] = cfg.training.wandb_run_id
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)
            print(f"[rank 0] wandb run: {wandb_run.url}", flush=True)
        except ImportError:
            print("[rank 0] wandb requested but not installed; stdout only")
        except Exception as e:
            print(f"[rank 0] wandb.init() failed ({e}); stdout only", flush=True)

    # ---- 7. Resume -------------------------------------------------------
    # start_step = optimizer updates already completed. Restore
    # model/optimizer/scheduler, then seek the data mixture to the SAME point by
    # arithmetic: k micro-batches = start_step * grad_accum (prefetch-safe).
    start_step = 0
    resume = cfg.training.resume_from or latest_ckpt(cfg.training.checkpoint_dir)
    if resume:
        start_step = load_ckpt(model, optimizer, resume, scheduler=scheduler)
        if rinfo.is_main:
            print(f"[rank 0] resumed from {resume} at step {start_step}", flush=True)
    mixed.seek(start_step * cfg.training.grad_accum)
    batch_iter = iter(mixed)

    # ---- 8. Train --------------------------------------------------------
    model.train()
    if rinfo.is_main:
        print("[rank 0] entering continual-pretraining loop", flush=True)
    t_start = time.time()
    last_log_time = t_start
    peak_tflops = _GPU_PEAK_TFLOPS[args.gpu]
    n_params = sum(p.numel() for p in model.parameters())
    if _dist_initialized():
        t = torch.tensor(float(n_params), device=device)
        torch.distributed.all_reduce(t)
        n_params = int(t.item())

    for step in range(start_step, cfg.training.total_steps):
        loss, grad_norm = train_step(
            model=model, optimizer=optimizer, batch_iter=batch_iter,
            grad_accum=cfg.training.grad_accum, grad_clip=cfg.training.grad_clip,
            dtype=cfg.training.dtype, device=device,
            fused_ce=cfg.training.use_fused_ce,
        )
        scheduler.step()

        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt = now - last_log_time
            steps_window = cfg.training.log_every if step > start_step else 1
            tok_per_sec = (steps_window * tokens_per_step) / max(dt, 1e-9)
            flops = 6.0 * n_params * tokens_per_step
            mfu = (flops * steps_window / max(dt, 1e-9)
                   / (peak_tflops * 1e12 * rinfo.world_size))
            lr_now = optimizer.param_groups[0]["lr"]
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
                wandb_run.log({"loss": loss, "grad_norm": grad_norm, "lr": lr_now,
                               "tokens_per_second": tok_per_sec, "mfu": mfu, "step": step})

        steps_done = step + 1
        if steps_done % cfg.training.save_every == 0:
            ckpt_path = save_ckpt(model, optimizer, steps_done,
                                  cfg.training.checkpoint_dir, scheduler=scheduler)
            if rinfo.is_main:
                deleted = cleanup_old_ckpts(cfg.training.checkpoint_dir,
                                            keep_last=cfg.training.keep_last_n_checkpoints,
                                            milestone_every=cfg.training.milestone_every)
                msg = f"[rank 0] saved checkpoint -> {ckpt_path}"
                if deleted:
                    msg += f"  (pruned {len(deleted)} old)"
                print(msg, flush=True)

    # ---- 9. Final save ---------------------------------------------------
    if cfg.training.total_steps > 0 and cfg.training.total_steps % cfg.training.save_every != 0:
        final_path = save_ckpt(model, optimizer, cfg.training.total_steps,
                               cfg.training.checkpoint_dir, scheduler=scheduler)
        if rinfo.is_main:
            cleanup_old_ckpts(cfg.training.checkpoint_dir,
                              keep_last=cfg.training.keep_last_n_checkpoints,
                              milestone_every=cfg.training.milestone_every)
            print(f"[rank 0] saved final checkpoint -> {final_path}", flush=True)

    if rinfo.is_main:
        elapsed = time.time() - t_start
        print(f"[rank 0] continual pretraining finished in {elapsed:.1f}s", flush=True)
        print("[rank 0] now measure BOTH axes:\n"
              "  python eval.py --checkpoint=<final ckpt> --config=" + (args.config or "configs/cpt_qwen3_0.6b.yaml") + " \\\n"
              "      --qa results/corpus/qa_heldout.jsonl --forgetting", flush=True)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed()


def _dist_initialized() -> bool:
    try:
        return torch.distributed.is_initialized()
    except Exception:
        return False


if __name__ == "__main__":
    main()
