"""The Module 17 DPO/IPO entrypoint. Launch via `torchrun`.

```
# Single A100-80GB (still goes through torchrun) — the canonical demo
torchrun --standalone --nproc_per_node=1 train.py --config=configs/dpo_qwen3_1.7b.yaml

# 8-GPU node — drop grad_accum to keep the effective batch constant
torchrun --standalone --nproc_per_node=8 train.py \\
    --config=configs/dpo_qwen3_1.7b.yaml --training.grad_accum=2

# Dev smoke test — points at Qwen/Qwen3-0.6B (already chat) for a cheap dry-run
torchrun --standalone --nproc_per_node=1 train.py --config=configs/dpo_demo.yaml
```

The composition mirrors Module 15 — same FSDP wrap, same AdamW, same DCP
checkpoint, same schedule. What changes:

  model:    build_policy + build_reference — TWO models loaded.
  data:     PreferenceDataset (chosen/rejected pairs).
  loop:     train_step does TWO forwards (policy + reference no_grad) per
            micro-batch and reports a richer metrics dict.

The activation-checkpointing + FSDP wrap order is the same as Module 15:
AC wraps each decoder layer FIRST, then FSDP wraps the wrapped layers. The
reference is FSDP-wrapped for sharding parity on multi-GPU but does NOT get
AC (no backward, no memory to save).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_policy, build_reference, count_params
from optim import build_optimizer
from schedule import build_scheduler
from data import make_dataloader, cycle
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt, cleanup_old as cleanup_old_ckpts
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed
from efficiency import apply_activation_checkpointing, gpu_utilization_snapshot
from loop import train_step


_GPU_PEAK_TFLOPS = {"A100": 312, "H100": 990, "V100": 125}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Module 17 Preference Optimization (DPO/IPO) entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/dpo_qwen3_1.7b.yaml\n"
            "  torchrun --standalone --nproc_per_node=8 train.py "
            "--config=configs/dpo_qwen3_1.7b.yaml --training.grad_accum=2\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/dpo_demo.yaml\n"
            "  # Try IPO instead of DPO with one flag override:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/dpo_qwen3_1.7b.yaml --preference.loss_type=ipo\n"
        ),
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu", type=str, default="A100",
                   choices=list(_GPU_PEAK_TFLOPS.keys()))
    args, extra = p.parse_known_args(argv)
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
        else:
            raise SystemExit(
                f"unrecognized arg: {tok!r}. "
                "Use --section.field=value form, e.g. --preference.beta=0.05"
            )
    return args, overrides


def _log_startup(rinfo, cfg: TrainConfig, policy, frac_chosen: float, frac_rejected: float, device: str) -> None:
    if not rinfo.is_main:
        return
    counts = count_params(policy)
    tokens_per_step = (
        # 2× because each pair = chosen + rejected
        2 * cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )
    print("=" * 64)
    print("Module 17 — Preference Optimization (DPO/IPO)")
    print("=" * 64)
    print(f"world_size: {rinfo.world_size}   device: {device}   dtype: {cfg.training.dtype}")
    print(f"policy:    {cfg.model.name}   params: {counts['total']/1e6:.1f}M "
          f"(rank-local count under FSDP; ×world for the global total)")
    print(f"reference: {cfg.model.resolved_ref_name()}   "
          f"({'== policy (default)' if cfg.model.ref_name == '' else 'overridden'})")
    print(f"loss:      {cfg.preference.loss_type}   beta={cfg.preference.beta}   "
          f"label_smoothing={cfg.preference.label_smoothing}   "
          f"reference_free={cfg.preference.reference_free}")
    print(f"optimizer: {cfg.optimizer.type}  peak_lr={cfg.optimizer.lr}  "
          f"betas={cfg.optimizer.betas}  wd={cfg.optimizer.weight_decay}")
    print(f"schedule:  {cfg.schedule.type}  warmup={cfg.schedule.warmup_steps}  "
          f"min_lr_ratio={cfg.schedule.min_lr_ratio}")
    print(f"data:      source={cfg.data.source}  split={cfg.data.split}  "
          f"seq_len={cfg.data.seq_len}  per-device-pairs={cfg.data.batch_size_per_device}  "
          f"chosen-resp-frac={frac_chosen:.1%}  rejected-resp-frac={frac_rejected:.1%}")
    print(f"training:  total_steps={cfg.training.total_steps}  "
          f"grad_accum={cfg.training.grad_accum}  "
          f"eff_pairs={cfg.data.batch_size_per_device*cfg.training.grad_accum*rinfo.world_size}  "
          f"tokens/step={tokens_per_step:,}")
    print(f"           grad_clip={cfg.training.grad_clip}  "
          f"activation_checkpointing={cfg.training.activation_checkpointing}")
    print(f"checkpoints: {cfg.training.checkpoint_dir}  every {cfg.training.save_every}")
    print("=" * 64, flush=True)


def main(argv=None):
    args, overrides = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- 1. Load + override config ----------------------------------------
    cfg = load_yaml(args.config) if args.config else TrainConfig()
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()

    # ---- 2. Distributed init ----------------------------------------------
    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 3. Data first (so the banner can report mask fractions) ---------
    loader = make_dataloader(cfg.data, tokenizer_name=cfg.model.name)
    frac_chosen = getattr(loader.dataset, "frac_chosen_response_tokens", float("nan"))
    frac_rejected = getattr(loader.dataset, "frac_rejected_response_tokens", float("nan"))
    batch_iter = cycle(loader)

    # ---- 4. Build BOTH models, apply efficiency + sharding ---------------
    # Policy: full FT, AC + FSDP.
    policy = build_policy(cfg.model).to(device)
    if cfg.training.activation_checkpointing:
        apply_activation_checkpointing(policy)
    policy = apply_fsdp(policy, dtype=cfg.training.dtype)

    # Reference: FSDP for sharding only (no AC — no backward).
    reference = build_reference(cfg.model).to(device)
    reference = apply_fsdp(reference, dtype=cfg.training.dtype)

    _log_startup(rinfo, cfg, policy, frac_chosen, frac_rejected, device)

    # ---- 5. Optimizer + scheduler (POLICY only) --------------------------
    optimizer = build_optimizer(policy, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.schedule)

    # ---- 6. Optional W&B (rank-0 only) -----------------------------------
    wandb_run = None
    if rinfo.is_main and cfg.training.wandb_project:
        try:
            import wandb
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
            print("[rank 0] wandb requested but not installed; logging to stdout only")
        except Exception as e:
            print(f"[rank 0] wandb.init() failed ({e}); logging to stdout only.")

    # ---- 7. Resume if requested ------------------------------------------
    start_step = 0
    resume = cfg.training.resume_from or latest_ckpt(cfg.training.checkpoint_dir)
    if resume:
        start_step = load_ckpt(policy, optimizer, resume, scheduler=scheduler)
        if rinfo.is_main:
            print(f"[rank 0] resumed from {resume} at step {start_step}  "
                  f"(scheduler.last_epoch={scheduler.last_epoch}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e})", flush=True)

    # ---- 8. Train --------------------------------------------------------
    policy.train()
    reference.eval()
    if rinfo.is_main:
        print("[rank 0] entering training loop", flush=True)
    t_start = time.time()
    last_log_time = t_start
    tokens_per_step = (
        2 * cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )
    peak_tflops = _GPU_PEAK_TFLOPS[args.gpu]
    n_params = sum(p.numel() for p in policy.parameters())
    if dist_initialized():
        n_params_t = torch.tensor(float(n_params), device=device)
        torch.distributed.all_reduce(n_params_t)
        n_params = int(n_params_t.item())

    for step in range(start_step, cfg.training.total_steps):
        stats = train_step(
            policy=policy,
            reference=reference,
            optimizer=optimizer,
            batch_iter=batch_iter,
            grad_accum=cfg.training.grad_accum,
            grad_clip=cfg.training.grad_clip,
            pref_cfg=cfg.preference,
            dtype=cfg.training.dtype,
            device=device,
        )
        scheduler.step()

        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt_window = now - last_log_time
            steps_window = cfg.training.log_every if step > start_step else 1
            tok_per_sec = (steps_window * tokens_per_step) / max(dt_window, 1e-9)
            # MFU: 6N per parameter per token. Double the tokens because policy+reference
            # both forward; only the policy backwards (so 2N + 4N = 6N per token still
            # under-estimates by the ref's forward, but it's within ~15% and consistent
            # with Module 11's accounting).
            flops_per_step = 6.0 * n_params * tokens_per_step
            mfu = (flops_per_step * steps_window / max(dt_window, 1e-9)
                   / (peak_tflops * 1e12 * rinfo.world_size))
            lr_now = optimizer.param_groups[0]["lr"]
            gpu = gpu_utilization_snapshot(0)
            gpu_str = ""
            if gpu is not None:
                gpu_str = (f"  sm {gpu['sm_util']:.0f}%  "
                           f"mem {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.0f}GB")
            print(f"step {step:6d}  loss {stats['loss']:.4f}  "
                  f"margin {stats['margin']:+.3f}  acc {stats['accuracy']:.1%}  "
                  f"r_chosen {stats['chosen_rewards']:+.3f}  "
                  f"r_rejected {stats['rejected_rewards']:+.3f}  "
                  f"grad_norm {stats['grad_norm']:.3f}  lr {lr_now:.2e}  "
                  f"tok/s {tok_per_sec/1e3:.1f}k  mfu {mfu*100:.1f}%"
                  f"{gpu_str}", flush=True)
            last_log_time = now

            if wandb_run is not None:
                log_payload = {
                    "loss": stats["loss"],
                    "margin": stats["margin"],
                    "accuracy": stats["accuracy"],
                    "chosen_rewards": stats["chosen_rewards"],
                    "rejected_rewards": stats["rejected_rewards"],
                    "grad_norm": stats["grad_norm"],
                    "lr": lr_now,
                    "tokens_per_second": tok_per_sec,
                    "mfu": mfu,
                    "step": step,
                }
                if gpu is not None:
                    log_payload.update({
                        "gpu/sm_util": gpu["sm_util"],
                        "gpu/mem_used_gb": gpu["mem_used_gb"],
                        "gpu/mem_used_pct": gpu["mem_used_pct"],
                    })
                wandb_run.log(log_payload)

        steps_done = step + 1
        if steps_done % cfg.training.save_every == 0:
            ckpt_path = save_ckpt(policy, optimizer, steps_done,
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

    # ---- 9. Final save + cleanup -----------------------------------------
    if cfg.training.total_steps > 0:
        already_saved = cfg.training.total_steps % cfg.training.save_every == 0
        if not already_saved:
            final_path = save_ckpt(policy, optimizer, cfg.training.total_steps,
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
        print("[rank 0] next: eval the policy and compare against the reference:\n"
              "  python eval.py --checkpoint=<final ckpt> "
              "--prompts \"Write a haiku about Python\"", flush=True)
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
