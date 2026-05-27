"""The Module 14 SFT entrypoint. Launch via `torchrun`.

```
# Single A100-80GB (still goes through torchrun) — the canonical demo
torchrun --standalone --nproc_per_node=1 train.py --config=configs/sft_qwen3_1.7b.yaml

# 8-GPU node — drop grad_accum to keep the effective batch constant
torchrun --standalone --nproc_per_node=8 train.py \\
    --config=configs/sft_qwen3_1.7b.yaml --training.grad_accum=2

# Dev smoke test on your exported Module 11 checkpoint (minutes, any GPU)
torchrun --standalone --nproc_per_node=1 train.py --config=configs/sft_demo.yaml
```

The composition mirrors Module 11 — the whole point of the "framework
directory" promise is that SFT reuses the pretraining substrate. What
changes is upstream of the loop: the model is *loaded* not *built*, the
data carries an assistant-only loss mask, and the LR is ~30× lower.

  config.py:    load YAML, apply CLI overrides, sync cross-section invariants
  model:        build_model — load HF causal LM by name (this module)
  efficiency:   apply_activation_checkpointing BEFORE apply_fsdp (Module 10/11)
  fsdp_setup:   init_distributed + apply_fsdp (Module 11)
  optim:        build_optimizer (AdamW, this module)
  schedule:     build_scheduler (warmup-cosine, Module 09/11)
  data:         make_dataloader — ChatDataset with the loss mask (this module)
  checkpoint:   DCP save/load (Module 11)
  loop:         train_step (this module — attention-mask aware)

A note on data resume: SFT uses an in-memory map-style `ChatDataset` with a
`DistributedSampler`, not Module 12's indexed corpus. Resume restores the
model/optimizer/scheduler/step exactly, but the data sampler restarts its
shuffle from the current epoch rather than seeking to the exact sample. For
a 10k-example dataset cycled ~3×, that re-shuffle is immaterial; if you need
bit-exact data resume on a huge SFT mix, lift in Module 12's indexed pipeline.
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
from data import make_dataloader, cycle
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt, cleanup_old as cleanup_old_ckpts
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed
from efficiency import apply_activation_checkpointing, gpu_utilization_snapshot
from loop import train_step


# BF16 peak FLOPS (TFLOPs) for the MFU estimate. Override with --gpu.
_GPU_PEAK_TFLOPS = {"A100": 312, "H100": 990, "V100": 125}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Module 14 SFT entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/sft_qwen3_1.7b.yaml\n"
            "  torchrun --standalone --nproc_per_node=8 train.py "
            "--config=configs/sft_qwen3_1.7b.yaml --training.grad_accum=2\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/sft_demo.yaml\n"
        ),
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML config path (e.g. configs/sft_qwen3_1.7b.yaml)")
    p.add_argument("--gpu", type=str, default="A100",
                   choices=list(_GPU_PEAK_TFLOPS.keys()),
                   help="GPU class for MFU calculation (default: A100)")
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


def _log_startup(rinfo, cfg: TrainConfig, model, frac_assistant: float, device: str) -> None:
    """Rank-0 startup summary. Catches misconfiguration before the run starts."""
    if not rinfo.is_main:
        return
    counts = count_params(model)
    # tokens/step counts every position processed (incl. padding) — the right
    # denominator for throughput/MFU. Loss is applied to assistant tokens only.
    tokens_per_step = (
        cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )
    print("=" * 64)
    print("Module 14 — Supervised Fine-Tuning")
    print("=" * 64)
    print(f"world_size: {rinfo.world_size}   device: {device}   dtype: {cfg.training.dtype}")
    print(f"base model: {cfg.model.name}   params: {counts['total']/1e6:.1f}M "
          f"(rank-local count under FSDP; ×world for the global total)")
    print(f"optimizer:  {cfg.optimizer.type}  peak_lr={cfg.optimizer.lr}  "
          f"betas={cfg.optimizer.betas}  wd={cfg.optimizer.weight_decay}")
    print(f"schedule:   {cfg.schedule.type}  warmup={cfg.schedule.warmup_steps}  "
          f"min_lr_ratio={cfg.schedule.min_lr_ratio}")
    print(f"data:       source={cfg.data.source}  seq_len={cfg.data.seq_len}  "
          f"per-device-bs={cfg.data.batch_size_per_device}  "
          f"assistant-token frac={frac_assistant:.1%}")
    print(f"training:   total_steps={cfg.training.total_steps}  "
          f"grad_accum={cfg.training.grad_accum}  "
          f"eff_batch={cfg.data.batch_size_per_device*cfg.training.grad_accum*rinfo.world_size}  "
          f"tokens/step={tokens_per_step:,}")
    print(f"            grad_clip={cfg.training.grad_clip}  "
          f"activation_checkpointing={cfg.training.activation_checkpointing}")
    print(f"checkpoints: {cfg.training.checkpoint_dir}  every {cfg.training.save_every}")
    print("=" * 64, flush=True)


def main(argv=None):
    args, overrides = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- 1. Load + override config ----------------------------------------
    cfg = load_yaml(args.config) if args.config else TrainConfig()
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()      # propagate training.total_steps -> schedule.total_steps; check seq_len

    # ---- 2. Distributed init (no-op for non-torchrun launches) -----------
    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 3. Data first (so the startup banner can report the mask fraction)
    # SFT always uses the model's own tokenizer — the chat template lives on it.
    loader = make_dataloader(cfg.data, tokenizer_name=cfg.model.name)
    frac_assistant = getattr(loader.dataset, "frac_assistant_tokens", float("nan"))
    batch_iter = cycle(loader)

    # ---- 4. Build model, apply efficiency + sharding ---------------------
    # Order matters: activation_checkpointing wraps each decoder layer; FSDP
    # then shards the wrapped layer. Reverse order silently breaks FSDP.
    model = build_model(cfg.model).to(device)
    if cfg.training.activation_checkpointing:
        apply_activation_checkpointing(model)
    model = apply_fsdp(model, dtype=cfg.training.dtype)

    _log_startup(rinfo, cfg, model, frac_assistant, device)

    # ---- 5. Optimizer + scheduler ----------------------------------------
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.schedule)

    # ---- 6. Optional W&B (rank-0 only) -----------------------------------
    # `pip install wandb` then `wandb login` once. Set training.wandb_project
    # to enable; on resume set training.wandb_run_id to append to that run.
    wandb_run = None
    if rinfo.is_main and cfg.training.wandb_project:
        try:
            import wandb
            init_kwargs = dict(
                project=cfg.training.wandb_project,
                name=cfg.training.wandb_run_name or None,
                config=cfg.to_dict(),
                tags=list(cfg.training.wandb_tags) or None,
                # console="off": don't let wandb redirect stdout/stderr — under
                # torchrun / non-TTY environments the fd redirect can deadlock.
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

    # ---- 7. Resume if requested ------------------------------------------
    # `start_step` = number of optimizer updates already completed. Restores
    # model/optimizer/scheduler/RNG; the data sampler restarts its shuffle
    # (see the module docstring — not bit-exact for SFT, and that's fine).
    start_step = 0
    resume = cfg.training.resume_from or latest_ckpt(cfg.training.checkpoint_dir)
    if resume:
        start_step = load_ckpt(model, optimizer, resume, scheduler=scheduler)
        if rinfo.is_main:
            print(f"[rank 0] resumed from {resume} at step {start_step}  "
                  f"(scheduler.last_epoch={scheduler.last_epoch}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e})", flush=True)

    # ---- 8. Train --------------------------------------------------------
    model.train()
    if rinfo.is_main:
        print("[rank 0] entering training loop", flush=True)
    t_start = time.time()
    last_log_time = t_start
    tokens_per_step = (
        cfg.data.batch_size_per_device * cfg.data.seq_len
        * cfg.training.grad_accum * rinfo.world_size
    )
    peak_tflops = _GPU_PEAK_TFLOPS[args.gpu]
    n_params = sum(p.numel() for p in model.parameters())   # local shard under FSDP
    if dist_initialized():
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
        )
        scheduler.step()

        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt_window = now - last_log_time
            steps_window = cfg.training.log_every if step > start_step else 1
            tok_per_sec = (steps_window * tokens_per_step) / max(dt_window, 1e-9)
            flops_per_step = 6.0 * n_params * tokens_per_step
            mfu = (flops_per_step * steps_window / max(dt_window, 1e-9)
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
                log_payload = {
                    "loss": loss, "grad_norm": grad_norm, "lr": lr_now,
                    "tokens_per_second": tok_per_sec, "mfu": mfu, "step": step,
                }
                if gpu is not None:
                    log_payload.update({
                        "gpu/sm_util": gpu["sm_util"],
                        "gpu/mem_used_gb": gpu["mem_used_gb"],
                        "gpu/mem_used_pct": gpu["mem_used_pct"],
                    })
                wandb_run.log(log_payload)

        # Save AFTER scheduler.step(). The persisted "step" is updates completed
        # (step + 1), so a resume with start_step=N starts at update N+1.
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

    # ---- 9. Final save + cleanup -----------------------------------------
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
        print("[rank 0] next: export to HF format, then eval generation:\n"
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
