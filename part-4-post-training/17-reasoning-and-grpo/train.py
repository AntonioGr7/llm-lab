"""The Module 17 GRPO entrypoint. Launch via `torchrun`.

```
# Canonical demo on a single A100-80GB — Qwen3-1.7B (SFT'd) on GSM8K
torchrun --standalone --nproc_per_node=1 train.py --config=configs/grpo_qwen3_1.7b.yaml

# 8-GPU node — keep effective batch constant by dropping data.prompts_per_step
torchrun --standalone --nproc_per_node=8 train.py \\
    --config=configs/grpo_qwen3_1.7b.yaml --data.prompts_per_step=1

# Dev smoke test — points at Qwen/Qwen3-0.6B for a cheap dry-run
torchrun --standalone --nproc_per_node=1 train.py --config=configs/grpo_demo.yaml
```

The composition mirrors Module 16 — same FSDP wrap, same AdamW, same DCP
checkpoint, same schedule. What changes:

  data:     GSM8K prompts (no chosen/rejected — completions are generated).
  rollout:  NEW PHASE. Before each gradient step we generate G completions
            per prompt, score them, and build per-token advantage + ref_logps
            into a `Rollout` dataclass.
  loop:     train_step runs on the rollout (one forward+backward per step).

Each "step" in this loop is therefore EXPENSIVE compared to DPO/SFT — it
contains G·P generation passes (~tens of seconds on A100 for Qwen3-1.7B
at G=8 and max_new=512) plus the gradient forward+backward. Don't expect
the tok/s number to look like SFT's; the relevant metric here is "reward
trajectory over steps", not throughput.

Activation checkpointing + FSDP wrap order is the same as Module 16:
AC wraps each decoder layer FIRST, then FSDP wraps the wrapped layers.
The reference is FSDP-wrapped for sharding parity on multi-GPU but does
NOT get AC (no backward, no memory to save).
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
from data import make_dataloader, cycle, load_tokenizer
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt, cleanup_old as cleanup_old_ckpts
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed
from efficiency import apply_activation_checkpointing, gpu_utilization_snapshot
from rollout import generate_rollout
from loop import train_step


_GPU_PEAK_TFLOPS = {"A100": 312, "H100": 990, "V100": 125}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Module 17 Reasoning and GRPO entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/grpo_qwen3_1.7b.yaml\n"
            "  torchrun --standalone --nproc_per_node=8 train.py "
            "--config=configs/grpo_qwen3_1.7b.yaml --data.prompts_per_step=1\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/grpo_demo.yaml\n"
            "  # Lower the KL anchor (more drift):\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/grpo_qwen3_1.7b.yaml --rl.kl_beta=0.01\n"
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
                "Use --section.field=value form, e.g. --rl.kl_beta=0.01"
            )
    return args, overrides


def _log_startup(rinfo, cfg: TrainConfig, policy, device: str) -> None:
    if not rinfo.is_main:
        return
    counts = count_params(policy)
    rollouts_per_step = cfg.data.prompts_per_step * cfg.rl.group_size * rinfo.world_size
    max_tokens_per_step = rollouts_per_step * (cfg.data.seq_len + cfg.rl.max_new_tokens)
    print("=" * 64)
    print("Module 17 — Reasoning and GRPO")
    print("=" * 64)
    print(f"world_size: {rinfo.world_size}   device: {device}   dtype: {cfg.training.dtype}")
    print(f"policy:    {cfg.model.name}   params: {counts['total']/1e6:.1f}M "
          f"(rank-local count under FSDP; ×world for the global total)")
    print(f"reference: {cfg.model.resolved_ref_name()}   "
          f"({'== policy (default)' if cfg.model.ref_name == '' else 'overridden'})")
    print(f"data:      source={cfg.data.source}/{cfg.data.subset}  split={cfg.data.split}  "
          f"prompts_per_step={cfg.data.prompts_per_step}  seq_len={cfg.data.seq_len}")
    print(f"rl:        group_size={cfg.rl.group_size}  T={cfg.rl.temperature}  "
          f"top_p={cfg.rl.top_p}  max_new={cfg.rl.max_new_tokens}")
    print(f"           kl_beta={cfg.rl.kl_beta}  clip_ratio={cfg.rl.clip_ratio}  "
          f"mu_epochs={cfg.rl.mu_epochs}")
    print(f"reward:    w_format={cfg.reward.w_format}  w_accuracy={cfg.reward.w_accuracy}")
    print(f"optimizer: {cfg.optimizer.type}  peak_lr={cfg.optimizer.lr}  "
          f"betas={cfg.optimizer.betas}  wd={cfg.optimizer.weight_decay}")
    print(f"schedule:  {cfg.schedule.type}  warmup={cfg.schedule.warmup_steps}  "
          f"min_lr_ratio={cfg.schedule.min_lr_ratio}")
    print(f"training:  total_steps={cfg.training.total_steps}  "
          f"completions/step={rollouts_per_step}  "
          f"max_tokens/step={max_tokens_per_step:,}")
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

    # ---- 3. Tokenizer + data ----------------------------------------------
    tokenizer = load_tokenizer(cfg.model.name, padding_side="left")
    loader = make_dataloader(cfg.data, tokenizer_name=cfg.model.name)
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

    _log_startup(rinfo, cfg, policy, device)

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
        print("[rank 0] entering GRPO training loop", flush=True)
    t_start = time.time()
    last_log_time = t_start

    for step in range(start_step, cfg.training.total_steps):
        # ---- 8a. Rollout: generate G completions per prompt, score them ----
        batch = next(batch_iter)
        rollout = generate_rollout(
            policy=policy,
            reference=reference,
            tokenizer=tokenizer,
            batch=batch,
            data_cfg=cfg.data,
            reward_cfg=cfg.reward,
            rl_cfg=cfg.rl,
            device=device,
        )

        # ---- 8b. Gradient step(s) — K=mu_epochs updates on this rollout ----
        for k in range(cfg.rl.mu_epochs):
            stats = train_step(
                policy=policy,
                optimizer=optimizer,
                rollout=rollout,
                rl_cfg=cfg.rl,
                grad_clip=cfg.training.grad_clip,
                dtype=cfg.training.dtype,
                device=device,
            )
        scheduler.step()

        # ---- 8c. Log ---------------------------------------------------------
        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt_window = now - last_log_time
            steps_window = cfg.training.log_every if step > start_step else 1
            sec_per_step = dt_window / max(steps_window, 1)
            lr_now = optimizer.param_groups[0]["lr"]
            gpu = gpu_utilization_snapshot(0)
            gpu_str = ""
            if gpu is not None:
                gpu_str = (f"  sm {gpu['sm_util']:.0f}%  "
                           f"mem {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.0f}GB")
            print(f"step {step:5d}  reward {stats['mean_reward']:.3f}  "
                  f"acc {stats['mean_accuracy']:.1%}  fmt {stats['mean_format']:.1%}  "
                  f"len {stats['mean_response_length']:.0f}  "
                  f"kl {stats['mean_kl']:.4f}  clipfrac {stats['clip_frac']:.3f}  "
                  f"loss {stats['loss']:+.4f}  grad_norm {stats['grad_norm']:.2f}  "
                  f"lr {lr_now:.2e}  s/step {sec_per_step:.1f}"
                  f"{gpu_str}", flush=True)
            # Print one example completion every save_every steps so you can
            # see the schema / reasoning visibly emerge in the logs.
            if step % max(cfg.training.save_every, 1) == 0 and rollout.completions_text:
                example = rollout.completions_text[0]
                ex_preview = example.replace("\n", " ")[:280]
                print(f"  [rollout example] {ex_preview!r}", flush=True)
            last_log_time = now

            if wandb_run is not None:
                log_payload = {**stats, "lr": lr_now, "s_per_step": sec_per_step, "step": step}
                if gpu is not None:
                    log_payload.update({
                        "gpu/sm_util": gpu["sm_util"],
                        "gpu/mem_used_gb": gpu["mem_used_gb"],
                    })
                wandb_run.log(log_payload)

        # ---- 8d. Periodic checkpoint ----------------------------------------
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
        print("[rank 0] next: eval the policy:\n"
              "  python eval.py --checkpoint=<final ckpt> --gsm8k", flush=True)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
