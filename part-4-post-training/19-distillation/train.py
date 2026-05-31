"""The Module 19 distillation entrypoint. Launch via `torchrun`.

```
# SDFT canonical — Qwen3-1.7B on tool-use + GSM8K eval, single A100
torchrun --standalone --nproc_per_node=1 train.py --config=configs/sdft_qwen3_1.7b.yaml

# On-policy distillation — Qwen3-1.7B teacher -> Qwen3-0.6B student
torchrun --standalone --nproc_per_node=1 train.py --config=configs/on_policy_distill.yaml

# Demo smoke — Qwen3-0.6B, tiny corpus, 30 steps
torchrun --standalone --nproc_per_node=1 train.py --config=configs/sdft_demo.yaml
```

Composition by mode:

  sdft:        ONE model in memory (the student); used as teacher under
               no_grad + demos-in-context during rollout. Lowest memory.
  on_policy:   TWO models in memory (student + separate larger teacher);
               teacher is FSDP-wrapped for sharding parity (same pattern
               as Modules 17/17).
  offline:     handled outside this file — see README §2 (config-only
               over Module 15's SFT code).

Each step:
  1. Pull a batch of tool-use prompts.
  2. `generate_rollout` — teacher-side generation + log-prob extraction.
  3. `train_step` — student forward (with grad) + KL loss + step.
  4. Periodic two-axis eval (tool-use new-skill + GSM8K prior-skill).

The two-axis eval is the "have your cake and eat it" demo: new-skill
accuracy grows AND prior-skill accuracy is preserved (vs. plain SFT
which would catastrophically forget on the prior task).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_student, build_teacher, count_params
from optim import build_optimizer
from schedule import build_scheduler
from data import make_dataloader, cycle, load_tokenizer
from checkpoint import save as save_ckpt, load as load_ckpt, latest as latest_ckpt, cleanup_old as cleanup_old_ckpts
from fsdp_setup import init_distributed, apply_fsdp, cleanup_distributed
from efficiency import apply_activation_checkpointing, gpu_utilization_snapshot
from rollout import generate_rollout
from loop import train_step


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Module 19 Distillation entrypoint (SDFT + on-policy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/sdft_qwen3_1.7b.yaml\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/on_policy_distill.yaml\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/sdft_demo.yaml\n"
            "  # Switch to reverse KL with one flag override:\n"
            "  torchrun --standalone --nproc_per_node=1 train.py "
            "--config=configs/sdft_qwen3_1.7b.yaml --distill.kl_direction=reverse\n"
        ),
    )
    p.add_argument("--config", type=str, default=None)
    args, extra = p.parse_known_args(argv)
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
        else:
            raise SystemExit(
                f"unrecognized arg: {tok!r}. Use --section.field=value form."
            )
    return args, overrides


def _log_startup(rinfo, cfg: TrainConfig, student, teacher_info: str, device: str) -> None:
    if not rinfo.is_main:
        return
    counts = count_params(student)
    print("=" * 64)
    print("Module 19 — Distillation")
    print("=" * 64)
    print(f"world_size: {rinfo.world_size}   device: {device}   dtype: {cfg.training.dtype}")
    print(f"mode:      {cfg.distill.mode}")
    print(f"student:   {cfg.model.name}   params: {counts['total']/1e6:.1f}M "
          f"(rank-local count under FSDP)")
    print(f"teacher:   {teacher_info}")
    print(f"distill:   kl_direction={cfg.distill.kl_direction}  "
          f"T={cfg.distill.temperature}  "
          f"n_demos={cfg.distill.n_demonstrations}")
    print(f"sampling:  T={cfg.distill.sampling_temperature}  "
          f"top_p={cfg.distill.top_p}  max_new={cfg.distill.max_new_tokens}")
    print(f"data:      corpus={cfg.data.corpus_dir}  "
          f"prompts_per_step={cfg.data.prompts_per_step}  seq_len={cfg.data.seq_len}")
    print(f"optimizer: {cfg.optimizer.type}  peak_lr={cfg.optimizer.lr}  "
          f"betas={cfg.optimizer.betas}  wd={cfg.optimizer.weight_decay}")
    print(f"schedule:  {cfg.schedule.type}  warmup={cfg.schedule.warmup_steps}")
    print(f"training:  total_steps={cfg.training.total_steps}  "
          f"grad_clip={cfg.training.grad_clip}  AC={cfg.training.activation_checkpointing}")
    print(f"checkpoints: {cfg.training.checkpoint_dir}  every {cfg.training.save_every}")
    print(f"eval:      every {cfg.training.eval_every} steps "
          f"(tool-use + {cfg.training.eval_n_gsm8k} GSM8K problems)")
    print("=" * 64, flush=True)


def main(argv=None):
    args, overrides = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- 1. Load + override config ----------------------------------------
    cfg = load_yaml(args.config) if args.config else TrainConfig()
    apply_dotted_overrides(cfg, overrides)
    cfg.sync()

    if cfg.distill.mode == "offline":
        raise SystemExit(
            "Offline distillation is README-only in Module 19 (the loss is "
            "identical to SFT). See ../15-sft/ — point its data loader at "
            "pre-generated (prompt, teacher_completion) data."
        )

    # ---- 2. Distributed init ----------------------------------------------
    rinfo = init_distributed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 3. Tokenizer + data ----------------------------------------------
    tokenizer = load_tokenizer(cfg.model.name, padding_side="left")
    loader = make_dataloader(cfg.data)
    demonstrations = loader.demonstrations[: cfg.distill.n_demonstrations]
    batch_iter = cycle(loader)

    # ---- 4. Build models, apply efficiency + sharding ---------------------
    student = build_student(cfg.model).to(device)
    if cfg.training.activation_checkpointing:
        apply_activation_checkpointing(student)
    student = apply_fsdp(student, dtype=cfg.training.dtype)

    if cfg.distill.mode == "on_policy":
        teacher = build_teacher(cfg.model).to(device)
        teacher = apply_fsdp(teacher, dtype=cfg.training.dtype)
        teacher_info = f"{cfg.model.resolved_teacher_name()} (separate, frozen)"
    else:
        # SDFT: the student IS the teacher (queried under no_grad + demos).
        teacher = None
        teacher_info = (
            f"{cfg.model.name} (same as student — SDFT, "
            f"conditioned on {len(demonstrations)} demos in-context)"
        )

    _log_startup(rinfo, cfg, student, teacher_info, device)

    # ---- 5. Optimizer + scheduler (STUDENT only) --------------------------
    optimizer = build_optimizer(student, cfg.optimizer)
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
        start_step = load_ckpt(student, optimizer, resume, scheduler=scheduler)
        if rinfo.is_main:
            print(f"[rank 0] resumed from {resume} at step {start_step}", flush=True)

    # ---- 8. Train --------------------------------------------------------
    student.train()
    if teacher is not None:
        teacher.eval()
    if rinfo.is_main:
        print("[rank 0] entering distillation loop", flush=True)
    t_start = time.time()
    last_log_time = t_start

    for step in range(start_step, cfg.training.total_steps):
        batch = next(batch_iter)
        rollout = generate_rollout(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            batch=batch,
            demonstrations=demonstrations,
            data_cfg=cfg.data,
            distill_cfg=cfg.distill,
            device=device,
        )
        stats = train_step(
            student=student,
            optimizer=optimizer,
            rollout=rollout,
            distill_cfg=cfg.distill,
            grad_clip=cfg.training.grad_clip,
            dtype=cfg.training.dtype,
            device=device,
        )
        scheduler.step()

        if step % cfg.training.log_every == 0 and rinfo.is_main:
            now = time.time()
            dt = now - last_log_time
            steps_window = cfg.training.log_every if step > start_step else 1
            sec_per_step = dt / max(steps_window, 1)
            lr_now = optimizer.param_groups[0]["lr"]
            gpu = gpu_utilization_snapshot(0)
            gpu_str = ""
            if gpu is not None:
                gpu_str = (f"  sm {gpu['sm_util']:.0f}%  "
                           f"mem {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.0f}GB")
            print(f"step {step:5d}  kl/tok {stats['kl_per_token']:.4f}  "
                  f"top_p {stats['teacher_top_prob']:.3f}  "
                  f"argmax_match {stats['argmax_agreement']:.1%}  "
                  f"len {stats['mean_completion_length']:.0f}  "
                  f"grad_norm {stats['grad_norm']:.2f}  "
                  f"lr {lr_now:.2e}  s/step {sec_per_step:.1f}"
                  f"{gpu_str}", flush=True)

            # Periodically print one example for qualitative inspection.
            if step % max(cfg.training.save_every, 1) == 0 and rollout.completion_texts:
                ex = rollout.completion_texts[0].replace("\n", " ")[:200]
                print(f"  [rollout example] teacher_input={rollout.teacher_input_summary}",
                      flush=True)
                print(f"  [rollout example] completion: {ex!r}", flush=True)
            last_log_time = now

            if wandb_run is not None:
                log_payload = {**stats, "lr": lr_now,
                               "s_per_step": sec_per_step, "step": step}
                if gpu is not None:
                    log_payload.update({
                        "gpu/sm_util": gpu["sm_util"],
                        "gpu/mem_used_gb": gpu["mem_used_gb"],
                    })
                wandb_run.log(log_payload)

        steps_done = step + 1

        # Periodic checkpoint.
        if steps_done % cfg.training.save_every == 0:
            ckpt_path = save_ckpt(student, optimizer, steps_done,
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

        # Periodic two-axis eval (tool-use new-skill + GSM8K prior-skill).
        # We import here to avoid circular imports and to keep the eval code
        # in eval.py as a single source of truth.
        if cfg.training.eval_every > 0 and steps_done % cfg.training.eval_every == 0 and rinfo.is_main:
            from eval import quick_two_axis_eval
            metrics = quick_two_axis_eval(
                student=student, tokenizer=tokenizer, cfg=cfg,
                n_tooluse=20, n_gsm8k=cfg.training.eval_n_gsm8k, device=device,
            )
            print(f"[rank 0] [eval @ step {steps_done}] "
                  f"tooluse correct: {metrics['tooluse_correct']:.1%}  "
                  f"schema_ok: {metrics['tooluse_schema_ok']:.1%}  "
                  f"GSM8K acc: {metrics['gsm8k_accuracy']:.1%}", flush=True)
            if wandb_run is not None:
                wandb_run.log({**{f"eval/{k}": v for k, v in metrics.items()},
                               "step": steps_done})

    # ---- 9. Final save + cleanup -----------------------------------------
    if cfg.training.total_steps > 0:
        already_saved = cfg.training.total_steps % cfg.training.save_every == 0
        if not already_saved:
            final_path = save_ckpt(student, optimizer, cfg.training.total_steps,
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
        print("[rank 0] next: full eval\n"
              "  python eval.py --config=<config> --checkpoint=<final ckpt> --full", flush=True)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
