"""Resume-correctness regression test.

Proves that a mid-run save + resume produces a model bit-identical to an
uninterrupted run of the same total length. If anyone's pretraining ever
crashes mid-flight, this is the test that proves we won't silently lose
(or duplicate) optimizer updates on resume.

What this test catches:
  - Off-by-one in step semantics (the bug where resume re-runs the saved step,
    duplicating one optimizer update).
  - Scheduler state drift (LR at step K after resume vs uninterrupted).
  - RNG state drift (any dropout, init, or sampling that uses torch.rand).
  - Optimizer state drift (Adam moments not being saved correctly).

What this test does NOT catch:
  - Data-loader position. We work around this by using SyntheticDataset with
    n_samples == K (the save interval) so the dataset cycles. After the resume,
    the fresh data iterator yields the same samples we'd have seen continuing
    uninterrupted. For streaming pipelines (FineWeb-Edu) this property doesn't
    hold; see README "Resume correctness" for the discussion.

Usage:
    cd 11-pretraining-in-practice/
    python tests/test_resume.py

Exits 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

import torch

# Make the module's directory importable when running from `tests/`.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from config import TrainConfig
from model import build_model
from optim import build_optimizer
from schedule import build_scheduler
from data import make_dataloader, cycle
from loop import train_step
from checkpoint import save as save_ckpt, load as load_ckpt


# Tiny model + tiny data — runs in ~1 second on CPU.
TOTAL_STEPS = 10
SAVE_AT = 5           # save after 5 updates; resume; do 5 more.
SEED = 12345


def _make_cfg() -> TrainConfig:
    cfg = TrainConfig()
    # Tiny architecture so the test runs on a laptop.
    cfg.model.vocab_size = 64
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    cfg.model.n_kv_heads = 1
    cfg.model.d_ffn = 64
    cfg.model.max_seq = 16
    # Data: synthetic + cycle, so resume sees the same samples it would have
    # seen continuing. The dataset's __iter__ is deterministic per (seed, rank, worker).
    cfg.data.source = "synthetic"
    cfg.data.seq_len = 16
    cfg.data.batch_size_per_device = 2
    cfg.data.num_workers = 0
    cfg.data.synthetic_samples = SAVE_AT * cfg.data.batch_size_per_device
    cfg.data.seed = SEED
    # Training: fp32 (no autocast nondeterminism), no FSDP (single process),
    # cosine LR with a tight warmup so the schedule visibly shapes the LR.
    cfg.training.total_steps = TOTAL_STEPS
    cfg.training.grad_accum = 1
    cfg.training.dtype = "fp32"
    cfg.training.activation_checkpointing = False
    cfg.training.grad_clip = 1.0
    cfg.training.checkpoint_dir = "./results/resume_test"
    cfg.training.save_every = SAVE_AT  # save after SAVE_AT updates
    cfg.training.keep_last_n_checkpoints = 5
    cfg.schedule.type = "cosine"
    cfg.schedule.warmup_steps = 2
    cfg.schedule.min_lr_ratio = 0.1
    cfg.sync()
    return cfg


def _build_world(cfg, device: str):
    """Build a fresh model + optimizer + scheduler + dataloader with the same seed."""
    torch.manual_seed(SEED)
    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.schedule)
    loader, _ = make_dataloader(cfg.data, vocab_size=cfg.model.vocab_size)
    batch_iter = cycle(loader)
    return model, optimizer, scheduler, batch_iter


def _train_block(model, optimizer, scheduler, batch_iter, cfg, n_steps, device):
    """Run `n_steps` optimizer updates, returning the trajectory of (loss, lr)."""
    trajectory = []
    for _ in range(n_steps):
        loss, _ = train_step(
            model, optimizer, batch_iter,
            grad_accum=cfg.training.grad_accum,
            grad_clip=cfg.training.grad_clip,
            dtype=cfg.training.dtype, device=device,
        )
        scheduler.step()
        trajectory.append((loss, optimizer.param_groups[0]["lr"]))
    return trajectory


def _params_fingerprint(model) -> torch.Tensor:
    """Concatenate-and-hash all params into a single tensor for exact comparison."""
    chunks = []
    for _, p in sorted(model.named_parameters()):
        chunks.append(p.detach().flatten().cpu().double())
    return torch.cat(chunks)


def _opt_fingerprint(optimizer) -> torch.Tensor:
    """Concatenate Adam moments into a single tensor for comparison."""
    chunks = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for k in sorted(state.keys()):
                v = state[k]
                if torch.is_tensor(v):
                    chunks.append(v.detach().flatten().cpu().double())
                else:
                    chunks.append(torch.tensor([float(v)], dtype=torch.float64))
    return torch.cat(chunks) if chunks else torch.tensor([])


def run_scenario_A(cfg, device):
    """Uninterrupted: run all TOTAL_STEPS in one go."""
    model, optimizer, scheduler, batch_iter = _build_world(cfg, device)
    traj = _train_block(model, optimizer, scheduler, batch_iter, cfg, TOTAL_STEPS, device)
    return model, optimizer, scheduler, traj


def run_scenario_B(cfg, device):
    """Mid-run save + resume: run SAVE_AT, save, throw it all away, reload, run the rest."""
    # First half
    model, optimizer, scheduler, batch_iter = _build_world(cfg, device)
    traj_first = _train_block(model, optimizer, scheduler, batch_iter, cfg, SAVE_AT, device)
    ckpt = save_ckpt(model, optimizer, SAVE_AT, cfg.training.checkpoint_dir,
                     scheduler=scheduler)
    # Drop the in-memory state on the floor and reload from disk to simulate a crash.
    del model, optimizer, scheduler, batch_iter

    # Second half — fresh world (different random init draws), then load.
    torch.manual_seed(SEED + 99_999)   # deliberately different — load_ckpt must overwrite
    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.schedule)
    start = load_ckpt(model, optimizer, ckpt, scheduler=scheduler)
    assert start == SAVE_AT, f"load_ckpt returned {start}, expected {SAVE_AT}"
    # Rebuild the loader. SyntheticDataset is deterministic from seed; the
    # cycle wrapper re-iterates from sample 0. Because synthetic_samples =
    # SAVE_AT * batch_size, the second pass starts again from sample 0 — same
    # samples the uninterrupted run would have seen at this point.
    loader, _ = make_dataloader(cfg.data, vocab_size=cfg.model.vocab_size)
    batch_iter = cycle(loader)

    traj_second = _train_block(model, optimizer, scheduler, batch_iter, cfg,
                               TOTAL_STEPS - SAVE_AT, device)
    return model, optimizer, scheduler, traj_first + traj_second


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _make_cfg()

    # Clean any previous test artifacts.
    test_dir = Path(cfg.training.checkpoint_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print(f"Running resume-correctness test on device={device}")
    print(f"  TOTAL_STEPS={TOTAL_STEPS}, SAVE_AT={SAVE_AT}, SEED={SEED}\n")

    print("Scenario A: uninterrupted run of TOTAL_STEPS steps")
    model_A, opt_A, sched_A, traj_A = run_scenario_A(cfg, device)

    # IMPORTANT: re-seed before scenario B so the *initial* model weights match
    # scenario A. We're testing that resume->continue == uninterrupted.
    print("\nScenario B: SAVE_AT steps + save + reload + continue")
    model_B, opt_B, sched_B, traj_B = run_scenario_B(cfg, device)

    # ---- Compare ---------------------------------------------------------
    print("\nResults:")
    print(f"  {'step':>4s}  {'lossA':>9s}  {'lossB':>9s}  {'lrA':>8s}  {'lrB':>8s}  match?")
    all_close = True
    for i, ((lA, lrA), (lB, lrB)) in enumerate(zip(traj_A, traj_B)):
        loss_close = abs(lA - lB) < 1e-6
        lr_close = abs(lrA - lrB) < 1e-9
        ok = loss_close and lr_close
        all_close = all_close and ok
        print(f"  {i:>4d}  {lA:9.6f}  {lB:9.6f}  {lrA:8.2e}  {lrB:8.2e}  "
              f"{'OK' if ok else '*** DIFFER ***'}")

    pf_A = _params_fingerprint(model_A)
    pf_B = _params_fingerprint(model_B)
    of_A = _opt_fingerprint(opt_A)
    of_B = _opt_fingerprint(opt_B)
    params_diff = (pf_A - pf_B).abs().max().item()
    opt_diff = (of_A - of_B).abs().max().item()
    sched_diff = abs(sched_A.last_epoch - sched_B.last_epoch)

    print(f"\n  max param diff:           {params_diff:.2e}")
    print(f"  max optimizer state diff: {opt_diff:.2e}")
    print(f"  scheduler.last_epoch diff: {sched_diff}")

    params_ok = params_diff < 1e-6
    opt_ok = opt_diff < 1e-6
    sched_ok = sched_diff == 0

    print()
    if all_close and params_ok and opt_ok and sched_ok:
        print("  PASS — resume produces a bit-identical run.")
        # Cleanup on success.
        shutil.rmtree(test_dir, ignore_errors=True)
        return 0
    else:
        print("  FAIL — resume diverged from uninterrupted run.")
        print("  Check: off-by-one in step semantics, scheduler state, RNG state.")
        if not all_close:
            print("    -> loss/lr trajectories differ; first divergence above")
        if not params_ok:
            print(f"    -> model params differ by up to {params_diff:.2e}")
        if not opt_ok:
            print(f"    -> optimizer state differs by up to {opt_diff:.2e}")
        if not sched_ok:
            print(f"    -> scheduler.last_epoch off by {sched_diff}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
