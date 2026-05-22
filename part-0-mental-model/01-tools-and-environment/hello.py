"""Sanity check: verifies torchrun + NCCL + CUDA + PyTorch are wired correctly.

Every training script in this course launches with `torchrun`. Use this on
your cloud pod after setup to confirm the distributed stack works, before you
spend money on a real run.

Run:
    torchrun --nproc_per_node=1 hello.py
    torchrun --nproc_per_node=2 hello.py   # if 2 GPUs are available

If this fails, fix it now — every later module depends on it working.
"""
import os

import torch
import torch.distributed as dist


def main() -> None:
    # NCCL is the only backend that's fast on GPU. Gloo is the CPU fallback
    # so this script stays runnable on a laptop for shape-only checks.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # torchrun sets LOCAL_RANK as an env var. We need it to pin this process
    # to one specific GPU on a multi-GPU host. (rank can span nodes;
    # local_rank is always the GPU index on *this* host.)
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"[rank {rank} / world_size {world_size}] hello from {device}")

    # Each rank contributes 1.0; the all-reduce should sum to world_size.
    # If it doesn't, NCCL is broken and nothing else in this course will work.
    x = torch.ones(1, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    if rank == 0:
        expected = float(world_size)
        status = "✓" if abs(x.item() - expected) < 1e-6 else "✗"
        print(
            f"[rank 0] all-reduce check: tensor sum = {x.item()} "
            f"(expected {expected}) {status}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
