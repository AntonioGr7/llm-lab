"""Muon optimizer (Jordan 2024) — Newton-Schulz orthogonalized momentum.

Implements the 2024-vintage Muon update for the 2D weight matrices in a
transformer, plus an `AdamW` fallback for everything else (embeddings,
norms, biases). The two updates are dispatched per param-group in a single
`MuonAdamW` Optimizer subclass, so it composes with any standard
`torch.optim.lr_scheduler.LRScheduler`.

The recipe:

- 2D Linear weights (attention QKV/O, FFN matrices): Muon. Momentum SGD
  followed by Newton-Schulz orthogonalization of the (Nesterov-style)
  momentum, giving an update with approximately-unit singular values.
  Scaled by `sqrt(max(d_out / d_in, 1))` to compensate for non-square shapes.
- Everything else (embeddings, RMSNorm gamma, biases, LM head): AdamW.

Status: research-validated up to ~16B parameters (Moonlight 2025). Not yet
the default at frontier production scale (Llama, DeepSeek, Qwen are all
still AdamW as of early 2026). See Module 08 README § 4 for context.

Reference: Keller Jordan, "Muon: An optimizer for hidden layers in neural
networks" (2024). https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate `G`'s matrix-sign via 5 Newton-Schulz iterations.

    For a 2D matrix `G`, returns a matrix with the same singular vectors but
    all singular values pushed toward 1. This is the "orthogonalization" that
    gives Muon its name and its update behavior.

    The quintic polynomial coefficients (a, b, c) are Jordan's choice; they
    converge in ~5 iterations for any input with bounded singular values.

    Args:
        G: a 2D tensor (any shape, any dtype). Will be cast to bfloat16 for
            the inner loop; the result is cast back to G's dtype.
        steps: number of NS iterations. 5 is the canonical choice.
        eps: small constant to prevent division by zero in the normalization.

    Returns:
        Tensor of the same shape and dtype as `G`, with sing vals ~= 1.
    """
    assert G.ndim == 2, f"Newton-Schulz expects 2D, got {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315  # Jordan's quintic coefficients

    X = G.to(torch.bfloat16)
    # Operate on the "tall-skinny" form for numerical stability; transpose if needed.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    # Spectral normalization so the initial sing vals are <= 1.
    X = X / (X.norm() + eps)

    for _ in range(steps):
        # X_{n+1} = a X + (b A + c A^2) X, where A = X X^T
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# The hybrid optimizer
# ---------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """Hybrid optimizer: Muon for 2D weight matrices, AdamW for everything else.

    Pass param groups tagged with `optimizer_type='muon'` or `'adamw'`. The
    `step()` method dispatches to the right update per group. Both share the
    same `param_groups` list, so any LRScheduler operating on `param_group['lr']`
    works uniformly — the scheduler doesn't know or care which is which.

    Args:
        params: an iterable of param groups, each a dict with at minimum
            `params`, `lr`, and `optimizer_type`. Use `split_muon_adamw_groups`
            below for the canonical Jordan-style split.

    Per-group hyperparameters that are read by `step()`:
        - Muon groups: `lr`, `momentum`, `ns_steps`, `nesterov` (default True)
        - AdamW groups: `lr`, `betas` (tuple), `eps`, `weight_decay`
    """

    def __init__(self, params):
        # `defaults` is required by Optimizer's __init__ and is used as the
        # fallback when a param group dict doesn't specify a field. We provide
        # the union of Muon and AdamW defaults; per-group `optimizer_type`
        # selects which keys actually matter.
        defaults = dict(
            optimizer_type="muon",
            lr=2e-2,
            # Muon
            momentum=0.95,
            ns_steps=5,
            nesterov=True,
            # AdamW
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            otype = group["optimizer_type"]
            if otype == "muon":
                self._muon_step(group)
            elif otype == "adamw":
                self._adamw_step(group)
            else:
                raise ValueError(f"unknown optimizer_type: {otype!r}")
        return loss

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        ns_steps = group["ns_steps"]
        nesterov = group.get("nesterov", True)

        for p in group["params"]:
            if p.grad is None:
                continue
            assert p.ndim == 2, (
                f"Muon expects 2D weights, got {tuple(p.shape)}. Move this "
                "parameter to an 'adamw' group via `split_muon_adamw_groups`."
            )
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            # Nesterov-style lookahead: g + momentum * buf (otherwise just buf).
            g_eff = g.add(buf, alpha=momentum) if nesterov else buf

            # Orthogonalize. The result has approximately-unit singular values;
            # we re-scale to account for the non-square shape so the update
            # magnitude is comparable across layer dimensions.
            update = newton_schulz5(g_eff, steps=ns_steps)
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]

            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # Decoupled weight decay.
            if wd > 0:
                p.mul_(1 - lr * wd)

            # Bias-corrected step.
            bc1 = 1 - beta1 ** t
            bc2 = 1 - beta2 ** t
            denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
            p.addcdiv_(m, denom, value=-lr / bc1)


# ---------------------------------------------------------------------------
# Param-group split (Jordan-style)
# ---------------------------------------------------------------------------

def split_muon_adamw_groups(
    model: nn.Module,
    *,
    muon_lr: float = 2e-2,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> list[dict]:
    """Split a model's parameters into Muon and AdamW groups.

    The convention (from Keller Jordan's recipe):

    - 1D parameters (norms, biases): AdamW.
    - Embeddings and LM head: AdamW.
    - All other 2D weights (the "hidden" Linear matrices): Muon.

    Returns a list of param-group dicts suitable for `MuonAdamW(groups)`.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # AdamW path: anything 1D (norms, biases) or any embedding/output head.
        is_1d = p.ndim < 2
        is_embed = "embed" in name.lower() or "lm_head" in name.lower() or "unembed" in name.lower()
        if is_1d or is_embed:
            adamw_params.append(p)
        else:
            # Sanity: must be 2D for Muon's Newton-Schulz to make sense.
            assert p.ndim == 2, (
                f"Muon expects 2D Linear weights; '{name}' has shape {tuple(p.shape)}. "
                "Either route it to AdamW (e.g. extend the embed-name heuristic) or skip it."
            )
            muon_params.append(p)

    groups: list[dict] = []
    if muon_params:
        groups.append(dict(
            params=muon_params, optimizer_type="muon",
            lr=muon_lr, momentum=muon_momentum, ns_steps=muon_ns_steps, nesterov=True,
            name="muon",
        ))
    if adamw_params:
        groups.append(dict(
            params=adamw_params, optimizer_type="adamw",
            lr=adamw_lr, betas=adamw_betas, eps=adamw_eps, weight_decay=weight_decay,
            name="adamw",
        ))
    return groups


if __name__ == "__main__":
    # Smoke test: train a tiny model with the hybrid optimizer for a few steps,
    # verify loss-at-init ~ ln(V) and loss decreases.
    import torch.nn.functional as F

    torch.manual_seed(0)
    V, D = 256, 64
    model = nn.Sequential(
        nn.Embedding(V, D),
        nn.Linear(D, 4 * D, bias=False),
        nn.GELU(),
        nn.Linear(4 * D, D, bias=False),
        nn.LayerNorm(D),
        nn.Linear(D, V, bias=False),
    )
    groups = split_muon_adamw_groups(model)
    opt = MuonAdamW(groups)

    # Sanity log of the split.
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  group '{g['name']}': lr={g['lr']:.2e}  n_params={n:,}")

    # Use a COPY task (target = input) so the loss has actual signal to learn:
    # the model just needs to learn to invert the embedding through the MLP.
    losses = []
    for step in range(60):
        ids = torch.randint(0, V, (4, 16))
        x = ids
        for layer in model:
            x = layer(x)
        logits = x                    # (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"\nloss[0]  = {losses[0]:.4f}  (expected ~{math.log(V):.4f})")
    print(f"loss[30] = {losses[30]:.4f}")
    print(f"loss[59] = {losses[59]:.4f}")
    assert losses[59] < losses[0] - 0.5, f"loss should decrease meaningfully on copy task"
    print("\nMuonAdamW smoke test passed.")
