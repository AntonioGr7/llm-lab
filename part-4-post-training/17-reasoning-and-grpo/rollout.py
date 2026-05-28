"""On-policy rollouts for GRPO.

For each prompt in the training batch, we:

  1. Sample G completions with `policy.generate(num_return_sequences=G)`.
  2. Decode each completion to text.
  3. Score each text with the verifiable reward function (rewards.py).
  4. Compute group-relative advantages — within each prompt's G rewards,
     subtract the mean and divide by std. THIS is the "GR" in GRPO; it
     replaces the value head a PPO critic would have learned.
  5. Score every generated token under both the policy (the "old"
     logprob — cached for the importance ratio) and the reference
     (the KL anchor).

The output is a `Rollout` dataclass that's then consumed by `loop.py`'s
`compute_grpo_loss` and `train_step`.

Implementation notes:

- **Left padding throughout.** `model.generate` requires left-padded
  prompts so the generated tokens append on the right. Data pipeline
  already does this; we forward the contract through generation.

- **One prompt × G at a time.** Generating B prompts × G completions in
  ONE `generate()` call is faster but pads to the longest prompt across
  the batch, wasting tokens. Iterating one prompt at a time avoids that
  and keeps each call's KV cache small (~G × max_new_tokens tokens).
  The cost: B kernel launches. For B=4 and G=8 this is fine; for very
  large batch or G push to batched generation with attention_mask.

- **`generation_mode` context manager flips KV cache on/off.** The policy
  is in training mode (no cache, `requires_grad=True`) the rest of the
  step; we flip it for the duration of `generate()` only.

- **`no_grad` everywhere in rollout.** Old_logps and ref_logps are stored
  as detached tensors; the actual gradient happens later in `train_step`
  when we re-forward the policy on the same sequences.

- **Loss positions.** The model produces logits[t] predicting token[t+1].
  So a completion of length C predicts at positions [prompt_len-1,
  prompt_len, ..., prompt_len+C-2]. The `labels` tensor follows the
  Module 16 convention: input_ids = full[:-1], labels = full[1:], mask
  IGNORE_INDEX outside completion. `loop.gather_token_logps` consumes
  this directly.

- **EOS handling.** `model.generate` continues until `max_new_tokens` OR
  hitting `eos_token_id`. After EOS, padding is appended. We mask
  everything after the first EOS in `labels` (don't score the model on
  pad tokens it didn't actually predict).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

from config import DataConfig, RewardConfig, RLConfig
from rewards import RewardBreakdown, compute_rewards
from model import generation_mode


IGNORE_INDEX = -100


# =============================================================================
# Rollout container
# =============================================================================

@dataclass
class Rollout:
    """One micro-batch of on-policy data ready to feed into a GRPO update.

    All tensors are shape [N=P·G, S], where P is the number of prompts in
    the batch and G is the per-prompt group size. The first G rows belong
    to prompt 0, the next G to prompt 1, etc.

    Fields:
      input_ids:        Long[N, S-1] — full (prompt + completion) shifted left by 1.
      labels:           Long[N, S-1] — input_ids shifted right by 1, IGNORE_INDEX
                        outside the completion span (so logp gather contributes 0
                        on prompt and on post-EOS padding).
      attention_mask:   Long[N, S-1] — 1 over real tokens, 0 over padding (both
                        left-padded prompt and right-padded post-EOS region).
      response_mask:    Bool[N, S-1] — True at exactly the positions we score
                        (= `labels != IGNORE_INDEX`). Kept explicit so the loss
                        code doesn't have to recompute it.
      old_logps:        Float[N, S-1] — per-token logp of `labels[t]` under the
                        policy AT ROLLOUT TIME. For K=1 these equal the logps
                        re-computed in the gradient step; we cache them so
                        K>1 generalizes cleanly.
      ref_logps:        Float[N, S-1] — same, under the frozen reference.
      advantages:       Float[N] — group-normalized advantage scalar per
                        completion. Broadcast across response tokens inside
                        the loss.
      rewards:          list[RewardBreakdown] — for logging.
      response_lengths: Long[N] — number of scored tokens per completion
                        (useful for mean-over-tokens vs mean-over-sequences
                        loss aggregation; we use mean-over-tokens).
      completions_text: list[str] — decoded completions, for the qualitative
                        readout printed at log time (rank 0 only).
    """
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logps: torch.Tensor
    ref_logps: torch.Tensor
    advantages: torch.Tensor
    rewards: list[RewardBreakdown]
    response_lengths: torch.Tensor
    completions_text: list[str]


# =============================================================================
# Per-token logp gather (shared with loop.py — exported here for tests)
# =============================================================================

def per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log π(label_t | context) gathered from logits.

    Args:
        logits: Float[N, S, V] — model output at every position.
        labels: Long[N, S]     — target token at every position; IGNORE_INDEX
                                 where we don't score.

    Returns:
        Float[N, S] — log-probabilities, with 0.0 (not -inf) at IGNORE_INDEX
        positions so the masked mean / sum stays numerically clean.

    Numerical detail: log_softmax in FP32. BF16 log_softmax over a 150k
    vocab gets shaky in the low-probability tail — and GRPO rollouts
    routinely sample low-prob tokens (we have temperature=1.0). The cast
    is free relative to the full transformer forward.
    """
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    per_token = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return per_token * mask.float()


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def _sample_completions(
    policy: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_attn: torch.Tensor,
    rl_cfg: RLConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample G completions for a SINGLE prompt.

    Args:
        prompt_ids: Long[S_prompt] — the left-padded prompt token ids.
        prompt_attn: Long[S_prompt] — 0/1 attention mask.

    Returns:
        (full_ids, completion_lengths)
            full_ids: Long[G, S_full] — prompt + generated tokens. Right-padded
                with pad_id past the first EOS for each row.
            completion_lengths: Long[G] — count of GENERATED tokens per row
                up to (and INCLUDING) the first EOS (or max_new_tokens if no
                EOS was emitted).

    Generation uses `model.generate(num_return_sequences=G)`. Sampling
    parameters come from `rl_cfg`.

    The reason we sample one prompt at a time (rather than batching P
    prompts × G completions in a single call): batched generation would
    pad the whole batch to the longest prompt, wasting KV cache slots on
    tokens that aren't real. With P=4 and G=8 in our default config, the
    extra kernel launches are < 1% of step time; not worth optimizing.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    prompt_ids_batch = prompt_ids.unsqueeze(0).to(device)
    prompt_attn_batch = prompt_attn.unsqueeze(0).to(device)

    # `generation_mode` enables KV cache + eval mode for the duration of this
    # block, restoring afterwards. See model.py for why this toggle matters.
    with generation_mode(policy):
        out = policy.generate(
            input_ids=prompt_ids_batch,
            attention_mask=prompt_attn_batch,
            do_sample=True,
            temperature=rl_cfg.temperature,
            top_p=rl_cfg.top_p,
            top_k=rl_cfg.top_k if rl_cfg.top_k > 0 else None,
            max_new_tokens=rl_cfg.max_new_tokens,
            num_return_sequences=rl_cfg.group_size,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=False,
        )

    # `out` is shape [G, S_full]. Slice off the prompt to count completion
    # length (where each row first emits eos OR hits max_new_tokens).
    G = rl_cfg.group_size
    S_prompt = prompt_ids.shape[0]
    completion_only = out[:, S_prompt:]                        # [G, max_new]
    completion_lengths = torch.full((G,), completion_only.shape[1],
                                    dtype=torch.long, device=device)
    if eos_id is not None:
        for g in range(G):
            row = completion_only[g]
            eos_positions = (row == eos_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                # +1 because we INCLUDE the EOS token itself as a generated token
                # (the model emitted it; we score the decision to emit EOS).
                completion_lengths[g] = int(eos_positions[0, 0].item()) + 1

    return out, completion_lengths


# =============================================================================
# Build training tensors from a generated [G, S_full]
# =============================================================================

def _build_labels_and_masks(
    full_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    completion_lens: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (input_ids[:-1], labels[1:]-with-IGNORE-mask, attention_mask[:-1]).

    Args:
        full_ids:        Long[N, S_full]
        prompt_lens:     Long[N] — number of REAL prompt tokens (excluding
                         left pad). The prompt is left-padded so REAL tokens
                         occupy positions [S_full - prompt_len - completion_total : ?].
                         To keep this simple we expose prompt_lens as the
                         length BEFORE generation began; the rollout caller
                         derives it from prompt_attention_mask.sum().
        completion_lens: Long[N] — number of GENERATED tokens including the
                         first EOS (or max_new_tokens if no EOS).
        pad_id:          right-pad token (we use the same id as left pad).

    Returns:
        input_ids:      Long[N, S_full - 1]  (full_ids[:, :-1])
        labels:         Long[N, S_full - 1]  (full_ids[:, 1:] with IGNORE_INDEX
                        outside the completion span)
        attention_mask: Long[N, S_full - 1]  (1 over [left pad] + prompt +
                        completion (up to EOS); 0 elsewhere)

    Geometry: for row i, the left pad runs over [0, S_full - prompt_lens[i] -
    completion_lens[i] - n_right_pad), the real prompt over the next
    prompt_lens[i] positions, the completion over the next completion_lens[i],
    and the right pad fills the tail. Since the data pipeline pre-pads ALL
    prompts to seq_len (with left padding), and `model.generate` appends on
    the right, the prompt occupies positions [S_full - max_new - prompt_lens[i]
    : S_full - max_new] — wait, no.

    Let's restate. After `generate`, full_ids has shape [N, S_prompt_buf +
    max_new_tokens] where S_prompt_buf is the prompt buffer length (=seq_len
    from the data pipeline, all rows the same). The REAL prompt tokens for
    row i live in positions [S_prompt_buf - prompt_lens[i], S_prompt_buf).
    The generated tokens live in [S_prompt_buf, S_prompt_buf + max_new) —
    but only the first completion_lens[i] of these are "real generations";
    the rest is pad.

    So the completion span (where labels should be valid) in `full_ids` is:
        [S_prompt_buf, S_prompt_buf + completion_lens[i])

    After the shift-by-1 to build (input_ids, labels), the same span in
    `labels` becomes:
        [S_prompt_buf - 1, S_prompt_buf - 1 + completion_lens[i])
    """
    N, S_full = full_ids.shape
    device = full_ids.device

    # S_prompt_buf is the same for all rows — the data pipeline padded every
    # prompt to seq_len. So:
    S_prompt_buf = S_full - int(completion_lens.max().item())
    # But completion_lens may vary across rows; we still slice from a uniform
    # S_prompt_buf because generate() pads the completion zone uniformly too.
    # Note we need the *max* across the batch since generate pads to that.
    # Actually generate returns shape [N, S_prompt_buf + max_new_tokens] for
    # all N rows — the "max_new" dimension is fixed. So:
    # S_prompt_buf = S_full - max_new_tokens. We pass it in implicitly via
    # full_ids.shape — but max_new_tokens isn't passed here. So just use:
    # S_prompt_buf = S_full - max_completion_len; for rows shorter than
    # max_completion_len, the trailing positions are pad.

    # input_ids[t] = full_ids[t], labels[t] = full_ids[t+1] for t in [0, S_full-1).
    input_ids = full_ids[:, :-1].contiguous()
    raw_labels = full_ids[:, 1:].contiguous()

    # Build the response mask: True at positions where labels[t] (= full_ids[t+1])
    # is a generated completion token, NOT a prompt token and NOT post-EOS pad.
    pos = torch.arange(S_full - 1, device=device).unsqueeze(0).expand(N, -1)   # [N, S_full-1]
    # full_ids[t+1] is a completion token iff (t+1) is in [S_prompt_buf,
    #   S_prompt_buf + completion_lens[i]), i.e. t is in [S_prompt_buf - 1,
    #   S_prompt_buf - 1 + completion_lens[i]).
    lo = S_prompt_buf - 1
    hi = (S_prompt_buf - 1) + completion_lens.unsqueeze(1)                     # [N, 1]
    response_mask_bool = (pos >= lo) & (pos < hi)                              # [N, S_full-1]

    labels = raw_labels.clone()
    labels[~response_mask_bool] = IGNORE_INDEX

    # Attention mask is 1 over (real prompt + completion-up-to-EOS), 0 over
    # left-pad and right-pad.
    # Real prompt positions in input_ids (= full_ids[:-1]) are [S_prompt_buf -
    # prompt_lens[i], S_prompt_buf). Completion positions are [S_prompt_buf,
    # S_prompt_buf + completion_lens[i]). They JOIN to [S_prompt_buf -
    # prompt_lens[i], S_prompt_buf + completion_lens[i]).
    attn_lo = S_prompt_buf - prompt_lens.unsqueeze(1)                          # [N, 1]
    attn_hi = S_prompt_buf + completion_lens.unsqueeze(1)                      # [N, 1]
    # The shift cuts the last input_ids position off, so attn_hi can be at
    # most S_full - 1.
    attn_hi = torch.minimum(attn_hi, torch.tensor(S_full - 1, device=device))
    attention_mask = ((pos >= attn_lo) & (pos < attn_hi)).long()

    return input_ids, labels, attention_mask


# =============================================================================
# Group-relative advantages
# =============================================================================

def group_normalize_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    """Within-group normalize: A_i = (r_i - mean_g) / (std_g + eps).

    Args:
        rewards: Float[P·G] — flat, prompt-major ordering (prompt 0's G
            completions, then prompt 1's G, ...). This is what HF's
            `num_return_sequences=G` produces naturally.
        group_size: G.

    Returns:
        Float[P·G] — within-prompt standardized advantage. Mean across
        each group is 0 by construction; per-group variance is ~1.

        Edge case: if all G rewards within a group are identical (very
        common when ALL miss the format/accuracy reward), `std = 0` and
        the normalized advantage is 0 for every member of that group.
        That group contributes ZERO gradient at this step — which is
        correct: the policy gets no signal from a group where it can't
        tell which response is better than which.
    """
    N = rewards.shape[0]
    assert N % group_size == 0, f"rewards length {N} not divisible by G={group_size}"
    P = N // group_size
    grouped = rewards.view(P, group_size)
    mean_g = grouped.mean(dim=1, keepdim=True)
    std_g = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - mean_g) / (std_g + eps)
    # If a group is degenerate (std == 0), the standardization makes all rows 0
    # already via (r - mean = 0). The +eps in the denominator prevents inf/NaN.
    return advantages.view(N)


# =============================================================================
# Top-level rollout
# =============================================================================

@torch.no_grad()
def generate_rollout(
    policy: nn.Module,
    reference: nn.Module,
    tokenizer,
    batch: dict,
    data_cfg: DataConfig,
    reward_cfg: RewardConfig,
    rl_cfg: RLConfig,
    device: str = "cuda",
) -> Rollout:
    """End-to-end rollout for one training step.

    Pipeline:
      1. For each prompt in `batch['prompt_input_ids']`, sample G completions.
      2. Decode each completion to text.
      3. Compute reward per completion (format + accuracy).
      4. Group-normalize advantages.
      5. Compute per-token old_logps (policy) and ref_logps (reference) on
         the (prompt + completion) sequences.
      6. Bundle into a Rollout dataclass.

    All forward passes here are `no_grad`. The gradient step happens in
    `loop.train_step`, which re-forwards the policy on the SAME sequences
    to get fresh log-probs that the autograd graph keeps.
    """
    prompt_ids = batch["prompt_input_ids"].to(device)            # [P, S_prompt]
    prompt_attn = batch["prompt_attention_mask"].to(device)      # [P, S_prompt]
    ground_truths: list[int] = batch["ground_truth"]
    P = prompt_ids.shape[0]
    G = rl_cfg.group_size

    pad_id = tokenizer.pad_token_id

    # ---- 1. Sample G completions per prompt ----------------------------------
    all_full: list[torch.Tensor] = []
    all_completion_lens: list[torch.Tensor] = []
    for i in range(P):
        full_i, lens_i = _sample_completions(
            policy=policy,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids[i],
            prompt_attn=prompt_attn[i],
            rl_cfg=rl_cfg,
            device=device,
        )
        all_full.append(full_i)
        all_completion_lens.append(lens_i)

    # Stack: [P*G, S_full]. (S_full is the same for every prompt because the
    # prompt buffer is uniform and max_new_tokens is fixed.)
    full_ids = torch.cat(all_full, dim=0)                        # [P*G, S_full]
    completion_lens = torch.cat(all_completion_lens, dim=0)      # [P*G]

    # ---- 2. Decode completions to text ---------------------------------------
    S_prompt_buf = prompt_ids.shape[1]
    completion_token_ids = full_ids[:, S_prompt_buf:]            # [P*G, max_new]
    completions_text: list[str] = []
    for i in range(P * G):
        clen = int(completion_lens[i].item())
        text = tokenizer.decode(completion_token_ids[i, :clen], skip_special_tokens=True)
        completions_text.append(text)

    # ---- 3. Compute rewards --------------------------------------------------
    # Expand ground truths to match the P*G ordering (prompt 0's G, then 1's G, ...).
    expanded_gts: list[int] = []
    for gt in ground_truths:
        expanded_gts.extend([gt] * G)
    rewards: list[RewardBreakdown] = compute_rewards(completions_text, expanded_gts, reward_cfg)
    rewards_total = torch.tensor([r.total for r in rewards], dtype=torch.float32, device=device)

    # ---- 4. Group-normalize advantages ---------------------------------------
    advantages = group_normalize_advantages(rewards_total, G, rl_cfg.adv_eps)

    # ---- 5. Build (input_ids, labels, attention_mask) ------------------------
    prompt_lens_per_prompt = prompt_attn.sum(dim=1).long()       # [P]
    # Tile per-prompt lengths to match P*G rows.
    prompt_lens = prompt_lens_per_prompt.repeat_interleave(G)     # [P*G]
    input_ids, labels, attention_mask = _build_labels_and_masks(
        full_ids, prompt_lens, completion_lens, pad_id,
    )

    # ---- 6. Score per-token logps under policy and reference (both no_grad) --
    # Policy "old" logps — what the policy thought of each generated token at
    # the moment it was generated. For K=1 these will equal the policy logps
    # recomputed in train_step (within numerical noise); we cache them so the
    # importance ratio is well-defined and K>1 generalizes.
    policy_out = policy(input_ids=input_ids, attention_mask=attention_mask)
    policy_logits = policy_out.logits if hasattr(policy_out, "logits") else policy_out
    old_logps = per_token_logps(policy_logits, labels)            # [P*G, S-1]

    # Reference logps — frozen, BF16 native; no autocast needed.
    ref_out = reference(input_ids=input_ids, attention_mask=attention_mask)
    ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out
    ref_logps = per_token_logps(ref_logits, labels)               # [P*G, S-1]

    response_mask = (labels != IGNORE_INDEX)
    response_lengths = response_mask.sum(dim=1)                   # [P*G]

    return Rollout(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        response_mask=response_mask,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        rewards=rewards,
        response_lengths=response_lengths,
        completions_text=completions_text,
    )


# =============================================================================
# Smoke test: build a tiny fake setup, exercise the pieces that don't need
# real weights (advantages + label/mask construction).
# =============================================================================

if __name__ == "__main__":
    print("--- rollout.py smoke test (advantage normalization + mask construction) ---")

    # Advantage normalization: 3 prompts, G=4 each. Within-group standardize.
    rewards_flat = torch.tensor([
        1.0, 1.0, 0.1, 0.1,   # prompt 0: 2 hits, 2 misses
        0.1, 0.1, 0.1, 0.1,   # prompt 1: degenerate — all same
        1.0, 0.1, 1.0, 1.0,   # prompt 2: 3 hits, 1 miss
    ])
    adv = group_normalize_advantages(rewards_flat, group_size=4, eps=1e-6)
    print(f"  rewards     : {rewards_flat.tolist()}")
    print(f"  advantages  : {[round(x, 3) for x in adv.tolist()]}")
    print(f"  group 0 mean: {adv[:4].mean().item():+.4f}  std: {adv[:4].std(unbiased=False).item():.4f}")
    print(f"  group 1 mean: {adv[4:8].mean().item():+.4f}  (degenerate — all 0)")

    # Label/mask construction: P=2 prompts, G=2, S_prompt_buf=6, max_new=4.
    # full_ids = [
    #   prompt0_completion0,  prompt0_completion1,
    #   prompt1_completion0,  prompt1_completion1,
    # ]
    # Each row is shape [10] = [S_prompt_buf=6 + max_new=4].
    # The prompt's REAL part lives in positions [4, 6) (i.e., last 2 of the
    # left-padded prompt). The completion lives in [6, 6+completion_len).
    full_ids = torch.tensor([
        [0, 0, 0, 0, 11, 12, 21, 22, 23, 99],     # prompt 0, comp 0 — eos at idx 9 (irrelevant for mask)
        [0, 0, 0, 0, 11, 12, 31, 32, 0, 0],       # prompt 0, comp 1 — 2 tokens then pad
        [0, 0, 0, 0, 13, 14, 41, 99, 0, 0],       # prompt 1, comp 0 — 2 tokens (incl eos at idx 7)
        [0, 0, 0, 0, 13, 14, 51, 52, 53, 54],     # prompt 1, comp 1 — full 4 tokens
    ])
    prompt_lens = torch.tensor([2, 2, 2, 2])
    completion_lens = torch.tensor([4, 2, 2, 4])
    inp, lab, attn = _build_labels_and_masks(full_ids, prompt_lens, completion_lens, pad_id=0)
    print(f"\n  full_ids shape: {tuple(full_ids.shape)}  -> after shift: {tuple(inp.shape)}")
    print(f"  labels for row 0 (4 completion tokens expected, others IGNORE):")
    print(f"    {lab[0].tolist()}")
    print(f"  attention_mask for row 1 (real prompt 2 + completion 2 = 4 ones):")
    print(f"    {attn[1].tolist()}  (sum={attn[1].sum().item()})")
