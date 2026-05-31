"""Teacher-rollout pipeline for SDFT + on-policy distillation.

Both distillation flavors share the same data structure: per training
step we produce a batch of (student_input, completion, teacher_logits)
triples. The student then does a fresh forward in `loop.py` to compute
per-token logits at the same response positions, and the forward KL is
computed between teacher and student distributions.

What differs by mode:

  - **sdft**:        the *student model itself* is the teacher, queried
                     in `torch.no_grad` with K demonstrations prepended
                     in-context. No second model in memory.
  - **on_policy**:   a *separate* teacher model (typically larger) scores
                     the student's rollouts. Two models in memory; the
                     student generates, the teacher scores.

In both cases the rollout returns the same `DistillRollout` dataclass —
loop.py doesn't need to know which mode produced it.

Key implementation details:

1. **Left-padding both teacher and student inputs** so that completion
   tokens land at a fixed column index. This makes the per-row position
   arithmetic (extract logits at response positions) a single slice.
2. **Completion IDs are shared** between teacher and student. Whoever
   generates them (the teacher in SDFT — running the student model with
   demos; the student in on_policy), the SAME token ids are then
   appended to the student-side input for the gradient forward.
3. **Teacher log-probs are computed in FP32 and stored detached**. The
   student forward in loop.py computes its own log-probs in FP32 (under
   autocast for the model body but log_softmax in FP32 — same numerical
   pattern as M17/M18).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from config import DataConfig, DistillationConfig
from make_tooluse_corpus import Example
from model import generation_mode


IGNORE_INDEX = -100


# =============================================================================
# Rollout container
# =============================================================================

@dataclass
class DistillRollout:
    """One micro-batch of distillation training data.

    Shape conventions (B = batch size, L_s = student-input length,
    C = completion length after right-padding to the batch-max):

      student_input_ids:        Long[B, L_s + C] — left-padded student
                                inputs concatenated with completion tokens.
      student_attention_mask:   Long[B, L_s + C].
      completion_start:         int — column index in student_input_ids
                                where the FIRST completion token lives
                                (L_s after left-padding). Same for all rows
                                because we left-padded uniformly.
      completion_lengths:       Long[B] — real completion length per row
                                (up to and INCLUDING the first EOS, or
                                max_new_tokens if no EOS).
      response_mask:            Bool[B, C] — True where the position is a
                                valid completion token (False past EOS /
                                in right-pad).
      teacher_log_probs:        Float[B, C, V] — teacher's log-prob
                                distribution at each completion position.
                                FP32, detached (the KL target).
      completion_texts:         list[str] — decoded for logging.
      teacher_input_summary:    str — one-line "what the teacher saw"
                                description for the startup banner /
                                logs (e.g. "[system + 8 demos + prompt]"
                                for SDFT, "[system + prompt]" for on_policy).
    """
    student_input_ids: torch.Tensor
    student_attention_mask: torch.Tensor
    completion_start: int
    completion_lengths: torch.Tensor
    response_mask: torch.Tensor
    teacher_log_probs: torch.Tensor
    completion_texts: list[str]
    teacher_input_summary: str


# =============================================================================
# Chat-template rendering
# =============================================================================

def _build_messages_for_teacher(
    user: str,
    demonstrations: list[Example],
    system_prompt: str,
    use_demos: bool,
) -> list[dict]:
    """Assemble the chat-template message list for the teacher's input.

    Two modes:
      - `use_demos=True` (SDFT): system + K demo turns + final user turn.
      - `use_demos=False` (on_policy): just system + user.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if use_demos:
        for d in demonstrations:
            messages.append({"role": "user", "content": d.user})
            messages.append({"role": "assistant", "content": d.assistant})
    messages.append({"role": "user", "content": user})
    return messages


def _build_messages_for_student(user: str, system_prompt: str) -> list[dict]:
    """Student NEVER sees demos. System + user only."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    return messages


def _render(tokenizer, messages: list[dict]) -> list[int]:
    """Apply chat template with `add_generation_prompt=True` + `enable_thinking=False`."""
    base = dict(tokenize=True, return_dict=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **base)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **base)


def _left_pad_batch(ids_list: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Left-pad a list of variable-length id sequences to a uniform tensor.

    Returns:
      ids:           Long[B, L_max]
      attn_mask:     Long[B, L_max] — 0 over pad, 1 over real tokens.
      L_max:         int.
    """
    L_max = max(len(ids) for ids in ids_list)
    B = len(ids_list)
    out_ids = torch.full((B, L_max), pad_id, dtype=torch.long)
    out_attn = torch.zeros((B, L_max), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        n = len(ids)
        out_ids[i, L_max - n:] = torch.tensor(ids, dtype=torch.long)
        out_attn[i, L_max - n:] = 1
    return out_ids, out_attn, L_max


# =============================================================================
# Teacher generation
# =============================================================================

@torch.no_grad()
def _generate_with_teacher(
    teacher_model: nn.Module,
    teacher_ids: torch.Tensor,
    teacher_attn: torch.Tensor,
    tokenizer,
    distill_cfg: DistillationConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate completions from a batch of teacher inputs (B prompts at once).

    Returns:
      teacher_full_ids:     Long[B, L_t_max + max_new] — input + completion.
      completion_lengths:   Long[B] — actual length up to (and including)
                            first EOS or max_new_tokens.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    L_t_max = teacher_ids.shape[1]

    with generation_mode(teacher_model):
        out = teacher_model.generate(
            input_ids=teacher_ids.to(device),
            attention_mask=teacher_attn.to(device),
            do_sample=True,
            temperature=distill_cfg.sampling_temperature,
            top_p=distill_cfg.top_p,
            top_k=distill_cfg.top_k if distill_cfg.top_k > 0 else None,
            max_new_tokens=distill_cfg.max_new_tokens,
            num_return_sequences=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=False,
        )

    # Real completion length per row = position of first EOS (+1 to include
    # it as a generated token), capped at max_new_tokens.
    completion_only = out[:, L_t_max:]                          # [B, max_new]
    B, max_new = completion_only.shape
    completion_lengths = torch.full((B,), max_new, dtype=torch.long, device=device)
    if eos_id is not None:
        for i in range(B):
            pos = (completion_only[i] == eos_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                completion_lengths[i] = int(pos[0, 0].item()) + 1
    return out, completion_lengths


# =============================================================================
# Teacher logits at completion positions
# =============================================================================

@torch.no_grad()
def _score_teacher_logprobs(
    teacher_model: nn.Module,
    teacher_full_ids: torch.Tensor,
    teacher_attn: torch.Tensor,
    L_t_max: int,
    max_new: int,
    device: str,
) -> torch.Tensor:
    """Forward the teacher on its (input + completion) and extract LOG-PROBS
    at the completion positions.

    Recall the autoregressive shift: logits[t] predicts position t+1.
    The first completion token is at column L_t_max (just past the input),
    predicted from logits[L_t_max - 1]. So the C completion log-probs sit
    at logit columns [L_t_max - 1, L_t_max - 1 + max_new).

    Returns Float[B, max_new, V] — FP32 log-probs.
    """
    teacher_out = teacher_model(
        input_ids=teacher_full_ids.to(device),
        attention_mask=teacher_attn.to(device),
    )
    logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out
    resp_logits = logits[:, L_t_max - 1 : L_t_max - 1 + max_new, :]   # [B, max_new, V]
    log_probs = torch.log_softmax(resp_logits.float(), dim=-1)
    return log_probs


# =============================================================================
# Top-level rollout
# =============================================================================

@torch.no_grad()
def generate_rollout(
    student_model: nn.Module,
    teacher_model: Optional[nn.Module],
    tokenizer,
    batch: dict,
    demonstrations: list[Example],
    data_cfg: DataConfig,
    distill_cfg: DistillationConfig,
    device: str = "cuda",
) -> DistillRollout:
    """End-to-end rollout for one distillation step.

    Args:
        student_model: the trainable model. Used as TEACHER in sdft mode
            (no_grad + demos in context); used as STUDENT in both modes
            (but the student forward happens in loop.train_step, not here).
        teacher_model: the *separate* teacher. None for sdft.
        batch: dict from the dataloader — {"user": [B str], ...}.
        demonstrations: K demo Examples (used iff mode == "sdft").

    Returns:
        DistillRollout with everything loop.train_step needs.

    Pipeline by mode:

      sdft:
        1. Teacher input = system + K demos + user.
           Run student_model.generate(teacher_input) under no_grad → completion.
        2. Forward student_model on (teacher_input + completion) under no_grad
           → teacher_log_probs at completion positions.
        3. Build student input = system + user (no demos).
           Concatenate with completion ids → student_full_ids.

      on_policy:
        1. Student input = teacher input = system + user (no demos).
           Run student_model.generate(student_input) under no_grad → completion.
        2. Forward teacher_model on (input + completion) under no_grad
           → teacher_log_probs at completion positions.
        3. student_full_ids = same input + completion (= what we just used).
    """
    users: list[str] = batch["user"]
    B = len(users)

    if distill_cfg.mode == "sdft":
        # ---- SDFT path: same model, with demos as teacher ----
        teacher_msgs = [
            _build_messages_for_teacher(u, demonstrations, data_cfg.system_prompt, use_demos=True)
            for u in users
        ]
        student_msgs = [
            _build_messages_for_student(u, data_cfg.system_prompt)
            for u in users
        ]
        teacher_summary = (
            f"[system + {len(demonstrations)} demos + prompt] (SDFT — same model)"
        )
        # The model that GENERATES the rollout. In SDFT it's the student model
        # itself, just with demos in its input.
        generator = student_model
        # The model that SCORES the teacher log-probs.
        scorer = student_model

    elif distill_cfg.mode == "on_policy":
        if teacher_model is None:
            raise ValueError("on_policy mode requires a separate teacher model")
        teacher_msgs = [
            _build_messages_for_teacher(u, [], data_cfg.system_prompt, use_demos=False)
            for u in users
        ]
        student_msgs = teacher_msgs   # same input in on_policy
        teacher_summary = (
            f"[system + prompt] (on-policy — separate teacher)"
        )
        # Student generates; teacher scores. The classic GKD-style flow.
        generator = student_model
        scorer = teacher_model
    else:
        raise ValueError(
            f"rollout.generate_rollout doesn't support mode {distill_cfg.mode!r} "
            f"(offline is handled outside this module)"
        )

    pad_id = tokenizer.pad_token_id

    # ---- 1. Tokenize + left-pad the teacher inputs --------------------------
    teacher_ids_list = [_render(tokenizer, m) for m in teacher_msgs]
    teacher_ids, teacher_attn, L_t_max = _left_pad_batch(teacher_ids_list, pad_id)

    # ---- 2. Generate completions from the teacher-side input ----------------
    teacher_full_ids, completion_lengths = _generate_with_teacher(
        generator, teacher_ids, teacher_attn, tokenizer, distill_cfg, device,
    )
    max_new = teacher_full_ids.shape[1] - L_t_max
    completion_ids = teacher_full_ids[:, L_t_max:]      # [B, max_new] — what was generated

    # Build extended attention mask: real teacher tokens + completion tokens.
    teacher_full_attn = torch.cat([
        teacher_attn.to(device),
        torch.ones((B, max_new), dtype=torch.long, device=device),
    ], dim=1)
    # Zero out attention past EOS in each row.
    pos_in_completion = torch.arange(max_new, device=device).unsqueeze(0).expand(B, -1)
    completion_valid = pos_in_completion < completion_lengths.unsqueeze(1)         # [B, max_new]
    teacher_full_attn[:, L_t_max:] = completion_valid.long()

    # ---- 3. Score the teacher's log-probs at the completion positions -------
    teacher_log_probs = _score_teacher_logprobs(
        scorer, teacher_full_ids, teacher_full_attn, L_t_max, max_new, device,
    )

    # ---- 4. Build the STUDENT-side input (system + user, no demos) ---------
    student_ids_list = [_render(tokenizer, m) for m in student_msgs]
    student_ids, student_attn, L_s_max = _left_pad_batch(student_ids_list, pad_id)

    student_full_ids = torch.cat([student_ids.to(device), completion_ids], dim=1)
    student_full_attn = torch.cat([
        student_attn.to(device),
        completion_valid.long(),
    ], dim=1)

    # Response mask over the C completion positions.
    response_mask = completion_valid                      # Bool[B, max_new]

    # ---- 5. Decode completions for logging ----------------------------------
    completion_texts: list[str] = []
    for i in range(B):
        clen = int(completion_lengths[i].item())
        text = tokenizer.decode(completion_ids[i, :clen].cpu(), skip_special_tokens=True)
        completion_texts.append(text)

    return DistillRollout(
        student_input_ids=student_full_ids,
        student_attention_mask=student_full_attn,
        completion_start=L_s_max,
        completion_lengths=completion_lengths,
        response_mask=response_mask,
        teacher_log_probs=teacher_log_probs.detach(),
        completion_texts=completion_texts,
        teacher_input_summary=teacher_summary,
    )


# =============================================================================
# Smoke test — exercise the pieces that don't need real weights
# =============================================================================

if __name__ == "__main__":
    print("--- rollout.py smoke test (rendering + padding only) ---")

    # Tiny fake demonstrations
    demos = [
        Example(user="What's the weather in Paris?", tool="get_weather",
                args={"city": "Paris", "when": "today"}),
        Example(user="Calculate 7 * 8.", tool="calculator",
                args={"expression": "7 * 8"}),
    ]
    system = "Tool-use assistant."

    teacher_msgs_sdft = _build_messages_for_teacher(
        "What is 3 + 5?", demos, system, use_demos=True,
    )
    teacher_msgs_op = _build_messages_for_teacher(
        "What is 3 + 5?", demos, system, use_demos=False,
    )
    student_msgs = _build_messages_for_student("What is 3 + 5?", system)

    print(f"  SDFT teacher messages: {len(teacher_msgs_sdft)} turns "
          f"(1 system + {len(demos)} demos × 2 turns + 1 user)")
    print(f"  on_policy teacher messages: {len(teacher_msgs_op)} turns "
          f"(1 system + 1 user — same as student)")
    print(f"  student messages: {len(student_msgs)} turns")

    # Left-pad demo (no tokenizer needed)
    ids_list = [[1, 2, 3, 4, 5], [6, 7, 8]]
    out_ids, out_attn, L_max = _left_pad_batch(ids_list, pad_id=0)
    print(f"\n  _left_pad_batch on [{ids_list[0]}, {ids_list[1]}]:")
    print(f"    L_max = {L_max}")
    print(f"    ids = {out_ids.tolist()}")
    print(f"    attn = {out_attn.tolist()}")
