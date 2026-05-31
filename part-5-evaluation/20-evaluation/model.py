"""Model loading + the two forward modes evaluation needs.

Evaluation touches a model in exactly two ways, and they are different enough
that a lot of harness disagreement comes from conflating them:

  1. **Likelihood scoring** (`continuation_logprob`) — for multiple-choice.
     No generation. Teacher-force the prompt+continuation in ONE forward pass
     and read off the summed log-prob of the continuation tokens. Cheap,
     deterministic, exactly reproducible.

  2. **Generation** (`generate`) — for generative + instruction-following
     suites and for the judge. Autoregressive decode, greedy or sampled.

`build_model` loads a HF causal LM (optionally a Module 13-18 DCP checkpoint),
matching the load conventions used across the course. Nothing here touches the
GPU at import time.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


def load_tokenizer(name: str, padding_side: str = "left"):
    """Load the tokenizer; left-pad for generation (HF generate convention)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(cfg, device: Optional[str] = None):
    """Load the model to evaluate. `cfg` is an EvalConfig.ModelConfig.

    If `cfg.checkpoint` is set, loads the architecture from `cfg.name` and
    overlays the DCP checkpoint (the format Modules 13-18 save). Otherwise
    evaluates `cfg.name` as published. Always `.eval()`, no grad.
    """
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU + bf16 is slow/unsupported for some ops; fall back to fp32 on CPU.
    if device == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.name, dtype=dtype, use_cache=True)
    if cfg.checkpoint:
        _load_dcp_checkpoint(model, cfg.checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_dcp_checkpoint(model, ckpt_dir: str):
    """Load a distributed-checkpoint (the format train.py modules save).

    Tries the course's `checkpoint.load` helper if present in the path,
    else falls back to torch.distributed.checkpoint's model-state load.
    """
    try:
        from checkpoint import load as load_ckpt  # course helper, if vendored here
        load_ckpt(model, optimizer=None, ckpt_dir=ckpt_dir)
        return
    except Exception:
        pass
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, set_model_state_dict)
    sd = get_model_state_dict(model)
    dcp.load({"model": sd}, checkpoint_id=ckpt_dir)
    set_model_state_dict(model, sd)


@contextmanager
def generation_mode(model):
    """Toggle KV cache + eval mode for `.generate()`, then restore.

    Same load-bearing pattern as Modules 18/18: generation wants `use_cache`
    on, training/scoring wants it off. Here the model is eval-only anyway, but
    we keep the context manager so generation and scoring don't fight over the
    cache flag.
    """
    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", True)
    model.eval()
    model.config.use_cache = True
    try:
        yield model
    finally:
        model.config.use_cache = prev_cache
        if was_training:
            model.train()


@torch.no_grad()
def continuation_logprob(model, tok, prompt: str, continuation: str,
                         device: str) -> tuple[float, int, int]:
    """Sum log-prob of `continuation` given `prompt` — the MC scoring primitive.

    Teacher-forces [prompt + continuation] in one forward pass, then sums the
    log-probs of exactly the continuation tokens (shifted by one: the logits
    at position t predict token t+1). Returns:
        (summed_logprob, n_continuation_tokens, n_continuation_bytes)
    so the caller can apply raw / per-token / per-byte normalization
    (benchmarks.score_mc).

    Implementation detail that bites people: tokenize prompt and prompt+cont
    SEPARATELY and diff the lengths, because tokenizing the continuation alone
    can merge across the boundary (the space before the first cont token may
    bond differently). We tokenize the full string and locate the continuation
    by re-tokenizing the prompt to find the split index.
    """
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + continuation, add_special_tokens=False)["input_ids"]
    start = len(prompt_ids)
    cont_ids = full_ids[start:]
    if not cont_ids:
        return float("-inf"), 0, len(continuation.encode("utf-8"))

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits  # [1, T, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # logits at position t-1 predict token t; continuation tokens are at
    # positions [start, len(full)) so their predictors are [start-1, len-1).
    total = 0.0
    for i, tok_id in enumerate(cont_ids):
        pos = start + i - 1
        total += log_probs[0, pos, tok_id].item()
    n_bytes = len(continuation.encode("utf-8"))
    return total, len(cont_ids), n_bytes


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new_tokens: int,
             greedy: bool, temperature: float, top_p: float, device: str) -> str:
    """Chat-template generation. Returns the decoded completion text only."""
    try:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False).to(device)
    except TypeError:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(device)
    with generation_mode(model):
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(not greedy),
            temperature=temperature if not greedy else 1.0,
            top_p=top_p if not greedy else 1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def complete(model, tok, prompt: str, max_new_tokens: int, greedy: bool,
             temperature: float, top_p: float, device: str) -> str:
    """Raw (non-chat) completion — for few-shot prompts that aren't chat turns."""
    input_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with generation_mode(model):
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=(not greedy),
            temperature=temperature if not greedy else 1.0,
            top_p=top_p if not greedy else 1.0,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
