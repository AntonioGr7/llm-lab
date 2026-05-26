"""
landscape.py — tiny utilities for the Module 13 notebook.

This module has no training code. Module 13 is the concepts/map module for Part 4;
the actual post-training runs start in Module 14. These helpers exist so the
notebook can demonstrate the three things every student should see before they
write a line of post-training code:

  1) The gap.            What a base model produces vs what an aligned model
                         produces, given the same prompt.
  2) The structure.      How a chat template turns a list of role-tagged
                         messages into a single token stream the model can be
                         trained on.
  3) The alignment tax.  How much next-token modeling ability is sacrificed
                         (measured as perplexity on raw web text) when a model
                         goes through SFT/DPO/RL.

CPU-friendly. The default model pair is Qwen3-0.6B-Base / Qwen3-0.6B — ~600M
parameters each, slow on CPU but bearable for a handful of short generations.
On a GPU it's instant. The fine-tuning runs themselves (Modules 14-17) target
Qwen3-1.7B on a single A100/H100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

DEFAULT_BASE = "Qwen/Qwen3-0.6B-Base"
DEFAULT_INSTRUCT = "Qwen/Qwen3-0.6B"


@dataclass
class ModelPair:
    """A base model and its post-trained counterpart, sharing a tokenizer."""

    base: torch.nn.Module
    instruct: torch.nn.Module
    tokenizer: object
    base_name: str
    instruct_name: str


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_pair(
    base_name: str = DEFAULT_BASE,
    instruct_name: str = DEFAULT_INSTRUCT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ModelPair:
    """Load a (base, instruct) pair that share a tokenizer.

    Both models are loaded with `torch_dtype=dtype` and moved to `device`.
    The instruct tokenizer is the source of truth — we assume the base and
    instruct were aligned on the same tokenizer (true for the Qwen3 line).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    dtype = dtype or pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(instruct_name)
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(device)
    instruct = AutoModelForCausalLM.from_pretrained(instruct_name, torch_dtype=dtype).to(device)
    base.eval()
    instruct.eval()
    return ModelPair(base=base, instruct=instruct, tokenizer=tokenizer,
                     base_name=base_name, instruct_name=instruct_name)


# ---------------------------------------------------------------------------
# 1) The gap — same prompt, two completions
# ---------------------------------------------------------------------------

@torch.no_grad()
def complete(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Greedy-ish completion. Used for the side-by-side base vs instruct demo."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compare_completions(
    pair: ModelPair,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    instruct_uses_chat_template: bool = True,
) -> dict[str, str]:
    """Run the same prompt through both models. Returns {model_name: completion}.

    `instruct_uses_chat_template=True` wraps the prompt in the instruct model's
    chat template (turning a raw string into a proper user-turn before the
    model generates an assistant turn). The base model always sees the raw
    string — that *is* the comparison we're making.
    """
    base_text = complete(pair.base, pair.tokenizer, prompt,
                         max_new_tokens=max_new_tokens, temperature=temperature)

    if instruct_uses_chat_template:
        messages = [{"role": "user", "content": prompt}]
        instruct_prompt = pair.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        instruct_prompt = prompt

    instruct_text = complete(pair.instruct, pair.tokenizer, instruct_prompt,
                             max_new_tokens=max_new_tokens, temperature=temperature)
    return {pair.base_name: base_text, pair.instruct_name: instruct_text}


# ---------------------------------------------------------------------------
# 2) The structure — chat templates
# ---------------------------------------------------------------------------

def render_chat_template(tokenizer, messages: list[dict]) -> str:
    """Render a list of role-tagged messages into the model's chat format.

    The output is the exact token stream the instruct model was trained to see.
    Print this in the notebook — students should *look* at the special tokens
    (`<|im_start|>`, `<|im_end|>`, etc.) and understand that SFT data is just
    a giant pile of these strings with the loss masked to "assistant" turns.
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def assistant_response_token_mask(tokenizer, messages: list[dict]) -> torch.Tensor:
    """Return a 1-D bool mask over the tokenized chat: True for tokens that
    belong to assistant turns (the ones SFT computes loss on).

    This is illustrative — real SFT trainers (TRL, Axolotl) compute this mask
    by re-rendering the conversation with each assistant turn redacted and
    diffing the token ids. We do the same trick here.
    """
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    mask = torch.zeros(len(full_ids), dtype=torch.bool)

    cursor = 0
    for i, msg in enumerate(messages):
        prefix_ids = tokenizer.apply_chat_template(
            messages[:i + 1], tokenize=True, add_generation_prompt=False,
        )
        if msg["role"] == "assistant":
            mask[cursor:len(prefix_ids)] = True
        cursor = len(prefix_ids)

    return mask


# ---------------------------------------------------------------------------
# 3) The alignment tax — perplexity on raw text
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_nll_per_token(
    model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 512,
) -> float:
    """Mean negative-log-likelihood per token over a list of strings.

    Lower is better. Perplexity is `exp(mean_nll)`. We report nll directly
    because the difference between base and instruct is what matters; the
    absolute number is meaningless without a reference.
    """
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits[:, :-1, :].float()
        targets = ids[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += targets.numel()
    return total_nll / max(total_tokens, 1)


def alignment_tax(pair: ModelPair, texts: list[str], max_length: int = 512) -> dict:
    """Compare base vs instruct perplexity on the same raw-text corpus.

    A *positive* tax means the instruct model is worse at next-token modeling
    than the base model — the alignment process traded some pretraining
    ability for instruction-following. This is the price you pay; the rest
    of Part 4 is about minimizing it.

    Returns: {"base_nll": float, "instruct_nll": float, "tax_nats_per_tok": float}
    """
    import math
    base_nll = mean_nll_per_token(pair.base, pair.tokenizer, texts, max_length)
    inst_nll = mean_nll_per_token(pair.instruct, pair.tokenizer, texts, max_length)
    return {
        "base_nll": base_nll,
        "instruct_nll": inst_nll,
        "tax_nats_per_tok": inst_nll - base_nll,
        "base_ppl": math.exp(base_nll),
        "instruct_ppl": math.exp(inst_nll),
    }


# ---------------------------------------------------------------------------
# Tiny CLI sanity check (not a training entrypoint)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {DEFAULT_BASE} and {DEFAULT_INSTRUCT}...")
    pair = load_pair()
    out = compare_completions(pair, "Write a haiku about a cat.")
    for name, text in out.items():
        print(f"\n=== {name} ===\n{text}")
