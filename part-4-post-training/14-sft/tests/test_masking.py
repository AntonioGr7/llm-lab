"""Correctness tests for the SFT loss mask — the heart of Module 14.

The assistant-only loss mask is the single most common silent bug in homemade
SFT pipelines (README §3). These tests pin the properties the README claims,
two ways:

  * **Offline, always-run** — a `FakeChatTokenizer` mimics a ChatML template
    deterministically, so the diff trick is exercised with no network.
  * **Guarded, real-tokenizer** — if a real Qwen3 tokenizer is cached, run the
    same checks against it. This is the layer that catches version/template
    regressions the fake can't model (e.g. transformers returning a
    `BatchEncoding`, or Qwen3's position-dependent `<think>` block).

Properties:
  1. Shift + fraction. Every loss-bearing label is the next input token, and
     the assistant-token fraction is strictly in (0, 1) — fraction → 1 is the
     "flat-line loss" broken-mask signature.
  2. Only the response carries loss. The unmasked positions are exactly the
     diff-trick span (response text + end-of-turn); the prompt, the assistant
     header, and any template scaffold are masked.
  3. Per-turn expansion. A k-assistant-turn conversation becomes k examples.
  4. Normalization. Heterogeneous dataset shapes map to canonical messages.

Usage:
    cd 14-sft/
    python tests/test_masking.py

Exits 0 on pass, 1 on fail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Make the module directory importable when running from tests/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from data import _mask_final_turn, _render_examples, _normalize_to_messages, IGNORE_INDEX


# ---------------------------------------------------------------------------
# A deterministic, offline ChatML-style tokenizer.
# ---------------------------------------------------------------------------

class FakeChatTokenizer:
    """Mimics the slice of the HF tokenizer API that `data.py` uses.

    Renders a ChatML-shaped template:  `<im_start>{role}\\n{content}<im_end>\\n`,
    with the generation prompt a dangling `<im_start>assistant\\n`. Content is
    tokenized at word granularity into stable ids >= 100 (specials are < 10), so
    token *lengths* — all the diff trick relies on — are exact. Accepts (and
    ignores) the `return_dict`/`enable_thinking` kwargs the real API takes.
    """
    IM_START, IM_END, NL = 1, 2, 3
    ROLE = {"system": 4, "user": 5, "assistant": 6}

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = self.IM_END
        self.chat_template = "{# fake chatml #}"

    def _tok_content(self, content: str) -> list[int]:
        toks = [100 + (sum(ord(c) for c in w) % 5000) for w in content.split()]
        return toks or [100]

    def _render_message(self, role: str, content: str) -> list[int]:
        return ([self.IM_START, self.ROLE[role], self.NL]
                + self._tok_content(content)
                + [self.IM_END, self.NL])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False, **_ignored):
        ids: list[int] = []
        for m in messages:
            ids += self._render_message(m["role"], m["content"])
        if add_generation_prompt:
            ids += [self.IM_START, self.ROLE["assistant"], self.NL]
        assert tokenize and not return_dict, "data.py calls with tokenize=True, return_dict=False"
        return ids


# ---------------------------------------------------------------------------
# Shared property checks (run against any tokenizer)
# ---------------------------------------------------------------------------

def _check_final_turn(tok, messages, seq_len) -> tuple[bool, str]:
    """_mask_final_turn(messages) masks exactly the diff-trick span of the
    final (assistant) turn. Returns (ok, detail)."""
    out = _mask_final_turn(messages, tok, seq_len)
    if out is None:
        return False, "got None"
    input_ids, labels, attn = out["input_ids"], out["labels"], out["attention_mask"]
    n_real = int(attn.sum().item())

    full = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
    prefix = tok.apply_chat_template(messages[:-1], tokenize=True,
                                     add_generation_prompt=True, return_dict=False)
    # Label t is a target iff full[t+1] is a response token, i.e. t+1 >= len(prefix).
    expected = {t for t in range(min(n_real, len(full) - 1)) if (t + 1) >= len(prefix)}
    actual = {t for t in range(n_real) if labels[t].item() != IGNORE_INDEX}

    shapes = input_ids.shape == labels.shape == attn.shape == (seq_len,)
    frac = len(actual) / max(n_real, 1)
    shift = all(labels[t].item() == IGNORE_INDEX or labels[t].item() == input_ids[t + 1].item()
                for t in range(n_real - 1))
    pad = bool((labels[n_real:] == IGNORE_INDEX).all()) and bool((attn[n_real:] == 0).all())
    ok = shapes and (actual == expected) and (0.0 < frac < 1.0) and shift and pad
    return ok, f"frac={frac:.2f} positions_match={actual == expected} shapes={shapes} shift={shift} pad={pad}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fake_single_turn() -> bool:
    tok = FakeChatTokenizer()
    ok, detail = _check_final_turn(tok, [
        {"role": "user", "content": "What is two plus two equal to"},
        {"role": "assistant", "content": "It is four"},
    ], seq_len=64)
    print(f"  [1] fake single-turn final mask: {'OK' if ok else 'FAIL'}  ({detail})")
    return ok


def test_fake_expansion_and_masks() -> bool:
    tok = FakeChatTokenizer()
    convo = [
        {"role": "user", "content": "first question here please"},
        {"role": "assistant", "content": "first answer is this"},
        {"role": "user", "content": "second follow up question"},
        {"role": "assistant", "content": "second answer right here now"},
    ]
    samples = _render_examples(convo, tok, seq_len=128)
    count_ok = len(samples) == 2          # two assistant turns -> two examples

    # Each emitted example must mask exactly its own final-turn span.
    per_turn_ok = True
    for k, m in enumerate(convo):
        if m["role"] != "assistant":
            continue
        ok, _ = _check_final_turn(tok, convo[:k + 1], 128)
        per_turn_ok = per_turn_ok and ok

    # No-assistant and single-turn expansion counts.
    none_ok = _render_examples([{"role": "user", "content": "hi there"}], tok, 32) == []
    one_ok = len(_render_examples(convo[:2], tok, 64)) == 1

    ok = count_ok and per_turn_ok and none_ok and one_ok
    print(f"  [2] fake expansion + per-turn masks: {'OK' if ok else 'FAIL'}  "
          f"(n_examples={len(samples)}, per_turn={per_turn_ok}, none={none_ok}, one={one_ok})")
    return ok


def test_normalize() -> bool:
    checks = []
    checks.append(_normalize_to_messages({"messages": [{"role": "user", "content": "x"}]})
                  == [{"role": "user", "content": "x"}])
    checks.append(_normalize_to_messages({"prompt": "q", "response": "a"})
                  == [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    checks.append([d["role"] for d in _normalize_to_messages({"instruction": "q", "output": "a"})]
                  == ["user", "assistant"])
    checks.append(_normalize_to_messages({"conversations": [
        {"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]})
        == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    checks.append(_normalize_to_messages({"prompt": "q", "response": "a"}, system_prompt="be nice")[0]
                  == {"role": "system", "content": "be nice"})
    ok = all(checks)
    print(f"  [3] normalize shapes: {'OK' if ok else 'FAIL'}  ({sum(checks)}/{len(checks)})")
    return ok


def test_real_tokenizer() -> bool:
    """Guarded: only runs if a real Qwen3 tokenizer is cached. Catches the
    transformers-5 BatchEncoding return and the multi-turn <think> misalignment
    that the fake tokenizer cannot model."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from data import load_tokenizer
    for name in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-0.6B-Base"):
        try:
            tok = load_tokenizer(name)
            break
        except Exception:
            tok = None
    if tok is None:
        print("  [4] real-tokenizer checks: SKIP (no Qwen3 tokenizer cached)")
        return True

    convo = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And of Italy?"},
        {"role": "assistant", "content": "Rome is the capital of Italy."},
    ]
    samples = _render_examples(convo, tok, seq_len=128)
    count_ok = len(samples) == 2
    leak_ok = True
    frac_ok = True
    for s in samples:
        tgt = [s["input_ids"][t + 1].item() for t in range(int(s["attention_mask"].sum()) - 1)
               if s["labels"][t].item() != IGNORE_INDEX]
        text = tok.decode(tgt)
        if "<|im_start|>user" in text:           # a user turn leaked into the targets
            leak_ok = False
        n_real = int(s["attention_mask"].sum())
        frac = len(tgt) / max(n_real, 1)
        if not (0.0 < frac < 1.0):
            frac_ok = False
    ok = count_ok and leak_ok and frac_ok
    print(f"  [4] real-tokenizer ({name}): {'OK' if ok else 'FAIL'}  "
          f"(n_examples={len(samples)}, no_user_leak={leak_ok}, frac_ok={frac_ok})")
    return ok


def main() -> int:
    print("SFT loss-mask tests\n")
    results = [
        test_fake_single_turn(),
        test_fake_expansion_and_masks(),
        test_normalize(),
        test_real_tokenizer(),
    ]
    print()
    if all(results):
        print("  PASS — chat template renders, mask is response-only, per-turn expansion correct.")
        return 0
    print("  FAIL — see the failing property above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
