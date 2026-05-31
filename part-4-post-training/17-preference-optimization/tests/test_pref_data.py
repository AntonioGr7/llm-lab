"""Correctness tests for the preference-pair data pipeline.

Uses the same FakeChatTokenizer pattern as Module 15's test_masking.py — a
deterministic ChatML-shaped renderer with stable token lengths — so the
diff-mask logic is exercised end-to-end without HF downloads.

Properties checked:
  1. `_normalize_to_pair` accepts the three canonical shapes (messages-list,
     string-completion, and `messages`/`rejected_messages`).
  2. Malformed rows (missing fields, last turn not assistant) return None.
  3. With the fake tokenizer, a chosen/rejected pair yields the right
     dictionary of six tensors with IGNORE_INDEX outside the response.
  4. Different chosen vs. rejected content -> different unmasked target tokens.

Usage:
    cd 17-preference-optimization/
    python tests/test_pref_data.py

Exits 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from data import _normalize_to_pair, _mask_final_turn, IGNORE_INDEX


# ---------------------------------------------------------------------------
# Fake tokenizer (same shape as Module 15's, copied for self-containedness)
# ---------------------------------------------------------------------------

class FakeChatTokenizer:
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
# Tests
# ---------------------------------------------------------------------------

def test_normalize_messages_shape() -> bool:
    """UltraFeedback-style: chosen/rejected are message lists."""
    row = {
        "prompt": "Hello?",
        "chosen":   [{"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "Hi!"}],
        "rejected": [{"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "go away"}],
    }
    out = _normalize_to_pair(row)
    ok = (out is not None
          and out[0][-1]["role"] == "assistant" and out[0][-1]["content"] == "Hi!"
          and out[1][-1]["role"] == "assistant" and out[1][-1]["content"] == "go away")
    print(f"  [1] _normalize_to_pair messages-list shape: {'OK' if ok else 'FAIL'}")
    return ok


def test_normalize_string_shape() -> bool:
    """HH-RLHF style: chosen/rejected are bare strings — wrap as 2-turn convos."""
    row = {"prompt": "Hello?", "chosen": "Hi!", "rejected": "go away"}
    out = _normalize_to_pair(row)
    ok = (out is not None
          and out[0] == [{"role": "user", "content": "Hello?"},
                         {"role": "assistant", "content": "Hi!"}]
          and out[1] == [{"role": "user", "content": "Hello?"},
                         {"role": "assistant", "content": "go away"}])
    print(f"  [2] _normalize_to_pair string shape: {'OK' if ok else 'FAIL'}")
    return ok


def test_normalize_messages_rejected_messages_shape() -> bool:
    """`messages` + `rejected_messages` style."""
    row = {
        "messages": [{"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "Hi!"}],
        "rejected_messages": [{"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "go away"}],
    }
    out = _normalize_to_pair(row)
    ok = out is not None and out[0][-1]["content"] == "Hi!" and out[1][-1]["content"] == "go away"
    print(f"  [3] _normalize_to_pair messages/rejected_messages shape: {'OK' if ok else 'FAIL'}")
    return ok


def test_normalize_rejects_malformed() -> bool:
    """Missing fields / non-assistant last turn -> None."""
    none_for_missing = _normalize_to_pair({"prompt": "hi"}) is None
    bad_role_chosen = _normalize_to_pair({
        "chosen": [{"role": "user", "content": "x"}],   # no assistant turn
        "rejected": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    }) is None
    bad_role_rejected = _normalize_to_pair({
        "chosen": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        "rejected": [{"role": "user", "content": "x"}],
    }) is None
    ok = none_for_missing and bad_role_chosen and bad_role_rejected
    print(f"  [4] _normalize_to_pair rejects malformed: {'OK' if ok else 'FAIL'}  "
          f"(missing={none_for_missing}, bad_chosen={bad_role_chosen}, bad_rejected={bad_role_rejected})")
    return ok


def test_system_prompt_prepend() -> bool:
    """system_prompt gets prepended to both branches when given."""
    row = {"prompt": "Hello?", "chosen": "Hi!", "rejected": "go away"}
    out = _normalize_to_pair(row, system_prompt="be nice")
    ok = (out is not None
          and out[0][0] == {"role": "system", "content": "be nice"}
          and out[1][0] == {"role": "system", "content": "be nice"})
    print(f"  [5] system_prompt prepend on both branches: {'OK' if ok else 'FAIL'}")
    return ok


def test_pair_masking_disjoint() -> bool:
    """chosen and rejected each get their own response-only mask. The unmasked
    target tokens should be the rendered response text of each side."""
    tok = FakeChatTokenizer()
    chosen_msgs = [{"role": "user", "content": "what is one plus one"},
                   {"role": "assistant", "content": "the answer is two friend"}]
    rejected_msgs = [{"role": "user", "content": "what is one plus one"},
                     {"role": "assistant", "content": "go away please"}]

    seq_len = 64
    chosen = _mask_final_turn(chosen_msgs, tok, seq_len)
    rejected = _mask_final_turn(rejected_msgs, tok, seq_len)
    if chosen is None or rejected is None:
        print(f"  [6] pair masking: FAIL  (returned None)")
        return False

    # Different responses -> different unmasked target tokens
    chosen_targets = [chosen["labels"][t].item() for t in range(seq_len)
                      if chosen["labels"][t].item() != IGNORE_INDEX]
    rejected_targets = [rejected["labels"][t].item() for t in range(seq_len)
                        if rejected["labels"][t].item() != IGNORE_INDEX]

    # Shared user prefix should NOT appear as a target on either side.
    user_word_id_hello = tok._tok_content("one plus one")
    user_leaked = any(t in user_word_id_hello for t in chosen_targets) \
                  or any(t in user_word_id_hello for t in rejected_targets)

    # Sanity on shape + non-empty + different content
    shapes_ok = (chosen["input_ids"].shape == (seq_len,) and
                 chosen["labels"].shape == (seq_len,) and
                 chosen["attention_mask"].shape == (seq_len,))
    nonempty = len(chosen_targets) > 0 and len(rejected_targets) > 0
    different = chosen_targets != rejected_targets

    ok = shapes_ok and nonempty and different and (not user_leaked)
    print(f"  [6] pair masking disjoint: {'OK' if ok else 'FAIL'}  "
          f"(shapes={shapes_ok}, nonempty={nonempty}, different={different}, "
          f"no_user_leak={not user_leaked}, "
          f"chosen_n_targets={len(chosen_targets)}, rejected_n_targets={len(rejected_targets)})")
    return ok


def main() -> int:
    print("Preference-pair data tests\n")
    results = [
        test_normalize_messages_shape(),
        test_normalize_string_shape(),
        test_normalize_messages_rejected_messages_shape(),
        test_normalize_rejects_malformed(),
        test_system_prompt_prepend(),
        test_pair_masking_disjoint(),
    ]
    print()
    if all(results):
        print("  PASS — normalization handles three input shapes, rejects malformed, "
              "and the diff-mask is applied independently to each side.")
        return 0
    print("  FAIL — see the failing property above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
