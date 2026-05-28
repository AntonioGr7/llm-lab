"""Evaluate a DPO/IPO checkpoint.

Three signals, in order of importance:

  1. **Generation A/B** — the headline qualitative test. Same prompt rendered
     through the chat template, two completions: one from the trained policy,
     one from the reference (`--also-reference`). Did the policy actually
     change behavior in the direction we wanted (more helpful, more polite,
     better-formatted), without losing the SFT'd capability?

  2. **Held-out preference accuracy** — the policy's implicit reward, scored
     against a held-out slice of preference pairs. For each `(prompt, chosen,
     rejected)`, compute the policy's logratio margin and check if it ranks
     `chosen` above `rejected`. Reports the headline DPO metric `accuracy ∈
     [0, 1]` (random = 0.5, perfect = 1.0). On ultrafeedback_binarized after
     full training you should see ~70-85%.

  3. **Average chosen/rejected rewards and KL drift** — pulled from the same
     scoring pass. Tells you whether the policy moved both sides as designed
     (chosen up, rejected down) and how far it drifted from the reference.

Usage:

    # Generate side-by-side with the reference (the most useful first check)
    python eval.py --checkpoint=results/checkpoints/step_00001000 \\
        --prompts "Explain why the sky is blue" "Write a haiku about Python" \\
        --also-reference

    # Score held-out preference accuracy (default = 200 test_prefs pairs)
    python eval.py --checkpoint=results/checkpoints/step_00001000 --accuracy

    # Baseline (no fine-tune) — the reference's accuracy on the same slice
    python eval.py --base --accuracy
"""
from __future__ import annotations

import argparse

import torch

from config import TrainConfig, load_yaml, apply_dotted_overrides
from model import build_policy, build_reference
from data import PreferenceDataset, load_tokenizer
from loop import gather_response_logps, compute_dpo_loss
from checkpoint import load as load_ckpt
from fsdp_setup import init_distributed, cleanup_distributed


# ---------------------------------------------------------------------------
# Generation with the chat template
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_chat(
    model,
    tok,
    prompts: list[str],
    system_prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cuda",
) -> list[tuple[str, str]]:
    """Render each prompt through the chat template, generate, return (prompt,
    completion). Identical to Module 15's `generate_chat`."""
    model.config.use_cache = True
    model.eval()

    out = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=False,
            ).to(device)
        except TypeError:
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(device)
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_k=top_k,
            pad_token_id=tok.pad_token_id,
        )
        completion = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        out.append((prompt, completion.strip()))
    return out


# ---------------------------------------------------------------------------
# Held-out preference accuracy + average rewards
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_preference_accuracy(
    policy,
    reference,
    cfg: TrainConfig,
    eval_split: str = "test_prefs",
    n_examples: int = 200,
    device: str = "cuda",
) -> dict[str, float]:
    """Score held-out preference pairs with the policy and reference.

    For each pair:
      logratio_chosen   = log π_policy(y_w|x) - log π_ref(y_w|x)
      logratio_rejected = log π_policy(y_l|x) - log π_ref(y_l|x)
      margin            = logratio_chosen - logratio_rejected

    Returns:
      accuracy:       % pairs where margin > 0 (policy ranks chosen > rejected)
      margin:         mean margin (should be positive after training)
      chosen_rewards: β · mean logratio_chosen
      rejected_rewards: β · mean logratio_rejected
      ref_accuracy:   what % the *reference alone* gets right (baseline). For
                      a healthy DPO run, ref_accuracy < accuracy.

    Uses a held-out split of the configured dataset. Defaults to
    `test_prefs` (ultrafeedback_binarized's held-out slice).
    """
    from copy import deepcopy
    eval_data_cfg = deepcopy(cfg.data)
    eval_data_cfg.split = eval_split
    eval_data_cfg.max_examples = n_examples
    ds = PreferenceDataset(eval_data_cfg, tokenizer_name=cfg.model.name)

    policy.eval()
    reference.eval()

    n_correct = 0
    n_correct_ref = 0
    sum_margin = 0.0
    sum_chosen_logratio = 0.0
    sum_rejected_logratio = 0.0
    n = 0

    for i in range(len(ds)):
        s = ds[i]
        # Stack chosen/rejected for one 2-row forward on each model
        input_ids = torch.stack([s["chosen_input_ids"], s["rejected_input_ids"]]).to(device)
        labels = torch.stack([s["chosen_labels"], s["rejected_labels"]]).to(device)
        attn = torch.stack([s["chosen_attention_mask"], s["rejected_attention_mask"]]).to(device)

        p_logits = policy(input_ids=input_ids, attention_mask=attn).logits
        r_logits = reference(input_ids=input_ids, attention_mask=attn).logits

        p_logps = gather_response_logps(p_logits, labels)   # [2]
        r_logps = gather_response_logps(r_logits, labels)   # [2]

        p_chosen, p_rejected = p_logps[0], p_logps[1]
        r_chosen, r_rejected = r_logps[0], r_logps[1]

        chosen_logratio = float((p_chosen - r_chosen).item())
        rejected_logratio = float((p_rejected - r_rejected).item())
        margin = chosen_logratio - rejected_logratio

        sum_chosen_logratio += chosen_logratio
        sum_rejected_logratio += rejected_logratio
        sum_margin += margin
        if margin > 0:
            n_correct += 1
        # Reference baseline: does the reference itself rank chosen above rejected?
        if float((r_chosen - r_rejected).item()) > 0:
            n_correct_ref += 1
        n += 1

    beta = cfg.preference.beta
    return {
        "accuracy": n_correct / max(n, 1),
        "ref_accuracy": n_correct_ref / max(n, 1),
        "margin": sum_margin / max(n, 1),
        "chosen_rewards": beta * sum_chosen_logratio / max(n, 1),
        "rejected_rewards": beta * sum_rejected_logratio / max(n, 1),
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Write a haiku about Python.",
    "Explain recurrent neural networks to a 10-year-old.",
    "What's a good weeknight dinner I can make in 20 minutes?",
    "I'm feeling overwhelmed at work. Any advice?",
]


def _parse_args():
    p = argparse.ArgumentParser(description="Eval a DPO/IPO checkpoint")
    p.add_argument("--config", type=str, default="configs/dpo_qwen3_1.7b.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to a DPO checkpoint dir saved by train.py")
    p.add_argument("--base", action="store_true",
                   help="skip the checkpoint and eval the reference (the SFT model) — DPO baseline")
    p.add_argument("--prompts", type=str, nargs="+", default=None,
                   help="user prompts to generate from (default: a canned set)")
    p.add_argument("--also-reference", action="store_true",
                   help="also generate from the reference for side-by-side A/B")
    p.add_argument("--accuracy", action="store_true",
                   help="compute held-out preference accuracy + reward stats")
    p.add_argument("--eval_split", type=str, default="test_prefs",
                   help="dataset split to score (default 'test_prefs' for ultrafeedback_binarized)")
    p.add_argument("--n_eval_examples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args, extra = p.parse_known_args()
    overrides = []
    for tok in extra:
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
        else:
            raise SystemExit(f"unrecognized arg: {tok!r}")
    return args, overrides


def main():
    args, overrides = _parse_args()
    if not args.base and args.checkpoint is None:
        raise SystemExit("pass --checkpoint=<dir>, or --base to eval just the reference")

    cfg = load_yaml(args.config)
    apply_dotted_overrides(cfg, overrides)

    rinfo = init_distributed()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not rinfo.is_main:
        cleanup_distributed()
        return

    # Reference is always needed (for the eval baseline + the A/B + the
    # preference-accuracy computation). Load it first; it's BF16 so cheap.
    reference = build_reference(cfg.model).to(device)

    if args.base:
        # Score the reference itself — "what does the SFT model get out of the box?"
        policy = reference
        print(f"[eval] base mode: policy IS the reference ({cfg.model.resolved_ref_name()!r})", flush=True)
    else:
        policy = build_policy(cfg.model).to(device)
        step = load_ckpt(policy, None, args.checkpoint)
        print(f"[eval] loaded DPO checkpoint at step {step}", flush=True)

    tok = load_tokenizer(cfg.model.name)

    prompts = args.prompts or _DEFAULT_PROMPTS
    print("[eval] generating from policy ...", flush=True)
    policy_results = generate_chat(
        policy, tok, prompts,
        system_prompt=cfg.data.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    ref_results = None
    if args.also_reference and not args.base:
        print("[eval] generating from reference ...", flush=True)
        ref_results = generate_chat(
            reference, tok, prompts,
            system_prompt=cfg.data.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

    for i, (prompt, completion) in enumerate(policy_results):
        print(f"\nUSER:       {prompt}")
        if ref_results is not None:
            print(f"REFERENCE:  {ref_results[i][1]}")
        print(f"POLICY:     {completion}")
    print()

    if args.accuracy:
        print(f"[eval] held-out preference accuracy on split={args.eval_split!r} ...", flush=True)
        r = evaluate_preference_accuracy(
            policy, reference, cfg,
            eval_split=args.eval_split,
            n_examples=args.n_eval_examples,
            device=device,
        )
        print(f"\n  pairs scored:           {r['n_pairs']}")
        print(f"  policy accuracy:        {r['accuracy']:.1%}  "
              f"(% pairs where policy ranks chosen above rejected)")
        print(f"  reference accuracy:     {r['ref_accuracy']:.1%}  "
              f"(same metric on reference alone — your baseline)")
        print(f"  mean margin:            {r['margin']:+.3f}  (chosen logratio - rejected logratio)")
        print(f"  mean chosen reward:     {r['chosen_rewards']:+.3f}  (β · logratio_chosen)")
        print(f"  mean rejected reward:   {r['rejected_rewards']:+.3f}  (β · logratio_rejected)")
        if r['accuracy'] > r['ref_accuracy']:
            print(f"  -> DPO improved preference ranking by "
                  f"{(r['accuracy'] - r['ref_accuracy'])*100:.1f} percentage points\n")
        else:
            print(f"  -> WARNING: policy not beating reference. Check LR / β / steps.\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
