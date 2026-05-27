# llm-lab

A from-scratch course in pretraining and post-training a modern LLM, using the techniques that frontier labs actually use in 2026 — including the advances from DeepSeek and other Chinese labs (MoE, MLA, GRPO, FP8).

The whole course is sized to finish for around **$50 of GPU credits**.

## Who this is for

Applied AI Engineers who want to be the person their team trusts to train, fine-tune, and ship a model — not just call an API.

Not for: researchers looking for new techniques. We borrow shamelessly from the literature; we don't derive anything from first principles.

## The promise

By the end of this course you will have:

- Pretrained a small language model from scratch, with the same architecture choices as the modern frontier families (DeepSeek-V3 and Qwen 3.6 are our two reference points throughout).
- Turned that base model into an instruction-following model (SFT), a preference-aligned model (DPO), and a reasoning model (GRPO).
- Built an evaluation suite that tells you whether anything actually worked.
- Internalized the decisions — not the derivations — that scale from a $50 run to a $50M one.

We anchor on DeepSeek-V3 and Qwen 3.6 1.7B because their architectures, training recipes, and post-training stacks are the best-documented points on the current frontier. Newer releases (DeepSeek-V4, the next Qwen, whatever comes after) are direct extensions of these — once you understand the anchors, the new ones read as small deltas.

## Course structure

Six Parts, twenty-two modules. The order is deliberate.

| Part | Topic | Modules |
|---|---|---|
| [Part 0](part-0-mental-model/) | Mental Model | 00–01 |
| [Part 1](part-1-data/) | Data | 02–03 |
| [Part 2](part-2-architecture/) | Architecture | 04–07 |
| [Part 3](part-3-pretraining/) | Pretraining | 08–13 |
| [Part 4](part-4-post-training/) | Post-Training | 14–18 |
| [Part 5](part-5-evaluation/) | Evaluation | 19 |
| [Part 6](part-6-bigger-picture/) | Bigger Picture | 20–21 |

Start at [Module 00](part-0-mental-model/00-what-we-are-building/).

## How each module is laid out

```
NN-module-name/
├── README.md        ← the lecture: motivation + concept + decision it informs
├── notebook.ipynb   ← the narrative + experiments (when there is code to run)
├── *.py             ← clean, reusable code
└── results/         ← pre-run logs and checkpoints
```

Read the README first. Run the notebook if you have GPU time. Read the `.py` modules when you want a production-quality reference. Use the `results/` if you want to skip ahead.

## Compute & budget at a glance

| If you have... | What to expect |
|---|---|
| No GPU budget | Read every README, run notebook cells on CPU with tiny tensors, inspect provided `results/`. ~60% of the value. |
| ~$50 of GPU | The intended path. Run small experiments yourself, use provided checkpoints for the expensive runs. |
| ~$200+ of GPU | Run everything, vary hyperparameters, compare against the reference runs. Where real intuition lives. |

Recommended hardware: a single A100 80GB on RunPod or Lambda Labs. H100 is nicer (unlocks FP8 demos) but not required. We do **not** recommend V100s or consumer GPUs — they'll burn your time on workarounds instead of concepts.

Full per-module cost breakdown lives in Module 01.

## Prerequisites

- Comfortable with Python and PyTorch basics (tensors, `nn.Module`, optimizers).
- Vague familiarity with what a transformer is. You don't need to know how attention works — Module 04 covers that.
- Willing to read a code module top-to-bottom rather than ctrl-F to the one function you care about.

If you've never trained a neural network end-to-end before, this course will be steep. Doable, but steep.

## Repo layout

```
part-0-mental-model/ … part-6-bigger-picture/   ← the modules
common/                                          ← shared model + training utilities
configs/                                         ← YAML configs (never hardcoded hyperparams)
data/                                            ← data pipelines + gitignored cache
tests/                                           ← shape and sanity checks
```

## Status

Under active construction.
