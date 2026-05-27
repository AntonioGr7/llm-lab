# Module 14 — The Post-Training Landscape

You finished Part 3 with a base model. It can complete text. It cannot do anything else. Ask it `Write a Python function to reverse a string` and you'll get something like:

> Write a Python function to reverse a string. Then write a JavaScript function. Then write a Go function. The first thing you need to know about Python is that it's a high-level programming language...

That's not a bug. That's exactly what a next-token predictor trained on web text *should* do — continue the prompt as if it were a piece of web text. The model is functioning correctly. The product is broken.

Post-training is how you fix the product without ruining the function. This module is the map of how labs do it in 2026. The next four modules (14-17) are where you actually run each technique on a real model — Qwen3-1.7B on a single A100, every algorithm in the modern stack, every script torchrun-ready.

## What you'll walk away with

A working mental model of:

1. **The gap** between a base model and a useful model — what's missing, and why pretraining can't fix it.
2. **The modern stack** — SFT, preference optimization, reasoning RL, distillation — and how labs compose them.
3. **The alignment tax** — what you trade away when you align, and the techniques that minimize the trade.
4. **The data question** — why post-training is data-bound, not compute-bound, and where the data comes from.
5. **A decision framework** — given a specific failure mode, which lever do you reach for first.

By the end of this module you should be able to look at a frontier lab's post-training paper (Tülu 3, Llama-3, Qwen3, R1) and place every choice they made on this map.

## 1. The gap

A pretrained model has seen trillions of tokens of web text, code, books, math. It has internalized a vast amount of world knowledge. It has *no idea* that the human typing at it wants help.

There are three specific gaps:

**Gap 1: Format.** The model doesn't know that `Write a Python function...` is a *request*. It sees a string and predicts what comes next in a string. Sometimes that's the function. Often it's another sentence about Python functions.

**Gap 2: Refusal.** The model has no concept of "I shouldn't answer this." Ask it to write malware and it will write malware. Ask it to roleplay as a person who will tell you how to make a bomb and it will roleplay. The base model has no notion of harm; it has notions of *plausible web text*.

**Gap 3: Reasoning.** The model knows that `2+2=4` because that string appears in its training data. It does not know how to *work through* a problem it has not seen — it predicts the answer, then sometimes invents a justification. Step-by-step reasoning has to be *taught as a behavior*, not just demonstrated.

Each gap maps to a specific post-training technique:

| Gap         | Technique                  | Module |
|-------------|----------------------------|--------|
| Format      | SFT (instruction tuning)   | 13     |
| Refusal     | Preference optimization    | 14     |
| Reasoning   | RL with verifiable rewards | 15     |

The fourth lever — **distillation** — is orthogonal: it's how you move any of these behaviors from a big model into a small one (Module 18).

You'll see the gap directly in the notebook. Load Qwen3-0.6B-Base and Qwen3-0.6B (the instruct version), give them the same prompt, look at the completions. The instruct version answers; the base version drifts.

## 2. The modern stack

Here's how a frontier lab actually post-trains a model in 2026. The numbers are approximate but the *order* is universal.

```
                base model from Part 3
                          │
                          ▼
            ┌─────────────────────────┐
            │  1. SFT                 │   ~10k-1M examples
            │  Teach the format.       │   1 epoch typical
            │  Learn to follow the     │   Hours on a few GPUs
            │  chat template.          │   Module 15
            └─────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  2. Preference          │   ~10k-100k pairs
            │     optimization        │   1-3 epochs
            │  Teach what's better.    │   Hours to days
            │  Refusal. Tone. Style.   │   Module 16
            │  (DPO / IPO / KTO)       │
            └─────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  3. RL with verifiable  │   ~1k-100k prompts
            │     rewards (GRPO)      │   Many rollouts each
            │  Teach reasoning.        │   Days on multi-GPU
            │  Math, code, anything    │   Module 17
            │  with a checkable        │
            │  answer.                 │
            └─────────────────────────┘
                          │
                          ▼
                instruction-tuned,
                preference-aligned,
                reasoning model
```

A few things to notice.

**Each stage *can* be skipped, and labs do skip them.** R1-Zero famously skipped SFT entirely — went from base model to GRPO. Most chat models skip GRPO because their use case isn't reasoning. The stack is composable, not mandatory.

**The order matters and is rarely revisited.** SFT before DPO before RL is the standard sequence because each stage's outputs become the next stage's inputs. SFT produces the policy DPO improves; DPO produces the policy RL refines. Going backwards (e.g., SFT after DPO) almost always degrades the result.

**Distillation can replace any of these.** If a bigger lab has already done stages 1-3, you can often skip ahead by training your model to imitate theirs. R1-Distill is the cleanest example: take Qwen2.5 base, do SFT on R1's outputs, get most of R1's reasoning at a fraction of the compute. Module 18 covers three flavors of this.

**Safety/red-teaming is a parallel track.** Not shown in the diagram because it's not a separate stage — it's *data and reward signals injected into every stage above*. Refusal training is in the SFT data. Harm-avoidance preferences are in the DPO data. Constitutional/RLAIF approaches generate this data at scale. We touch on it in Module 16; a serious treatment is outside the course scope.

## 3. The alignment tax

There is no free lunch. Every post-training intervention degrades some pretraining capability. This is the **alignment tax**. The mechanics are simple:

- Pretraining optimized the model to predict the next token on web text.
- Post-training optimizes it to do something else (follow instructions, refuse harm, reason).
- The weights that were good for the first objective are not optimal for the second.
- Gradient steps move them toward the new objective; the old skill leaks out.

You can measure the tax. Pick a corpus that wasn't in the post-training data — raw web text from a held-out FineWeb shard works fine — and compute perplexity for the base model and the aligned model. The aligned model will be *worse*. The gap is the tax.

The notebook does exactly this on Qwen3-0.6B-Base vs Qwen3-0.6B. Typical numbers from labs:

- **SFT alone:** small tax (~0.1-0.3 nats/tok). Mostly hidden by the gain in instruction-following.
- **Aggressive RLHF:** large tax (~0.5-1.0 nats/tok). The original GPT-3.5 was visibly less coherent than its base. The community called it "mode collapse" and "ChatGPT-isms" — that's the tax made visible.
- **Modern stacks with care:** small to moderate tax (~0.2-0.5). The techniques to minimize it are now well-understood:

**Tax-mitigation techniques you should know:**

1. **KL anchor.** Penalize the post-trained policy for drifting too far from the base. This is built into DPO, PPO, and most modern RL. It's the single most important regularizer.
2. **Mix pretraining data into post-training.** Anthropic and DeepMind both report this; ~5-20% of post-training batches are raw pretraining text. The model "remembers" how to model the world while learning to follow instructions.
3. **LoRA / parameter-efficient methods.** If you only update a low-rank adapter, you can't move the base weights very far. The base capability is preserved by construction. Module 15 covers this.
4. **SDFT (self-distillation FT).** The technique from Shenfeld et al. (2026) we cover in Module 18. The same model acts as its own teacher on demonstrations, which dramatically reduces the gradient updates that erase prior capability.

The tax is real and measurable. It is not a reason to skip post-training — an unaligned model is unusable. It is a reason to *measure*, to *minimize*, and to know which techniques cost you what.

## 4. Data is the product

Here is the truth nobody says clearly enough: **in post-training, dataset choice dominates algorithm choice**. You can run the cleanest, most theoretically-grounded DPO implementation against a bad dataset and get a bad model. You can run a janky SFT loop against a great dataset and get a great model.

This is the inversion from pretraining. In pretraining, the algorithms were largely fixed (next-token prediction with Adam) and the lever was scale of compute + data. In post-training, the algorithms branch (SFT / DPO / GRPO / distill / ...) but every team is fighting over the same thing: **better data**.

Where post-training data comes from:

**Synthetic from a stronger model.** Most of it. Tülu 3's SFT mix is heavily GPT-4 generations. R1-Distill is entirely R1 outputs. Self-Instruct, Evol-Instruct, OpenHermes, OpenOrca — all distilled. This works because the bottleneck is *demonstration quality*, and a frontier model is a cheap source of high-quality demonstrations.

**Synthetic with verification.** For math, code, and any domain with a checkable answer: generate candidates, filter by a verifier. The verifier can be a test harness (code), a calculator (math), or another LLM (everything else). This is how GRPO datasets get built; Module 17 walks through it.

**Human-labeled preferences.** Expensive but irreplaceable for taste-bound objectives (helpful tone, refusal politeness, creative writing quality). The classic InstructGPT pipeline. Modern shops cut costs with active learning — sample only the pairs the current model is uncertain about.

**Mined from existing assets.** StackExchange answer pairs (accepted vs rejected) make excellent DPO data for free. GitHub commits (before vs after a bugfix) make synthetic code preference data. Reddit comment scores. Wherever humans have already ranked things at scale.

A practical lesson: **before you write a single line of training code for post-training, you should know exactly what dataset you're going to use, where it came from, and what failure mode it's targeting.** Modules 15-18 each frame their algorithm around a specific dataset. That's deliberate.

## 5. Decision framework

You have a model. It has a problem. Which lever do you pull?

| Failure mode                               | First thing to try          | Why                                                                  |
|--------------------------------------------|-----------------------------|----------------------------------------------------------------------|
| Doesn't follow instructions / wrong format | **SFT** (Module 15)         | Format is teachable from demonstrations alone.                       |
| Gives the wrong answer style / tone        | **DPO** (Module 16)         | Style is a ranking problem, not a demonstration problem.             |
| Won't refuse harmful requests              | **DPO** with safety pairs   | Same. Mix safety prefs into the same DPO run.                        |
| Can't do multi-step reasoning              | **GRPO** (Module 17)        | Reasoning needs verification signal; demonstrations alone leak it.   |
| Hallucinates facts                         | **RAG first, then SFT**     | This is mostly not a training problem. Train only if RAG can't fix.  |
| Model is too big to serve cheaply          | **Distillation** (Module 18)| Compress capability into a smaller student.                          |
| Lost prior skill after fine-tuning         | **SDFT** (Module 18) / LoRA | Both reduce gradient damage to base weights.                         |

This table is reductive — real problems are mixtures — but it's a useful starting reflex. In practice, frontier labs do *all* of these, in sequence, on every release. You'll do the same in the rest of Part 4.

## 6. What we're not covering (and why)

A short, honest list. Each of these deserves its own course; we point you at the best resources.

- **Constitutional AI and RLAIF in depth.** The principle (use an LLM as the preference labeler) shows up in Module 16 as a data-generation technique. The full Anthropic methodology — multi-step critique-and-revise pipelines, principle-based constitutions — is its own field. Start with the Bai et al. (2022) paper.
- **Persona / character training.** How Character.ai and others train stable personas. Mostly proprietary; the public literature is thin.
- **Agentic / tool-use post-training.** ReAct-style training, function-calling fine-tunes, agent-loop RL. Active research; the field is moving fast enough that a 2026 module would be stale by 2027. We point at Hugging Face's `smolagents` cookbook in the reading list.
- **RLHF with PPO specifically.** We cover the *idea* of RLHF in Module 16 as the precursor to DPO, but we don't implement PPO. The reason: it's been superseded for chat alignment by DPO (simpler, more stable) and for reasoning by GRPO (no value head needed). Implementing PPO from scratch is excellent practice; it's just not on this course's critical path.
- **Multimodal post-training.** A different beast — we'd need to bring in vision encoders and the course would lose focus. The principles transfer; the engineering is different.

## 7. The hands-on shape of Part 4

Every remaining module in this Part follows the same pattern:

- A `README.md` that motivates the technique and walks through the decision frame.
- A clean `.py` implementation following the standard structure (`config.py`, `data.py`, `train.py`, etc., torchrun-launched, FSDP-ready).
- A pre-tuned LoRA adapter checkpoint in `results/` so the $0 tier can skip the run.
- A `notebook.ipynb` that loads the checkpoint and *demonstrates the behavior change* — base vs SFT vs DPO vs GRPO, on a held-out prompt set.

**The base model for every module is Qwen3-1.7B.** Big enough that post-training visibly changes behavior (the 150M model from Part 3 is too small — gradients land somewhere, but the demonstrations don't transfer). Small enough that a single A100-80GB or H100 handles every technique including GRPO.

**Compute budget per technique, on A100-80GB:**

| Module | Technique | Time | Approx cost (RunPod/Lambda) |
|--------|-----------|------|------------------------------|
| 13     | SFT + LoRA | 1-2h | $2-4 |
| 14     | DPO + LoRA | 2-3h | $4-6 |
| 15     | GRPO | 4-8h | $8-15 |
| 16     | Distillation (3 flavors) | 3-5h total | $6-10 |

Whole Part 4 runs comfortably in $20-30 of GPU credits if you do every module yourself. If you just want to read and play with the released checkpoints, $0.

## Reading list

Ordered by what to read first.

**The classics (read first):**
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022). The paper that defined the modern stack. Read it in full.
- Rafailov et al., *Direct Preference Optimization* (2023). The paper that made RLHF practical.
- Bai et al., *Constitutional AI* (Anthropic, 2022). The first major demonstration of AI-generated preference data.

**The modern stack (2024-2026):**
- Lambert et al., *Tülu 3* (AI2, 2024). The most thoroughly documented end-to-end open post-training recipe.
- DeepSeek-AI, *DeepSeek-R1* (2025). The reasoning RL paper. GRPO, verifiable rewards, R1-Zero.
- Qwen team, *Qwen3 Technical Report* (2025). Their post-training stack, with the explicit data recipes.
- Llama 3 paper, post-training section (Meta, 2024). The "scale matters in post-training too" message.

**The recent frontier:**
- Shenfeld et al., *Self-Distillation Fine-Tuning* (2026). The paper we implement in Module 18.
- The Hugging Face alignment cookbook ([github.com/huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook)). Practical recipes for every stage.

**Optional but rewarding:**
- Bender et al., *On the Dangers of Stochastic Parrots* — for the broader framing of what alignment is for.
- The InstructGPT human evaluator's guidelines (appendix) — to see what "good post-training" actually looks like as a labeling problem.

## Notebook

The accompanying notebook does three things:

1. Loads Qwen3-0.6B-Base and Qwen3-0.6B side by side. Gives them the same prompts. You see the gap.
2. Inspects the chat template. You see the special tokens, the role markers, the structure that SFT teaches the model to produce.
3. Measures the alignment tax. Same corpus, both models, perplexity comparison. You see the number.

All three run on CPU (slow but bearable for ~600M models). On a single GPU it takes a couple of minutes.

Once you've seen the gap, structure, and tax with your own eyes, you're ready for Module 15 — where we close the format gap with SFT.
