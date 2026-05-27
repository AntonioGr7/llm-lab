# Module 13 — Continual Pretraining

You have a base model that someone spent millions of dollars and trillions of tokens to train. You also have something they don't: **3 billion tokens of your company's private data** — support tickets, internal wikis, contracts, code, research notes — that has never touched the public internet. You want the model to *know* this. Not to look it up at inference time; to have internalized it the way it internalized Wikipedia.

This is the question every applied team eventually asks, and the answer is almost never the one they reach for first.

## The thesis: knowledge lives in pretraining, not post-training

The instinct is to fine-tune. You have an instruct model, you have documents, surely SFT can teach it the documents? **No — and it's worse than ineffective, it's actively harmful.**

SFT and preference optimization (Part 4) are *behavioral* interventions. They teach the model a **form**: follow instructions, adopt a tone, refuse certain requests, lay out a chain of thought. They reshape how the model uses knowledge it already has. They are a terrible vehicle for *new* knowledge, for one sharp reason:

> **Fine-tuning a model on facts it doesn't already know teaches it to hallucinate.** Gekhman et al. (2024) showed this directly: when you SFT on knowledge outside the model's existing distribution, it doesn't absorb the facts — it learns the *behavior* "state things in this confident format," and then generalizes that confidence to everything, degrading calibration across the board. You teach the surface, not the substance.

Knowledge acquisition is a **pretraining** phenomenon. It happens through next-token prediction over a large, diverse corpus, with a learning rate high enough to actually move weights. That's why this module lives at the end of Part 3, not in Part 4: **the technique that internalizes your private data is the one you already built in Modules 08–12.** You're going to run it again.

## "But 3B tokens can't train a language model"

Correct. You're not training from scratch — that would need *trillions* of tokens to learn language, world knowledge, and reasoning before it ever got to your data. You start from a **finished base model** that already has all of that, and you **continue** its pretraining on a mixture that includes your corpus. The literature calls this **Continued Pre-Training (CPT)**, or **Domain-Adaptive Pretraining (DAPT)** after Gururangan et al.'s 2020 "Don't Stop Pretraining." 3B tokens is far too little to *create* a model and plenty to *adapt* one.

The entire difficulty of CPT is one tension:

> Push hard enough that the model genuinely internalizes the new corpus — but not so hard that it forgets how to be a language model. That failure mode has a name: **catastrophic forgetting**, and fighting it is what every lever below is for.

## What you'll be able to do at the end of this module

- Continue-pretrain a finished base model on a private corpus using the Part 3 stack.
- Build a **replay mixture** and choose a domain:general ratio that adapts without forgetting.
- Set a **re-warm / re-decay** learning-rate schedule — the part everyone gets wrong.
- Use **synthetic augmentation** (paraphrase + QA generation) to make injected knowledge *extractable*, not just stored.
- Evaluate on two axes: **knowledge acquired** (held-out closed-book QA) and **capability retained** (before/after benchmark delta).
- Know when CPT is the wrong tool and **RAG** is the right one.

## Directory layout

```
13-continual-pretraining/
├── README.md              ← this lecture
├── make_corpus.py         ← generate the fictional-company corpus: raw docs + paraphrase
│                            augmentation + synthetic QA + a HELD-OUT QA probe set;
│                            tokenize into a Module-12 indexed corpus. $0 offline; optional --llm.
├── data.py                ← MixedDataLoader: interleaves domain + general "replay" at a
│                            configured TOKEN ratio, on top of Module 12's IndexedDataset.
├── schedule.py            ← re-warm + re-decay (WSD / cosine), the CPT-specific schedule.
├── config.py              ← the composed TrainConfig + CPTConfig (replay_ratio, rewarm_peak, …).
├── model.py               ← build_model = AutoModelForCausalLM.from_pretrained (base checkpoint).
├── optim.py · fsdp_setup.py · checkpoint.py · efficiency.py   ← copied from Modules 11/15.
├── train.py               ← torchrun + FSDP2 continual-pretraining entrypoint.
├── eval.py                ← the headline: closed-book QA (held-out facts) before vs after,
│                            + general-benchmark retention before vs after (the forgetting bill).
├── configs/
│   ├── cpt_demo.yaml        ← tiny, ~$0: small corpus, few steps, runs on a modest GPU.
│   └── cpt_qwen3_0.6b.yaml  ← the real demo: Qwen3-0.6B-Base, ~$1–3 on one A100.
├── notebook.ipynb         ← CPU-runnable: probe base (knows nothing) → mix → short CPT →
│                            re-probe (now knows) → forgetting table.
├── tests/
│   ├── test_mix_ratio.py    ← the replay loader actually hits the configured token ratio.
│   └── test_qa_holdout.py   ← held-out QA facts never leak into the training corpus.
└── results/                 ← pre-run loss curve, QA before/after, forgetting table.
```

Like Modules 11 and 12, this is a **framework directory**: `cp -r 13-continual-pretraining/ ~/my-cpt-repo/`, point it at *your* corpus and *your* base model, and you have a working continual-pretraining stack. That's not incidental — it's the whole point. The enterprise scenario in the opening paragraph is exactly what you'd lift this out to do.

## The four levers

### 1. Replay — the single most important trick

Do **not** continue-pretrain on your domain corpus alone. A model trained on a narrow distribution rapidly forgets the broad one. The fix, from Ibrahim et al. (2024, *"Simple and Scalable Strategies to Continually Pre-train LLMs"*), is almost embarrassingly simple: **mix in a fraction of general pretraining-style data.** Their headline result is that even a **~5% replay fraction** sharply reduces forgetting at negligible cost to domain adaptation.

For a *small* domain corpus like ours you go more aggressive — typically **domain:general from 1:1 to 1:4** — because you want to repeat the domain data without over-repeating it *relative to* the general stream. You almost never have the model's original pretraining mix (nobody publishes it), so use a strong open proxy: FineWeb-Edu (which you already have indexed from Module 12!), DCLM, or Dolma.

`data.py` implements this as a **token-ratio interleave** over two Module-12 indexed datasets. The ratio is the knob; `test_mix_ratio.py` asserts the loader actually delivers it.

### 2. Re-warm, then re-decay — the part everyone gets wrong

A base checkpoint was annealed down to a near-zero learning rate at the end of its pretraining. You have two failure modes:

- **Resume at that tiny LR** → the model barely moves; your corpus doesn't get learned.
- **Jump to the original peak LR** → you blow away existing knowledge; catastrophic forgetting.

The recipe (Gupta et al. 2023, *"How to re-warm your model?"*; Ibrahim et al. 2024):

1. **Re-warm** with a short warmup (~1% of steps)…
2. …to a peak **well below the original** — typically **~10–30% of the original pretraining peak** (concretely, often somewhere in `2e-5` to `1e-4` for a small model).
3. **Re-decay** (cosine or WSD) back down toward zero over your CPT token budget.

> This one knob — the re-warm peak — *is* your forgetting/adaptation dial. Higher = more adaptation **and** more forgetting. Tuning CPT is mostly tuning this number against the two-axis eval in §"Evaluation."

`schedule.py` is Module 9's scheduler configured for this shape. Note this is the same WSD (warmup-stable-decay) idea from Part 3 — we just start it from a trained model instead of random init.

### 3. Synthetic augmentation — store *and* extract

This is the most underrated lever, and where teams leave the most on the table. **Raw documents are a shockingly token-inefficient way to inject knowledge.** Two results you must internalize:

- **Allen-Zhu & Li, *"Physics of Language Models" (Parts 3.1 / 3.3)*** — a fact seen in only *one* phrasing gets **stored** but is barely **extractable**: the model can't answer a question about it. Extraction emerges only when the fact appears in **multiple rewrites** during training. Knowledge augmentation is *necessary*, not optional.
- **Maini et al. (2024), *"Rephrasing the Web" (WRAP)*** — rephrasing each document into several styles (QA, encyclopedic, simple, dialogue) yields ~3× data efficiency and large downstream gains.

So `make_corpus.py` does not feed raw docs once. It produces, per source document:
- the **raw** document,
- several **paraphrases** (different phrasings of the same facts),
- and **synthetic QA pairs** generated from it.

The QA form is what makes knowledge *retrievable* at inference instead of merely latent. This is the difference between a model that has "read" your docs and one that can actually answer about them. It also conveniently turns 3B raw tokens into ~10–20B tokens of diverse views — which makes your "small corpus" worry mostly disappear.

> Two paths in `make_corpus.py`: a **templated, offline, $0** generator (deterministic, runs in CI, the default) and an optional `--llm` path that calls a strong model to rephrase + generate QA at real quality. The course demo uses the offline path; the technique is identical at scale.

### 4. Data repetition — how many epochs is safe?

With a finite corpus you *will* repeat. **Muennighoff et al. (2023, *"Scaling Data-Constrained Language Models"*)** quantified the limit: up to **~4 epochs** of repeated data is nearly as valuable as fresh tokens; past that, returns decay quickly and you risk verbatim memorization. So: 2–4 passes over your (augmented) domain set, kept diluted by fresh replay, is the safe zone.

## Where to inject: the mid-training / annealing insight

The current framing (2024–2026) treats this less as a bolt-on and more as **mid-training** — a recognized stage between pretraining and post-training. The key empirical finding (visible in OLMo 2's recipe and NVIDIA's *"Reuse, Don't Retrain"* report) is that **the model is most plastic to high-value data during the LR-decay/annealing phase.** So you don't sprinkle your corpus uniformly: you front-load general + domain during the stable/re-warm portion and **concentrate your highest-quality curated domain data + synthetic QA into the decay tail.** The schedule in lever 2 isn't just hygiene — it's a *delivery mechanism*.

## The demo: a fictional company, on `Qwen3-0.6B-Base`

To *prove* knowledge injection you need knowledge the base model provably cannot already have. Real "recent" data can't give you that guarantee — you can never fully rule out contamination, and you'd have to hand-build an answer key. So the demo invents its own world.

`make_corpus.py` procedurally generates a **fictional company** — randomized proper nouns, products, people, dates, and *numeric* specs (e.g. "the Q-90's rated payload is 4,200 kg", "Project Halcyon shipped in Q3 of fiscal 2031"). Because the specifics are randomly generated, they are **guaranteed novel**: no pretraining corpus contains them, and you hold the ground-truth answer key by construction. This mirrors the real enterprise scenario — a private corpus no public model has seen — in miniature, and it's exactly how researchers isolate knowledge injection (the synthetic-biography setup in the *Physics of LM* work).

We use **`Qwen3-0.6B-Base`** because: it's a *true base* checkpoint (CPT operates on base, not instruct); it's the same family as the SFT module so the course narrative is continuous; it's small enough to continue-pretrain on one consumer GPU; and — critically — it scores **clearly above chance** on MMLU/GSM8K, so the *forgetting* half of the evaluation is actually measurable. A 135M toy model sits at random on those benchmarks, which makes the retention demo meaningless.

You do **not** need 3B tokens to see a clean signal at 0.6B — a few million tokens of *augmented* fictional corpus makes the before/after contrast obvious while teaching the identical recipe. The scaling laws don't change; only the token counts do.

## Running it

Same `torchrun` contract as the rest of Part 3 — multi-GPU from day one, FSDP2, single-GPU is just a one-rank cluster.

```bash
# 0. Build the corpus: raw + paraphrase + QA, plus a held-out QA probe set,
#    tokenized into a Module-12 indexed corpus. Offline & deterministic.
python make_corpus.py --out results/corpus --n-entities 64 --augment 6 --qa-per-doc 4

# 1. Probe the BASE model first — establish it knows nothing (closed-book QA).
python eval.py --base Qwen/Qwen3-0.6B-Base --qa results/corpus/qa_heldout.jsonl

# 2. Continual pretraining (re-warm/re-decay + replay mix), 1 GPU:
torchrun --standalone --nproc_per_node=1 train.py --config=configs/cpt_qwen3_0.6b.yaml

#    …or 8 GPUs — same code, change the launch flag:
torchrun --standalone --nproc_per_node=8 train.py --config=configs/cpt_qwen3_0.6b.yaml \
         --optim.grad_accum=2

# 3. Re-probe the continually-pretrained model, AND measure forgetting:
python eval.py --checkpoint results/checkpoints/final \
               --qa results/corpus/qa_heldout.jsonl --forgetting mmlu,gsm8k --base Qwen/Qwen3-0.6B-Base
```

The `cpt_demo.yaml` config is the ~$0 dev path (tiny corpus, a handful of steps); `cpt_qwen3_0.6b.yaml` is the real ~$1–3 run.

## What you should see

The payoff is a single before/after table. Closed-book accuracy on the held-out fictional facts:

```
                          held-out QA acc   MMLU    GSM8K
Qwen3-0.6B-Base (before)       3%           ~XX%    ~XX%     ← knows nothing about the fiction
+ CPT, domain only             71%          ~XX-Δ   ~XX-Δ    ← learned it, but forgot more
+ CPT, with replay + QA aug    78%          ~XX-δ   ~XX-δ    ← learned it, kept its mind
```

Two stories in one table. The QA column going from ~chance to high is **acquisition** — the thing you came for. The MMLU/GSM8K columns are the **bill**: how much general capability you paid. The replay+augmentation row should learn *more* (QA augmentation aids extraction) while forgetting *less* (replay protects the general distribution) — which is the entire argument of this module made concrete. (The general-benchmark numbers are filled in from your own run; a pre-run table ships in `results/`.)

## Evaluation: always two axes

A CPT run is never "good" on one number. You must measure both:

- **Acquisition** — closed-book QA on a **held-out** slice of domain facts that were *never in training* (`test_qa_holdout.py` enforces this). If you test on facts you trained on, you're measuring memorization, not generalization of the knowledge.
- **Retention** — a fixed general benchmark suite (MMLU, GSM8K, …) run **before and after**. The delta is your forgetting bill. This is the number you tune the re-warm peak and replay ratio against.

This is also why CPT comes *before* post-training in the pipeline: CPT damages instruction-following, so you re-apply Part 4 afterward. The full sequence is **CPT → SFT (Module 15) → preference optimization (Module 16)**. You internalize knowledge on the base, *then* re-teach behavior.

## When CPT is the wrong tool: RAG and LoRA

Two honest caveats, because using CPT where it doesn't belong is a common and expensive mistake.

- **CPT vs RAG is not either/or.** If the knowledge is **volatile** (changes weekly), needs **citations**, or is enormous, retrieval beats baking it into weights — you don't retrain to fix one fact. CPT shines for **pervasive domain language, reasoning patterns, and stable core knowledge** you want the model to *think in* without retrieval latency. The strongest enterprise stacks do **both**: CPT to internalize the domain's vocabulary and reasoning, RAG for fresh, citable specifics.
- **LoRA is the wrong tool for knowledge.** Biderman et al.'s *"LoRA Learns Less and Forgets Less"* names the trade-off exactly: low-rank adapters have limited capacity to *absorb new knowledge* (great for style, weak for facts). For knowledge injection, **full-parameter CPT** wins. LoRA's "forgets less" is real, but you pay for it with weak acquisition — the opposite of what you want here.

## Stretch goals

- **Sweep the re-warm peak** (`1e-5 → 3e-4`) and plot acquisition vs forgetting. You'll see the dial directly; pick the knee.
- **Ablate replay ratio** (0%, 5%, 25%, 50%) and watch the forgetting column move.
- **Ablate augmentation**: raw-only vs raw+paraphrase vs raw+paraphrase+QA. The QA column should separate dramatically — that's the storage-vs-extraction gap, measured.
- **Annealing placement**: uniform domain data vs domain concentrated in the decay tail. Same tokens, different schedule, different uptake.
- **Real `--llm` corpus**: swap the templated generator for a strong-model rephraser/QA-generator and compare extraction quality.
- **Scale up**: point `make_corpus.py` at a real private corpus and `cpt_qwen3_0.6b.yaml` at a larger base. Nothing but the numbers changes.

## Reading list

Ordered by how directly it maps to this module's decisions:

- **Ibrahim et al. (2024), *Simple and Scalable Strategies to Continually Pre-train LLMs*** — the replay + re-warm/re-decay recipe this module is built on. Read first.
- **Gupta et al. (2023), *Continual Pre-Training of LLMs: How to (re)warm your model?*** — the LR-schedule half, in detail.
- **Gururangan et al. (2020), *Don't Stop Pretraining*** — the original DAPT/TAPT result that domain-adaptive continued pretraining works.
- **Allen-Zhu & Li, *Physics of Language Models, Part 3.1 & 3.3*** — knowledge **storage vs extraction**, and why augmentation is mandatory. The conceptual core of lever 3.
- **Maini et al. (2024), *Rephrasing the Web (WRAP)*** — practical synthetic rephrasing for data efficiency.
- **Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*** — the ~4-epoch data-repetition limit.
- **Gekhman et al. (2024), *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*** — why you don't SFT for facts. The justification for this module's existence.
- **OLMo 2** (Allen AI, 2024) and **NVIDIA, *Reuse, Don't Retrain*** — mid-training / annealing as a recognized stage; where to place high-value data.
- **Biderman et al. (2024), *LoRA Learns Less and Forgets Less*** — the PEFT-vs-full-FT trade-off for knowledge.

---

*Previous: [12 — Production Data Pipelines](../12-production-data-pipelines/) — whose indexed corpus is what you mix against replay here. Next: [Part 4 — Post-Training](../../part-4-post-training/) — now that the base model knows your domain, you teach it how to behave, starting with [14 — The Post-Training Landscape](../../part-4-post-training/14-post-training-landscape/).*
