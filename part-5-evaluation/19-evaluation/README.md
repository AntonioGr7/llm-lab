# Module 19 — Evaluation That Actually Means Something

You have spent eighteen modules building and post-training a model. This module answers the only question that decides whether any of it worked: **is the model actually good, and how would you know?**

That question is harder than it sounds, and getting it wrong is the most common way serious teams waste months. A team can ship a "new SOTA" that's a contaminated benchmark mirage. A team can kill a genuinely better model because a 1.5-point benchmark drop was pure noise. A team can optimize a number for a quarter and discover the number had nothing to do with what users wanted. Every one of these is an *evaluation* failure, not a modeling failure.

The thesis of this module:

> **A benchmark number is an estimate produced by a measurement instrument, and every instrument has a bias and an error bar. Evaluation is the discipline of knowing both.** The labs that win are not the ones with the best benchmark scores — they're the ones who trust their own internal evals enough to ignore the public ones.

This is why we make evaluation its own Part. By the end you'll be able to look at any leaderboard row and name three things that could be wrong with it, and — more importantly — build an eval for *your* use case that you'd actually bet a release on.

## What you'll be able to do at the end

1. Explain what each benchmark family (MC, generative+verifier, instruction-following, human/judge preference) measures and where it lies (§3–§4).
2. Put a confidence interval on any score and run a *paired* significance test before claiming model B beats model A (§5, `metrics.py`).
3. Run LLM-as-judge correctly — position-swapped, length-aware, calibrated against humans — and recognize when it's measuring its own biases (§6, `judge.py`).
4. Detect benchmark contamination with n-gram overlap and canary scans, and explain why a private held-out set is the only real defense (§7, `contamination.py`).
5. Evaluate a model *at each stage of the pipeline* (base → SFT → DPO → GRPO) and read the capability gains and the alignment tax (§8).
6. Build a use-case-specific eval suite you'd trust to ship (§9).

---

## 1. Directory layout

```
19-evaluation/
├── README.md              you are here
├── notebook.ipynb         CPU-only tour: every failure mode made visceral, with plots
├── config.py              EvalConfig — model, which suites, judge, contamination knobs
├── model.py               load model + the two forward modes: likelihood scoring & generation
├── data.py                benchmark loaders (MMLU/GSM8K/IFEval/pairwise) + $0 synthetic fallbacks
├── benchmarks.py          MC scoring (3 norms) + format styles + generative scoring + IFEval checkers
├── judge.py               LLM-as-judge: position-swap, length bias, pointwise rubric, κ calibration
├── contamination.py       n-gram overlap detector + BIG-bench canary scan
├── metrics.py             bootstrap CIs, paired bootstrap, McNemar, pass@k / avg@k / maj@k
├── harness.py             orchestrate a model + suites + metrics into a Scorecard
├── eval.py                CLI: score a model, judge pairwise, scan contamination, --compare two runs
├── configs/
│   ├── eval_qwen3_1.7b.yaml   canonical — score the post-trained Qwen3-1.7B at any stage
│   └── eval_demo.yaml         offline $0 smoke (synthetic data, dummy judge, CPU)
├── tests/
│   ├── test_metrics.py        CIs, paired-vs-unpaired, McNemar, pass@k closed form
│   ├── test_contamination.py  n-gram overlap, canary, partial-leak fractions
│   ├── test_judge.py          position-swap resolution, length-bias corr, κ
│   └── test_benchmarks.py     MC norm/format sensitivity, extraction, constraint checkers
└── results/                   pre-run scorecards + the stage-by-stage progression table
```

The four pure modules (`metrics`, `contamination`, `judge`'s logic, `benchmarks`' scoring) have no torch dependency and are fully covered by offline tests — because the parts of evaluation you most need to trust are exactly the parts that should be testable in milliseconds. The model-touching code (`model.py`, the `run_*` in `harness.py`) is thin by design.

We deliberately **build the harness from scratch** rather than calling `lm-eval-harness`. Not because you should reimplement it in production (you shouldn't — see §10), but because the failure modes only become real once you've seen that *the same model and the same data give different scores depending on choices you didn't know you were making*.

---

## 2. The mental model: evaluation is measurement

Borrow the framing from experimental science. A benchmark is an **instrument**. Like any instrument it has:

- **Bias** — a systematic offset. MC benchmarks reward models that learned the "answer is a letter" convention; judge benchmarks reward verbosity; contaminated benchmarks reward memorization. The number is consistently off in a direction.
- **Variance** — the error bar. A 200-item benchmark scored at 60% has a 95% CI of roughly ±7 points. Two models 3 points apart are, as far as that instrument can tell, *identical*.
- **Validity** — does it measure what you care about? GSM8K measures grade-school arithmetic word problems. If your product is a legal assistant, GSM8K validity for your use case is approximately zero, no matter how clean the number.

Everything in this module is an attempt to characterize one of those three for the instruments you'll actually use. **A score reported without an error bar and a contamination check is not a measurement. It's a vibe with a decimal point.**

---

## 3. Perplexity and the pretraining signal (necessary, not sufficient)

The first eval you ever compute is the training loss itself. Held-out **perplexity** (exp of the mean per-token negative log-likelihood) or its tokenizer-independent cousin **bits-per-byte** is the cleanest, cheapest, hardest-to-game signal you have, and it's what labs watch *continuously* during a pretraining run.

```
bits_per_byte = (loss_in_nats / ln(2)) * (n_tokens / n_bytes)
```

Why labs track it live: it's a smooth, low-variance curve, so a regression (a bad data shard, a diverging optimizer, a bug in a kernel) shows up within hundreds of steps — long before any downstream benchmark would. This is the single most important eval during pretraining, and it's the one this module says the *least* about because Modules 11–13 already compute it.

**Why it's not sufficient:** perplexity measures how well the model predicts text, not whether it can *do* anything. A model can have excellent perplexity and be unable to follow an instruction, refuse a harmful request, or reason through a problem. Lower perplexity correlates with downstream capability *within a model family on similar data*, but the correlation breaks across families, tokenizers, and data mixes — which is exactly why you can't rank two different models by perplexity. The moment you post-train, perplexity on the SFT data goes *down* while the thing you care about (helpfulness) goes *up* in ways perplexity can't see.

So labs use perplexity as a **training-health monitor** and reach for downstream benchmarks to measure capability. The rest of this module is about those.

---

## 4. The benchmark families, and where each one lies

There are really only four shapes of automatic-or-semi-automatic eval. Know the shape and you know the failure mode.

### 4a. Multiple-choice, scored by likelihood (MMLU, ARC, HellaSwag, GPQA, MMLU-Pro)

The model never generates. You give it the question and each candidate answer, and rank the options by the model's log-probability. The argmax is the prediction. `benchmarks.score_mc` does this; `model.continuation_logprob` produces the per-option scores.

This is the workhorse of pretraining-era evals because it's cheap (one forward pass per option, no decoding) and deterministic. **MMLU** (57 subjects, 4-way) is the canonical knowledge benchmark; **GPQA Diamond** (graduate-level, "Google-proof" science) is its modern, harder-to-saturate successor; **MMLU-Pro** raised the option count to 10 and pruned noise.

**Where it lies — three places, all demonstrated in `benchmarks.py` and the notebook:**

1. **Normalization sensitivity.** Longer answers have lower raw log-prob just for having more tokens. So do you rank by the *summed* logprob (`raw`), the *per-token* average (`token`, the HellaSwag `acc_norm`), or the *per-byte* average (`byte`, tokenizer-independent)? **These give different accuracies on the same model.** `score_mc` implements all three; the canonical config pins `token`. The Open LLM Leaderboard pins the exact normalization for exactly this reason.

2. **Prompt-format sensitivity.** Present the options as lettered "A/B/C/D" and score the letter, or present them cloze-style and score the answer text? `build_mc_prompt` does both. A model that learned the "answer is a letter after `Answer:`" convention scores higher in lettered format; a model that didn't gets punished for a formatting convention, not a knowledge gap. HELM and the Eleuther harness both document multi-point swings from format alone.

3. **The MC artifact.** A model can exploit statistical regularities in the *options* (the longest option is often right; "all of the above" is often right) without understanding the question. This is why GPQA and others now report alongside a "without the question" sanity baseline.

The practical consequence, and the reason this is the first failure mode in the notebook: **when you read "Model X scores 71.2 on MMLU," that number is meaningless without the harness, the few-shot count, the normalization, and the prompt format.** Two papers reporting MMLU for the same model routinely differ by 5+ points purely from these choices. Pin them in your config (`config.BenchmarkConfig`) and report them in your scorecard.

### 4b. Generative + verifier (GSM8K, MATH, AIME, HumanEval, MBPP, SWE-bench)

The model generates free text; a *rule* checks it. For math, parse the final integer and compare (`benchmarks.extract_final_int`). For code, run the unit tests. These are the most honest automatic benchmarks **where a verifier exists**, because the verifier is objective.

The traps:

- **Answer extraction is lossy and model-favoring.** A model that emits a clean `#### 42` or `\boxed{42}` scores easily; a verbose model that says "...so the answer is forty-two" scores 0 even when correct. This asymmetry can *manufacture* a gap between two equally-capable models. The fix labs use: a fixed few-shot prompt that pins the output format so extraction is reliable for everyone. `extract_final_int` prefers `\boxed{}`, then the last integer — and the notebook shows it failing on a correct-but-unformatted answer.

- **Tiny high-variance sets.** **AIME** has *30 problems per year*. A single greedy pass@1 on 30 items has a ±~18-point CI — one lucky problem swings the headline. This is why frontier reasoning models report **avg@k** (mean over k samples, e.g. avg@64) or **maj@k / cons@k** (majority vote) instead of a single decode. `metrics.pass_at_k` (the unbiased Codex estimator), `avg_at_k`, and `majority_at_k` implement these. **If you ever see a single number on AIME with no k and no error bar, distrust it.**

- **Saturation + contamination.** GSM8K and HumanEval are largely solved by frontier models (>95%) and are *thoroughly* in everyone's training data. They retain value as regression checks and for small models, but a frontier "98.x on GSM8K" carries almost no information. The field has migrated to contamination-resistant successors: **LiveCodeBench** (time-gated — only problems published after a model's training cutoff count), **SWE-bench Verified** (real GitHub issues), and freshly-authored math (**AIME 2025**, **GSM1k**).

### 4c. Instruction-following with verifiable constraints (IFEval)

A rare island of trustworthy automatic eval. **IFEval** (Zhou et al. 2023) uses instructions whose compliance can be checked *programmatically*: "respond in JSON", "use exactly 3 bullet points", "write at least 200 words", "don't use any commas". No judge, no extraction ambiguity — the score is a fact about the string. `benchmarks.CONSTRAINT_CHECKERS` implements a representative set, and IFEval reports two granularities: **prompt-level** (strict — did it obey *every* constraint?) and **instruction-level** (loose — fraction of constraints met). Both matter; the loose rate is a smoother training signal, the strict rate is what the user feels.

This is the cleanest way to measure the *instruction-following* capability that SFT is supposed to install, separate from knowledge or reasoning.

### 4d. Open-ended quality, scored by humans or a judge (Chatbot Arena, MT-Bench, AlpacaEval, Arena-Hard)

Once the output is a paragraph of chat, no rule scores it. Two options, both with teeth:

- **Human preference at scale.** **Chatbot Arena / LMArena** shows the same prompt to two anonymous models and asks a human which is better, then fits an **Elo / Bradley-Terry** rating from millions of votes. This is the closest thing the field has to a ground-truth ranking of general chat quality — and it's still gameable (a model tuned for chatty, formatted, agreeable answers can out-rank a more correct but terse one; the early-2025 "style over substance" and benchmark-specific-tuning controversies are the cautionary tales) and slow/expensive to run.

- **LLM-as-judge.** Use a strong model as the grader. This is §6 and the heart of `judge.py`. It's how MT-Bench, AlpacaEval, and Arena-Hard automate the human-preference signal — and how almost every internal "is the new checkpoint better?" eval works, because it's the only thing that scales to nightly runs.

---

## 5. Error bars: the statistics nobody reports (`metrics.py`)

This is the most quietly important section in the module, and the cheapest fix for the most expensive mistakes.

**Every benchmark number is a sample mean.** "62.3%" means "we scored a sample of problems and got 0.623." The questions that decide a release are statistical:

### How wide is the error bar?

`metrics.bootstrap_ci` gives a distribution-free 95% CI by resampling problems with replacement. `metrics.wilson_interval` gives the closed-form proportion CI as a fast cross-check. The sobering number from `metrics.min_n_for_halfwidth`: **a ±2-point CI on a ~60% accuracy needs ~2,300 problems.** Most public benchmark *subsets* people run are 200–1000 items, giving ±3–7 points. So:

> **Most 1–2 point leaderboard gaps are statistically indistinguishable from noise.** When you see "we improved MMLU from 67.1 to 68.0," the correct response is "what's the CI?" — and the honest answer is usually "they overlap completely."

### Is B *really* better than A?

Don't compare two marginal CIs — compare the **paired difference**, because both models were scored on the *same* problems and problem difficulty (the dominant variance source) cancels. `metrics.paired_bootstrap_diff` resamples problem *indices* so the A–B correlation is preserved, and the test in `test_metrics.py` shows the paired CI is **~5× tighter** than the unpaired comparison on correlated models. The accompanying significance test is **McNemar's** (`mcnemar_test`): of the problems where the two models *disagree*, is the win/loss split far enough from 50/50 to be unlikely under "equally good"? `eval.py --compare a.json b.json` runs this on two saved scorecards and prints `SIGNIFICANT` / `not significant`.

This is the single highest-leverage habit in the module. Anthropic's **"Adding Error Bars to Evals"** (Miller, 2024) is the one-paper version: report a CI, and for A-vs-B use the paired/clustered standard error. Almost no public benchmark report does this. Yours should.

### Sampling-based metrics for generative tasks

When the model samples (temperature > 0), one decode is one draw from a distribution. Report:

- **pass@k** — probability that k samples contain ≥1 correct. The naive "generate k, did any pass?" is biased and noisy for small k; the **unbiased Codex estimator** `1 − C(n−c,k)/C(n,k)` lets you draw n=100 *once* and read off pass@1, pass@10, pass@100 from the same samples (`metrics.pass_at_k`).
- **avg@k** — mean correctness over k samples; the stable replacement for a single greedy decode on tiny sets like AIME.
- **maj@k / self-consistency** (Wang et al. 2022) — take the modal answer across k samples; usually beats avg@k because errors are diffuse but the correct answer is the single most common attractor.

`harness.run_gsm8k` computes all three automatically when `generation.n_samples > 1`.

---

## 6. LLM-as-judge, done right (`judge.py`)

The judge is the workhorse of modern post-training eval and the easiest instrument to fool. Three biases, three defenses, all in `judge.py`:

### Position bias → swap and resolve

Judges systematically favor whichever answer is shown first (some models, second). **Defense:** run every pair in *both* orders; only count a win if it survives the swap. If the judge flips when the answers flip, it was responding to position, not substance — score it a tie. `pairwise_judge(..., swap=True)` does this and returns a `position_bias` flag; `aggregate_win_rate` reports the **position-bias rate** as a first-class diagnostic. `test_judge.py` constructs an "always pick the first" judge and confirms it's caught and neutralized.

### Verbosity / length bias → measure it, then control

Judges reward longer answers regardless of quality. This is why naive AlpacaEval was gameable by padding, and why **AlpacaEval 2.0 length-controlled** (Dubois et al. 2024) regresses length out of the win rate. `aggregate_win_rate` reports the **length–win correlation**; a strongly positive value means your win rate is partly a length artifact. The honest fixes: length-control the metric, or hold answer length roughly constant across the models you compare.

### Self-preference bias → judge with a different family

A model rates its own family higher (Panickssery et al. 2024). **Defense:** use a judge from a *different* model family than the one under test (the canonical config judges Qwen with a non-Qwen model), and never let a model be the sole judge of its own outputs.

### The decisive practice: calibrate against humans

Before you trust a judge on thousands of pairs, check it against human labels on a small gold set. `judge.agreement` computes **Cohen's κ** (chance-corrected — raw agreement lies when the label distribution is skewed; `test_judge.py` shows a judge with 80% raw agreement and κ≈0). Rule of thumb: **κ ≥ 0.4 before you trust the judge at scale.** A judge you haven't calibrated is measuring something, but you don't know what.

`judge.py` ships a `Judge` Protocol with two backends: `LocalJudge` (a real HF model, loaded lazily) and `DummyJudge` (deterministic, for the $0 path and tests). Pointwise rubric scoring (`POINTWISE_TEMPLATE`, `parse_score`) is included for absolute grading; pairwise is preferred when you have a baseline because relative judgments are far more reliable than absolute 1–10 scores.

---

## 7. Contamination: the dirty secret (`contamination.py`)

A benchmark measures *generalization* only if the model never saw the answers. But pretraining corpora are scraped from the whole web, and the whole web contains GSM8K, MMLU, and HumanEval — often with answers, often many times. When the test set leaked into training, the "score" is partly a memorization readout.

**How big is the effect?** When Scale AI built **GSM1k** (Zhang et al. 2024) — fresh grade-school problems matched in difficulty to GSM8K — some models dropped up to **~13 points**, while genuinely-strong models barely moved. That gap *is* the contamination-plus-overfitting tax (`contamination.overfit_gap`).

Two detectors you can run without model internals:

1. **N-gram overlap** (the GPT-3 / Llama / FineWeb method). Build an index of the training corpus's n-grams; flag any test item whose n-grams are largely present (`contamination_report`, default 13-gram, threshold 0.5). Catches verbatim leakage. The in-memory `set` here is the teaching version; production uses Bloom filters or suffix arrays (FineWeb's exact-substring dedup, Google's `deduplicate-text-datasets`).

2. **Canary strings.** BIG-bench and others embed a fixed GUID in their files specifically so you can grep your corpus for it (`find_canary`, `BIG_BENCH_CANARY`). A hit means benchmark files are in your training data.

Two more that need the model (described, not shipped): **perplexity gaps** (memorized text has anomalously low loss — compare loss on the benchmark vs. paraphrases of it) and the **order-sensitivity probe** (Oren et al. 2023 — a model that memorized a dataset scores higher on the canonical example order than on a shuffled order).

**The only real defense is a private, never-published, freshly-authored eval set** — which is exactly why frontier labs treat held-out evals as core infrastructure and refresh them constantly. The Italian eval probes you curated in Module 03a are a small instance of this idea: a held-out set you authored, so nobody trained on it.

---

## 8. Evaluating at each stage of the pipeline

The course built a model through stages: base → CPT (M13) → SFT (M15) → DPO (M16) → GRPO (M17) / distillation (M18). Labs evaluate **at every stage**, because each stage has a different goal and a different failure mode, and because that's how you localize a regression.

Point `model.checkpoint` in `eval_qwen3_1.7b.yaml` at each stage's checkpoint and run the same scorecard. What to expect and watch for:

| Stage | Primary eval | What should improve | The failure mode to watch |
|---|---|---|---|
| **Base / pretrained** | perplexity, few-shot MMLU/ARC | knowledge, raw capability | — (the floor you build on) |
| **CPT (M13)** | held-out domain QA + retention perplexity | domain knowledge | **catastrophic forgetting** of general benchmarks |
| **SFT (M15)** | IFEval, MT-Bench-style judge | instruction-following, format | **capability regression** — knowledge benchmarks can *drop* (the "alignment tax") |
| **DPO (M16)** | pairwise judge win-rate vs the SFT model | helpfulness, preference alignment | **reward hacking** — longer/sycophantic answers that judges love but users don't; **over-refusal** |
| **GRPO (M17)** | GSM8K/MATH avg@k + maj@k | reasoning, verifiable accuracy | **format collapse**, **prior-skill loss** on non-reasoning evals |
| **Distillation (M18)** | the two-axis (new skill + prior skill) | the target capability | **forgetting** the prior skill (the whole SDFT point) |

The two recurring cross-stage signals: the **alignment tax** (post-training for helpfulness/safety can cost raw capability — measure it explicitly by running MMLU before and after, with a paired test) and **regression** (a new stage silently breaks something the last stage could do — which is why labs keep a *regression suite* of capabilities that must never drop). `eval.py --compare` on two stages' scorecards is exactly the paired test that quantifies both.

---

## 9. How labs *actually* evaluate (and what they publish)

There's a gap between the benchmark tables in a model card and how the decision to ship was actually made. What frontier teams really lean on:

- **Private, contamination-free, continuously-refreshed eval sets** — the real arbiter. Public benchmarks are for the press release and external comparison; the internal held-out sets decide what ships.
- **Continuous eval during training** — downstream benchmarks tracked on a schedule alongside the loss curve, so a regression is caught in hours, not after the run.
- **Capability-specific evals built for the product** — a coding assistant team builds a coding eval that mirrors real usage; a search team builds a grounding/faithfulness eval. Generic benchmarks are necessary, never sufficient.
- **Human evaluation with rubrics** — expert raters doing side-by-side comparisons against a detailed rubric, for the capabilities that matter most and where judges aren't trusted yet.
- **Red-teaming and safety evals** — adversarial prompting, jailbreak robustness, harmful-content refusal rates. These are go/no-go gates, not leaderboard rows. (See the published **system cards** from OpenAI, Anthropic, and Google for the disclosed slice.)
- **Online A/B tests** — the final arbiter for a deployed product: real users, real metrics (thumbs, retention, task completion), the new model behind a flag against the old one.
- **The vibe check** — fast, qualitative, real. Karpathy's point that you should *talk to your model* on prompts you care about. It catches gross failures no benchmark covers and builds the intuition that tells you which benchmark moves to believe. It is a complement to systematic eval, never a replacement.

The synthesis you should internalize: **public benchmarks are for orientation and external comparison; private evals, human judgment, regression suites, and online tests are for decisions.** A lab that ships on public-benchmark scores alone is a lab that's about to get caught by contamination or Goodhart.

### Goodhart, in one line

> When a measure becomes a target, it ceases to be a good measure.

Every benchmark you optimize against directly will eventually be gamed by your own training — sometimes deliberately, often accidentally (the benchmark leaks into a synthetic data pipeline). The defense is the same as the contamination defense: keep your *decision* evals separate from your *optimization* signal, and refresh them.

---

## 10. Running it

```bash
# Offline / $0 smoke — synthetic data, dummy judge, runs on CPU.
python eval.py --config=configs/eval_demo.yaml --judge

# Canonical: score the post-trained Qwen3-1.7B on all three suites.
python eval.py --config=configs/eval_qwen3_1.7b.yaml

# Score a specific stage's checkpoint (e.g. the GRPO model from Module 17),
# GSM8K only, with pass@k from 8 samples.
python eval.py --config=configs/eval_qwen3_1.7b.yaml \
    --model.checkpoint=../17-reasoning-and-grpo/results/checkpoints/step_00000300 \
    --benchmarks.suites=gsm8k --generation.greedy=false --generation.n_samples=8

# See format/normalization sensitivity: same model, different harness.
python eval.py --config=configs/eval_qwen3_1.7b.yaml --benchmarks.suites=mmlu \
    --benchmarks.mc_style=cloze --benchmarks.mc_norm=byte --out=results/mmlu_cloze_byte.json

# LLM-as-judge win-rate vs a baseline (use a non-Qwen judge).
python eval.py --config=configs/eval_qwen3_1.7b.yaml --judge \
    --benchmarks.suites=  # skip automatic suites, judge only

# Contamination scan against a corpus glob.
python eval.py --config=configs/eval_qwen3_1.7b.yaml --contamination \
    --contamination.corpus_glob='../../part-3-pretraining/**/*.jsonl'

# Paired significance test between two stages' scorecards (no model load).
python eval.py --compare results/sft.json results/dpo.json
```

Every run writes a JSON scorecard and a markdown twin (the `Scorecard.to_markdown` table) into `results/`. Each headline number carries its 95% CI, and the MMLU row carries the normalization sweep so the sensitivity is visible, not hidden.

**Cost:** ~$2–5 on an A100 for the canonical config — evaluation is far cheaper than training. The MC suite is forward-only; the cost is in GSM8K/IFEval generation. Bump `n_per_suite` toward 1000+ for publication-grade CIs (and watch the CI tighten in the scorecard).

---

## 11. Gotchas (the ones that bite in practice)

1. **Reporting a score without a CI.** The default failure. A number with no error bar invites you to over-interpret noise. `metrics.bootstrap_ci` is one line — there's no excuse.
2. **Comparing two models with unpaired CIs.** You'll call a real improvement "not significant" or a noise blip "significant." Always pair (`paired_bootstrap_diff`) when the items are the same.
3. **Trusting an uncalibrated judge.** Run `agreement` against ~50 human labels first. A judge with κ < 0.4 is a random number generator with good PR.
4. **Forgetting to position-swap.** A 5–15 point chunk of your judge win-rate can be pure position bias. `swap=True` is nearly free; leave it on always.
5. **Reading a single AIME/small-set number.** 30 problems → ±18pt. Demand avg@k or maj@k with k stated.
6. **Comparing MMLU across papers.** Different harness/few-shot/normalization/format → 5+ point artifacts. Only compare numbers from the *same* harness config.
7. **Assuming a high public-benchmark score generalizes.** Contamination + Goodhart. Validate on a private, use-case-specific set before believing it.
8. **Optimizing the eval you decide with.** Keep the optimization signal and the decision eval separate, or you'll Goodhart yourself.

---

## 12. Stretch goals

- **Build your own GSM1k.** Hand-author (or carefully synthesize + human-verify) 50 fresh math problems matched to GSM8K difficulty, run both, and measure your model's overfit gap.
- **Length-controlled win rate.** Extend `aggregate_win_rate` to fit AlpacaEval-2.0's logistic length control and report the de-biased rate alongside the raw one.
- **Wire in `lm-eval-harness`.** Run the same MMLU through EleutherAI's harness and reconcile the number with yours — the discrepancy *is* the lesson about harness sensitivity.
- **Perplexity-gap contamination probe.** Add a detector that compares the model's loss on benchmark items vs. paraphrases of them; a large gap flags memorization.
- **A real regression suite.** Freeze a set of capability probes that must never drop, and make `--compare` fail loudly if any regress beyond their CI.
- **Multi-judge ensembles + tie-breaking.** Use 3 different judge families and report agreement among them; treat unanimous disagreement with humans as a calibration alarm.

---

## 13. Reading list

- **Adding Error Bars to Evals** — Miller, Anthropic 2024. The one paper to internalize: CIs and paired/clustered tests for LLM evals.
- **A Careful Examination of LLM Performance on Grade School Arithmetic** (GSM1k) — Zhang et al. 2024. The contamination/overfitting tax, measured.
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al. 2023. The foundational LLM-as-judge + Arena Elo paper, including position and verbosity bias.
- **Length-Controlled AlpacaEval** — Dubois et al. 2024. Regressing length out of judge win-rates.
- **LLM Evaluators Recognize and Favor Their Own Generations** — Panickssery et al. 2024. Self-preference bias, quantified.
- **HELM** — Liang et al. 2022. The case for multi-metric, multi-scenario evaluation and the documentation of prompt-format sensitivity.
- **IFEval: Instruction-Following Eval** — Zhou et al. 2023. Verifiable-constraint instruction following.
- **GPQA: A Graduate-Level Google-Proof Q&A Benchmark** — Rein et al. 2023. Contamination-resistant knowledge eval.
- **Proving Test Set Contamination in Black-Box Language Models** — Oren et al. 2023. The order-sensitivity contamination probe.
- **Self-Consistency Improves Chain-of-Thought Reasoning** — Wang et al. 2022. maj@k / self-consistency.
- **Evaluating Large Language Models Trained on Code** (Codex) — Chen et al. 2021. The unbiased pass@k estimator.
- **`lm-evaluation-harness`** (EleutherAI) — the production standard you should use in real life; read its task configs to see how many knobs a "single benchmark" actually has.
- **System cards** — OpenAI, Anthropic, Google DeepMind. The disclosed slice of how frontier labs evaluate, including safety/red-team methodology.

---

**Next:** Part 6 — the bigger picture. Module 20 (Reading the Literature) and Module 21 (Scaling Intuitions / Career). You now have the full loop: build a model, post-train it, and — the skill most people skip — *know whether it actually got better.*
