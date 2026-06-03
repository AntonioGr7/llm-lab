# Module 21 — Reading the Literature

The field you just learned moves weekly. The specific techniques in this course — the exact post-training recipe, the current best attention variant, the SOTA eval — will drift. The half-life of a "current best practice" in LLMs is about eighteen months. So the most durable thing this course can give you is not a technique. It's a **method for keeping up** without quitting your job to read arXiv full-time.

That method has three parts, and this module teaches all three:

1. **Triage** — how to read a paper in 30 minutes and know whether it's worth 3 hours.
2. **Discernment** — how to tell a genuine advance from a well-marketed non-result.
3. **Reproduction** — the one skill that separates people who *understand* the literature from people who've merely *read* it.

This is the shortest module in the course and the one with the longest shelf life. Everything else taught you what was true in 2026. This teaches you how to find out what's true in 2030.

## What you'll walk away with

- A repeatable **30-minute triage procedure** you can run on any paper, with a checklist you fill in (in the notebook, as data).
- A **red-flag / green-flag scanner** for separating signal from noise — the recurring tells of an over-claimed result.
- A **curated reading list** of the ~18 papers a graduate of this course should know, as a queryable dataset you can sort by area, tier, and the course module that used it.
- A way to **find the seminal papers in any new area** using the citation graph — so you start at the load-bearing work, not a random recent preprint.

## 1. The problem: you cannot read everything

Roughly a hundred LLM papers hit arXiv every day. If you tried to read them all you would do nothing else, and you would still fall behind. Everyone who operates in this field — including the researchers at frontier labs — reads a tiny fraction, deeply, and triages the rest in minutes.

The skill is not *reading faster*. It's **deciding what not to read**, quickly and well, and extracting the one idea from a paper without committing to the whole thing. A senior researcher and a new grad student read the same number of papers per week. The senior researcher just spends 90% of that time on the 10% that matters, and knows which 10% that is in five minutes.

## 2. The 30-minute triage

Don't read a paper front-to-back on the first pass. Read it in **three passes**, escalating your investment only if it survives the previous one. [litkit.py](litkit.py) encodes this as nine questions across the three passes; the notebook lets you fill them in.

**Pass 1 — 5 minutes. Is this even relevant?**
Read the title, the abstract, the figures, and the conclusion. Skip everything else. Answer two questions:
- *What is the single concrete claim?* In one sentence, no hedging. If you can't state it, the paper hasn't stated it either — a bad sign.
- *What's the delta?* What does it do that the previous best didn't? "We get SOTA" is not a delta; "we remove the reward model from RLHF" is.

If it's not relevant to anything you care about, stop here. You spent five minutes and you have the headline. That's a successful read.

**Pass 2 — 15 minutes. Is the claim supported?**
Now read the method and the main results table. Answer:
- *What is the method, in two sentences a peer could re-implement from?* If you can't compress it, you don't understand it yet (or it's underspecified).
- *What are the baselines — are they strong and current?* This is where most papers are weakest. A gain over a weak or untuned baseline is not a gain. (DPO over a badly-tuned PPO tells you nothing.)
- *What experiment supports the claim, on what data, at what scale?*
- *Which design choices are ablated, and which are asserted on faith?* The ablation table is where a paper proves it understands its own method.

**Pass 3 — 10 minutes. Would it survive contact with reality?**
- *Is there code and data? Could **you** reproduce the headline number?* Reproducibility is the strongest signal of all.
- *What does the paper admit it does not do — and what does it quietly avoid mentioning?*
- *Does the result transfer to your scale and setting, or only to theirs?* A method that works at 70B may do nothing at 1.7B (and vice versa).

After 30 minutes you either have a deep-read candidate or you have the paper's core idea and a reason it didn't make the cut. Both outcomes are wins. The notebook's `Triage` object tracks your completeness — a read below ~70% of the questions answered is a skim, not a triage, and you don't yet know enough to judge.

## 3. Signal vs. noise

Most papers aren't fraudulent. They're **over-claimed** — a real but small effect, dressed up as a breakthrough. After a few hundred papers you internalize the tells. Until then, run the checklist. `litkit.py`'s `SIGNALS` scores a paper on green flags and red flags:

**Green flags (trust accrues):**
- Compared against the *current strongest* method, *tuned fairly*.
- Code and enough detail to reproduce, released.
- Results with **error bars** / multiple seeds (you built the tooling for this in Module 20 — most 1–2 point gaps are noise).
- Key design choices **ablated**.
- Limitations stated concretely, not as a throwaway paragraph.
- The effect shown at **more than one scale**.

**Red flags (suspicion accrues):**
- The headline rests on **cherry-picked qualitative examples** ("look at this one great generation").
- The whole claim hangs on a **single benchmark number**.
- Baselines are suspiciously weak or obviously untuned.
- The method is **too vague to re-implement**.
- The gain could be **contamination** — the model saw the test set (Module 20's lesson).

None of these is a verdict on its own. A great paper can lack error bars; a bad one can have code. The checklist's value is that it forces you to *ask*, instead of being carried along by good writing. Good writing is not evidence. It's just good writing.

A specific habit worth building: **when a result seems important, look for the replication.** The Chinchilla scaling paper (which you'll meet in Module 22) is the canonical example — hugely influential, headline correct, but a 2024 replication attempt found its published constants internally inconsistent. The shape was right; the table was quietly wrong, and it took two years and an adversarial re-read to surface. Trust the shape. Verify the digits.

## 4. The reading list, as data

[litkit.py](litkit.py) ships an opinionated reading list — about 18 papers a graduate of *this course* should know — not as a flat bibliography but as a **queryable dataset**. Each entry carries its area (transformers / scaling / pretraining / post-training / eval / systems), a tier, the course module that used it, and one sentence on why it matters.

Three tiers, and you read them differently:
- **Foundational** — read in full, more than once. Attention Is All You Need, GPT-3, Chinchilla, InstructGPT, DPO. The load-bearing walls of the field.
- **Important** — read carefully once. LoRA, QLoRA, FlashAttention, ZeRO, R1, the eval-error-bars paper.
- **Frontier** — track, read as the need arises. The technical reports (DeepSeek-V3, Llama 3, Tülu 3) and the bleeding edge (SDFT).

The notebook sorts this list into reading order (foundational first, oldest first within a tier, because ideas build on each other) and lets you filter by area — so when you want to go deep on, say, post-training, you get exactly those papers in dependency order.

This is also a map back over the whole course: nearly every paper here links to the module where you *implemented* its idea. You didn't just read DPO — you built it. That's the difference this course is trying to make.

## 5. Finding the seminal papers in a new area

When you enter an area you don't know, the fastest orientation is the **citation graph**. The papers everyone cites are the ones to read first; recent preprints citing them are the frontier. You don't start at a random 2026 paper — you start at the work it stands on.

`litkit.py` encodes a toy dependency graph over the reading list (what each paper directly builds on) and ranks every paper by **how much work transitively depends on it**. Run `most_foundational()` and "Attention Is All You Need" comes out on top — fourteen of the other papers descend from it — followed by GPT-3. That ranking *is* your reading order for orienting in the field: start where the most arrows point, work outward toward the leaves (the frontier).

In real life you'd run this over Semantic Scholar or the arXiv citation graph (both have APIs). The principle is identical: **in-degree is a proxy for importance, and the leaves of the graph are the research frontier.** The notebook shows the toy version so the mechanic is concrete; the README's reading list points you at the real tools.

## 6. The skill nobody teaches: reproduction

Here is the most important sentence in this module. **Reproducing a result is the only way to know you actually understand it.**

Reading a paper gives you the *illusion* of understanding. You followed the argument, the figures made sense, you could summarize it at a meeting. Then you try to implement it and discover the three things the paper didn't tell you — the learning rate that actually matters, the data preprocessing the authors considered too obvious to mention, the initialization that makes it converge. Those three things are where the real knowledge lives, and you only find them by building it.

This is exactly what this course made you do. You didn't read about attention — you wrote it (Module 04). You didn't read about DPO — you derived the loss and trained with it (Module 17). Every module was a reproduction. That's why you can now read the source papers and *recognize* the choices, instead of taking them on faith.

So the highest-leverage thing you can do after this course is **pick one recent paper and reproduce its headline result at small scale.** Qwen3-0.6B on a single GPU is enough for most post-training and many architecture papers. The valuable part is not confirming the paper is right — it's finding where it's *fragile*: the setting where the effect vanishes, the baseline that was under-tuned, the claim that only holds at the authors' scale. A clean small-scale reproduction with an honest "here's where it breaks" is worth more, as a signal of skill, than ten papers read. It's also, not coincidentally, the single best thing you can put in a portfolio (see Module 22).

## 7. A sustainable weekly habit

You do not need to read full-time. A few hours a week, spent well, keeps you current:

- **Skim the firehose, triage a handful.** Follow a small number of high-signal sources (a few researchers, one or two newsletters, the labs' release feeds). Pass-1 triage maybe ten papers a week; deep-read one.
- **Read the technical reports in full.** When a major model ships (Qwen, Llama, DeepSeek, OLMo), its report is the best textbook in the field that week. Block the time.
- **Keep one benchmark you trust.** You built an eval harness in Module 20 — use it as your personal sniff test for new claims. A number you can reproduce beats a number you read.
- **Reproduce one thing a quarter.** Small, finished, honest. This is what compounds.

The goal is not to read everything. It's to never be surprised by the *shape* of what's coming — and to be able to go deep, fast, on the rare thing that matters to you.

## Reading list

This module's reading list is *about reading* — the rest of the field's reading list lives in `litkit.py` and in each module's own list. A few meta-resources:

- Keshav, *How to Read a Paper* (2007). The three-pass method, two pages, still the best thing written on this. Read it first.
- Andrew Ng's advice on reading research papers (the "read many abstracts, then go deep" strategy). Widely transcribed.
- The Chinchilla replication (Besiroglu et al., 2024) — not for the scaling content but as a worked example of adversarial reading finding a quiet error in a famous result.
- Any frontier model's technical report (Qwen3, Llama 3, DeepSeek-V3) — read one in full as an exercise in seeing the whole stack documented at once.

## Notebook

[notebook.ipynb](notebook.ipynb) makes the method concrete:

1. The triage checklist — fill in a `Triage` for a real paper and watch the completeness and verdict update.
2. The signal/noise scanner — score a strong paper and an over-claimed one side by side; see the net score separate them.
3. The reading list as data — filter by area and tier, sort into reading order, see the map back over the course.
4. The citation graph — rank the reading list by foundational reach, visualize which papers everything depends on, and read off where to start.

It runs in milliseconds on CPU — no model, no network. The point isn't computation; it's turning "read more papers" into a procedure you can actually run.

---

Next: the final module — **Module 22, Scaling Intuitions** — where the small runs you did become the arithmetic behind the runs that cost $50M, and we close the course by looking at what comes next.
