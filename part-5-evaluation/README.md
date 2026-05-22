# Part 5 — Evaluation

Evaluation is its own Part on purpose.

The most underrated skill in this field is knowing whether your model actually got better. It's harder than making it better. Plenty of researchers can train a model; far fewer can tell you, with evidence, that their new version is an improvement and not a regression hiding behind cherry-picked examples.

Most courses bolt evaluation onto the end of a fine-tuning notebook as an afterthought. We make it a first-class citizen. You'll leave this Part skeptical of benchmark numbers in a healthy way, and equipped to build evaluations that actually mean something for your specific use case.

## Modules

- **[17 — Evaluation That Actually Means Something](17-evaluation/)** — Why perplexity is necessary but not sufficient. Standard benchmarks and where they lie. LLM-as-judge. Vibe evals vs systematic evals. Contamination — the dirty secret of benchmark scores. How labs actually evaluate models internally vs what they publish.

## What you'll be able to do at the end of this Part

- Look at any leaderboard score and identify what could be wrong with it.
- Build an evaluation suite for a specific use case that you'd actually trust to ship a model.
- Use LLM-as-judge correctly — and know when it betrays you.

## Time and cost

- Reading + coding: ~3 hours.
- Compute cost: ~$2–5. Evaluation is much cheaper than training.
