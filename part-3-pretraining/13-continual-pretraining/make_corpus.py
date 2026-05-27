"""Generate the fictional-company corpus for the continual-pretraining demo.

The whole demo rests on one requirement: we need knowledge the base model
*provably* cannot already have. Real "recent" data can't give that guarantee
(you can never fully rule out contamination, and you'd have to hand-build an
answer key). So we invent a world.

This script procedurally generates a **fictional company** — randomized
products, people, and *numeric* specs. Because the specifics are random, they
are guaranteed novel: no pretraining corpus contains "the Atlas-7's rated
payload is 4200 kg", and we hold the ground-truth answer key by construction.
This mirrors the real enterprise scenario (a private corpus no public model has
seen) in miniature, and is how researchers isolate knowledge injection (the
synthetic-biography setup in the *Physics of Language Models* work).

What it writes (under `--out`, default `results/corpus`):

  domain.{bin,idx.npy,meta.json}   the training corpus: raw docs + paraphrases
                                   + train-QA, tokenized into a Module-12 corpus.
  replay.{bin,idx.npy,meta.json}   a generic-text corpus for the replay mix.
  qa_heldout.jsonl                 held-out QA probes: {question, answer, ...}.
                                   NEVER written into training — the eval set.
  facts.jsonl, domain_docs.jsonl   the structured facts + raw text, for the
                                   notebook and for inspection.

Two augmentation levers from README lever 3 are baked in:
  - **paraphrase**: each fact is rendered through several sentence templates,
    so the model sees it in multiple phrasings (storage -> extraction).
  - **synthetic QA**: facts are also rendered as Q/A text, which is what makes
    the knowledge retrievable at inference rather than merely latent.

The held-out probes use a *different* question phrasing from anything in
training (a distinct template set), so a model that scores well must have
generalized the knowledge, not memorized a question string. `tests/
test_qa_holdout.py` enforces that no held-out question leaks into training.

Tokenization uses the model's real tokenizer by default (so the corpus is
compatible with `Qwen3-0.6B-Base`). `--fake-tokenizer` builds a tiny word-level
tokenizer instead, for a fully offline `$0` smoke run on a tiny model (used by
the notebook); it writes `fake_tokenizer.json` so eval/notebook can decode.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from indexed_dataset import IndexedDatasetBuilder, best_dtype_for_vocab


# =============================================================================
# Word banks (deterministic; randomness comes only from the seeded RNG)
# =============================================================================

_PRODUCT_PREFIX = ["Atlas", "Helix", "Orion", "Vega", "Nimbus", "Pylon",
                   "Cobalt", "Drift", "Quanta", "Mesa", "Talos", "Lumen",
                   "Apex", "Sable", "Kestrel", "Borealis"]
_FIRST = ["Mara", "Devin", "Soren", "Priya", "Lena", "Kato", "Ingrid", "Rashid",
          "Noa", "Yuki", "Bram", "Tariq", "Elsa", "Owen", "Hala", "Dmitri"]
_LAST = ["Voss", "Okonkwo", "Albright", "Castellanos", "Nakamura", "Fenwick",
         "Delacroix", "Oyelaran", "Sandoval", "Eriksson", "Haddad", "Whitlock"]
_CITIES = ["Bracken Hollow", "Fort Dell", "Saltmarsh", "New Caraway", "Pell Ridge",
           "Granite Bay", "West Aldon", "Marrow Creek", "Cinderport", "Thale Junction"]
_COMPANY_A = ["Meridian", "Halcyon", "Aster", "Greywater", "Brightspire",
              "Ironwood", "Cleftstone", "Vantage", "Dunmore", "Sablefield"]
_COMPANY_B = ["Robotics", "Dynamics", "Systems", "Logistics", "Automation",
              "Instruments", "Mechanics", "Works"]


# Per-attribute specs. Each has: a value generator, the unit, the templates used
# to phrase the fact in DOCUMENTS (multiple -> paraphrase augmentation), a TRAIN
# question template, and a HELD-OUT question template that is intentionally
# different in surface form. `{p}` = product name, `{v}` = formatted value.
def _name(r):  # a person name
    return f"{r.choice(_FIRST)} {r.choice(_LAST)}"


_ATTRS = [
    dict(key="payload", unit="kg",
         gen=lambda r: r.randrange(500, 9000, 50),
         facts=["The {p} has a rated payload of {v} kg.",
                "Engineers rate the {p} to carry up to {v} kg.",
                "Payload capacity for the {p} is specified at {v} kg."],
         train_q="What is the rated payload of the {p}?",
         held_q="How many kilograms can the {p} carry at most?"),
    dict(key="speed", unit="km/h",
         gen=lambda r: r.randrange(3, 26),
         facts=["The {p} travels at a top speed of {v} km/h.",
                "Maximum speed of the {p} is {v} km/h.",
                "The {p} is governed to {v} km/h on the floor."],
         train_q="What is the top speed of the {p}?",
         held_q="How fast does the {p} go at its quickest?"),
    dict(key="battery", unit="kWh",
         gen=lambda r: r.randrange(2, 41),
         facts=["The {p} ships with a {v} kWh battery pack.",
                "A {v} kWh pack powers the {p}.",
                "Onboard energy storage on the {p} is {v} kWh."],
         train_q="What is the battery capacity of the {p}?",
         held_q="How large is the energy pack inside the {p}?"),
    dict(key="year", unit="",
         gen=lambda r: r.randrange(2024, 2039),
         facts=["The {p} first shipped in {v}.",
                "{p} units began rolling off the line in {v}.",
                "The {p} entered service in {v}."],
         train_q="In what year did the {p} first ship?",
         held_q="When did the {p} enter service?"),
    dict(key="factory", unit="",
         gen=lambda r: r.choice(_CITIES),
         facts=["The {p} is assembled in {v}.",
                "Final assembly of the {p} happens at the {v} plant.",
                "The {p} is built in {v}."],
         train_q="Where is the {p} assembled?",
         held_q="At which site is the {p} put together?"),
    dict(key="designer", unit="",
         gen=_name,
         facts=["The {p} was designed by {v}.",
                "Lead designer of the {p} is {v}.",
                "{v} led the design of the {p}."],
         train_q="Who designed the {p}?",
         held_q="Which engineer led the design of the {p}?"),
]


# =============================================================================
# Universe
# =============================================================================

def build_universe(seed: int = 0, n_entities: int = 64) -> dict:
    """Procedurally generate the fictional company + its products.

    Deterministic in `seed`: same seed => byte-identical corpus. The random
    *values* are what guarantee novelty — no real corpus asserts these specs.
    """
    r = random.Random(seed)
    company = f"{r.choice(_COMPANY_A)} {r.choice(_COMPANY_B)}"

    names: set[str] = set()
    products = []
    while len(products) < n_entities:
        name = f"{r.choice(_PRODUCT_PREFIX)}-{r.randrange(1, 99)}"
        if name in names:
            continue
        names.add(name)
        prod = {"name": name}
        for a in _ATTRS:
            prod[a["key"]] = a["gen"](r)
        products.append(prod)

    company_facts = {
        "ceo": _name(r),
        "founded": r.randrange(1998, 2020),
        "hq": r.choice(_CITIES),
    }
    return {"company": company, "products": products, **company_facts}


def universe_facts(u: dict) -> list[dict]:
    """Flatten the universe into structured (subject, attr, value) facts."""
    facts = []
    for p in u["products"]:
        for a in _ATTRS:
            facts.append({"subject": p["name"], "attr": a["key"],
                          "value": str(p[a["key"]]), "unit": a["unit"]})
    facts.append({"subject": u["company"], "attr": "ceo", "value": u["ceo"], "unit": ""})
    facts.append({"subject": u["company"], "attr": "founded", "value": str(u["founded"]), "unit": ""})
    facts.append({"subject": u["company"], "attr": "hq", "value": u["hq"], "unit": ""})
    return facts


# =============================================================================
# Text rendering: documents (raw + paraphrases), train QA, held-out QA
# =============================================================================

def render_documents(u: dict, seed: int = 0, augment: int = 4) -> list[str]:
    """One spec-sheet doc per product, plus `augment` paraphrases of each.

    A document strings together one fact-sentence per attribute, with the
    template choice and sentence order re-rolled per paraphrase. Same facts,
    many phrasings — the augmentation that turns *stored* knowledge into
    *extractable* knowledge (README lever 3).
    """
    r = random.Random(seed + 101)
    docs = []
    # Company overview (a few variants too).
    for _ in range(max(1, augment)):
        docs.append(
            f"{u['company']} is a robotics company headquartered in {u['hq']}, "
            f"founded in {u['founded']} and led by CEO {u['ceo']}. Its product "
            f"line includes {', '.join(p['name'] for p in u['products'][:6])} "
            f"and others."
        )
    for p in u["products"]:
        for variant in range(1 + max(0, augment)):
            attrs = list(_ATTRS)
            r.shuffle(attrs)
            sents = [f"The {p['name']} is a product of {u['company']}."]
            for a in attrs:
                tmpl = a["facts"][variant % len(a["facts"])] if variant else r.choice(a["facts"])
                sents.append(tmpl.format(p=p["name"], v=p[a["key"]]))
            docs.append(" ".join(sents))
    return docs


def render_train_qa(u: dict, seed: int = 0) -> list[str]:
    """Synthetic Q/A text included in TRAINING (teaches the QA-extraction form).

    Each block is `Q: ...\\nA: ...` — the format the model needs to have seen to
    answer questions, not just to have read the facts.
    """
    r = random.Random(seed + 202)
    blocks = []
    for p in u["products"]:
        for a in _ATTRS:
            q = a["train_q"].format(p=p["name"])
            v = p[a["key"]]
            ans = f"{v} {a['unit']}".strip()
            blocks.append(f"Q: {q}\nA: The answer is {ans}.")
    cf = [("Who is the CEO of {c}?", u["ceo"]),
          ("In what year was {c} founded?", str(u["founded"])),
          ("Where is {c} headquartered?", u["hq"])]
    for qt, ans in cf:
        blocks.append(f"Q: {qt.format(c=u['company'])}\nA: The answer is {ans}.")
    r.shuffle(blocks)
    return blocks


def build_heldout_qa(u: dict) -> list[dict]:
    """The probe set: held-out question phrasings + gold answers.

    The *facts* are in training (via docs + train-QA); only these specific
    question phrasings are held out. A model must therefore generalize the
    knowledge to score, not pattern-match a memorized question.
    """
    probes = []
    for p in u["products"]:
        for a in _ATTRS:
            probes.append({
                "question": a["held_q"].format(p=p["name"]),
                "answer": str(p[a["key"]]),
                "subject": p["name"], "attr": a["key"],
            })
    probes += [
        {"question": f"Who runs {u['company']} as chief executive?",
         "answer": u["ceo"], "subject": u["company"], "attr": "ceo"},
        {"question": f"What year was {u['company']} established?",
         "answer": str(u["founded"]), "subject": u["company"], "attr": "founded"},
        {"question": f"In which town does {u['company']} have its head office?",
         "answer": u["hq"], "subject": u["company"], "attr": "hq"},
    ]
    return probes


def assert_no_leak(train_texts: list[str], heldout: list[dict]) -> None:
    """Guarantee no held-out QUESTION string appears verbatim in training.

    (The answers DO appear in training — that's the knowledge. What must not
    leak is the question phrasing, or the probe would test memorization.)
    """
    blob = "\n".join(train_texts)
    leaked = [h["question"] for h in heldout if h["question"] in blob]
    if leaked:
        raise AssertionError(
            f"{len(leaked)} held-out question(s) leaked into training, e.g. "
            f"{leaked[0]!r}. Held-out and train question templates must differ."
        )


# =============================================================================
# Replay corpus (generic text for the anti-forgetting mix)
# =============================================================================

_REPLAY_SUBJECTS = ["The river", "A theorem", "Her argument", "The committee",
                    "Sunlight", "The old bridge", "An algorithm", "The harvest",
                    "His proposal", "The orchestra", "A glacier", "The market"]
_REPLAY_VERBS = ["shaped", "explained", "delayed", "revealed", "balanced",
                 "questioned", "preserved", "accelerated", "softened", "measured"]
_REPLAY_TAILS = ["the outcome in ways no one predicted.",
                 "a tension between speed and accuracy.",
                 "what the early reports had missed entirely.",
                 "the difference between theory and practice.",
                 "a pattern that held across every region.",
                 "the cost of ignoring the obvious."]


def build_replay(seed: int = 0, n_docs: int = 400) -> list[str]:
    """A generic-prose corpus used as the replay mix in the $0 demo.

    This is a *stand-in*. For a real run you point `data.replay_prefix` at
    Module 12's FineWeb-Edu corpus — genuine general data is what actually
    preserves general capability. This synthetic prose only keeps the offline
    demo self-contained.
    """
    r = random.Random(seed + 303)
    docs = []
    for _ in range(n_docs):
        sents = []
        for _ in range(r.randrange(3, 7)):
            sents.append(f"{r.choice(_REPLAY_SUBJECTS)} {r.choice(_REPLAY_VERBS)} "
                         f"{r.choice(_REPLAY_TAILS)}")
        docs.append(" ".join(sents))
    return docs


# =============================================================================
# Tokenizers: real (default) + a tiny offline word-level fallback
# =============================================================================

_WORD_RE = re.compile(r"\w+|[^\w\s]")


class WordTokenizer:
    """A tiny deterministic word/punct tokenizer for the offline `$0` path.

    Not a real BPE — just enough to turn the generated text into integer ids so
    a tiny model can train on it with no network. id 0 is reserved as EOS.
    """

    def __init__(self, vocab: dict[str, int], eos_id: int = 0):
        self.vocab = vocab
        self.inv = {i: w for w, i in vocab.items()}
        self.eos_id = eos_id
        self.vocab_size = len(vocab) + 1   # +1 for EOS at id 0

    @classmethod
    def build(cls, texts: list[str]) -> "WordTokenizer":
        words = sorted({w for t in texts for w in _WORD_RE.findall(t)})
        vocab = {w: i + 1 for i, w in enumerate(words)}   # ids start at 1; 0 = EOS
        return cls(vocab, eos_id=0)

    def encode(self, text: str) -> list[int]:
        return [self.vocab[w] for w in _WORD_RE.findall(text) if w in self.vocab]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inv.get(int(i), "") for i in ids if int(i) != self.eos_id)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab, "eos_id": self.eos_id}))

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        d = json.loads(Path(path).read_text())
        return cls({k: int(v) for k, v in d["vocab"].items()}, eos_id=int(d["eos_id"]))


class _HFTokAdapter:
    """Unifies a HuggingFace tokenizer to the (encode/decode/eos_id/vocab_size)
    surface the rest of this file uses."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.eos_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
        self.vocab_size = len(self.tok)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


def load_tokenizer(name: str):
    """Real-tokenizer path (HF). Used for the actual Qwen3-0.6B-Base corpus."""
    return _HFTokAdapter(name)


# =============================================================================
# Tokenize + write an indexed corpus
# =============================================================================

def write_indexed(texts: list[str], tok, prefix: str, *, source: str) -> dict:
    """Tokenize `texts`, append EOS per doc, write a Module-12 indexed corpus."""
    dtype = best_dtype_for_vocab(tok.vocab_size)
    b = IndexedDatasetBuilder(prefix, dtype=dtype)
    for t in texts:
        ids = tok.encode(t)
        ids.append(tok.eos_id)
        b.add_document(ids)
    return b.finalize(vocab_size=tok.vocab_size, eos_id=tok.eos_id,
                      extra={"source": source})


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/corpus", help="output dir prefix")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base",
                   help="HF tokenizer to tokenize with (must match the train model)")
    p.add_argument("--fake-tokenizer", action="store_true",
                   help="use a tiny offline word tokenizer instead (for $0 smoke / notebook)")
    p.add_argument("--n-entities", type=int, default=64, help="number of fictional products")
    p.add_argument("--augment", type=int, default=4, help="paraphrases per document")
    p.add_argument("--replay-docs", type=int, default=400, help="synthetic replay documents")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Generate text (deterministic, no tokenizer needed).
    u = build_universe(seed=args.seed, n_entities=args.n_entities)
    docs = render_documents(u, seed=args.seed, augment=args.augment)
    train_qa = render_train_qa(u, seed=args.seed)
    heldout = build_heldout_qa(u)
    replay = build_replay(seed=args.seed, n_docs=args.replay_docs)

    domain_texts = docs + train_qa
    assert_no_leak(domain_texts, heldout)   # the holdout guarantee

    # 2. Write the reference artifacts (text — for the notebook + inspection).
    (out / "facts.jsonl").write_text(
        "\n".join(json.dumps(f) for f in universe_facts(u)))
    (out / "domain_docs.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in domain_texts))
    (out / "qa_heldout.jsonl").write_text(
        "\n".join(json.dumps(h) for h in heldout))

    # 3. Tokenize + write the two indexed corpora.
    if args.fake_tokenizer:
        # vocab includes held-out probe text so the offline tokenizer can
        # encode the probes — held-out text is NOT added to TRAINING (no leak).
        probe_text = [h["question"] for h in heldout] + [h["answer"] for h in heldout]
        tok = WordTokenizer.build(domain_texts + replay + probe_text)
        tok.save(str(out / "fake_tokenizer.json"))
        print(f"[make_corpus] fake word tokenizer: vocab_size={tok.vocab_size}")
    else:
        tok = load_tokenizer(args.tokenizer)
        print(f"[make_corpus] tokenizer {args.tokenizer!r}: vocab_size={tok.vocab_size}")

    dm = write_indexed(domain_texts, tok, str(out / "domain"), source="fictional-company")
    rp = write_indexed(replay, tok, str(out / "replay"), source="synthetic-replay")

    print(f"[make_corpus] company={u['company']!r}  products={len(u['products'])}")
    print(f"[make_corpus] domain : {len(domain_texts):>5} docs, {dm['total_tokens']:>8,} tokens "
          f"-> {out/'domain'}.bin")
    print(f"[make_corpus] replay : {len(replay):>5} docs, {rp['total_tokens']:>8,} tokens "
          f"-> {out/'replay'}.bin")
    print(f"[make_corpus] probes : {len(heldout):>5} held-out QA -> {out/'qa_heldout.jsonl'}")
    print(f"[make_corpus] no held-out question leaks into training ✓")


if __name__ == "__main__":
    main()
