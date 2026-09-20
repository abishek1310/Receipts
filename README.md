# 📎 Receipts

**Marketing copy generation where the model never writes its own citations.**

Built for the CREWASIS External Hackathon 1.0.

---

## The idea

Ask a language model to cite its sources and it will happily produce a list of
source IDs. Some of them will be real. Some will be real but irrelevant. Some
will not exist. The output looks identical in all three cases, which is the
problem — a fabricated citation is *more* persuasive than no citation at all.

Receipts refuses to trust the model on this point. Citations are assigned and
verified in Python. The model's tags are treated as **claims to be checked**,
never as facts.

> **The invariant:** no generated sentence reaches the user unless every citation
> in it resolves to a signal that was actually retrieved for that generation.

Where this sits: HAZRA finds the opportunity. Winston explains the opportunity.
Receipts makes the resulting marketing claim defensible.

We found no public documentation of claim-level provenance enforcement — where
generated marketing claims are programmatically bound to retrieved source signals
and unsupported citations are rejected.

---

## What it does

1. Takes a market insight ("Gen Z athletes have an unmet need around post-workout
   facial redness").
2. Retrieves the consumer signals underneath it — real Reddit comments and Amazon
   reviews.
3. Generates TikTok hooks, ad copy and positioning lines.
4. **Validates** every citation tag against the retrieved set, and regenerates
   anything that fails.
5. Lets you revise conversationally — "softer", "more clinical", "focus on price"
   — with revisions bound to the same evidence set.

---

## Results

From `eval/results/report.md` (regenerate with `python eval/run_eval.py`):

| Metric | Value |
|---|---|
| **Catch rate** | **100%** — 30/30 injected bad citations rejected |
| **False-positive rate** | **7.7%** — 10/130 valid claims wrongly rejected |
| Diagnosis accuracy | 100% — caught *and* named the right failure mode |

The false positives are all one pattern, and we left it in rather than tuning it
away: a leading fragment like `SPF 50.` is treated as an uncited claim. The
sentence splitter errs toward splitting, so ambiguity costs a rewrite instead of
letting an uncited claim through. That is the trade we chose.

---

## The four ways a citation fails

Checked in this order. The third is the one that matters.

| Reason | Meaning |
|---|---|
| `UNPARSEABLE` | A claim-bearing sentence has no tag, or a tag in the wrong shape |
| `UNKNOWN_ID` | An ID that is not in the corpus at all — pure hallucination |
| `OUT_OF_SET` | A **real** corpus ID that was not retrieved for *this* generation |
| `EMPTY_CITATION` | A tag is present but carries no IDs |

`OUT_OF_SET` is the subtle one. A model can emit a perfectly real signal ID it saw
three turns ago. Every naive checker passes it — the ID resolves, the source is
genuine, the link works. It still fails here, because the claim was not grounded
in *this* retrieval. That distinction is the difference between a citation and a
coincidence.

A block that fails is regenerated on its own, up to twice. If it still fails, it
is shown to the user with a visible ❌ and the reason. Visible refusal beats
silence.

---

## The data

**Real public data only.** No synthetic comments — the realism of the evidence
layer is the credibility of the demo.

- **400 signals**, one narrow category (skincare), frozen IDs `sig_0001`–`sig_0400`
- **220 Reddit comments** from r/SkincareAddiction, via the public
  [`HuggingFaceGECLM/REDDIT_comments`](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments)
  dump. Permalinks are reconstructed from `link_id` + comment `id`, so every URL
  resolves to the original comment.
- **180 Amazon reviews** from
  [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
  (All_Beauty), filtered to skincare products by their real titles.
- Spread evenly across 10 themes; mostly 2021–2023.

Nothing is truncated. The spec permits cutting at 400 characters, but a comment
chopped mid-sentence reads like fabricated data to anyone who opens the expander,
so we select comments that are naturally short enough to quote whole.

*Built to consume CREWASIS signals, demonstrated on a representative public
dataset.*

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt                  # runtime + ingest + tests
cp .env.example .env                                 # then add your API key
```

The corpus is committed, so there is nothing to download:

```bash
streamlit run app.py
```

Terminal run of the whole chain:

```bash
python scripts/demo_run.py --insight ins_01 --revise "make it softer"
```

Tests (no LLM calls, no network):

```bash
python -m pytest
```

Evaluation:

```bash
python eval/build_eval_set.py && python eval/run_eval.py
```

Rebuilding the corpus from source (not usually needed — IDs are frozen in
`data/id_ledger.json` and reused):

```bash
python scripts/build_corpus.py --reddit-shard data/raw/<shard>.parquet
```

---

## Layout

```
src/validation.py   THE CORE — citation grammar, the four failure modes
src/graph.py        LangGraph: retrieve → generate → validate → regenerate → respond
src/generation.py   Prompt construction; serialises the evidence block
src/retrieval.py    Insight + query → evidence set
src/corpus.py       Loads signals.jsonl; the authority on which IDs are real
src/ingest.py       Corpus construction from the two public sources
src/llm.py          Provider adapter (Anthropic / OpenAI / scripted stub)
tests/              73 tests, no LLM calls
eval/               Catch rate, false-positive rate, honest failure case
```

---

## Two deliberate departures from the spec

Both are noted at the top of the files they affect.

1. **The Anthropic call uses the official SDK, not LangChain.** §3 says "via
   LangChain", but the requirement in the same row — one adapter, swappable — is
   what `src/llm.py` provides. Going direct keeps us off a wrapper that lags new
   model parameters. LangGraph, the part of the stack §3 actually depends on, is
   unchanged.

2. **`temperature` is not sent to current Claude models.** §7 asks for ~0.7.
   Sampling parameters were removed from Claude Opus 5, Sonnet 5 and the 4.7/4.8
   family and now return a 400; `output_config.effort` replaced them. The adapter
   still sends `temperature` to models that accept it. §7's actual point is
   preserved and is load-bearing here: safety comes from the validator, not from
   clamping the sampler.
