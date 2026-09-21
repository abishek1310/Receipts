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

**Catch rate is not the same as "the citation supports the claim."** Reading
real output from the deployed app turned up three ways a citation stays
perfectly valid while the sentence drifts away from it — sentiment inversion,
scope-and-certainty upgrade, and relevance drift. All three pass the validator,
none are fixable with a stricter regex, and fixing them properly would require
a model to judge what counts as support — which is the thing this project
exists to replace. They are written up with verified examples in
`eval/results/report.md`.

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

- **400 signals**, one narrow category (skincare), IDs frozen in `data/id_ledger.json`
- **224 Reddit comments** from r/SkincareAddiction, via the public
  [`HuggingFaceGECLM/REDDIT_comments`](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments)
  dump. Permalinks are reconstructed from `link_id` + comment `id`, so every URL
  resolves to the original comment.
- **176 Amazon reviews** from
  [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
  (All_Beauty), filtered to skincare products by their real titles.
- Spread evenly across 10 themes; mostly 2021–2023.

**Every URL resolves, and that took work.** §5.1 says a judge will click one, so
we checked. The Amazon dump is a 2023 snapshot and products get delisted: **31%
of the ASINs the first build selected already 404'd.** All 152 ASINs now in the
corpus have been loaded in a real browser and confirmed live; 69 dead ones are
denylisted in `data/dead_asins.json`. The check cannot be done with curl —
Amazon serves an identical 3,790-byte bot-block page for live and dead ASINs
alike, so a shell check reports everything healthy. The procedure, and when to
re-run it, is in `scripts/check_amazon_urls.md`.

Reddit permalinks need no such check: comment URLs are permanent, and a deleted
comment still resolves to its thread.

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

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (public, or private with Streamlit Cloud granted
   access).
2. At [share.streamlit.io](https://share.streamlit.io), **New app** -> pick the
   repo, branch `main`, main file `app.py`.
3. **Advanced settings -> Secrets**: paste the contents of
   `.streamlit/secrets.toml.example` with a real key filled in. Do not commit the
   real file — `.streamlit/secrets.toml` is gitignored.
4. Deploy.

Nothing else is needed. `data/signals.jsonl` is committed, so there is no build
step and no download at boot; Cloud installs `requirements.txt` only, which is
why the ingest-only dependency lives in `requirements-dev.txt`.

**Watch the quota.** Gemini's free tier allows **20 requests per day, per
model**. One chat turn is one request (plus one per citation retry). Rehearse on
one model and switch `GEMINI_MODEL` to a different one for the live run —
`gemini-3.6-flash` and `gemini-3.7-flash` have separate daily buckets.

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

## Three deliberate departures from the spec

Both are noted at the top of the files they affect.

1. **The Anthropic call uses the official SDK, not LangChain.** §3 says "via
   LangChain", but the requirement in the same row — one adapter, swappable — is
   what `src/llm.py` provides. Going direct keeps us off a wrapper that lags new
   model parameters. LangGraph, the part of the stack §3 actually depends on, is
   unchanged.

2. **Gemini is wired as a third provider.** §3 names Claude or OpenAI. The team
   has credentials for neither, and §3 *also* requires a live Streamlit Community
   Cloud URL — which rules out a locally hosted model, since Streamlit Cloud
   cannot reach one. Gemini's free tier satisfies both constraints. The part of
   §3 that carries the architecture — one adapter, swappable — is what made this
   cheap: only `src/llm.py` and `src/config.py` changed. `validation.py`,
   `generation.py` and `graph.py` are untouched, which is the whole argument for
   putting the provider behind an adapter in the first place.

3. **`temperature` is not sent to current Claude models.** §7 asks for ~0.7.
   Sampling parameters were removed from Claude Opus 5, Sonnet 5 and the 4.7/4.8
   family and now return a 400; `output_config.effort` replaced them. The adapter
   still sends `temperature` to models that accept it. §7's actual point is
   preserved and is load-bearing here: safety comes from the validator, not from
   clamping the sampler.
