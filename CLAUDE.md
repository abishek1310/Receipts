# Receipts — Implementation Spec

Marketing copy generation where **the model never writes its own citations**.

This file is the working context for Claude Code. Read it fully before writing code.
Built for the CREWASIS External Hackathon 1.0 (Sept 8–18, 2026). Team of 4.

---

## 1. The one invariant

> **No generated sentence reaches the user unless every citation in it resolves to a
> signal that was actually retrieved for that generation.**

Everything else in this project is negotiable. This is not.

If you find yourself writing a prompt that asks the model to "include source IDs" and
then trusting what comes back, stop — that is the failure mode this project exists to
prevent. Citations are assigned and verified in Python. The LLM's tags are treated as
*claims to be checked*, never as facts.

---

## 2. What the system does

1. Takes a market insight (e.g. "unmet need: post-workout facial redness, Gen Z athletes").
2. Retrieves the consumer signals underneath it — real Reddit/Amazon comments.
3. Generates marketing copy (TikTok hooks, ad copy, positioning lines).
4. **Validates** every citation tag against the retrieved set; rejects and regenerates
   any block that fails.
5. Lets the user revise conversationally ("softer", "more clinical", "focus on price"),
   with revisions **bound to the same evidence set**.

---

## 3. Stack

| Concern | Choice | Notes |
|---|---|---|
| Orchestration | **LangGraph** | Not CrewAI. Team already knows LangGraph. |
| LLM | Anthropic Claude (or OpenAI) via LangChain | Keep behind one adapter, swappable. |
| Retrieval | keyword/theme filter first; `sentence-transformers` + cosine only if time | ~400 signals — brute force is fine. |
| Storage | JSON/JSONL on disk | No database. Do not add one. |
| UI | **Streamlit** | `st.chat_message`, `st.chat_input`. |
| Deploy | Streamlit Community Cloud | Live URL required by demo day. |
| Config | `pydantic-settings` + `.env` | Never commit keys. |

Python 3.11+. Dependencies pinned in `requirements.txt`.

---

## 4. Repo structure

```
receipts/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── app.py                      # Streamlit entrypoint
├── data/
│   ├── raw/                    # scraped/downloaded source data
│   ├── signals.jsonl           # canonical corpus (the only file the app reads)
│   └── insights.json           # 3–5 hand-written insight objects
├── src/
│   ├── __init__.py
│   ├── schema.py               # Signal, Insight, CopyBlock, GenerationResult
│   ├── corpus.py               # load, index, lookup by ID
│   ├── retrieval.py            # insight/query -> List[Signal]
│   ├── generation.py           # LLM call, prompt construction
│   ├── validation.py           # THE CORE — citation parsing + verification
│   ├── graph.py                # LangGraph state machine
│   ├── llm.py                  # provider adapter
│   └── config.py
├── eval/
│   ├── build_eval_set.py       # inject unsupported claims
│   ├── run_eval.py             # catch rate / false-positive rate
│   └── results/                # metrics + logged natural failures
└── tests/
    └── test_validation.py      # must be thorough — this is the core
```

---

## 5. Data

### 5.1 Signal schema (`src/schema.py`)

```python
from pydantic import BaseModel
from datetime import date

class Signal(BaseModel):
    id: str            # stable, e.g. "sig_0183" — NEVER regenerate these
    text: str          # verbatim consumer comment, unedited
    source: str        # "reddit" | "amazon" | "tiktok"
    source_detail: str # "r/SkincareAddiction" | product name
    url: str           # clickable, must actually resolve
    timestamp: date
    theme: str         # coarse bucket: "redness", "price", "texture", ...
```

Rules:
- IDs are **assigned once** at ingest and frozen. Nothing downstream may mint an ID.
- `text` is verbatim. No cleaning beyond whitespace normalisation and truncation
  at 400 chars.
- `url` must resolve — a judge will click one.

### 5.2 Corpus

- **Real public data only.** No synthetic comments. CREWASIS issues no API keys, so
  the realism of the evidence layer is the credibility of the demo.
- Sources: Reddit (PRAW or a public dump), a HuggingFace Amazon reviews dataset.
- **One narrow category** — skincare. Do not broaden.
- Target **300–500 signals**. Small enough to eyeball, large enough for real retrieval.
- Output `data/signals.jsonl`, one Signal per line.

### 5.3 Insight schema

```python
class Insight(BaseModel):
    id: str
    statement: str      # "Gen Z athletes have an unmet need around post-workout redness"
    category: str       # "athletic skincare"
    audience: str       # "Gen Z, 18-26, trains 4+/week"
    themes: list[str]   # themes to retrieve against
    momentum: str       # "rising" — narrative only, do not compute
```

Hand-write 3–5 of these in `data/insights.json`. They stand in for HAZRA output.
Do **not** build ranking or trend detection — that is CREWASIS's product, not ours.

---

## 6. The validation core (`src/validation.py`)

This module is the project. Write it first, test it hardest.

### 6.1 Citation format

The model is instructed to tag every claim-bearing sentence:

```
Your gym glow shouldn't last three hours. [sig_0183, sig_0241, sig_0319]
```

Parse with a strict regex. Anything that doesn't match the exact shape is a
parse failure, which is a validation failure.

### 6.2 Algorithm

```python
def validate_block(block: CopyBlock, evidence_set: set[str]) -> ValidationResult:
    """
    Returns PASS or FAIL with a machine-readable reason.

    FAIL conditions, checked in order:
      1. UNPARSEABLE      — no citation tag on a claim-bearing sentence
      2. UNKNOWN_ID       — a tag not present in the corpus at all (pure hallucination)
      3. OUT_OF_SET       — a real corpus ID that was NOT in this generation's
                            evidence_set (the model reached outside its context)
      4. EMPTY_CITATION   — tag present but contains no IDs
    """
```

`OUT_OF_SET` is the subtle one and must not be skipped. A model can emit a real ID it
saw in an earlier turn. That still fails: the claim was not grounded in *this*
retrieval.

### 6.3 On failure

- Return the failing block, the reason, and the offending IDs to the graph.
- Regenerate **only the failing block**, not the whole response.
- Max 2 retries. On the third failure, surface the block to the user marked
  `UNSUPPORTED` rather than silently dropping it — visible refusal beats silence,
  and it is a good demo moment.
- **Log every rejection** to `eval/results/rejections.jsonl`. These logs are the
  source of the live demo's rejection example (see §11).

### 6.4 Tests (`tests/test_validation.py`)

Cover at minimum: clean pass; hallucinated ID; real-but-out-of-set ID; missing tag;
malformed tag; empty tag; multiple blocks with one failing; retry exhaustion.
No LLM calls in these tests — construct `CopyBlock` objects directly.

---

## 7. Generation (`src/generation.py`)

Prompt construction, strictly in this order:

1. Retrieve signals in Python.
2. Serialise them into the prompt as an explicit numbered evidence block, each with
   its real ID and verbatim text.
3. Instruct: *use only these signals; cite by ID; if the evidence does not support a
   claim, do not make it.*
4. Call the LLM.
5. Hand the raw output to `validation.py`. **Never render unvalidated output.**

The prompt must never ask the model to invent, infer, or recall IDs. The IDs are
given to it; its only job is to attribute correctly among them.

Keep `temperature` around 0.7 for copy quality — the validator is what provides
safety, not low temperature.

---

## 8. Retrieval (`src/retrieval.py`)

```python
def retrieve(insight: Insight, user_query: str | None, k: int = 40) -> list[Signal]
```

v1: filter by `theme` overlap, then keyword-match the query, return top-k by recency.
v2 (only if ahead of schedule): embed signals once at startup, cosine similarity.

Retrieval returns the **evidence set** for this turn. Store it in graph state — the
revision loop reuses it verbatim.

---

## 9. LangGraph (`src/graph.py`)

State:

```python
class GraphState(TypedDict):
    insight: Insight
    messages: list              # conversation history
    evidence_set: list[Signal]  # frozen for the turn; reused on revision
    blocks: list[CopyBlock]
    failures: list[ValidationResult]
    attempt: int
```

Nodes: `retrieve` → `generate` → `validate` → conditional edge → (`regenerate` loop,
max 2) → `respond`.

**Revision path:** when the user asks for a tone/angle change, skip `retrieve`.
Reuse `evidence_set` from state. This is the "evidence-preserving revision" feature
and it is a scored differentiator — do not let a rewrite trigger a fresh retrieval
unless the user explicitly asks for a different angle (e.g. "what about price?"),
in which case retrieve fresh and say so in the UI.

---

## 10. UI (`app.py`)

- `st.chat_input` / `st.chat_message`, standard Streamlit chrome. **No custom CSS,
  no dashboard styling.** Nobody scores pixel fidelity in a 4-minute demo.
- Each copy block renders as: the copy line, then `📎 47 supporting signals` as an
  `st.expander`.
- Expanded: each signal as verbatim text, source, date, and a clickable link.
- Rejected blocks render with a visible ❌ and the reason. Do not hide failures —
  they are the differentiator.
- Sidebar: insight selector, signal count, and a running tally of
  `blocks generated / blocks rejected`.
- Must run on Streamlit Community Cloud. Keep secrets in `st.secrets`.

---

## 11. Evaluation (`eval/`)

Run on **Sept 14**. Produces the single most differentiating slide in the deck.

`build_eval_set.py` — generate 10 briefs, inject ~30 unsupported claims across them
(mix of hallucinated IDs and out-of-set real IDs).

`run_eval.py` — report:
- **Catch rate**: injected bad citations correctly rejected / total injected.
- **False-positive rate**: valid claims wrongly rejected / total valid claims.
- One honest failure case, written up in plain language.

Also: scan `rejections.jsonl` for **naturally occurring** hallucinations and save the
best one. The live demo's rejection moment must use a real failure, not a prompt
engineered to fail — a judge can tell, and being caught staging it is worse than not
demoing it.

---

## 12. Build order

Do not reorder. Each step is demoable on its own.

1. `schema.py` + `corpus.py` + 400 real signals in `signals.jsonl`
2. `validation.py` + full test suite — **before any LLM code**
3. `retrieval.py`
4. `generation.py` → first end-to-end run in a terminal
5. `graph.py` — retry loop
6. Multi-turn revision with frozen evidence set
7. `app.py` — Streamlit chat + citation expanders
8. Deploy to Streamlit Community Cloud
9. `eval/` — metrics + logged natural failures
10. Freeze. Bug fixes only.

Milestone that matters most: **step 4 working by end of Sept 10.**

---

## 13. Non-goals — do not build these

- ❌ Trend detection, ranking, or momentum scoring (that is HAZRA — we consume, not rebuild)
- ❌ Image generation, media plans, budget allocation (fabricated numbers, zero evidence)
- ❌ Multi-agent swarms (commoditised; one graph with a validator is stronger)
- ❌ A vector database (400 records — in-memory is correct)
- ❌ Custom dashboard styling
- ❌ Auth, user accounts, persistence
- ❌ Live scraping at request time

Every hour spent here is an hour not spent on citation integrity.

---

## 14. Conventions

- Type hints everywhere; pydantic models for all data crossing a module boundary.
- No bare `except`. Validation failures are values, not exceptions.
- No LLM calls in tests.
- Never log API keys. `.env` stays gitignored; ship `.env.example`.
- Commit messages: `feat:`, `fix:`, `test:`, `docs:`.
- If a change would weaken the §1 invariant, don't make it — raise it instead.

---

## 15. Framing (for README and any generated docs)

- **What it is:** a conversational copy agent where the model never writes its own
  citations.
- **Division of labour:** HAZRA finds the opportunity. Winston explains the
  opportunity. Receipts makes the resulting marketing claim defensible.
- **Never claim** CREWASIS lacks citations. The defensible phrasing is: *we found no
  public documentation of claim-level provenance enforcement, where generated
  marketing claims are programmatically bound to retrieved source signals and
  unsupported citations are rejected.*
- **Data caveat, stated once, without apology:** built to consume CREWASIS signals,
  demonstrated on a representative public dataset.
