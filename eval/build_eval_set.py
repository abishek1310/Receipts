"""Build the evaluation set (CLAUDE.md §11).

Ten briefs, with ~30 unsupported claims injected across them — a mix of
hallucinated IDs and real-but-out-of-set IDs, which are the two failures §6.2
cares most about.

Two design choices, both so the number means something:

* **The copy is templated, not model-written.** The metric being measured is the
  validator's catch rate, not any model's honesty. Templated copy makes the run
  free, instant and byte-identical every time, so a regression in
  `validation.py` shows up as a changed number rather than as noise. (The
  *naturally* occurring failures §11 also asks for come from
  `rejections.jsonl`, which is real model output — see `run_eval.py`.)
* **The signals are real.** Evidence sets come from the actual corpus via the
  real `retrieve()`, so out-of-set injections are genuine corpus IDs that this
  brief's retrieval genuinely did not return.

The set deliberately includes *hard valid* blocks — calls to action, decimals,
multi-sentence copy, ten-ID citations. Without them a false-positive rate of
zero would only prove the eval was easy.

    python eval/build_eval_set.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import load_corpus, load_insights  # noqa: E402
from src.retrieval import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "eval_set.json"

SEED = 20260914  # the date §11 schedules the eval run for
N_BRIEFS = 10
N_INJECTIONS = 30

# Claim templates by theme. Bland on purpose — the validator does not read for
# quality, and interesting copy here would only make the fixtures harder to diff.
CLAIMS: dict[str, list[str]] = {
    "redness": [
        "The flush does not fade when the session ends",
        "Redness is the part people actually notice",
        "Calm skin is the result being asked for, not clear skin",
    ],
    "sweat": [
        "Sweat is what breaks the routine, not laziness",
        "The gym undoes the morning routine by lunchtime",
        "Training four times a week changes what skin needs",
    ],
    "sun": [
        "Reapplying is the step that never happens",
        "A white cast is a reason to skip protection entirely",
        "Sunscreen that stings the eyes gets left at home",
    ],
    "irritation": [
        "The barrier gives out before the actives do",
        "Stinging is read as the product working, and it is not",
        "Sensitive skin has learned to expect a trade-off",
    ],
    "acne": [
        "Breakouts follow the routine, not the diet",
        "Spot treatments are bought in hope, not confidence",
        "Clogged pores outlast every three-step system",
    ],
    "hydration": [
        "Dry and tight is the default state by evening",
        "Flaking makes every other step look worse",
        "Hydration is judged by how skin feels at hour eight",
    ],
    "texture": [
        "Anything tacky gets abandoned in week two",
        "Pilling under sunscreen ends the relationship",
        "Lightweight is the only texture that survives a routine",
    ],
    "fragrance": [
        "Scent reads as something to hide behind",
        "Fragrance-free is checked before the ingredient list",
        "A strong smell is a reason to return it",
    ],
    "price": [
        "The expensive option is assumed to be the same formula",
        "Dupes are searched for before the original is bought",
        "Value is judged per use, not per bottle",
    ],
    "routine": [
        "Every step added is a step eventually skipped",
        "Ten products is an aspiration, not a routine",
        "Simplicity is what actually gets repeated",
    ],
}

CTA = ["Shop now.", "Link in bio.", "Learn more.", "Try it today."]


def _hallucinated_id(corpus_ids: set[str], rng: random.Random) -> str:
    """An ID that looks right and does not exist — the pure-fabrication case."""
    while True:
        candidate = f"sig_{rng.randint(1, 9999):04d}"
        if candidate not in corpus_ids:
            return candidate


def _out_of_set_id(corpus_ids: set[str], evidence: set[str], rng: random.Random) -> str:
    """A real corpus ID this brief's retrieval did not return — the subtle case."""
    pool = sorted(corpus_ids - evidence)
    return rng.choice(pool)


def main() -> int:
    rng = random.Random(SEED)
    corpus = load_corpus()
    insights = load_insights()
    corpus_ids = corpus.ids

    briefs = []
    for i in range(N_BRIEFS):
        insight = insights[i % len(insights)]
        # Vary the query so the ten briefs are not five duplicated pairs.
        query = None if i < len(insights) else f"focus on {insight.themes[-1]}"
        evidence = retrieve(insight, query, k=40, corpus=corpus)
        ev_ids = [s.id for s in evidence]

        blocks = []
        # --- ordinary valid blocks -------------------------------------------------
        for n in range(6):
            theme = evidence[n % len(evidence)].theme
            claim = CLAIMS[theme][n % len(CLAIMS[theme])]
            cites = rng.sample(ev_ids, rng.randint(1, 3))
            blocks.append(
                {
                    "id": f"b{n}",
                    "kind": ["hook", "ad_copy", "positioning"][n % 3],
                    "text": f"{claim}. [{', '.join(cites)}]",
                    "label": "valid",
                    "injection": None,
                }
            )

        # --- hard valid blocks: the cases that produce honest false positives ------
        theme = evidence[0].theme
        c1, c2 = CLAIMS[theme][0], CLAIMS[theme][1]
        hard = [
            # A call to action needs no citation.
            (f"{c1}. [{ev_ids[0]}] {rng.choice(CTA)}", "cta"),
            # A decimal must not be read as a sentence boundary.
            (f"{c2} in 2.5 hours, not eight. [{ev_ids[1]}]", "decimal"),
            # Two cited sentences in one block.
            (f"{c1}. [{ev_ids[2]}] {c2}. [{ev_ids[3]}, {ev_ids[4]}]", "multi_sentence"),
            # A wide citation — ten IDs in one tag.
            (f"{c1}. [{', '.join(ev_ids[:10])}]", "wide_citation"),
            # No terminal punctuation, as hooks often have.
            (f"{c2} [{ev_ids[5]}]", "no_terminator"),
            # --- the cases that genuinely stress the sentence splitter -------------
            # An abbreviation mid-sentence.
            (f"Dr. Idriss says {c1.lower()}. [{ev_ids[6]}]", "abbreviation"),
            # "e.g." inside a clause.
            (f"{c1}, e.g. after hot yoga. [{ev_ids[7]}]", "inline_eg"),
            # A quoted sentence carrying its own terminal punctuation.
            (f'One reviewer put it plainly: "It stings." {c2}. [{ev_ids[8]}]', "quotation"),
            # An ellipsis used rhetorically, mid-thought.
            (f"{c1}... and nobody warns you. [{ev_ids[9]}]", "inline_ellipsis"),
            # A product number that ends in a digit-dot.
            (f"SPF 50. {c2}. [{ev_ids[10]}]", "spf_number"),
        ]
        for n, (text, variant) in enumerate(hard):
            blocks.append(
                {
                    "id": f"h{n}",
                    "kind": "hook",
                    "text": text,
                    "label": "valid",
                    "injection": None,
                    "variant": variant,
                }
            )

        briefs.append(
            {
                "brief_id": f"brief_{i:02d}",
                "insight_id": insight.id,
                "query": query,
                "evidence_ids": ev_ids,
                "blocks": blocks,
            }
        )

    # --- inject the unsupported claims --------------------------------------------
    # Spread evenly across briefs, alternating the two failure modes.
    slots = [(b, n) for b in range(N_BRIEFS) for n in range(6)]
    rng.shuffle(slots)
    for k, (bi, ni) in enumerate(slots[:N_INJECTIONS]):
        brief = briefs[bi]
        block = brief["blocks"][ni]
        ev = set(brief["evidence_ids"])
        if k % 2 == 0:
            bad = _hallucinated_id(corpus_ids, rng)
            kind, expected = "hallucinated_id", "UNKNOWN_ID"
        else:
            bad = _out_of_set_id(corpus_ids, ev, rng)
            kind, expected = "out_of_set_id", "OUT_OF_SET"

        # Replace the first cited ID with the bad one, leaving the rest intact —
        # a wholly fabricated tag is easier to catch than one bad ID in a good tag.
        head, _, tail = block["text"].partition("[")
        cites = tail.rstrip("]").split(", ")
        cites[0] = bad
        block["text"] = f"{head}[{', '.join(cites)}]"
        block["label"] = "injected"
        block["injection"] = {"type": kind, "bad_id": bad, "expected_reason": expected}

    payload = {
        "seed": SEED,
        "corpus_size": len(corpus),
        "n_briefs": len(briefs),
        "n_injected": N_INJECTIONS,
        "briefs": briefs,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_blocks = sum(len(b["blocks"]) for b in briefs)
    valid = sum(
        1 for b in briefs for blk in b["blocks"] if blk["label"] == "valid"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  briefs        {len(briefs)}")
    print(f"  blocks        {total_blocks}  ({valid} valid, {N_INJECTIONS} injected)")
    print(f"  injections    {N_INJECTIONS // 2} hallucinated, {N_INJECTIONS - N_INJECTIONS // 2} out-of-set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
