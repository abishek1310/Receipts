"""Run the evaluation (CLAUDE.md §11).

Reports:

* **Catch rate** — injected bad citations correctly rejected / total injected.
* **False-positive rate** — valid claims wrongly rejected / total valid claims.
* **Diagnosis accuracy** — of the caught ones, how many got the *right* reason.
  Catching a hallucinated ID and calling it OUT_OF_SET is still a catch, but it
  is the wrong story to tell a judge, so it is worth measuring separately.

It also scans `rejections.jsonl` for naturally occurring hallucinations and saves
the best one to `eval/results/natural_failure.json`. §11 is emphatic that the
live demo's rejection moment must use a real failure rather than a prompt
engineered to fail — this is where that example comes from.

    python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import load_corpus  # noqa: E402
from src.schema import CopyBlock  # noqa: E402
from src.validation import REJECTION_LOG, validate_block  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval" / "eval_set.json"
RESULTS = ROOT / "eval" / "results"

# Found by reading real output from the deployed app and opening every expander.
# These are not bugs to fix before demo day — they are the boundary of what §1
# claims, and stating the boundary is worth more than pretending it is not there.
# Each example was verified against the corpus; the IDs are frozen and stable.
ATTRIBUTION_LIMITS = """\
## Where valid attribution still goes wrong

The catch rate above measures one thing: **does every cited ID resolve to a
signal that was retrieved for this generation?** That is the §1 invariant and it
holds 100% of the time.

It is not the same as "the cited signal supports this sentence". Reading real
output from the deployed app and opening every expander turned up three ways a
citation stays perfectly valid while the claim drifts away from it. All three
pass the validator. None of them are fixable by a stricter regex, because none
of them are citation-format problems.

### 1. Sentiment inversion — the citation is verbatim, the stance is reversed

Generated copy:

> "Applying occlusive recovery creams to a flushed face is like spackling
> drywall over active inflammation." `[sig_0033]`

`sig_0033`, verbatim:

> "I found Avene cicalfate **helped** calm down the redness and inflammation,
> but it feels like you're spackling dry wall on your skin."

The phrase is genuinely theirs. The person is **recommending** the product — it
worked, it just felt thick. The copy uses their words to argue the opposite.

This is the most dangerous of the three precisely because it looks like the
*strongest* citation: a vivid phrase lifted word-for-word from a real comment.
A judge who opens the expander and reads past the first clause sees a satisfied
customer quoted as a complaint.

### 2. Scope and certainty upgrade — one hedged anecdote becomes a general law

Generated copy:

> "Leaving perspiration on the skin after exercise **can trigger** fungal
> infections like tinea versicolor." `[sig_0061]`

`sig_0061`, verbatim:

> "**Could be** a fungal infection like tinea versicolor. **I** get tinea **on
> my torso** when I sweat a lot."

One person, hedged, about their torso, becomes a general causal claim about
everyone's face. The clinical vocabulary is *not* invented — r/SkincareAddiction
users write "tinea versicolor" and "closed comedones" themselves — so a
banned-words list does not catch this. What changed is scope and confidence.

This gets worse under a "make it more medical" revision, which is why that is
the one direction we would not demo.

### 3. Relevance drift — on-theme, but not about this claim

A hook about sunscreen stinging the eyes cited `sig_0066`:

> "The sweat is no joke. The gym or any exercise is fantastic... I had acne just
> like yours"

Retrieved for the brief, genuinely about sweat and exercise, and says nothing
about sunscreen or eyes. Retrieval matched the theme; the sentence needed
something narrower.

### Why we did not "fix" these

Every fix requires judging whether a passage *supports* a claim, and that
judgement needs a language model. The moment a model decides what counts as
support, the guarantee becomes "a model thinks this is fine" — which is the
thing this project exists to replace. We would rather enforce one property
completely than four properties approximately.

What we ship instead: the evidence is one click away, verbatim and linked, so a
human can catch all three in seconds. That is strictly more than a pipeline
which either prints uncited copy or prints citations nobody can check.
"""


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def evaluate() -> dict:
    corpus = load_corpus()
    payload = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    caught = missed = 0
    correct_reason = 0
    valid_total = valid_rejected = 0
    reason_counts: Counter[str] = Counter()
    by_injection: Counter[str] = Counter()
    false_positives: list[dict] = []
    misses: list[dict] = []

    for brief in payload["briefs"]:
        evidence = set(brief["evidence_ids"])
        for blk in brief["blocks"]:
            block = CopyBlock(id=blk["id"], kind=blk["kind"], text=blk["text"])
            result = validate_block(block, evidence, corpus.ids)

            if blk["label"] == "injected":
                expected = blk["injection"]["expected_reason"]
                by_injection[blk["injection"]["type"]] += 1
                if result.passed:
                    missed += 1
                    misses.append(
                        {
                            "brief": brief["brief_id"],
                            "block": blk["id"],
                            "text": blk["text"],
                            "expected_reason": expected,
                        }
                    )
                else:
                    caught += 1
                    reason_counts[result.reason.value] += 1
                    if result.reason.value == expected:
                        correct_reason += 1
            else:
                valid_total += 1
                if not result.passed:
                    valid_rejected += 1
                    false_positives.append(
                        {
                            "brief": brief["brief_id"],
                            "block": blk["id"],
                            "variant": blk.get("variant", "ordinary"),
                            "text": blk["text"],
                            "reason": result.reason.value,
                            "detail": result.detail,
                        }
                    )

    injected_total = caught + missed
    return {
        "corpus_size": payload["corpus_size"],
        "n_briefs": payload["n_briefs"],
        "injected_total": injected_total,
        "caught": caught,
        "missed": missed,
        "catch_rate_pct": _pct(caught, injected_total),
        "diagnosis_accuracy_pct": _pct(correct_reason, caught),
        "valid_total": valid_total,
        "valid_rejected": valid_rejected,
        "false_positive_rate_pct": _pct(valid_rejected, valid_total),
        "caught_by_reason": dict(reason_counts),
        "injected_by_type": dict(by_injection),
        "false_positives": false_positives,
        "misses": misses,
    }


def best_natural_failure(path: Path = REJECTION_LOG) -> dict | None:
    """The most demo-worthy real rejection from the log.

    Preference order mirrors how damning each failure is: a fabricated ID beats a
    real ID from the wrong retrieval, which beats a formatting slip. Within a
    reason, the most recent wins — it reflects the prompt as it stands now.
    """
    if not path.exists():
        return None
    rank = {"UNKNOWN_ID": 3, "OUT_OF_SET": 2, "EMPTY_CITATION": 1, "UNPARSEABLE": 0}
    best = None
    best_key = (-1, "")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rank.get(record.get("reason") or "", -1), record.get("ts", ""))
        if key > best_key:
            best_key, best = key, record
    return best


def report(metrics: dict, natural: dict | None) -> str:
    lines = [
        "# Receipts — validator evaluation",
        "",
        f"Corpus: {metrics['corpus_size']} real signals · {metrics['n_briefs']} briefs",
        "",
        "## Headline",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Catch rate** | **{metrics['catch_rate_pct']}%** "
        f"({metrics['caught']}/{metrics['injected_total']} injected bad citations rejected) |",
        f"| **False-positive rate** | **{metrics['false_positive_rate_pct']}%** "
        f"({metrics['valid_rejected']}/{metrics['valid_total']} valid claims wrongly rejected) |",
        f"| Diagnosis accuracy | {metrics['diagnosis_accuracy_pct']}% "
        f"(caught *and* named the right failure mode) |",
        "",
        "## Breakdown",
        "",
        f"- Injected by type: `{metrics['injected_by_type']}`",
        f"- Caught by reason: `{metrics['caught_by_reason']}`",
        "",
    ]

    lines += ["## Honest failure case", ""]
    if metrics["misses"]:
        miss = metrics["misses"][0]
        lines += [
            f"The validator **missed {metrics['missed']}** injected citation(s). Example:",
            "",
            f"> {miss['text']}",
            "",
            f"Expected `{miss['expected_reason']}`, got a pass.",
            "",
        ]
    elif metrics["false_positives"]:
        fp = metrics["false_positives"][0]
        lines += [
            f"Nothing got through, but the validator is **over-strict in "
            f"{metrics['valid_rejected']} case(s)**. The clearest one is the "
            f"`{fp['variant']}` block:",
            "",
            f"> {fp['text']}",
            "",
            f"Rejected as `{fp['reason']}` — {fp['detail']}",
            "",
            "This is the trade we chose. The sentence splitter errs toward splitting, "
            "and an uncited fragment fails. That costs a rewrite; the opposite error "
            "would put an uncited claim in front of a customer.",
            "",
        ]
    else:
        lines += [
            "No misses and no false positives on this set — which says more about "
            "the set than about the validator. The honest caveat is that the eval "
            "copy is templated, so it exercises the citation grammar rather than the "
            "full range of things a model can do to a sentence. The naturally "
            "occurring failures below are the real check.",
            "",
        ]

    lines += [ATTRIBUTION_LIMITS, ""]

    lines += ["## Naturally occurring failure (from the live rejection log)", ""]
    if natural:
        lines += [
            f"- Reason: `{natural['reason']}`",
            f"- Offending IDs: `{', '.join(natural['offending_ids']) or '—'}`",
            f"- Insight: `{natural['insight_id']}`, attempt {natural['attempt']}",
            "",
            "Model output, verbatim:",
            "",
            f"> {natural['block_text']}",
            "",
            natural["detail"],
            "",
        ]
    else:
        lines += [
            "_No rejections logged yet._ Run the app or `scripts/demo_run.py` against "
            "a real model first — §11 requires the demo's rejection moment to use a "
            "genuine failure, not one engineered to fail.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    if not EVAL_SET.exists():
        print(f"{EVAL_SET} not found — run `python eval/build_eval_set.py` first")
        return 1

    metrics = evaluate()
    natural = best_natural_failure()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    text = report(metrics, natural)
    (RESULTS / "report.md").write_text(text, encoding="utf-8")
    if natural:
        (RESULTS / "natural_failure.json").write_text(
            json.dumps(natural, indent=2), encoding="utf-8"
        )

    print(text)
    print(f"\nwrote {(RESULTS / 'metrics.json').relative_to(ROOT)} "
          f"and {(RESULTS / 'report.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
