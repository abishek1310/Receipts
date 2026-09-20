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
