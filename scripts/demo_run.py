"""End-to-end run in a terminal (CLAUDE.md §12, step 4).

The first thing to get working, and the quickest way to check the whole chain
after a change. Needs a real API key.

    python scripts/demo_run.py                        # first insight, fresh brief
    python scripts/demo_run.py --insight ins_02
    python scripts/demo_run.py --revise "make it softer"
    python scripts/demo_run.py --revise "what about price?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.corpus import load_corpus, load_insights  # noqa: E402
from src.graph import run_turn  # noqa: E402
from src.llm import LLMError, get_client  # noqa: E402
from src.schema import GenerationResult  # noqa: E402

RULE = "─" * 78


def show(result: GenerationResult, corpus) -> None:
    if result.error:
        print(f"\n!! generation failed: {result.error}")
        return

    fresh = "retrieved fresh" if result.retrieved_fresh else "REUSED (frozen)"
    print(f"\nEvidence set: {len(result.evidence_set)} signals — {fresh}")
    print(f"  {result.evidence_note}")
    print(f"Attempts: {result.attempts}")
    print(RULE)

    for outcome in result.outcomes:
        mark = "X" if outcome.unsupported else "OK"
        print(f"\n[{mark}] {outcome.block.kind}")
        print(f"  {outcome.display_text}")
        if outcome.unsupported:
            print(f"  REJECTED ({outcome.result.reason.value}) — {outcome.result.detail}")
            continue
        print(f"  {len(outcome.supporting)} supporting signal(s):")
        for sig in outcome.supporting[:3]:
            print(f"    [{sig.id}] {sig.source_detail} · {sig.timestamp}")
            print(f"           \"{sig.text[:120]}...\"")
            print(f"           {sig.url}")
        if len(outcome.supporting) > 3:
            print(f"    ... and {len(outcome.supporting) - 3} more")

    print(f"\n{RULE}")
    print(f"generated {result.n_generated} · rejected {result.n_rejected}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insight", default=None, help="insight id, e.g. ins_01")
    ap.add_argument("--revise", default=None, help="run a second turn with this direction")
    args = ap.parse_args()

    settings = get_settings()
    if not settings.configured:
        print(
            f"No API key for provider '{settings.llm_provider}'.\n"
            "Copy .env.example to .env and fill in ANTHROPIC_API_KEY."
        )
        return 1

    corpus = load_corpus()
    insights = {i.id: i for i in load_insights()}
    insight = insights[args.insight] if args.insight else next(iter(insights.values()))

    print(f"{RULE}\nINSIGHT {insight.id}: {insight.statement}")
    print(f"corpus: {len(corpus)} signals · model: {settings.model}\n{RULE}")

    try:
        client = get_client(settings)
    except LLMError as exc:
        print(f"!! {exc}")
        return 1

    result = run_turn(insight, client=client, corpus=corpus, settings=settings)
    show(result, corpus)

    if args.revise:
        print(f"\n\n{RULE}\nREVISION: {args.revise!r}\n{RULE}")
        second = run_turn(
            insight,
            client=client,
            user_message=args.revise,
            evidence_set=result.evidence_set,
            previous_blocks=[o.block for o in result.outcomes],
            corpus=corpus,
            settings=settings,
        )
        show(second, corpus)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
