"""Build `data/signals.jsonl` from real public sources (CLAUDE.md §5.2).

Run once. The output and the ID ledger are committed; nobody should need to run
this again unless the corpus is deliberately rebuilt.

    python scripts/build_corpus.py --reddit-shard data/raw/skincareaddiction-00006.parquet

Re-running is safe: IDs already in `data/id_ledger.json` are reused, never reissued.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import (  # noqa: E402
    assign_ids,
    balance,
    download_amazon_reviews,
    extract_amazon,
    extract_reddit,
)
from src.schema import Signal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ROOT / "data" / "signals.jsonl"
LEDGER = ROOT / "data" / "id_ledger.json"
AMAZON_SLICE = ROOT / "data" / "raw" / "amazon_all_beauty_slice.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reddit-shard", type=Path, required=True)
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--amazon-mb", type=int, default=140)
    args = ap.parse_args()

    print(f"reddit  : reading {args.reddit_shard.name}")
    reddit = extract_reddit(args.reddit_shard)
    print(f"          {len(reddit):,} themed, quotable comments")

    print(f"amazon  : fetching {args.amazon_mb} MB of All_Beauty reviews")
    download_amazon_reviews(AMAZON_SLICE, megabytes=args.amazon_mb)
    amazon = extract_amazon(AMAZON_SLICE)
    print(f"          {len(amazon):,} themed, quotable skincare reviews")

    selected = balance(reddit, amazon, target=args.target)
    records, ledger = assign_ids(selected, LEDGER)

    # Round-trip through the pydantic model so a malformed corpus fails here, at
    # ingest, rather than three modules downstream.
    signals = [Signal.model_validate(r) for r in records]

    SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    with SIGNALS.open("w", encoding="utf-8", newline="\n") as fh:
        for sig in signals:
            fh.write(json.dumps(json.loads(sig.model_dump_json()), ensure_ascii=False) + "\n")
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nwrote {len(signals)} signals -> {SIGNALS.relative_to(ROOT)}")
    print(f"      ledger ({len(ledger)} frozen IDs) -> {LEDGER.relative_to(ROOT)}\n")
    by_source = Counter(s.source for s in signals)
    by_theme = Counter(s.theme for s in signals)
    print("by source:", dict(by_source))
    print("by theme :")
    for theme, n in by_theme.most_common():
        print(f"  {theme:12} {n:3}")
    years = Counter(s.timestamp.year for s in signals)
    print("by year  :", dict(sorted(years.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
