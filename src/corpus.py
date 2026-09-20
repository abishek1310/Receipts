"""Load and index the signal corpus (CLAUDE.md §4).

`data/signals.jsonl` is the only data file the app reads. No database (§13) — 400
records fit in memory many times over, and an in-memory dict is the fastest, most
auditable way to answer the one question validation asks: *is this ID real?*

The corpus is the authority for `UNKNOWN_ID`. If a lookup here misses, the model
invented the ID.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from src.schema import Insight, Signal

ROOT = Path(__file__).resolve().parents[1]
SIGNALS_PATH = ROOT / "data" / "signals.jsonl"
INSIGHTS_PATH = ROOT / "data" / "insights.json"


@dataclass(frozen=True)
class Corpus:
    """Every signal, indexed the three ways the rest of the app needs."""

    signals: tuple[Signal, ...]
    by_id: dict[str, Signal] = field(default_factory=dict)
    by_theme: dict[str, tuple[Signal, ...]] = field(default_factory=dict)

    @property
    def ids(self) -> set[str]:
        """The full set of real IDs. This is what separates UNKNOWN_ID from OUT_OF_SET."""
        return set(self.by_id)

    def __len__(self) -> int:
        return len(self.signals)

    def get(self, signal_id: str) -> Signal | None:
        return self.by_id.get(signal_id)

    def resolve(self, signal_ids: Iterable[str]) -> list[Signal]:
        """IDs to signals, silently dropping unknown ones.

        Callers must have validated first — this is for rendering a *passing*
        block's evidence, not for deciding whether it passes.
        """
        return [self.by_id[sid] for sid in signal_ids if sid in self.by_id]

    def themed(self, themes: Sequence[str]) -> list[Signal]:
        """Every signal in any of `themes`, de-duplicated, corpus order preserved."""
        seen: set[str] = set()
        out: list[Signal] = []
        for theme in themes:
            for sig in self.by_theme.get(theme, ()):
                if sig.id not in seen:
                    seen.add(sig.id)
                    out.append(sig)
        return out


def _build(signals: Sequence[Signal]) -> Corpus:
    by_id = {s.id: s for s in signals}
    grouped: dict[str, list[Signal]] = defaultdict(list)
    for s in signals:
        grouped[s.theme].append(s)
    return Corpus(
        signals=tuple(signals),
        by_id=by_id,
        by_theme={k: tuple(v) for k, v in grouped.items()},
    )


@lru_cache(maxsize=4)
def load_corpus(path: Path | str = SIGNALS_PATH) -> Corpus:
    """Read `signals.jsonl`. Cached — Streamlit re-runs this module on every keystroke."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it first:\n"
            "  python scripts/build_corpus.py --reddit-shard data/raw/<shard>.parquet"
        )

    signals: list[Signal] = []
    seen: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            signal = Signal.model_validate(record)
            if signal.id in seen:
                raise ValueError(
                    f"{p}:{lineno} duplicate signal id {signal.id!r} — IDs are frozen "
                    "and unique (§5.1); rebuild the corpus rather than patching by hand"
                )
            seen.add(signal.id)
            signals.append(signal)

    if not signals:
        raise ValueError(f"{p} is empty")
    return _build(signals)


@lru_cache(maxsize=4)
def load_insights(path: Path | str = INSIGHTS_PATH) -> tuple[Insight, ...]:
    """Read the hand-written insights that stand in for HAZRA output (§5.3)."""
    p = Path(path)
    records = json.loads(p.read_text(encoding="utf-8"))
    return tuple(Insight.model_validate(r) for r in records)


def corpus_from_signals(signals: Sequence[Signal]) -> Corpus:
    """Build a Corpus in memory. For tests and eval fixtures — never reads disk."""
    return _build(list(signals))
