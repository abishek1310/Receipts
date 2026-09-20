"""Retrieval (CLAUDE.md §8).

    retrieve(insight, user_query, k) -> list[Signal]

v1, as specified: filter by theme overlap, keyword-match the query, return top-k by
recency. No embeddings, no vector database (§13) — 400 records is a brute-force
problem, and a transparent scoring function is worth more here than a marginally
better ranking nobody can inspect.

Whatever comes back is **the evidence set** for the turn. The graph freezes it in
state and the revision loop reuses it verbatim, so this function's output is the
exact universe of IDs the model is allowed to cite.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.corpus import Corpus, load_corpus
from src.schema import Insight, Signal
from src.themes import ALL_THEMES, theme_hits

_WORD = re.compile(r"[a-z][a-z'-]{2,}")

_STOPWORDS = frozenset(
    """
    the and for with that this have has had you your are but not from what about
    more make makes making write written give some into they them their there
    when where which while would could should can will just like also than then
    something anything everything really very much many most our ours out off
    copy line lines hook hooks ad ads version tone angle please again another
    """.split()
)

# How strongly each contribution counts when ranking a signal for a query.
_QUERY_TERM_WEIGHT = 4
_QUERY_THEME_WEIGHT = 6
_INSIGHT_THEME_WEIGHT = 2


def query_terms(text: str | None) -> set[str]:
    """Content words from a user query, lowercased, stopwords removed."""
    if not text:
        return set()
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def query_themes(text: str | None) -> list[str]:
    """Themes the user's wording points at, strongest first.

    Used by the graph to tell "make it softer" (a tone change — reuse the evidence
    set) from "what about price?" (a new angle — retrieve fresh). See §9.

    Unlike corpus classification, a weak keyword counts here. A user query is a
    handful of deliberate words: someone who types "price" means price, even
    though the same bare word inside a 300-character review does not settle what
    that review is about.
    """
    if not text:
        return []
    scored = []
    for theme in ALL_THEMES:
        strong, weak = theme_hits(text, theme)
        if strong or weak:
            scored.append((strong * 3 + weak, theme))
    scored.sort(key=lambda pair: (-pair[0], ALL_THEMES.index(pair[1])))
    return [theme for _, theme in scored]


def _score(signal: Signal, terms: set[str], q_themes: Sequence[str], insight: Insight) -> int:
    text = signal.text.lower()
    score = 0
    if terms:
        score += _QUERY_TERM_WEIGHT * sum(1 for t in terms if t in text)
    if signal.theme in q_themes:
        score += _QUERY_THEME_WEIGHT * (len(q_themes) - list(q_themes).index(signal.theme))
    if signal.theme in insight.themes:
        score += _INSIGHT_THEME_WEIGHT * (
            len(insight.themes) - insight.themes.index(signal.theme)
        )
    return score


def _quotas(themes: Sequence[str], k: int) -> dict[str, int]:
    """Split k across themes, weighted by priority order (first theme gets most)."""
    weights = [len(themes) - i for i in range(len(themes))]
    total = sum(weights)
    quotas = {t: max(1, k * w // total) for t, w in zip(themes, weights)}
    # Hand any rounding remainder to the highest-priority theme.
    quotas[themes[0]] += k - sum(quotas.values())
    return quotas


def retrieve(
    insight: Insight,
    user_query: str | None = None,
    k: int = 40,
    corpus: Corpus | None = None,
) -> list[Signal]:
    """The evidence set for one generation.

    Candidates come from the insight's themes, widened to any theme the query
    names, and the budget is **split across those themes by priority** rather than
    handed to whichever theme happens to have the most signals. Without the split,
    an insight about redness-after-training retrieves forty comments about redness
    and none about sweat, and the copy has nothing to connect.

    Ranking inside a theme is query-term overlap, then recency; the final
    tie-break is the signal ID, so identical inputs always produce an identical
    evidence set — which is what makes a rejection reproducible.
    """
    corpus = corpus or load_corpus()
    terms = query_terms(user_query)
    q_themes = query_themes(user_query)

    # A theme the user named outranks the insight's own themes for this turn.
    themes = [t for t in dict.fromkeys([*q_themes, *insight.themes]) if corpus.by_theme.get(t)]
    if not themes:
        return sorted(corpus.signals, key=lambda s: (-s.timestamp.toordinal(), s.id))[:k]

    rank = lambda s: (-_score(s, terms, q_themes, insight), -s.timestamp.toordinal(), s.id)  # noqa: E731

    quotas = _quotas(themes, k)
    chosen: list[Signal] = []
    leftovers: list[Signal] = []
    for theme in themes:
        ranked = sorted(corpus.by_theme.get(theme, ()), key=rank)
        take = quotas[theme]
        chosen.extend(ranked[:take])
        leftovers.extend(ranked[take:])

    # A thin theme under-fills its quota; top up from the best of what is left.
    if len(chosen) < k:
        chosen.extend(sorted(leftovers, key=rank)[: k - len(chosen)])

    return sorted(chosen, key=rank)[:k]


def evidence_ids(signals: Sequence[Signal]) -> set[str]:
    """The ID set handed to the validator. Nothing outside this may be cited."""
    return {s.id for s in signals}
