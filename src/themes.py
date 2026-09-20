"""Coarse theme buckets for the corpus (CLAUDE.md §5.1).

Themes are assigned deterministically in Python at ingest — no model decides what a
comment is about. That matters for the same reason citations do: if a theme were
model-assigned, the retrieval that produces an evidence set would itself be
unverifiable.

Each theme has **strong** and **weak** keywords. A comment only enters a theme on a
strong hit; weak keywords add confidence but can never carry a comment alone. The
split exists because a single broad keyword list pulls in false friends — "I was
running out of it" is not a comment about working out, and "my occasional red
patches" is, and one bare keyword cannot tell them apart.

Keep the buckets coarse. They exist to make retrieval (§8) tractable over ~400
signals, not to be a taxonomy.
"""

from __future__ import annotations

import re

# Ordered — earlier themes win ties, so the more specific buckets come first.
THEME_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "redness": {
        "strong": (
            "redness", "red face", "flushing", "flushed", "rosacea", "blotchy",
            "red patches", "red and angry", "ruddy", "broken capillaries",
            "facial redness", "red spots",
        ),
        "weak": ("red", "pink", "angry skin", "inflamed", "inflammation"),
    },
    "sweat": {
        "strong": (
            "sweat", "sweaty", "sweating", "workout", "work out", "working out",
            "worked out", "post-workout", "the gym", "at the gym", "after the gym",
            "hot yoga", "spin class",
        ),
        "weak": ("exercise", "exercising", "cardio", "athlete", "training", "sports"),
    },
    "sun": {
        "strong": (
            "sunscreen", "sunblock", "sun screen", "spf", "white cast", "sunburn",
            "sun protection", "uv protection", "sun damage",
        ),
        "weak": ("uv", "reapply", "reapplying", "beach", "sun"),
    },
    "irritation": {
        "strong": (
            "irritation", "irritated", "irritating", "stinging", "stings",
            "sensitive skin", "itchy", "itching", "tingling", "purging",
            "barrier damage", "compromised barrier", "moisture barrier",
            "burning sensation", "allergic reaction",
        ),
        "weak": ("reaction", "burns", "burning", "raw", "sting", "sensitive"),
    },
    "acne": {
        "strong": (
            "acne", "breakout", "breakouts", "breaking out", "broke out", "pimple",
            "pimples", "whitehead", "whiteheads", "blackhead", "blackheads",
            "cystic", "clogged pores", "blemish", "blemishes", "zit", "zits",
        ),
        "weak": ("congestion", "spot treatment", "comedonal", "pustule"),
    },
    "hydration": {
        "strong": (
            "dry skin", "dryness", "flaky", "flaking", "dehydrated", "dehydration",
            "hydration", "hydrating", "moisturizing", "moisturising", "dry patches",
            "so dry", "really dry", "dry and tight",
        ),
        "weak": ("moisture", "parched", "tight", "dry", "moisturizer", "hydrated"),
    },
    "texture": {
        "strong": (
            "greasy", "sticky", "tacky", "pilling", "pills under", "oily film",
            "leaves a residue", "lightweight", "absorbs quickly", "absorbs fast",
            "sits on top", "goes on smooth", "glides on",
        ),
        "weak": ("absorbs", "absorbed", "heavy", "thick", "silicone", "residue", "melts"),
    },
    "fragrance": {
        "strong": (
            "fragrance", "fragrance-free", "fragrance free", "unscented", "scent",
            "scented", "perfume", "perfumey", "smells like", "smells so",
            "strong smell", "no smell",
        ),
        "weak": ("smell", "smells", "stinks", "odor", "odour"),
    },
    "price": {
        "strong": (
            "expensive", "pricey", "overpriced", "worth the money", "worth it",
            "affordable", "drugstore", "dupe", "for the price", "great value",
            "waste of money", "too much money", "on a budget",
        ),
        "weak": ("price", "cost", "cheap", "budget", "money", "splurge"),
    },
    "routine": {
        "strong": (
            "routine", "morning routine", "night routine", "am routine", "pm routine",
            "layering", "too many steps", "simplify my", "whole regimen", "regimen",
        ),
        "weak": ("layer", "step", "steps", "simplify", "order of"),
    },
}

ALL_THEMES: tuple[str, ...] = tuple(THEME_KEYWORDS)

# Word-boundary matching so "spot treatment" does not fire on "spotless" and
# "dry" does not fire on "laundry".
def _compile(words: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words)


_PATTERNS: dict[str, dict[str, tuple[re.Pattern[str], ...]]] = {
    theme: {kind: _compile(words) for kind, words in kinds.items()}
    for theme, kinds in THEME_KEYWORDS.items()
}

# A strong hit is worth more than a weak one when breaking ties between themes.
STRONG_WEIGHT = 3
WEAK_WEIGHT = 1


def theme_hits(text: str, theme: str) -> tuple[int, int]:
    """(strong hits, weak hits) for one theme."""
    pats = _PATTERNS[theme]
    strong = sum(1 for p in pats["strong"] if p.search(text))
    weak = sum(1 for p in pats["weak"] if p.search(text))
    return strong, weak


def score_themes(text: str) -> dict[str, int]:
    """Weighted score per theme. Zero unless at least one strong keyword matched."""
    scores: dict[str, int] = {}
    for theme in ALL_THEMES:
        strong, weak = theme_hits(text, theme)
        scores[theme] = 0 if strong == 0 else strong * STRONG_WEIGHT + weak * WEAK_WEIGHT
    return scores


def classify(text: str) -> str | None:
    """The single best theme for `text`, or None if no theme matched strongly.

    Ties break toward the earlier theme in `THEME_KEYWORDS`, which is why that
    dict is ordered specific-to-general.
    """
    scores = score_themes(text)
    best = max(scores.values(), default=0)
    if best == 0:
        return None
    for theme in ALL_THEMES:
        if scores[theme] == best:
            return theme
    return None
