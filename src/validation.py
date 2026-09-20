"""THE CORE (CLAUDE.md §6).

The one invariant (§1):

    No generated sentence reaches the user unless every citation in it resolves to
    a signal that was actually retrieved for that generation.

The model's citation tags are *claims to be checked*, never facts. Everything in
this module is deterministic Python — no LLM call, no heuristic scoring, no
"probably fine". A block either resolves against the evidence set or it fails.

Design notes worth knowing before you change anything here:

* **Over-strictness is the safe direction.** Where the sentence splitter is
  ambiguous it splits *more*, which means *more* sentences demanding a citation.
  That can cost a false positive (measured in §11); it can never let an uncited
  claim through.
* **Tags are validated wherever they appear**, even on sentences that do not
  require one. A hallucinated ID on a call-to-action is still a hallucinated ID
  reaching the user.
* **Failures are values, not exceptions** (§14). Nothing here raises on bad model
  output; bad model output is the expected case.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from src.schema import (
    FAILURE_PRIORITY,
    CopyBlock,
    FailureReason,
    Signal,
    ValidationResult,
)

# --------------------------------------------------------------------------------------
# Citation grammar (§6.1)
# --------------------------------------------------------------------------------------

# Any bracket group at all — used to detect that *a* tag is present, however broken.
_TAG_ANY = re.compile(r"\[[^\[\]]*\]")

# The exact shape, and nothing else: [sig_0183] or [sig_0183, sig_0241, sig_0319]
_TAG_STRICT = re.compile(r"^\[sig_\d{4}(?:,\s*sig_\d{4})*\]$")

# A tag that is present but carries no IDs: [] or [   ]
_TAG_EMPTY = re.compile(r"^\[\s*\]$")

# Pull IDs out of a tag already known to be strictly shaped.
_ID_IN_TAG = re.compile(r"sig_\d{4}")

_TERMINATORS = ".!?…"

# A dot after one of these is an abbreviation, not a full stop. Without this the
# splitter cuts "Dr. Idriss recommends it." into two fragments and fails the first
# one as uncited — a false positive on copy that was correctly cited.
_ABBREVIATIONS = frozenset(
    """
    dr mr mrs ms prof st ave inc co ltd vs etc approx no fig dept est
    jan feb mar apr jun jul aug sep sept oct nov dec
    e.g i.e a.m p.m u.s u.k
    """.split()
)
_ABBREV_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")

# A sentence that is *only* one of these is not making a claim, so it needs no
# citation. Matched whole, after lowercasing and stripping punctuation — so
# "Shop now." is exempt but "Shop now and clear your acne overnight." is not.
_CTA_EXACT = frozenset(
    {
        "shop now",
        "learn more",
        "read more",
        "try it",
        "try it now",
        "try it today",
        "get yours",
        "get yours now",
        "get yours today",
        "link in bio",
        "available now",
        "tap to shop",
        "swipe up",
        "order now",
    }
)

# Structural scaffolding the model sometimes emits around blocks, e.g. "Hook:".
_LABEL_RE = re.compile(r"^(hook|ad copy|ad|positioning|caption|headline|option \d+)\s*:?$")


@dataclass(frozen=True)
class Sentence:
    """One sentence of a block, with the citation tag that trails it (if any)."""

    body: str
    tag: str | None

    @property
    def raw(self) -> str:
        return f"{self.body} {self.tag}" if self.tag else self.body


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _is_boundary(text: str, term_end: int) -> bool:
    """Is the terminator run ending at `term_end` a real sentence boundary?

    Guards against splitting decimals ("3.5 hours") and initialisms, while still
    splitting aggressively enough that a missing tag is never hidden by a
    run-on. When in doubt this returns True — over-splitting fails safe.
    """
    n = len(text)
    k = term_end
    while k < n and text[k] in " \t":
        k += 1
    if k >= n:
        return True
    nxt = text[k]
    # A tag always closes a sentence.
    if nxt == "[":
        return True
    # "3.5" / "1.99" — a single dot between digits is a decimal point, not a stop.
    if (
        term_end - 1 >= 1
        and text[term_end - 1] == "."
        and text[term_end - 2].isdigit()
        and k == term_end
        and nxt.isdigit()
    ):
        return False
    # "Dr. Idriss" / "e.g. retinol" — a known abbreviation, not the end of a claim.
    if text[term_end - 1] == ".":
        m = _ABBREV_RE.search(text[:term_end])
        if m and m.group(1).lower().rstrip(".") in _ABBREVIATIONS:
            return False
    return nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘("


def split_sentences(text: str) -> list[Sentence]:
    """Split a block into sentences, attaching each trailing citation tag.

    A tag binds to the sentence it follows:

        "Redness fades fast. [sig_0001] Price still stings. [sig_0002]"
        -> [Sentence("Redness fades fast.", "[sig_0001]"),
            Sentence("Price still stings.", "[sig_0002]")]
    """
    sentences: list[Sentence] = []
    n = len(text)
    i = 0
    start = 0
    in_quote = False

    while i < n:
        ch = text[i]
        # A full stop inside a quotation belongs to the quotation, not to the
        # sentence carrying it: `One reviewer put it plainly: "It stings." Redness
        # is the part people notice. [sig_0020]` is one claim, with one citation.
        if ch == '"' or ch == "“":
            in_quote = not in_quote if ch == '"' else True
            i += 1
            continue
        if ch == "”":
            in_quote = False
            i += 1
            continue
        if in_quote or ch not in _TERMINATORS:
            i += 1
            continue

        j = i
        while j < n and text[j] in _TERMINATORS:
            j += 1

        if not _is_boundary(text, j):
            i = j
            continue

        k = j
        while k < n and text[k] in " \t":
            k += 1

        tag = None
        end = j
        m = _TAG_ANY.match(text, k)
        if m:
            tag = m.group(0)
            end = m.end()

        body = text[start:j].strip()
        if body:
            sentences.append(Sentence(body=body, tag=tag))
        i = end
        start = end

    # Trailing text with no terminal punctuation — common for hooks.
    rest = text[start:].strip()
    if rest:
        tag = None
        body = rest
        m = _TAG_ANY.search(rest)
        if m and rest[m.end() :].strip() == "":
            tag = m.group(0)
            body = rest[: m.start()].strip()
        if body or tag:
            sentences.append(Sentence(body=body, tag=tag))

    return sentences


def is_claim_bearing(body: str) -> bool:
    """Does this sentence assert something that needs evidence?

    Conservative by construction: everything is claim-bearing unless it matches a
    short, explicit allowlist of calls-to-action and structural labels. Widening
    this allowlist weakens §1 — do not do it to make the eval numbers look better.
    """
    stripped = body.strip()
    if not stripped:
        return False
    normalised = re.sub(r"[^\w\s]", "", stripped).strip().lower()
    if not normalised:
        return False
    if normalised in _CTA_EXACT:
        return False
    if _LABEL_RE.match(stripped.lower()):
        return False
    return True


def extract_citations(text: str) -> list[str]:
    """Every signal ID cited anywhere in the text, in order, de-duplicated."""
    seen: list[str] = []
    for tag in _TAG_ANY.findall(text):
        for sid in _ID_IN_TAG.findall(tag):
            if sid not in seen:
                seen.append(sid)
    return seen


def strip_citations(text: str) -> str:
    """Remove citation tags for display. Never call this before validating."""
    return re.sub(r"\s*\[[^\[\]]*\]", "", text).strip()


# --------------------------------------------------------------------------------------
# Validation (§6.2)
# --------------------------------------------------------------------------------------


def validate_block(
    block: CopyBlock,
    evidence_set: set[str] | Iterable[str],
    corpus_ids: set[str] | Iterable[str] | None = None,
) -> ValidationResult:
    """Return PASS or FAIL with a machine-readable reason.

    FAIL conditions, checked in the order of §6.2:

      1. UNPARSEABLE     — a claim-bearing sentence with no tag, or a tag that does
                           not match the exact grammar (`[sig_NNNN, sig_NNNN]`).
      2. UNKNOWN_ID      — an ID that is not in the corpus at all (pure hallucination).
      3. OUT_OF_SET      — a real corpus ID that was not in *this* generation's
                           evidence set. The model reached outside its context; the
                           claim was not grounded in this retrieval, so it fails.
      4. EMPTY_CITATION  — a tag is present but carries no IDs.

    `corpus_ids` is what makes (2) distinguishable from (3). When it is omitted the
    check still fails closed — every ID outside the evidence set is reported as
    OUT_OF_SET — but the diagnosis is coarser. Pass it in production.
    """
    evidence = set(evidence_set)
    corpus = set(corpus_ids) if corpus_ids is not None else None

    buckets: dict[FailureReason, list[str]] = {r: [] for r in FAILURE_PRIORITY}
    sentences_by_reason: dict[FailureReason, list[str]] = {r: [] for r in FAILURE_PRIORITY}
    cited: list[str] = []

    def flag(reason: FailureReason, sentence: str, ids: Sequence[str] = ()) -> None:
        for sid in ids:
            if sid not in buckets[reason]:
                buckets[reason].append(sid)
        if sentence not in sentences_by_reason[reason]:
            sentences_by_reason[reason].append(sentence)

    for sentence in split_sentences(block.text):
        needs_citation = is_claim_bearing(sentence.body)

        if sentence.tag is None:
            if needs_citation:
                flag(FailureReason.UNPARSEABLE, sentence.body)
            continue

        if _TAG_EMPTY.match(sentence.tag):
            flag(FailureReason.EMPTY_CITATION, sentence.raw)
            continue

        if not _TAG_STRICT.match(sentence.tag):
            # Malformed shape is a parse failure, per §6.1.
            flag(FailureReason.UNPARSEABLE, sentence.raw)
            continue

        ids = _ID_IN_TAG.findall(sentence.tag)
        if not ids:
            flag(FailureReason.EMPTY_CITATION, sentence.raw)
            continue

        for sid in ids:
            if sid not in cited:
                cited.append(sid)

        if corpus is not None:
            unknown = [s for s in ids if s not in corpus]
            if unknown:
                flag(FailureReason.UNKNOWN_ID, sentence.raw, unknown)
            out_of_set = [s for s in ids if s in corpus and s not in evidence]
        else:
            out_of_set = [s for s in ids if s not in evidence]
        if out_of_set:
            flag(FailureReason.OUT_OF_SET, sentence.raw, out_of_set)

    for reason in FAILURE_PRIORITY:
        if sentences_by_reason[reason]:
            return ValidationResult(
                status="FAIL",
                block_id=block.id,
                reason=reason,
                offending_ids=buckets[reason],
                offending_sentences=sentences_by_reason[reason],
                cited_ids=cited,
                detail=_explain(reason, buckets[reason], sentences_by_reason[reason]),
            )

    return ValidationResult(
        status="PASS",
        block_id=block.id,
        cited_ids=cited,
        detail=f"{len(cited)} citation(s) resolved against the evidence set",
    )


def _explain(reason: FailureReason, ids: list[str], sentences: list[str]) -> str:
    """Plain-language reason, shown to the user in the UI (§10)."""
    first = sentences[0] if sentences else ""
    snippet = first if len(first) <= 90 else first[:87] + "..."
    if reason is FailureReason.UNPARSEABLE:
        return f"No valid citation tag on a claim-bearing sentence: {snippet!r}"
    if reason is FailureReason.UNKNOWN_ID:
        return f"Cited {', '.join(ids)} — not in the corpus at all (hallucinated ID)"
    if reason is FailureReason.OUT_OF_SET:
        return (
            f"Cited {', '.join(ids)} — real signal(s), but not retrieved for this "
            "generation, so the claim is not grounded in this evidence set"
        )
    return f"Citation tag present but empty: {snippet!r}"


def validate_blocks(
    blocks: Sequence[CopyBlock],
    evidence_set: set[str] | Iterable[str],
    corpus_ids: set[str] | Iterable[str] | None = None,
) -> list[ValidationResult]:
    """Validate each block independently so only the failures are regenerated (§6.3)."""
    evidence = set(evidence_set)
    corpus = set(corpus_ids) if corpus_ids is not None else None
    return [validate_block(b, evidence, corpus) for b in blocks]


def resolve_signals(result: ValidationResult, by_id: dict[str, Signal]) -> list[Signal]:
    """Map a passing result's cited IDs back to Signal objects, for the UI expander."""
    return [by_id[sid] for sid in result.cited_ids if sid in by_id]


# --------------------------------------------------------------------------------------
# Rejection log (§6.3, §11)
# --------------------------------------------------------------------------------------

REJECTION_LOG = Path("eval/results/rejections.jsonl")


def log_rejection(
    result: ValidationResult,
    block: CopyBlock,
    *,
    insight_id: str,
    attempt: int,
    evidence_set: Iterable[str],
    path: Path | None = None,
) -> None:
    """Append one rejection. These logs are where the demo's real failure comes from.

    §11 is explicit that the live rejection moment must use a naturally occurring
    hallucination, not a prompt engineered to fail — so this has to capture every
    rejection as it happens, including ones that a later retry recovered from.
    """
    target = path or REJECTION_LOG
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "insight_id": insight_id,
        "attempt": attempt,
        "block_id": block.id,
        "block_kind": block.kind,
        "block_text": block.text,
        "reason": result.reason.value if result.reason else None,
        "offending_ids": result.offending_ids,
        "offending_sentences": result.offending_sentences,
        "cited_ids": result.cited_ids,
        "detail": result.detail,
        "evidence_set": sorted(evidence_set),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
