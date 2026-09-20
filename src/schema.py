"""Data models for Receipts.

Every object that crosses a module boundary is defined here (CLAUDE.md §14).

The types in this file encode the §1 invariant structurally: a `CopyBlock` carries
only what the model produced, and nothing downstream may render one without a
`ValidationResult` beside it. `BlockOutcome` is the only shape the UI accepts.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# A signal ID is `sig_` followed by exactly four digits. Frozen at ingest (§5.1).
SIGNAL_ID_RE = re.compile(r"^sig_\d{4}$")

SourceName = Literal["reddit", "amazon", "tiktok"]
BlockKind = Literal["hook", "ad_copy", "positioning"]

# Verbatim text is truncated at ingest, never cleaned beyond whitespace (§5.1).
MAX_SIGNAL_CHARS = 400


class Signal(BaseModel):
    """One real consumer comment. Immutable once ingested."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source: SourceName
    source_detail: str
    url: str
    timestamp: date
    theme: str

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not SIGNAL_ID_RE.match(v):
            raise ValueError(f"signal id must match sig_NNNN, got {v!r}")
        return v

    @field_validator("text")
    @classmethod
    def _text_length(cls, v: str) -> str:
        if len(v) > MAX_SIGNAL_CHARS:
            raise ValueError(
                f"signal text exceeds {MAX_SIGNAL_CHARS} chars — truncate at ingest, not here"
            )
        return v


class Insight(BaseModel):
    """A market insight. Stands in for HAZRA output (§5.3)."""

    model_config = ConfigDict(frozen=True)

    id: str
    statement: str
    category: str
    audience: str
    themes: list[str]
    momentum: str


class CopyBlock(BaseModel):
    """One unit of generated copy, exactly as the model emitted it.

    `text` still contains the raw citation tags. It is deliberately NOT cleaned
    here — `validation.py` needs the tags, and stripping them before validation
    would discard the only thing there is to check.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: BlockKind
    text: str


class FailureReason(str, Enum):
    """Machine-readable failure reasons, in the priority order of §6.2."""

    UNPARSEABLE = "UNPARSEABLE"
    UNKNOWN_ID = "UNKNOWN_ID"
    OUT_OF_SET = "OUT_OF_SET"
    EMPTY_CITATION = "EMPTY_CITATION"


# Checked in this order; the first reason that has any offending sentence wins (§6.2).
FAILURE_PRIORITY: tuple[FailureReason, ...] = (
    FailureReason.UNPARSEABLE,
    FailureReason.UNKNOWN_ID,
    FailureReason.OUT_OF_SET,
    FailureReason.EMPTY_CITATION,
)


class ValidationResult(BaseModel):
    """The verdict on one `CopyBlock`. A failure is a value, never an exception (§14)."""

    model_config = ConfigDict(frozen=True)

    status: Literal["PASS", "FAIL"]
    block_id: str
    reason: FailureReason | None = None
    offending_ids: list[str] = Field(default_factory=list)
    offending_sentences: list[str] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class BlockOutcome(BaseModel):
    """A block plus its verdict and resolved evidence. The only thing the UI renders.

    `unsupported=True` means validation failed and the retry budget is spent, so the
    block is surfaced with a visible ❌ rather than dropped silently (§6.3).
    """

    block: CopyBlock
    result: ValidationResult
    supporting: list[Signal] = Field(default_factory=list)
    unsupported: bool = False

    @property
    def display_text(self) -> str:
        """Copy with citation tags removed, for rendering."""
        from src.validation import strip_citations

        return strip_citations(self.block.text)


class GenerationResult(BaseModel):
    """Everything one turn of the graph produced."""

    insight_id: str
    outcomes: list[BlockOutcome] = Field(default_factory=list)
    evidence_set: list[Signal] = Field(default_factory=list)
    attempts: int = 1
    retrieved_fresh: bool = True
    # Set when the provider or the response shape failed outright — not a
    # citation rejection, which is carried per-block on the outcomes.
    error: str | None = None
    # Plain-language note for the UI about why the evidence set was or was not
    # re-retrieved this turn (§9).
    evidence_note: str = ""

    @property
    def n_generated(self) -> int:
        return len(self.outcomes)

    @property
    def n_rejected(self) -> int:
        return sum(1 for o in self.outcomes if o.unsupported)
