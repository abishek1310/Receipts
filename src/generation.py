"""Generation (CLAUDE.md §7).

Prompt construction, strictly in the order §7 specifies:

1. Signals are retrieved in Python (the caller does this — see `retrieval.py`).
2. They are serialised into the prompt as an explicit numbered evidence block,
   each with its real ID and verbatim text.
3. The model is told: use only these signals, cite by ID, and if the evidence does
   not support a claim, do not make the claim.
4. The LLM is called.
5. The raw output goes straight to `validation.py`. **Nothing here renders.**

The prompt never asks the model to invent, infer, or recall an ID. Every ID it can
legally write is printed in front of it. Its only job is to attribute correctly
among them — which is a task we can check, unlike recall, which is not.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from src.llm import LLMClient
from src.schema import BlockKind, CopyBlock, FailureReason, Insight, Signal, ValidationResult

# 3 hooks, 2 ad lines, 1 positioning statement — enough surface area for a demo
# without making a rejection look like a rounding error.
DEFAULT_MIX: tuple[tuple[BlockKind, int], ...] = (
    ("hook", 3),
    ("ad_copy", 2),
    ("positioning", 1),
)

_KIND_BRIEF: dict[BlockKind, str] = {
    "hook": "a scroll-stopping TikTok hook, one or two sentences, spoken-voice",
    "ad_copy": "a short paid-social ad line, two or three sentences",
    "positioning": "a positioning statement describing who this is for and why it wins",
}


class GenerationError(RuntimeError):
    """The model's response was not in the requested shape at all.

    Distinct from a validation failure: this is a malformed *response*, not a
    badly cited claim. The graph surfaces it as an error rather than a rejection.
    """


BLOCKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["hook", "ad_copy", "positioning"]},
                    "text": {"type": "string"},
                },
                "required": ["kind", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------


def format_evidence(signals: Sequence[Signal]) -> str:
    """The numbered evidence block (§7 step 2).

    Real IDs, verbatim text, nothing paraphrased. This listing is the complete and
    only universe of citable IDs for this generation.
    """
    lines = []
    for n, sig in enumerate(signals, 1):
        lines.append(
            f"{n}. [{sig.id}] ({sig.source} — {sig.source_detail}, {sig.timestamp}, "
            f"theme: {sig.theme})\n   \"{sig.text}\""
        )
    return "\n".join(lines)


_RULES = """\
RULES — these are enforced by a validator that runs on your output. A block that
breaks any of them is rejected and you are asked to write it again.

1. Use ONLY the signals listed above. They are the entire evidence base.
2. EVERY sentence that makes a claim must carry its own citation tag naming the
   signals that support it, in exactly this shape:

       Your gym glow shouldn't last three hours [sig_0183, sig_0241].

   Square brackets, four-digit IDs, separated by ", ", nothing else inside the
   brackets. A tag before or after the full stop is fine; what is not fine is a
   claim-bearing sentence with no tag of its own. If a block has three
   sentences, it needs three tags:

       Redness outlasts the workout [sig_0183]. Cooling gels sting broken skin
       [sig_0241]. Nothing on the shelf is built for this [sig_0319].

   A tag at the end of a paragraph does not cover the sentences before it.
3. Cite only IDs printed in the evidence block above. Do not write an ID you have
   not been given, do not adjust one, and do not reuse an ID from earlier in this
   conversation — the evidence for this request is only what is listed above.
4. If the evidence does not support a claim you want to make, do not make it.
   Write a weaker claim that the evidence does support, or write about something
   else the evidence does cover. Copy with no claim in it — a bare call to action
   like "Shop now." — needs no tag.
5. Never invent statistics, clinical results, ingredient claims, or percentages.
   None of that is in the evidence.

Write copy a person would actually stop scrolling for. The citations are checked
mechanically; the writing is the part only you can do."""


def build_system_prompt(insight: Insight, evidence: Sequence[Signal]) -> str:
    """The stable half of the prompt — identical across retries, so it caches."""
    return f"""\
You are a senior copywriter working from verified consumer research.

THE INSIGHT
  {insight.statement}
  Category : {insight.category}
  Audience : {insight.audience}
  Momentum : {insight.momentum}

EVIDENCE — {len(evidence)} consumer signals retrieved for this brief
{format_evidence(evidence)}

{_RULES}"""


def build_user_prompt(
    mix: Sequence[tuple[BlockKind, int]] = DEFAULT_MIX,
    direction: str | None = None,
) -> str:
    wanted = "\n".join(
        f"  - {count} x {kind}: {_KIND_BRIEF[kind]}" for kind, count in mix
    )
    prompt = f"Write:\n{wanted}"
    if direction:
        prompt += f"\n\nDirection from the user: {direction}"
    prompt += (
        "\n\nReturn every block. Put the citation tags inside each block's `text`."
    )
    return prompt


def build_revision_prompt(
    previous: Sequence[CopyBlock],
    direction: str,
) -> str:
    """A tone or angle revision, bound to the same evidence set (§9).

    The previous copy goes in so the model revises rather than restarts, but the
    evidence block in the system prompt is unchanged — which is the whole point of
    evidence-preserving revision.
    """
    current = "\n".join(f"  [{b.kind}] {b.text}" for b in previous)
    return f"""\
Here is the copy you wrote:

{current}

The user asks: {direction}

Rewrite every block to follow that direction. The evidence set has not changed —
re-cite from the same signal list above. A revised sentence needs a citation that
supports the revised claim, not the one it replaced."""


def build_regenerate_prompt(block: CopyBlock, result: ValidationResult) -> str:
    """Regenerate one failing block (§6.3), telling it precisely what broke."""
    guidance = {
        FailureReason.UNPARSEABLE: (
            "One or more sentences had no citation tag, or a tag in the wrong shape. "
            "Every claim-bearing sentence must end with [sig_XXXX] or "
            "[sig_XXXX, sig_YYYY] — four digits, comma-space separated, nothing else "
            "inside the brackets."
        ),
        FailureReason.UNKNOWN_ID: (
            "You cited an ID that does not exist. Every ID you may use is printed in "
            "the evidence block above; copy them exactly rather than reconstructing "
            "them."
        ),
        FailureReason.OUT_OF_SET: (
            "You cited a real signal that was not retrieved for this brief. Only the "
            "IDs in the evidence block above are available for this request."
        ),
        FailureReason.EMPTY_CITATION: (
            "You left a citation tag empty. Either cite the signals that support the "
            "sentence, or remove the claim."
        ),
    }
    reason = result.reason or FailureReason.UNPARSEABLE
    offending = (
        f"\nOffending IDs: {', '.join(result.offending_ids)}"
        if result.offending_ids
        else ""
    )
    return f"""\
This block was rejected by the citation validator.

  [{block.kind}] {block.text}

Reason: {reason.value}
{guidance[reason]}{offending}

Write the block again — same kind, same intent, corrected citations. Return exactly
one block."""


# --------------------------------------------------------------------------------------
# Calling the model
# --------------------------------------------------------------------------------------


def _parse_blocks(raw: str, start_index: int = 1) -> list[CopyBlock]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"model did not return JSON: {raw[:200]!r}") from exc

    items = payload.get("blocks") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise GenerationError(f"model returned no blocks: {raw[:200]!r}")

    blocks: list[CopyBlock] = []
    for offset, item in enumerate(items):
        if not isinstance(item, dict):
            raise GenerationError(f"malformed block: {item!r}")
        kind = item.get("kind")
        text = item.get("text")
        if kind not in ("hook", "ad_copy", "positioning") or not isinstance(text, str):
            raise GenerationError(f"malformed block: {item!r}")
        blocks.append(
            CopyBlock(id=f"block_{start_index + offset}", kind=kind, text=text.strip())
        )
    return blocks


def generate(
    client: LLMClient,
    insight: Insight,
    evidence: Sequence[Signal],
    *,
    mix: Sequence[tuple[BlockKind, int]] = DEFAULT_MIX,
    direction: str | None = None,
    previous: Sequence[CopyBlock] | None = None,
) -> list[CopyBlock]:
    """Produce copy blocks. Unvalidated — the caller must not render these."""
    system = build_system_prompt(insight, evidence)
    user = (
        build_revision_prompt(previous, direction)
        if previous and direction
        else build_user_prompt(mix, direction)
    )
    response = client.complete(system=system, user=user, json_schema=BLOCKS_SCHEMA)
    return _parse_blocks(response.text)


def regenerate(
    client: LLMClient,
    insight: Insight,
    evidence: Sequence[Signal],
    block: CopyBlock,
    result: ValidationResult,
) -> CopyBlock:
    """Rewrite one rejected block, keeping its ID so the UI can slot it back in."""
    system = build_system_prompt(insight, evidence)
    user = build_regenerate_prompt(block, result)
    response = client.complete(system=system, user=user, json_schema=BLOCKS_SCHEMA)
    rewritten = _parse_blocks(response.text)[0]
    return CopyBlock(id=block.id, kind=block.kind, text=rewritten.text)
