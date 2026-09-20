"""The LangGraph state machine (CLAUDE.md §9).

    retrieve -> generate -> validate -> (regenerate, max 2) -> respond

Two things in here carry real weight:

**The retry loop (§6.3).** A failing block is regenerated on its own — not the
whole response — and after the retry budget is spent it is surfaced to the user
marked UNSUPPORTED rather than dropped. Visible refusal beats silence. Every
rejection is logged on the way past, including ones a later retry recovered from,
because §11 needs naturally occurring failures and a recovered failure is still a
real one.

**Evidence-preserving revision (§9).** "Make it softer" reuses the evidence set
frozen in state; "what about price?" retrieves fresh and says so. `plan_turn()`
is where that decision is made, and it is made in Python from the user's wording
rather than by asking the model what it thinks it should do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.corpus import Corpus, load_corpus
from src.generation import DEFAULT_MIX, GenerationError, generate, regenerate
from src.llm import LLMClient, LLMError
from src.retrieval import evidence_ids, query_themes, retrieve
from src.schema import (
    BlockOutcome,
    CopyBlock,
    GenerationResult,
    Insight,
    Signal,
    ValidationResult,
)
from src.validation import log_rejection, validate_block, validate_blocks


class GraphState(TypedDict, total=False):
    """State as specified in §9, plus the wiring the nodes need."""

    insight: Insight
    messages: list[dict[str, str]]
    evidence_set: list[Signal]
    blocks: list[CopyBlock]
    failures: list[ValidationResult]
    attempt: int

    # --- wiring, not part of the §9 contract ---------------------------------------
    direction: str | None
    previous_blocks: list[CopyBlock]
    reuse_evidence: bool
    retrieved_fresh: bool
    outcomes: list[BlockOutcome]
    error: str | None
    _deps: "Deps"


@dataclass(frozen=True)
class Deps:
    """Everything the nodes call out to. Injected so tests can pass a stub client."""

    client: LLMClient
    corpus: Corpus
    settings: Settings
    log_path: Any = None


# --------------------------------------------------------------------------------------
# Turn planning — the §9 revision decision
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnPlan:
    reuse_evidence: bool
    reason: str
    new_themes: list[str]


def plan_turn(
    user_message: str | None,
    insight: Insight,
    has_evidence: bool,
) -> TurnPlan:
    """Decide whether this turn reuses the frozen evidence set or retrieves again.

    A tone or register change ("softer", "more clinical", "shorter") is a revision:
    the claims stay the same, so the evidence behind them must too. A new angle
    ("what about price?") asks about something the current evidence set may not
    cover, so it retrieves fresh — and the UI says so, because silently swapping
    the evidence under a revision is exactly the kind of thing this project exists
    to make visible.
    """
    if not has_evidence or not user_message:
        return TurnPlan(False, "first turn for this insight", [])

    named = [t for t in query_themes(user_message) if t not in insight.themes]
    if named:
        return TurnPlan(
            False,
            f"you asked about a new angle ({', '.join(named)}), so I retrieved fresh signals",
            named,
        )
    return TurnPlan(True, "revised against the same evidence set", [])


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------


def _retrieve(state: GraphState) -> dict:
    deps = state["_deps"]
    if state.get("reuse_evidence") and state.get("evidence_set"):
        return {"retrieved_fresh": False}
    signals = retrieve(
        state["insight"],
        state.get("direction"),
        k=deps.settings.retrieval_k,
        corpus=deps.corpus,
    )
    return {"evidence_set": signals, "retrieved_fresh": True}


def _generate(state: GraphState) -> dict:
    deps = state["_deps"]
    try:
        blocks = generate(
            deps.client,
            state["insight"],
            state["evidence_set"],
            mix=DEFAULT_MIX,
            direction=state.get("direction"),
            previous=state.get("previous_blocks") or None,
        )
    except (LLMError, GenerationError) as exc:
        return {"blocks": [], "error": str(exc)}
    return {"blocks": blocks, "attempt": 1, "error": None}


def _validate(state: GraphState) -> dict:
    deps = state["_deps"]
    blocks = state.get("blocks") or []
    if not blocks:
        return {"failures": []}

    evidence = evidence_ids(state["evidence_set"])
    results = validate_blocks(blocks, evidence, deps.corpus.ids)
    by_id = {b.id: b for b in blocks}
    attempt = state.get("attempt", 1)

    failures = []
    for result in results:
        if result.passed:
            continue
        failures.append(result)
        log_rejection(
            result,
            by_id[result.block_id],
            insight_id=state["insight"].id,
            attempt=attempt,
            evidence_set=evidence,
            path=deps.log_path,
        )
    return {"failures": failures}


def _regenerate(state: GraphState) -> dict:
    """Rewrite only the blocks that failed (§6.3), then re-validate just those."""
    deps = state["_deps"]
    evidence = evidence_ids(state["evidence_set"])
    by_id = {b.id: b for b in state["blocks"]}
    attempt = state.get("attempt", 1) + 1

    still_failing: list[ValidationResult] = []
    for failure in state["failures"]:
        original = by_id[failure.block_id]
        try:
            rewritten = regenerate(
                deps.client, state["insight"], state["evidence_set"], original, failure
            )
        except (LLMError, GenerationError):
            # A provider hiccup during a retry is not a reason to accept the
            # block — keep the original failure and let the budget run out.
            still_failing.append(failure)
            continue

        by_id[rewritten.id] = rewritten
        result = validate_block(rewritten, evidence, deps.corpus.ids)
        if not result.passed:
            still_failing.append(result)
            log_rejection(
                result,
                rewritten,
                insight_id=state["insight"].id,
                attempt=attempt,
                evidence_set=evidence,
                path=deps.log_path,
            )

    blocks = [by_id[b.id] for b in state["blocks"]]
    return {"blocks": blocks, "failures": still_failing, "attempt": attempt}


def _should_retry(state: GraphState) -> str:
    """Loop while something still fails and the budget (§6.3: max 2) holds."""
    deps = state["_deps"]
    if not state.get("failures"):
        return "respond"
    if state.get("attempt", 1) > deps.settings.max_retries:
        return "respond"
    return "regenerate"


def _respond(state: GraphState) -> dict:
    """Assemble what the UI renders: every block, passing or not."""
    deps = state["_deps"]
    blocks = state.get("blocks") or []
    evidence = evidence_ids(state["evidence_set"]) if state.get("evidence_set") else set()
    results = validate_blocks(blocks, evidence, deps.corpus.ids)

    outcomes = []
    for block, result in zip(blocks, results):
        outcomes.append(
            BlockOutcome(
                block=block,
                result=result,
                supporting=deps.corpus.resolve(result.cited_ids) if result.passed else [],
                unsupported=not result.passed,
            )
        )
    return {"outcomes": outcomes}


def build_graph() -> Any:
    g = StateGraph(GraphState)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("validate", _validate)
    g.add_node("regenerate", _regenerate)
    g.add_node("respond", _respond)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate", _should_retry, {"regenerate": "regenerate", "respond": "respond"}
    )
    g.add_conditional_edges(
        "regenerate", _should_retry, {"regenerate": "regenerate", "respond": "respond"}
    )
    g.add_edge("respond", END)
    return g.compile()


_GRAPH = None


def get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_turn(
    insight: Insight,
    *,
    client: LLMClient,
    user_message: str | None = None,
    evidence_set: Sequence[Signal] | None = None,
    previous_blocks: Sequence[CopyBlock] | None = None,
    corpus: Corpus | None = None,
    settings: Settings | None = None,
    log_path: Any = None,
) -> GenerationResult:
    """Run one conversational turn end to end.

    Pass `evidence_set` and `previous_blocks` from the last turn to enable
    evidence-preserving revision; omit them for a fresh brief.
    """
    settings = settings or get_settings()
    corpus = corpus or load_corpus()
    plan = plan_turn(user_message, insight, bool(evidence_set))

    state: GraphState = {
        "insight": insight,
        "messages": [],
        "evidence_set": list(evidence_set or []),
        "blocks": [],
        "failures": [],
        "attempt": 0,
        "direction": user_message,
        "previous_blocks": list(previous_blocks or []),
        "reuse_evidence": plan.reuse_evidence,
        "retrieved_fresh": not plan.reuse_evidence,
        "outcomes": [],
        "error": None,
        "_deps": Deps(client=client, corpus=corpus, settings=settings, log_path=log_path),
    }

    final = get_graph().invoke(state)

    return GenerationResult(
        insight_id=insight.id,
        outcomes=final.get("outcomes") or [],
        evidence_set=final.get("evidence_set") or [],
        attempts=final.get("attempt", 1),
        retrieved_fresh=final.get("retrieved_fresh", True),
        error=final.get("error"),
        evidence_note=plan.reason,
    )
