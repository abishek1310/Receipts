"""Receipts — Streamlit chat UI (CLAUDE.md §10).

Standard Streamlit chrome. No custom CSS, no dashboard styling (§13) — nobody
scores pixel fidelity in a four-minute demo, and the thing worth looking at is
the citation expander, not the theme.

Rejected blocks render with a visible ❌ and the machine-readable reason. Hiding
them would hide the differentiator.
"""

from __future__ import annotations

import streamlit as st

from src.config import get_settings, load_streamlit_secrets
from src.corpus import load_corpus, load_insights
from src.graph import plan_turn, run_turn
from src.llm import LLMError, get_client
from src.schema import BlockOutcome, GenerationResult, Insight

st.set_page_config(page_title="Receipts", page_icon="📎", layout="centered")

load_streamlit_secrets()

KIND_LABEL = {"hook": "TikTok hook", "ad_copy": "Ad copy", "positioning": "Positioning"}


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("history", [])  # list of (role, payload)
    st.session_state.setdefault("evidence_set", [])
    st.session_state.setdefault("previous_blocks", [])
    st.session_state.setdefault("generated", 0)
    st.session_state.setdefault("rejected", 0)
    st.session_state.setdefault("insight_id", None)


def _reset_for_insight(insight_id: str) -> None:
    """A new insight is a new brief — nothing carries over, least of all evidence."""
    st.session_state.history = []
    st.session_state.evidence_set = []
    st.session_state.previous_blocks = []
    st.session_state.insight_id = insight_id


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_outcome(outcome: BlockOutcome, key: str) -> None:
    label = KIND_LABEL.get(outcome.block.kind, outcome.block.kind)

    if outcome.unsupported:
        st.markdown(f"**{label}** ❌ `{outcome.result.reason.value}`")
        st.markdown(f"> {outcome.display_text or '_(no copy survived validation)_'}")
        st.caption(f"Rejected — {outcome.result.detail}")
        st.caption(
            "Regenerated twice and still unsupported, so it is shown rather than "
            "quietly dropped."
        )
        return

    st.markdown(f"**{label}**")
    st.markdown(f"> {outcome.display_text}")

    n = len(outcome.supporting)
    with st.expander(f"📎 {n} supporting signal{'s' if n != 1 else ''}", expanded=False):
        for sig in outcome.supporting:
            st.markdown(f"“{sig.text}”")
            st.caption(
                f"`{sig.id}` · {sig.source_detail} · {sig.timestamp} · "
                f"[open source]({sig.url})"
            )
            st.divider()


def render_result(result: GenerationResult, key: str) -> None:
    if result.error:
        st.error(f"Generation failed: {result.error}")
        return

    note = (
        f"Retrieved {len(result.evidence_set)} signals — {result.evidence_note}"
        if result.retrieved_fresh
        else f"Reusing the same {len(result.evidence_set)} signals — {result.evidence_note}"
    )
    st.caption(note)

    for i, outcome in enumerate(result.outcomes):
        render_outcome(outcome, key=f"{key}_{i}")

    if result.n_rejected:
        st.warning(
            f"{result.n_rejected} of {result.n_generated} blocks failed citation "
            "validation and are shown unsupported."
        )


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def main() -> None:
    _init_state()

    try:
        corpus = load_corpus()
        insights = load_insights()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
        return

    settings = get_settings()

    with st.sidebar:
        st.header("Receipts")
        st.caption("Marketing copy where the model never writes its own citations.")

        options = {f"{i.id} — {i.statement[:60]}…": i for i in insights}
        chosen_label = st.selectbox("Insight", list(options))
        insight: Insight = options[chosen_label]

        if st.session_state.insight_id != insight.id:
            _reset_for_insight(insight.id)

        st.divider()
        st.markdown(f"**Audience** {insight.audience}")
        st.markdown(f"**Themes** {', '.join(insight.themes)}")
        st.markdown(f"**Momentum** {insight.momentum}")

        st.divider()
        st.metric("Signals in corpus", len(corpus))
        col_a, col_b = st.columns(2)
        col_a.metric("Blocks generated", st.session_state.generated)
        col_b.metric("Blocks rejected", st.session_state.rejected)

        st.divider()
        st.caption(f"Model: `{settings.model}` · provider: `{settings.llm_provider}`")
        if not settings.configured:
            st.warning("No API key configured. Set it in `.env` or `st.secrets`.")
        st.caption(
            "Built to consume CREWASIS signals, demonstrated on a representative "
            "public dataset."
        )

    st.title("📎 Receipts")
    st.caption(
        "Every claim is bound to a retrieved consumer signal. Citations are assigned "
        "and verified in Python — the model's tags are checked, never trusted."
    )

    for i, (role, payload) in enumerate(st.session_state.history):
        with st.chat_message(role):
            if role == "user":
                st.markdown(payload)
            else:
                render_result(payload, key=f"h{i}")

    placeholder = (
        "Write copy for this insight"
        if not st.session_state.evidence_set
        else "Revise it — softer, more clinical, or ask about a new angle"
    )
    prompt = st.chat_input(placeholder)
    if not prompt:
        return

    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        plan = plan_turn(prompt, insight, bool(st.session_state.evidence_set))
        spinner = (
            "Revising against the frozen evidence set…"
            if plan.reuse_evidence
            else "Retrieving signals and generating…"
        )
        with st.spinner(spinner):
            try:
                client = get_client(settings)
            except LLMError as exc:
                st.error(str(exc))
                st.session_state.history.pop()
                return

            result = run_turn(
                insight,
                client=client,
                user_message=prompt,
                evidence_set=st.session_state.evidence_set or None,
                previous_blocks=st.session_state.previous_blocks or None,
                corpus=corpus,
                settings=settings,
            )

        render_result(result, key=f"live{len(st.session_state.history)}")

    st.session_state.history.append(("assistant", result))
    if result.evidence_set:
        st.session_state.evidence_set = result.evidence_set
    if result.outcomes:
        st.session_state.previous_blocks = [o.block for o in result.outcomes]
    st.session_state.generated += result.n_generated
    st.session_state.rejected += result.n_rejected

    # The sidebar was rendered at the top of this run, before the turn existed, so
    # the §10 tally would show the previous turn's totals — it read "0 generated"
    # immediately after producing six blocks. Re-run so the counts match what is
    # on screen. Replaying history is free: the result is already in state and no
    # LLM call is repeated.
    st.rerun()


main()
