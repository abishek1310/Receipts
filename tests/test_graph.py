"""Graph tests: the retry loop and evidence-preserving revision (§6.3, §6.4, §9).

No LLM calls (§14) — `ScriptedClient` replays canned JSON, so the retry budget,
the rejection log and the revision decision are all exercised deterministically.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.config import Settings
from src.corpus import corpus_from_signals
from src.graph import plan_turn, run_turn
from src.llm import ScriptedClient
from src.schema import FailureReason, Insight, Signal


def signal(n: int, theme: str = "redness") -> Signal:
    return Signal(
        id=f"sig_{n:04d}",
        text=f"A real consumer comment number {n} about {theme} that is long enough to read.",
        source="reddit",
        source_detail="r/SkincareAddiction",
        url=f"https://example.invalid/{n}",
        timestamp=date(2022, 6, 1),
        theme=theme,
    )


# 1-8 are redness, 9-12 are price. Retrieval will pick redness for this insight.
CORPUS = corpus_from_signals(
    [signal(n) for n in range(1, 9)] + [signal(n, "price") for n in range(9, 13)]
)

INSIGHT = Insight(
    id="ins_test",
    statement="Post-workout redness outlasts the session",
    category="athletic skincare",
    audience="Gen Z athletes",
    themes=["redness"],
    momentum="rising",
)

SETTINGS = Settings(
    anthropic_api_key="not-used", retrieval_k=8, max_retries=2, llm_provider="anthropic"
)


def response(*blocks: tuple[str, str]) -> str:
    return json.dumps({"blocks": [{"kind": k, "text": t} for k, t in blocks]})


def run(client, tmp_path, **kw):
    return run_turn(
        INSIGHT,
        client=client,
        corpus=CORPUS,
        settings=SETTINGS,
        log_path=tmp_path / "rejections.jsonl",
        **kw,
    )


# ======================================================================================
# Happy path
# ======================================================================================


def test_all_blocks_pass_on_the_first_attempt(tmp_path):
    client = ScriptedClient([
        response(
            ("hook", "Your gym glow shouldn't last three hours. [sig_0001]"),
            ("ad_copy", "Redness that outstays the workout. [sig_0002, sig_0003]"),
        )
    ])
    result = run(client, tmp_path)

    assert result.n_generated == 2
    assert result.n_rejected == 0
    assert all(o.result.passed for o in result.outcomes)
    assert not (tmp_path / "rejections.jsonl").exists()


def test_passing_blocks_resolve_their_supporting_signals(tmp_path):
    client = ScriptedClient([
        response(("hook", "Redness outlasts the workout. [sig_0001, sig_0002]"))
    ])
    outcome = run(client, tmp_path).outcomes[0]

    assert [s.id for s in outcome.supporting] == ["sig_0001", "sig_0002"]
    assert outcome.display_text == "Redness outlasts the workout."


# ======================================================================================
# Retry loop (§6.3)
# ======================================================================================


def test_a_failing_block_is_regenerated_and_can_recover(tmp_path):
    client = ScriptedClient([
        response(
            ("hook", "Good claim. [sig_0001]"),
            ("ad_copy", "Bad claim. [sig_9999]"),
        ),
        response(("ad_copy", "Corrected claim. [sig_0002]")),
    ])
    result = run(client, tmp_path)

    assert result.n_rejected == 0
    assert result.outcomes[1].block.text == "Corrected claim. [sig_0002]"
    assert result.attempts == 2


def test_only_the_failing_block_is_regenerated(tmp_path):
    client = ScriptedClient([
        response(
            ("hook", "Good claim. [sig_0001]"),
            ("ad_copy", "Bad claim. [sig_9999]"),
        ),
        response(("ad_copy", "Corrected claim. [sig_0002]")),
    ])
    result = run(client, tmp_path)

    # The passing block is untouched, and only one regeneration call was made.
    assert result.outcomes[0].block.text == "Good claim. [sig_0001]"
    assert len(client.calls) == 2


def test_retry_exhaustion_surfaces_the_block_as_unsupported(tmp_path):
    # Fails on the first attempt and on both retries — §6.3 says surface it, not drop it.
    bad = response(("hook", "Skin recovers in minutes. [sig_9999]"))
    client = ScriptedClient([bad, bad, bad])
    result = run(client, tmp_path)

    assert result.attempts == 3  # first attempt + 2 retries
    assert result.n_rejected == 1
    outcome = result.outcomes[0]
    assert outcome.unsupported is True
    assert outcome.result.reason is FailureReason.UNKNOWN_ID
    assert outcome.supporting == []


def test_a_failed_block_is_never_dropped_from_the_response(tmp_path):
    bad = response(
        ("hook", "Grounded. [sig_0001]"),
        ("ad_copy", "Fabricated. [sig_9999]"),
    )
    client = ScriptedClient([bad, response(("ad_copy", "Still fabricated. [sig_9999]")),
                             response(("ad_copy", "Still fabricated. [sig_9999]"))])
    result = run(client, tmp_path)

    assert result.n_generated == 2
    assert [o.unsupported for o in result.outcomes] == [False, True]


def test_out_of_set_id_is_caught_end_to_end(tmp_path):
    # sig_0012 is real (it is in CORPUS) but themed "price", so retrieval for this
    # redness insight never returned it. The model reached outside its context.
    bad = response(("hook", "Price is the real barrier. [sig_0012]"))
    client = ScriptedClient([bad, bad, bad])
    result = run(client, tmp_path)

    assert result.outcomes[0].result.reason is FailureReason.OUT_OF_SET
    assert result.outcomes[0].result.offending_ids == ["sig_0012"]


# ======================================================================================
# Rejection logging (§6.3, §11)
# ======================================================================================


def test_every_rejection_is_logged_including_recovered_ones(tmp_path):
    log = tmp_path / "rejections.jsonl"
    client = ScriptedClient([
        response(("hook", "Bad claim. [sig_9999]")),
        response(("hook", "Corrected claim. [sig_0001]")),
    ])
    result = run(client, tmp_path)

    assert result.n_rejected == 0  # it recovered...
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1  # ...but the rejection is still on the record
    assert records[0]["reason"] == "UNKNOWN_ID"
    assert records[0]["insight_id"] == "ins_test"
    assert records[0]["attempt"] == 1


def test_each_failed_attempt_gets_its_own_log_line(tmp_path):
    log = tmp_path / "rejections.jsonl"
    bad = response(("hook", "Fabricated. [sig_9999]"))
    run(ScriptedClient([bad, bad, bad]), tmp_path)

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["attempt"] for r in records] == [1, 2, 3]


# ======================================================================================
# Evidence-preserving revision (§9)
# ======================================================================================


@pytest.mark.parametrize(
    "message", ["make it softer", "more clinical please", "shorter", "punchier"]
)
def test_tone_changes_reuse_the_evidence_set(message):
    plan = plan_turn(message, INSIGHT, has_evidence=True)
    assert plan.reuse_evidence is True


@pytest.mark.parametrize(
    "message", ["what about price?", "focus on fragrance", "talk about sunscreen"]
)
def test_new_angles_trigger_a_fresh_retrieval(message):
    plan = plan_turn(message, INSIGHT, has_evidence=True)
    assert plan.reuse_evidence is False
    assert plan.new_themes


def test_the_first_turn_always_retrieves():
    assert plan_turn(None, INSIGHT, has_evidence=False).reuse_evidence is False


def test_a_revision_validates_against_the_frozen_set_not_a_new_one(tmp_path):
    first = run(
        ScriptedClient([response(("hook", "Redness lingers. [sig_0001]"))]), tmp_path
    )
    frozen = first.evidence_set

    revised = run(
        ScriptedClient([response(("hook", "Redness, gently put. [sig_0002]"))]),
        tmp_path,
        user_message="make it softer",
        evidence_set=frozen,
        previous_blocks=[o.block for o in first.outcomes],
    )

    assert revised.retrieved_fresh is False
    assert [s.id for s in revised.evidence_set] == [s.id for s in frozen]
    assert revised.n_rejected == 0


def test_an_angle_change_reports_that_it_re_retrieved(tmp_path):
    first = run(
        ScriptedClient([response(("hook", "Redness lingers. [sig_0001]"))]), tmp_path
    )
    second = run(
        ScriptedClient([response(("hook", "Worth the money. [sig_0009]"))]),
        tmp_path,
        user_message="what about price?",
        evidence_set=first.evidence_set,
        previous_blocks=[o.block for o in first.outcomes],
    )

    assert second.retrieved_fresh is True
    assert "price" in second.evidence_note
    assert any(s.theme == "price" for s in second.evidence_set)


# ======================================================================================
# Provider failure is an error, not a silent pass
# ======================================================================================


def test_a_provider_failure_surfaces_as_an_error_with_no_blocks(tmp_path):
    result = run(ScriptedClient([]), tmp_path)  # exhausted immediately
    assert result.outcomes == []
    assert result.error and "exhausted" in result.error


def test_malformed_model_output_does_not_reach_the_user(tmp_path):
    result = run(ScriptedClient(["this is not json"]), tmp_path)
    assert result.outcomes == []
    assert result.error
