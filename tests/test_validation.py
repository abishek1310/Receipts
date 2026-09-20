"""Tests for the validation core (CLAUDE.md §6.4).

No LLM calls (§14). Every `CopyBlock` here is constructed by hand, which is the
point: the validator must be judgeable without a model in the loop.
"""

from __future__ import annotations

import json

import pytest

from src.schema import CopyBlock, FailureReason
from src.validation import (
    extract_citations,
    is_claim_bearing,
    log_rejection,
    split_sentences,
    strip_citations,
    validate_block,
    validate_blocks,
)

# The corpus knows about these. Only the first three were retrieved this turn.
CORPUS = {f"sig_{i:04d}" for i in range(1, 50)}
EVIDENCE = {"sig_0001", "sig_0002", "sig_0003"}


def block(text: str, bid: str = "block_1", kind: str = "hook") -> CopyBlock:
    return CopyBlock(id=bid, kind=kind, text=text)


# ======================================================================================
# Clean pass
# ======================================================================================


def test_single_sentence_with_one_citation_passes():
    r = validate_block(block("Your gym glow shouldn't last three hours. [sig_0001]"), EVIDENCE, CORPUS)
    assert r.status == "PASS"
    assert r.cited_ids == ["sig_0001"]
    assert r.reason is None


def test_multiple_ids_in_one_tag_pass():
    r = validate_block(
        block("Redness outlasts the workout. [sig_0001, sig_0002, sig_0003]"), EVIDENCE, CORPUS
    )
    assert r.status == "PASS"
    assert r.cited_ids == ["sig_0001", "sig_0002", "sig_0003"]


def test_multi_sentence_block_all_cited_passes():
    text = (
        "Redness outlasts the workout. [sig_0001] "
        "Most cooling gels sting on broken skin. [sig_0002, sig_0003]"
    )
    r = validate_block(block(text), EVIDENCE, CORPUS)
    assert r.status == "PASS"
    assert set(r.cited_ids) == EVIDENCE


def test_hook_without_terminal_punctuation_passes():
    r = validate_block(block("Gym glow, not gym rash [sig_0002]"), EVIDENCE, CORPUS)
    assert r.status == "PASS"
    assert r.cited_ids == ["sig_0002"]


def test_repeated_id_reported_once():
    r = validate_block(block("A claim. [sig_0001] Another claim. [sig_0001]"), EVIDENCE, CORPUS)
    assert r.status == "PASS"
    assert r.cited_ids == ["sig_0001"]


# ======================================================================================
# 1. UNPARSEABLE — missing tag
# ======================================================================================


def test_missing_tag_fails_unparseable():
    r = validate_block(block("Your gym glow shouldn't last three hours."), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNPARSEABLE


def test_one_uncited_sentence_among_cited_ones_fails():
    text = (
        "Redness outlasts the workout. [sig_0001] "
        "Clinical studies prove it clears in ten minutes. "
        "Cooling gels sting. [sig_0002]"
    )
    r = validate_block(block(text), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNPARSEABLE
    assert any("Clinical studies" in s for s in r.offending_sentences)


def test_call_to_action_needs_no_citation():
    r = validate_block(block("Redness fades. [sig_0001] Shop now."), EVIDENCE, CORPUS)
    assert r.status == "PASS"


def test_call_to_action_with_a_claim_attached_still_needs_one():
    r = validate_block(
        block("Redness fades. [sig_0001] Shop now and clear your acne overnight."),
        EVIDENCE,
        CORPUS,
    )
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNPARSEABLE


# ======================================================================================
# 1. UNPARSEABLE — malformed tag (§6.1: anything not the exact shape)
# ======================================================================================


@pytest.mark.parametrize(
    "tag",
    [
        "[sig_1]",  # too few digits
        "[sig_00001]",  # too many digits
        "[signal_0001]",  # wrong prefix
        "[SIG_0001]",  # wrong case
        "[sig_0001; sig_0002]",  # wrong separator
        "[sig_0001 sig_0002]",  # no separator
        "[sig_0001,]",  # trailing comma
        "[0001]",  # bare number
        "[sig_0001, hallucinated]",  # mixed junk
    ],
)
def test_malformed_tags_fail_unparseable(tag):
    r = validate_block(block(f"A grounded claim. {tag}"), EVIDENCE, CORPUS)
    assert r.status == "FAIL", f"{tag} should not validate"
    assert r.reason is FailureReason.UNPARSEABLE


def test_malformed_tag_is_not_rescued_by_containing_a_valid_id():
    # "[sig_0001, hallucinated]" contains a real, in-set ID. It still fails: the
    # grammar is the contract, and a partially parseable tag is not parseable.
    r = validate_block(block("A claim. [sig_0001, hallucinated]"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNPARSEABLE


# ======================================================================================
# 2. UNKNOWN_ID — pure hallucination
# ======================================================================================


def test_hallucinated_id_fails_unknown_id():
    r = validate_block(block("Skin recovers in minutes. [sig_9999]"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNKNOWN_ID
    assert r.offending_ids == ["sig_9999"]


def test_unknown_id_beats_out_of_set_in_the_same_sentence():
    # sig_0042 is real but not retrieved; sig_9999 does not exist. §6.2 checks
    # UNKNOWN_ID first — pure fabrication is the more severe diagnosis.
    r = validate_block(block("A claim. [sig_9999, sig_0042]"), EVIDENCE, CORPUS)
    assert r.reason is FailureReason.UNKNOWN_ID
    assert r.offending_ids == ["sig_9999"]


def test_without_corpus_ids_everything_outside_evidence_is_out_of_set():
    # Degraded mode: still fails closed, just with a coarser diagnosis.
    r = validate_block(block("A claim. [sig_9999]"), EVIDENCE)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.OUT_OF_SET


# ======================================================================================
# 3. OUT_OF_SET — real ID, wrong generation (the subtle one, §6.2)
# ======================================================================================


def test_real_but_unretrieved_id_fails_out_of_set():
    r = validate_block(block("Texture matters more than price. [sig_0042]"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.OUT_OF_SET
    assert r.offending_ids == ["sig_0042"]


def test_mixing_one_out_of_set_id_into_a_good_tag_still_fails():
    r = validate_block(block("A claim. [sig_0001, sig_0042]"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.OUT_OF_SET
    assert r.offending_ids == ["sig_0042"]


def test_shrinking_the_evidence_set_turns_a_pass_into_out_of_set():
    # The same block, revalidated against a different turn's retrieval. This is
    # exactly the cross-turn leak §6.2 warns about.
    b = block("Redness outlasts the workout. [sig_0003]")
    assert validate_block(b, EVIDENCE, CORPUS).status == "PASS"
    assert validate_block(b, {"sig_0001"}, CORPUS).reason is FailureReason.OUT_OF_SET


# ======================================================================================
# 4. EMPTY_CITATION
# ======================================================================================


def test_empty_tag_fails_empty_citation():
    r = validate_block(block("Skin feels calmer. []"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.EMPTY_CITATION


def test_whitespace_only_tag_fails_empty_citation():
    r = validate_block(block("Skin feels calmer. [   ]"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.EMPTY_CITATION


def test_empty_tag_is_not_reported_as_missing_tag():
    # A present-but-empty tag is a different defect from no tag at all, and the
    # regenerate prompt says different things about each.
    r = validate_block(block("Skin feels calmer. []"), EVIDENCE, CORPUS)
    assert r.reason is not FailureReason.UNPARSEABLE


# ======================================================================================
# Priority ordering (§6.2)
# ======================================================================================


def test_unparseable_outranks_every_other_reason():
    text = "Uncited claim. A hallucination. [sig_9999] Out of set. [sig_0042] Empty. []"
    r = validate_block(block(text), EVIDENCE, CORPUS)
    assert r.reason is FailureReason.UNPARSEABLE


def test_out_of_set_outranks_empty_citation():
    r = validate_block(block("Out of set. [sig_0042] Empty. []"), EVIDENCE, CORPUS)
    assert r.reason is FailureReason.OUT_OF_SET


# ======================================================================================
# Tags on non-claim sentences are still checked
# ======================================================================================


def test_hallucinated_id_on_a_cta_still_fails():
    # The CTA did not *need* a tag, but it emitted one, and it is fabricated.
    r = validate_block(block("Redness fades. [sig_0001] Shop now. [sig_9999]"), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNKNOWN_ID


# ======================================================================================
# Multiple blocks — only the failing one is rejected (§6.3)
# ======================================================================================


def test_only_the_failing_block_is_marked_failed():
    blocks = [
        block("Redness outlasts the workout. [sig_0001]", "block_1"),
        block("Nothing on the shelf is built for this. [sig_9999]", "block_2"),
        block("Cooling gels sting. [sig_0002, sig_0003]", "block_3"),
    ]
    results = validate_blocks(blocks, EVIDENCE, CORPUS)
    assert [r.status for r in results] == ["PASS", "FAIL", "PASS"]
    assert results[1].reason is FailureReason.UNKNOWN_ID
    assert results[1].block_id == "block_2"


def test_block_ids_are_carried_through():
    results = validate_blocks([block("A claim.", "hook_7")], EVIDENCE, CORPUS)
    assert results[0].block_id == "hook_7"


# ======================================================================================
# Edge cases
# ======================================================================================


def test_empty_block_text_passes_vacuously():
    # Nothing asserted, nothing to ground. The graph treats an empty block as a
    # generation problem, not a citation problem.
    assert validate_block(block(""), EVIDENCE, CORPUS).status == "PASS"


def test_decimal_point_does_not_split_a_sentence():
    r = validate_block(block("Redness fades in 2.5 hours, not eight. [sig_0001]"), EVIDENCE, CORPUS)
    assert r.status == "PASS"


def test_ellipsis_and_exclamations_are_handled():
    r = validate_block(block("Wait for it... [sig_0001] It actually works! [sig_0002]"), EVIDENCE, CORPUS)
    assert r.status == "PASS"


def test_tag_before_the_sentence_does_not_count_as_citing_it():
    # A leading tag belongs to no sentence, so the claim after it is uncited.
    r = validate_block(block("[sig_0001] Redness outlasts the workout."), EVIDENCE, CORPUS)
    assert r.status == "FAIL"
    assert r.reason is FailureReason.UNPARSEABLE


# ======================================================================================
# Helpers
# ======================================================================================


def test_split_sentences_attaches_tags():
    got = split_sentences("One. [sig_0001] Two. [sig_0002]")
    assert [(s.body, s.tag) for s in got] == [
        ("One.", "[sig_0001]"),
        ("Two.", "[sig_0002]"),
    ]


def test_strip_citations_removes_tags_and_tidies_spacing():
    assert strip_citations("One. [sig_0001] Two. [sig_0002]") == "One. Two."


def test_extract_citations_preserves_order_and_dedupes():
    assert extract_citations("A [sig_0002] B [sig_0001] C [sig_0002]") == ["sig_0002", "sig_0001"]


@pytest.mark.parametrize("text,expected", [
    ("Shop now.", False),
    ("Link in bio!", False),
    ("Hook:", False),
    ("   ", False),
    ("...", False),
    ("Redness outlasts the workout.", True),
    ("Shop now for calmer skin.", True),
])
def test_is_claim_bearing(text, expected):
    assert is_claim_bearing(text) is expected


# ======================================================================================
# Rejection logging (§6.3)
# ======================================================================================


def test_log_rejection_appends_a_readable_record(tmp_path):
    path = tmp_path / "rejections.jsonl"
    b = block("Skin recovers in minutes. [sig_9999]")
    result = validate_block(b, EVIDENCE, CORPUS)

    log_rejection(result, b, insight_id="ins_01", attempt=2, evidence_set=EVIDENCE, path=path)
    log_rejection(result, b, insight_id="ins_01", attempt=3, evidence_set=EVIDENCE, path=path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["reason"] == "UNKNOWN_ID"
    assert record["offending_ids"] == ["sig_9999"]
    assert record["insight_id"] == "ins_01"
    assert record["block_text"] == b.text
    assert sorted(record["evidence_set"]) == sorted(EVIDENCE)
