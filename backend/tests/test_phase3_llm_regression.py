"""Phase 3 — optional LLM-on regression (RUN_LLM_TESTS=1 + LLM_API_KEY)."""

from __future__ import annotations

import pytest

from tests.helpers.yaml_scenario_runner import (
    llm_tests_enabled,
    load_paraphrase_cases,
    load_yaml_scenarios,
    run_paraphrase_case,
    run_yaml_scenario,
)

pytestmark = [
    pytest.mark.llm,
    pytest.mark.django_db,
    pytest.mark.skipif(not llm_tests_enabled(), reason="Set RUN_LLM_TESTS=1 and LLM_API_KEY"),
]

# Critical subset — full paraphrase suite is slow/costly with live LLM.
_LLM_PARAPHRASE_IDS = {
    "show_draft_during_pending_reason__1",
    "show_draft_during_pending_reason__8",
    "show_draft_during_pending_reason__19",
    "meta_complaint_not_generic_leave_start__1",
    "slot_answer_reason_pending__1",
}

_LLM_TRANSCRIPT_IDS = {
    "leave_transcript_parallel_block_then_summary",
    "leave_transcript_meta_complaint_not_restart",
    "leave_transcript_where_is_leave",
}


@pytest.mark.parametrize(
    "case_id",
    sorted(_LLM_PARAPHRASE_IDS),
    ids=sorted(_LLM_PARAPHRASE_IDS),
)
def test_llm_paraphrase_critical(case_id: str):
    cases = {c["id"]: c for c in load_paraphrase_cases()}
    assert case_id in cases
    run_paraphrase_case(cases[case_id], use_llm=True)


@pytest.mark.parametrize(
    "scenario_id",
    sorted(_LLM_TRANSCRIPT_IDS),
    ids=sorted(_LLM_TRANSCRIPT_IDS),
)
def test_llm_transcript_critical(scenario_id: str):
    scenarios = {s["id"]: s for s in load_yaml_scenarios()}
    assert scenario_id in scenarios
    run_yaml_scenario(scenarios[scenario_id], use_llm=True)
