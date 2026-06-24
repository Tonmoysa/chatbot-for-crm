"""Phase 3 — YAML paraphrase regression (rules path, many synonyms per intent)."""

from __future__ import annotations

import pytest

from tests.helpers.yaml_scenario_runner import load_paraphrase_cases, run_paraphrase_case

_CASES = load_paraphrase_cases()
assert _CASES, "expected paraphrase_groups in tests/scenarios/*.yaml"

_PARAMS = [
    pytest.param(case, id=case["id"], marks=[pytest.mark.django_db, pytest.mark.paraphrase])
    for case in _CASES
]


@pytest.mark.parametrize("case", _PARAMS)
def test_paraphrase_case_rules_path(case):
  run_paraphrase_case(case, use_llm=False)


def test_paraphrase_catalog_minimum_coverage():
    """Guard — paraphrase groups cover navigation, slot, meta, and commands."""
    by_group: dict[str, int] = {}
    for case in _CASES:
        by_group[case["group_id"]] = by_group.get(case["group_id"], 0) + 1
    assert by_group.get("show_draft_during_pending_reason", 0) >= 15
    assert by_group.get("slot_answer_reason_pending", 0) >= 5
    assert by_group.get("meta_complaint_not_generic_leave_start", 0) >= 4
    assert by_group.get("bare_modify_not_reason_value", 0) >= 2
