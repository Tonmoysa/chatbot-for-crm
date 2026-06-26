"""LLM-off paths must not guess modify amounts — show clear unavailable message."""

from __future__ import annotations

from unittest.mock import patch

from chat.services.platform.field_extractors.expense import (
    _turn_has_actionable_patches,
    expense_turn_llm_blocked,
    expense_turn_to_field_updates,
    interpret_expense_draft_turn,
)
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.yaml_scenario_runner import llm_disabled


def _review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        pending_confirmation="submit",
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-25",
                    "items": [
                        {"category": "lunch", "amount": 100.0, "id": "a"},
                        {"category": "bike", "amount": 150.0, "id": "b"},
                    ],
                },
            )
        },
    )


def _single_lunch_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        pending_confirmation="submit",
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-25",
                    "items": [{"category": "lunch", "amount": 100.0, "id": "a"}],
                },
            )
        },
    )


def test_modify_review_with_patches_is_actionable():
    turn = {
        "intent": "modify_review",
        "item_patches": [{"action": "update", "category": "lunch", "amount": 120}],
    }
    assert _turn_has_actionable_patches(turn)


def test_correction_uses_llm_modify_review_patches():
    memory = _single_lunch_memory()
    msg = "lunch ta ami vule 100 taka diyechi ashole ota hobe 120 taka"
    llm_turn = {
        "intent": "modify_review",
        "item_patches": [
            {
                "action": "update",
                "category": "lunch",
                "match_amount": 100,
                "amount": 120,
            }
        ],
        "delete_indices": [],
        "reasoning": "User corrects lunch from 100 to 120.",
    }
    with patch("chat.services.llm_client.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = True
        mock_cls.return_value.chat_json.return_value = llm_turn
        turn = interpret_expense_draft_turn(msg, memory, trace_id="modify-review-llm")
    assert turn["intent"] == "modify_review"
    assert turn.get("llm_used") is True
    assert turn["item_patches"][0]["amount"] == 120


def test_correction_message_returns_llm_unavailable_when_llm_off():
    memory = _review_memory()
    msg = "lunch ta ami vule 100 taka diyechi ashole ota hobe 120 taka"
    with llm_disabled():
        turn, updates = expense_turn_to_field_updates(
            msg, memory, trace_id="llm-unavail-correction"
        )
    assert turn["intent"] == "llm_unavailable"
    assert updates == []
    assert expense_turn_llm_blocked(turn, memory)


def test_lunch_amount_unchanged_when_llm_off():
    memory = _review_memory()
    msg = "lunch ta ami vule 100 taka diyechi ashole ota hobe 120 taka"
    with llm_disabled():
        turn, updates = expense_turn_to_field_updates(
            msg, memory, trace_id="llm-unavail-amount"
        )
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields["items"][0]["amount"] == 100.0
    assert turn["intent"] == "llm_unavailable"
    assert not updates
