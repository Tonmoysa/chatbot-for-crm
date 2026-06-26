"""Regression: expense_turn must not leak patches across turns."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.field_extractors.expense import (
    empty_expense_turn,
    expense_entities_for_turn,
    is_expense_draft_mutation_message,
)
from chat.services.platform.intent_rules import (
    is_expense_navigation_message,
    is_resume_workflow_request,
    is_workflow_show_request,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import TargetRef, UnderstandingResult, UnderstandingAction
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def _four_item_review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-25",
                    "items": [
                        {"category": "lunch", "amount": 100.0, "id": "ab85197b"},
                        {"category": "bike", "amount": 150.0, "id": "c0f97ef2"},
                        {"category": "bus", "amount": 20.0, "id": "3ca2e912"},
                        {"category": "bus", "amount": 45.0, "id": "97223828"},
                    ],
                },
            )
        },
        last_entities={
            "expense_turn": {
                "intent": "add",
                "item_patches": [
                    {
                        "action": "append",
                        "item_index": 3,
                        "item_id": "97223828",
                        "match_amount": 45,
                        "category": "bus",
                        "amount": 35,
                    }
                ],
                "delete_indices": [],
            },
            "expense_intent": "add",
        },
    )


def test_delete_message_not_treated_as_navigation_or_show():
    msg = "expense e je lunch ta ache ashole ota delete kore daw"
    memory = _four_item_review_memory()
    assert is_expense_draft_mutation_message(msg, memory)
    assert not is_expense_navigation_message(msg)
    assert not is_workflow_show_request(msg, workflow_id="expense")
    assert not is_resume_workflow_request(msg, workflow_id="expense")


def test_expense_entities_for_turn_clears_stale_append_on_delete_action():
    stale = {
        "intent": "add",
        "item_patches": [{"action": "append", "category": "bus", "amount": 35}],
        "delete_indices": [],
    }
    entities = expense_entities_for_turn(
        {"expense_intent": "add"},
        stale,
        expense_intent="delete",
        action=UnderstandingAction.DELETE.value,
    )
    turn = entities["expense_turn"]
    assert turn["intent"] == "delete"
    assert turn["item_patches"] == []
    assert entities["expense_intent"] == "delete"


def test_empty_expense_turn_resets_stale_state():
    entities = expense_entities_for_turn({"expense_intent": "add", "expense_turn": {"intent": "add"}}, None)
    assert entities["expense_turn"] == empty_expense_turn()
    assert "expense_intent" not in entities


def test_lunch_delete_wins_over_stale_bus_append():
    memory = _four_item_review_memory()
    pipeline = WorkflowPipeline()
    engine = PendingQuestionEngine()
    msg = "expense e je lunch ta ache ashole ota delete kore daw"
    understanding = UnderstandingResult(
        goal="delete expense item",
        workflow="expense",
        action=UnderstandingAction.MODIFY.value,
        confidence=0.9,
        targets=[TargetRef(field="items", item_index=0)],
        reasoning="User wants to delete the lunch expense item.",
        source="llm",
        answers_pending_field=False,
    )
    decision = engine.classify(
        msg,
        memory=memory,
        trace_id="stale-turn-delete",
        conversation_history=[],
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.DELETE_DATA

    handle_with_rules_understanding(
        pipeline,
        msg,
        memory=memory,
        pq_decision=decision,
        trace_id="stale-turn-delete",
        route_source="active",
    )
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 3
    assert not any(it.get("category") == "lunch" for it in items)
    assert not any(it.get("amount") == 35 for it in items)
