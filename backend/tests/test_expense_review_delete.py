"""Expense review delete — delete_indices, numbered items, reducer."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import (
    _normalize_expense_delete_turn,
    patches_to_field_updates,
    sanitize_expense_review_updates,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import FieldUpdate
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def _five_item_review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 140.0},
                        {"category": "metro", "amount": 90.0},
                        {"amount": 150.0, "category": "bus"},
                        {"category": "bus", "amount": 40.0},
                    ],
                },
            )
        },
        pending_confirmation="submit",
    )


def test_normalize_delete_index_four_number():
    memory = _five_item_review_memory()
    turn = _normalize_expense_delete_turn(
        {"intent": "delete", "delete_indices": [0], "item_patches": []},
        "4 number expense ta delete koro",
        memory,
    )
    assert turn["delete_indices"] == [3]
    assert turn["item_patches"][0]["item_index"] == 3


def test_delete_indices_produce_field_updates():
    memory = _five_item_review_memory()
    fields = dict(memory.active_draft().fields)
    turn = {
        "intent": "delete",
        "delete_indices": [3],
        "item_patches": [{"action": "delete", "item_index": 3}],
    }
    updates = patches_to_field_updates(fields, turn)
    assert len(updates) == 1
    assert updates[0].action == "delete"
    assert updates[0].item_index == 3


def test_sanitize_review_allows_delete():
    memory = _five_item_review_memory()
    updates = sanitize_expense_review_updates(
        [
            FieldUpdate(field="items", value={}, item_index=0, action="delete"),
        ],
        "4 number expense ta delete koro",
        memory=memory,
    )
    assert len(updates) == 1
    assert updates[0].item_index == 3
    assert updates[0].action == "delete"


def test_review_delete_via_pipeline():
    memory = _five_item_review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.DELETE_DATA,
        confidence=0.9,
        reasoning="delete item 4",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    handle_with_rules_understanding(
        pipeline,
        "4 number expense ta delete koro",
        memory=memory,
        pq_decision=pq,
        trace_id="review-delete-4",
        route_source="active",
    )
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 4
    assert items[-1]["amount"] == 40.0


def test_llm_delete_indices_only_turn():
    memory = _five_item_review_memory()
    fields = dict(memory.active_draft().fields)
    turn = _normalize_expense_delete_turn(
        {
            "intent": "delete",
            "delete_indices": [0],
            "item_patches": [],
            "reasoning": "wrong index from mock",
        },
        "4 number expense ta delete koro",
        memory,
    )
    updates = patches_to_field_updates(fields, turn)
    assert updates and updates[0].item_index == 3
