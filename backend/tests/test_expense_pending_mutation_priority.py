"""Regression: pending_question must not hijack delete/modify/show intents."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.field_extractors.expense import (
    expense_turn_to_field_updates,
    is_expense_draft_mutation_message,
    is_expense_pending_field_value_answer,
    normalize_expense_mutation_turn,
)
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)
from tests.helpers.yaml_scenario_runner import llm_disabled


def _pending_bus_route_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Expense 3 — Bus — 20.0 taka: where did you travel from and to?",
            workflow_id="expense",
            asked_at_turn=11,
            item_index=2,
        ),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-25",
                    "items": [
                        {"category": "lunch", "amount": 100.0, "id": "ab85197b"},
                        {"category": "bike", "amount": 150.0, "id": "c0f97ef2"},
                        {"category": "bus", "amount": 20.0, "id": "3ca2e912", "missing_fields": ["route"]},
                        {"category": "bus", "amount": 45.0, "id": "97223828", "missing_fields": ["route"]},
                    ],
                },
            )
        },
    )


def test_pending_route_plus_bike_regret_delete_bypasses_pending():
    memory = _pending_bus_route_memory()
    msg = "bike ta shole ar lagbe nah..eta ami vule add dyechilam"
    assert is_expense_draft_mutation_message(msg, memory)
    assert not is_expense_pending_field_value_answer(msg, memory)

    wrong_llm = {
        "intent": "clarify_modify",
        "item_patches": [],
        "clarify": {
            "kind": "which_item",
            "candidate_indices": [2, 3],
            "category": "bus",
            "field": "route",
        },
    }
    fixed = normalize_expense_mutation_turn(wrong_llm, msg, memory)
    assert fixed["intent"] == "delete"
    assert fixed["delete_indices"] == [1]
    assert fixed["item_patches"][0]["item_index"] == 1

    with llm_disabled():
        turn, updates = expense_turn_to_field_updates(msg, memory, trace_id="pending-delete-bike")
    assert turn["intent"] == "delete"
    assert turn["delete_indices"] == [1]
    assert updates == []

    engine = PendingQuestionEngine()
    understanding = FieldEngine().ground_expense_understanding(
        msg,
        UnderstandingResult(
            goal="delete bike expense",
            workflow="expense",
            action=UnderstandingAction.DELETE.value,
            confidence=0.9,
            entities={
                "expense_intent": "delete",
                "expense_turn": turn,
            },
            answers_pending_field=False,
            source="rules",
        ),
        memory=memory,
        trace_id="pending-delete-bike",
    )
    decision = engine.classify(
        msg,
        memory=memory,
        trace_id="pending-delete-bike",
        conversation_history=[],
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.DELETE_DATA


def test_pending_route_plus_modify_lunch_not_slot_answer():
    memory = _pending_bus_route_memory()
    msg = "lunch ta 120 taka hobe"
    assert is_expense_draft_mutation_message(msg, memory)
    assert not is_expense_pending_field_value_answer(msg, memory)

    wrong_llm = {
        "intent": "answer_pending",
        "item_patches": [
            {"action": "update", "item_index": 2, "from_location": "x", "to_location": "y"},
        ],
    }
    fixed = normalize_expense_mutation_turn(wrong_llm, msg, memory)
    assert fixed["intent"] in ("modify_review", "update")
    assert fixed["item_patches"][0]["item_index"] == 0


def test_pending_route_plus_show_summary_wins_over_slot():
    memory = _pending_bus_route_memory()
    msg = "expense er list daw"
    assert not is_expense_draft_mutation_message(msg, memory)
    assert not is_expense_pending_field_value_answer(msg, memory)

    understanding = UnderstandingResult(
        goal="review expense",
        workflow="expense",
        action=UnderstandingAction.REVIEW.value,
        confidence=1.0,
        entities={"expense_intent": "show_summary"},
        interrupt_workflow="expense",
        answers_pending_field=False,
        source="llm",
    )
    decision = PendingQuestionEngine().classify(
        msg,
        memory=memory,
        trace_id="pending-show-summary",
        conversation_history=[],
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.SHOW_REVIEW
