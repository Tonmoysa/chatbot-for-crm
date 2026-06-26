"""Intent priority: actionable commands beat open pending_question slots."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.field_extractors.expense import (
    expense_pending_interrupted_by_updates,
    expense_turn_to_field_updates,
    normalize_expense_clarify_turn,
)
from chat.services.platform.field_engine import deserialize_field_updates
from chat.services.platform.schemas import FieldUpdate, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)
from tests.helpers.yaml_scenario_runner import llm_disabled


def _memory_with_pending_category() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        pending_question=PendingQuestion(
            field="item_category",
            prompt="Expense 4 — ? — 2.0 taka: what category was it?",
            workflow_id="expense",
            asked_at_turn=21,
            item_index=3,
        ),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-25",
                    "items": [
                        {"category": "lunch", "amount": 100.0, "id": "a"},
                        {
                            "category": "bike",
                            "amount": 150.0,
                            "id": "b",
                            "from_location": "jetaar",
                            "to_location": "ashole",
                        },
                        {
                            "category": "bus",
                            "amount": 45.0,
                            "id": "c",
                            "from_location": "Mirpur",
                            "to_location": "Dhaka",
                        },
                        {"amount": 2.0, "id": "d", "missing_fields": ["category"]},
                    ],
                },
            )
        },
    )


def _memory_with_pending_route() -> SessionMemory:
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
                        {"category": "lunch", "amount": 100.0, "id": "a"},
                        {"category": "bike", "amount": 150.0, "id": "b"},
                        {
                            "category": "bus",
                            "amount": 20.0,
                            "id": "c",
                            "missing_fields": ["route"],
                        },
                    ],
                },
            )
        },
    )


def test_pending_category_plus_modify_other_expense_executes():
    memory = _memory_with_pending_category()
    msg = "bike er location ta vul ota chage kore mirpur to badda daw"
    wrong_llm = {
        "intent": "clarify_modify",
        "item_patches": [],
        "clarify": {
            "kind": "which_item",
            "candidate_indices": [1, 2],
        },
    }
    turn = normalize_expense_clarify_turn(wrong_llm, msg, memory)
    if not turn.get("item_patches"):
        with llm_disabled():
            turn, _updates = expense_turn_to_field_updates(
                msg, memory, trace_id="intent-priority-modify"
            )
    assert turn["intent"] in ("modify_review", "update")
    assert turn["item_patches"][0]["item_index"] == 1
    assert turn["item_patches"][0]["from_location"] == "Mirpur"
    assert turn["item_patches"][0]["to_location"] == "Badda"

    understanding = FieldEngine().ground_expense_understanding(
        msg,
        UnderstandingResult(
            goal="Update expense draft",
            workflow="expense",
            action=UnderstandingAction.MODIFY.value,
            confidence=0.9,
            entities={"expense_intent": turn["intent"], "expense_turn": turn},
            answers_pending_field=False,
            source="rules",
        ),
        memory=memory,
        trace_id="intent-priority-modify",
    )
    decision = PendingQuestionEngine().classify(
        msg,
        memory=memory,
        trace_id="intent-priority-modify",
        conversation_history=[],
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.MODIFY_DATA
    assert decision.extracted_entities.get("interrupts_pending") is True


def test_pending_route_plus_delete_executes():
    memory = _memory_with_pending_route()
    msg = "bike ta shole ar lagbe nah..eta ami vule add dyechilam"
    with llm_disabled():
        turn, _updates = expense_turn_to_field_updates(
            msg, memory, trace_id="intent-priority-delete"
        )
    assert turn["intent"] == "delete"
    assert turn["delete_indices"] == [1]

    understanding = FieldEngine().ground_expense_understanding(
        msg,
        UnderstandingResult(
            goal="delete bike",
            workflow="expense",
            action=UnderstandingAction.DELETE.value,
            confidence=0.9,
            entities={"expense_intent": "delete", "expense_turn": turn},
            answers_pending_field=False,
            source="rules",
        ),
        memory=memory,
        trace_id="intent-priority-delete",
    )
    decision = PendingQuestionEngine().classify(
        msg,
        memory=memory,
        trace_id="intent-priority-delete",
        conversation_history=[],
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.DELETE_DATA


def test_add_expense_while_pending_category():
    memory = _memory_with_pending_category()
    msg = "snack 50 taka add koro"
    with llm_disabled():
        turn, _updates = expense_turn_to_field_updates(
            msg, memory, trace_id="intent-priority-add"
        )
    assert turn["intent"] == "add"
    assert turn["item_patches"][0]["category"] == "snack"


def test_pending_category_answer_continues():
    memory = _memory_with_pending_category()
    msg = "lunch"
    from chat.services.platform.field_extractors.expense import normalize_expense_category

    cat = normalize_expense_category(msg)
    assert cat == "lunch"
    understanding = UnderstandingResult(
        goal="answer category",
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.9,
        entities={
            "expense_intent": "answer_pending",
            "expense_turn": {
                "intent": "answer_pending",
                "item_patches": [
                    {"action": "update", "item_index": 3, "category": "lunch"},
                ],
            },
        },
        field_updates=[
            FieldUpdate(
                field="items",
                value={"category": "lunch"},
                item_index=3,
                action="update",
            )
        ],
        answers_pending_field=True,
        source="rules",
    )
    decision = PendingQuestionEngine().classify(
        msg,
        memory=memory,
        trace_id="intent-priority-answer",
        conversation_history=[],
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.ANSWER_PENDING


def test_modify_on_other_item_clears_pending_slot():
    memory = _memory_with_pending_category()
    updates = deserialize_field_updates(
        [
            {
                "field": "items",
                "value": {
                    "category": "bike",
                    "from_location": "Mirpur",
                    "to_location": "Badda",
                },
                "item_index": 1,
                "action": "update",
            }
        ]
    )
    assert expense_pending_interrupted_by_updates(updates, memory) is True
