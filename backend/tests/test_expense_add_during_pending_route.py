"""Regression: new expense line while pending route must append, not modify pending item."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.field_extractors.expense import (
    finalize_expense_turn_patches,
    interpret_expense_draft_turn,
)
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)
from tests.helpers.yaml_scenario_runner import llm_disabled


def _bike_route_pending_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting", draft_id="expense"),
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Expense 2 — Bike — 150.0 taka: where did you travel from and to?",
            workflow_id="expense",
            item_index=1,
        ),
        workflow_drafts={
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-26",
                    "items": [
                        {"category": "lunch", "amount": 100.0, "status": "complete"},
                        {
                            "category": "bike",
                            "amount": 150.0,
                            "missing_fields": ["route"],
                            "status": "incomplete",
                        },
                    ],
                },
            ),
        },
    )


def test_finalize_compound_add_beats_llm_modify_on_pending_bike():
    memory = _bike_route_pending_memory()
    llm_turn = {
        "intent": "modify_review",
        "item_patches": [
            {"action": "update", "item_index": 1, "category": "bike", "amount": 150},
        ],
    }
    final = finalize_expense_turn_patches(llm_turn, "bus 60 taka mirpur to badda", memory)
    assert final.get("intent") == "add"
    patch = final["item_patches"][0]
    assert patch.get("action") == "append"
    assert patch.get("category") == "bus"
    assert patch.get("amount") == 60.0
    assert patch.get("from_location") == "Mirpur"
    assert patch.get("to_location") == "Badda"


def test_rules_interpreter_appends_bus_without_llm():
    memory = _bike_route_pending_memory()
    with llm_disabled():
        turn = interpret_expense_draft_turn(
            "bus 60 taka mirpur to badda",
            memory,
            trace_id="test-bus-append",
        )
    assert turn.get("intent") == "add"
    patch = turn["item_patches"][0]
    assert patch.get("category") == "bus"
    assert patch.get("amount") == 60.0


def test_explicit_new_expense_add_beats_llm_modify_at_review():
    """'new expense add koro bus …' must append even when bus already exists in draft."""
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        pending_confirmation="submit",
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-27",
                    "items": [
                        {
                            "category": "bus",
                            "amount": 25.0,
                            "from_location": "gulshan",
                            "to_location": "baridhara",
                        },
                        {"category": "lunch", "amount": 120.0},
                    ],
                },
            )
        },
    )
    llm_turn = {
        "intent": "modify_review",
        "item_patches": [
            {
                "action": "update",
                "item_index": 0,
                "category": "bus",
                "amount": 30,
                "from_location": "mirpur",
                "to_location": "badda",
            }
        ],
    }
    msg = "okay..ekta new expense add koro bus 30 taka mirpur to badda"
    final = finalize_expense_turn_patches(llm_turn, msg, memory)
    assert final.get("intent") == "add"
    patch = final["item_patches"][0]
    assert patch.get("action") == "append"
    assert patch.get("category") == "bus"
    assert patch.get("amount") == 30.0
    assert "item_index" not in patch


def test_bare_yes_on_pending_route_no_crash():
    memory = _bike_route_pending_memory()
    understanding = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
        confidence=0.88,
        source="llm_expense",
    )
    decision = PendingQuestionEngine().classify(
        "yes",
        memory=memory,
        conversation_history=[],
        trace_id="test-yes-route",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.CLARIFICATION_NEEDED
