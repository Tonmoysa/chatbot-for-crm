"""Phase 0/1 — expense conversation context in draft interpreter."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import (
    expense_turn_to_field_updates,
    interpret_expense_draft_turn,
    sync_expense_draft,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
)
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def _route_pending_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 280.0},
                        {"category": "snack", "amount": 70.0},
                        {
                            "category": "metro",
                            "amount": 90.0,
                            "from_location": "Mirpur",
                            "to_location": "Agargaon",
                        },
                        {"amount": 150.0},
                    ],
                },
            )
        },
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Expense 1 — Bus — 120.0 taka: where did you travel from and to?",
            workflow_id="expense",
            asked_at_turn=2,
            item_index=0,
        ),
    )


def test_expense_draft_payload_includes_conversation_history():
    memory = _route_pending_memory()
    history = [
        "User: dhanmondi to mirpur",
        "Assistant: Still needed - Route",
    ]
    with mock_expense_llm():
        turn = interpret_expense_draft_turn(
            "ami tomake route diyechi..add koro",
            memory,
            trace_id="test-payload-history",
            conversation_history=history,
        )
    assert turn.get("intent") in ("answer_pending", "fix_mistake")
    patches = turn.get("item_patches") or []
    assert patches
    assert patches[0].get("from_location") == "Dhanmondi"
    assert patches[0].get("to_location") == "Mirpur"


def test_route_message_applies_to_pending_bus():
    memory = _route_pending_memory()
    history = [
        "Assistant: Expense 1 — Bus — 120.0 taka: where did you travel from and to?",
    ]
    with mock_expense_llm():
        turn, updates = expense_turn_to_field_updates(
            "dhanmondi to mirpur",
            memory,
            trace_id="test-route-dhanmondi",
            conversation_history=history,
        )
    assert turn.get("intent") == "answer_pending"
    assert updates
    pipeline = WorkflowPipeline()
    state = StatePatchBuffer(memory)
    from chat.services.platform.registry import get_workflow_definition

    pipeline._finish_expense_update_turn(
        memory,
        get_workflow_definition("expense"),
        updates=updates,
        message="dhanmondi to mirpur",
        lang="en",
        state=state,
    )
    state.flush()
    sync_expense_draft(memory.active_draft())
    bus = memory.active_draft().fields["items"][0]
    assert bus.get("from_location") == "Dhanmondi"
    assert bus.get("to_location") == "Mirpur"


def test_back_reference_route_not_duplicate_amount_item():
    memory = _route_pending_memory()
    pipeline = WorkflowPipeline()
    history = [
        "User: dhanmondi to mirpur",
        "Assistant: Still needed - Route",
    ]
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="route back-reference",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        handle_with_rules_understanding(
            pipeline,
            "ami tomake route diyechi..add koro",
            memory=memory,
            pq_decision=pq,
            conversation_history=history,
            trace_id="test-back-ref-route",
            route_source="active",
        )
    sync_expense_draft(memory.active_draft())
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 5
    bus = items[0]
    assert bus.get("from_location") == "Dhanmondi"
    assert bus.get("to_location") == "Mirpur"
