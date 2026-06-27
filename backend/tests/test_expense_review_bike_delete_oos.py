"""Regression: expense review delete/summary must not be blocked by HR scope or suspension."""

from __future__ import annotations

from unittest.mock import patch

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.hr_assistant_scope import resolve_hr_assistant_scope
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
)
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def _review_memory_with_bike() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit", draft_id="default"),
        pending_confirmation="submit",
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-27",
                    "items": [
                        {"category": "bus", "amount": 20.0, "id": "6d5d2b04"},
                        {"category": "lunch", "amount": 150.0, "id": "a00969a8"},
                        {"category": "bike", "amount": 150.0, "id": "66f8f707"},
                    ],
                },
            )
        },
    )


def _suspended_expense_memory() -> SessionMemory:
    items = [
        {"category": "bus", "amount": 20.0, "id": "6d5d2b04"},
        {"category": "lunch", "amount": 150.0, "id": "a00969a8"},
        {"category": "bike", "amount": 150.0, "id": "66f8f707"},
    ]
    return SessionMemory(
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={"incurred_date": "2026-06-27", "items": items},
            ),
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={"incurred_date": "2026-06-27", "items": items},
                version=3,
            ),
        },
        suspended_workflows=[
            SuspendedWorkflow(
                workflow_id="expense",
                stage="confirm_submit",
                draft_id="expense",
                suspended_at_turn=6,
            )
        ],
        pending_confirmation="submit",
    )


def test_hr_scope_skips_expense_review_delete():
    memory = _review_memory_with_bike()
    scope_oos = UnderstandingResult(
        goal="Out of scope",
        workflow="none",
        action=UnderstandingAction.NONE.value,
        confidence=1.0,
        is_out_of_scope=True,
        reasoning="User is asking to delete an item, which is not a valid action in the expense workflow.",
        source="llm_hr_scope",
    )
    with patch(
        "chat.services.platform.hr_assistant_scope.LLMClient"
    ) as mock_client:
        mock_client.return_value.is_configured.return_value = True
        mock_client.return_value.chat_json.return_value = {
            "in_scope": False,
            "confidence": 1.0,
            "reasoning": scope_oos.reasoning,
        }
        assert resolve_hr_assistant_scope("bike ta delete koro", memory) is None


def test_bike_delete_at_review_not_out_of_scope():
    memory = _review_memory_with_bike()
    scope_oos = UnderstandingResult(
        goal="Out of scope",
        workflow="none",
        action=UnderstandingAction.NONE.value,
        confidence=1.0,
        is_out_of_scope=True,
        reasoning="User is asking to delete an item, which is not a valid action in the expense workflow.",
        source="llm_hr_scope",
    )
    with patch(
        "chat.services.platform.hr_assistant_scope.resolve_hr_assistant_scope",
        return_value=scope_oos,
    ):
        with llm_disabled():
            understanding = AIUnderstandingLayer().understand(
                "bike ta delete koro",
                memory=memory,
                conversation_history=[],
                trace_id="bike-delete-review",
            )

    assert not understanding.is_out_of_scope
    assert understanding.action == UnderstandingAction.DELETE.value
    decision = PendingQuestionEngine().classify(
        "bike ta delete koro",
        memory=memory,
        conversation_history=[],
        trace_id="bike-delete-review",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.DELETE_DATA


def test_bike_delete_removes_item_from_review():
    memory = _review_memory_with_bike()
    pipeline = WorkflowPipeline()
    with llm_disabled():
        understanding = AIUnderstandingLayer().understand(
            "bike ta delete koro",
            memory=memory,
            conversation_history=[],
            trace_id="bike-delete-pipeline",
        )
    decision = PendingQuestionEngine().classify(
        "bike ta delete koro",
        memory=memory,
        conversation_history=[],
        trace_id="bike-delete-pipeline",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.DELETE_DATA
    handle_with_rules_understanding(
        pipeline,
        "bike ta delete koro",
        memory=memory,
        pq_decision=decision,
        trace_id="bike-delete-pipeline",
        route_source="active",
    )
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 2
    assert all(item.get("category") != "bike" for item in items)


def test_expense_summary_after_oos_suspend():
    memory = _suspended_expense_memory()
    pipeline = WorkflowPipeline()
    with llm_disabled():
        understanding = AIUnderstandingLayer().understand(
            "expesne er summery ta daw",
            memory=memory,
            conversation_history=[],
            trace_id="expense-summary-suspended",
        )
    decision = PendingQuestionEngine().classify(
        "expesne er summery ta daw",
        memory=memory,
        conversation_history=[],
        trace_id="expense-summary-suspended",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.SHOW_REVIEW
    msg, meta = handle_with_rules_understanding(
        pipeline,
        "expesne er summery ta daw",
        memory=memory,
        pq_decision=decision,
        trace_id="expense-summary-suspended",
        route_source="pending",
    )
    assert "No active draft" not in msg
    assert "bike" in msg.lower() or "lunch" in msg.lower()
    assert meta.get("outcome") == "INFORMATIONAL"
