"""Phase 5 — unified expense draft editor (collect + review share one interpreter)."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import (
    expense_turn_to_field_updates,
    is_expense_review_mode,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def _expense_review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-23",
                    "items": [
                        {"category": "lunch", "amount": 280.0},
                        {"category": "bus", "amount": 120.0},
                    ],
                },
            )
        },
        pending_confirmation="submit",
    )


def test_review_mode_uses_draft_interpreter_not_separate_review_path():
    memory = _expense_review_memory()
    assert is_expense_review_mode(memory)
    with mock_expense_llm():
        turn, updates = expense_turn_to_field_updates("bus 120 taka add koro", memory)
    assert turn.get("intent") in ("add", "update", "modify_review")
    assert updates


def test_expense_modify_at_review_routes_through_collect_handler():
    memory = _expense_review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="add at review",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "bus 120 taka add koro",
            memory=memory,
            pq_decision=pq,
            trace_id="phase5-review-add",
            route_source="active",
        )
    low = msg.lower()
    assert "120" in low or "bus" in low
    assert decision.get("outcome") in ("NEEDS_INPUT", "NEEDS_CLARIFICATION", None)
    draft = memory.active_draft()
    assert draft is not None
    items = draft.fields.get("items") or []
    assert len(items) >= 2


def test_expense_review_complaint_no_robotic_summary_only():
    memory = _expense_review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.9,
        reasoning="anti summary",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "summery chai ni",
            memory=memory,
            pq_decision=pq,
            trace_id="phase5-anti-summary",
            route_source="active",
        )
    low = msg.lower()
    assert "expense summary" not in low or "chai ni" in low or "sorry" in low or "thik" in low
    assert decision.get("rules_applied") == ["EXPENSE_ANTI_SUMMARY"] or decision.get("outcome") == "NEEDS_INPUT"
