"""Regression: expense/leave workflow switching while one draft is suspended."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.workflow_show import resolve_workflow_show_target
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
)


def _leave_active_expense_suspended() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="leave"),
        workflow_drafts={
            "leave": WorkflowDraft(
                workflow_id="leave",
                fields={"start_date": "2026-06-27"},
            ),
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-26",
                    "items": [
                        {"category": "lunch", "amount": 100.0, "status": "complete"},
                        {
                            "category": "bus",
                            "amount": 20.0,
                            "missing_fields": ["route"],
                            "status": "incomplete",
                        },
                    ],
                },
            ),
        },
        suspended_workflows=[
            SuspendedWorkflow(
                workflow_id="expense",
                stage="collecting",
                draft_id="expense",
                suspended_at_turn=2,
            ),
        ],
    )


def test_plan_shortcut_expense_continue_switches_not_leave_show():
    memory = _leave_active_expense_suspended()
    decision = PendingQuestionEngine.detect_plan_shortcut(
        "expense continue",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "expense"


def test_plan_shortcut_expense_e_jao_switches():
    memory = _leave_active_expense_suspended()
    decision = PendingQuestionEngine.detect_plan_shortcut(
        "expense e jao",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "expense"


def test_plan_shortcut_expense_summary_switches():
    memory = _leave_active_expense_suspended()
    decision = PendingQuestionEngine.detect_plan_shortcut(
        "amar expense summery daww",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "expense"


def test_show_target_rules_beat_active_workflow_llm_misfire():
    """Explicit 'expense' in message must resolve to expense, not active leave."""
    memory = _leave_active_expense_suspended()
    target = resolve_workflow_show_target(
        "amar expense summery daww",
        memory,
        active_workflow_id="leave",
        trace_id="test-show-rules",
    )
    assert target == "expense"
