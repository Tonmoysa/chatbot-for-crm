"""Phase C — helpful guidance + workflow switch copy (Fix 6 + 7)."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.response_composer import ResponseComposer
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.platform.intent_rules import is_workflow_interrupt_leave
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
)
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def _expense_active_leave_suspended_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit", draft_id="expense"),
        workflow_drafts={
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-23",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 280.0},
                    ],
                },
            ),
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "osusto",
                },
            ),
        },
        suspended_workflows=[
            SuspendedWorkflow(
                workflow_id="leave",
                stage="confirm_submit",
                draft_id="default",
                suspended_at_turn=2,
            )
        ],
        pending_confirmation="submit",
    )


def test_is_workflow_interrupt_leave_during_expense():
    assert is_workflow_interrupt_leave("amar leave lagbe", active_workflow="expense")
    assert not is_workflow_interrupt_leave("amar leave lagbe", active_workflow="leave")


def test_detect_interrupt_leave_during_expense():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting", draft_id="expense"),
        workflow_drafts={"expense": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    interrupt = WorkflowManager().detect_interrupt("amar leave lagbe", memory)
    assert interrupt is not None
    assert interrupt.to_workflow == "leave"


def test_switch_confirm_message_banglish():
    msg = WorkflowManager.switch_confirm_message("expense", "leave", lang="banglish")
    low = msg.lower()
    assert "incomplete" in low or "ekhono" in low
    assert "leave" in low
    assert "pause" in low or "switch" in low


def test_amar_leave_lagbe_during_expense_shows_helpful_switch_copy():
    memory = _expense_active_leave_suspended_memory()
    memory.last_entities = {"reply_language": "banglish"}
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.SWITCH_WORKFLOW,
        confidence=0.9,
        reasoning="resume suspended leave",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, meta = handle_with_rules_understanding(
        pipeline,
        "amar leave lagbe",
        memory=memory,
        pq_decision=pq,
        trace_id="phase-c-leave-resume",
        route_source="active",
    )
    low = msg.lower()
    assert "switched to **leave**" not in low
    assert "leave" in low
    assert "expense" in low
    assert meta.get("awaiting_confirmation") or "pause" in low or "switch" in low or "incomplete" in low


def test_amar_leave_lagbe_resumes_suspended_leave_with_guidance():
    memory = _expense_active_leave_suspended_memory()
    memory.last_entities = {"reply_language": "banglish"}
    memory.pending_confirmation = None
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.SWITCH_WORKFLOW,
        confidence=0.9,
        reasoning="resume suspended leave",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, _ = handle_with_rules_understanding(
        pipeline,
        "amar leave lagbe",
        memory=memory,
        pq_decision=pq,
        trace_id="phase-c-leave-resume-direct",
        route_source="active",
    )
    low = msg.lower()
    assert "pause" in low
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "leave"


def test_review_ready_message_includes_submit_footer():
    memory = _expense_active_leave_suspended_memory()
    composer = ResponseComposer()
    msg = composer.review_ready_message(
        "Saved.",
        "**Expense — Review**\n\n_Reply yes to submit._",
        lang="banglish",
        memory=memory,
    )
    low = msg.lower()
    assert "submit" in low
    assert "modify" in low or "cancel" in low


def test_workflow_switch_resumed_copy():
    memory = _expense_active_leave_suspended_memory()
    composer = ResponseComposer()
    msg = composer.workflow_switch_resumed(
        memory,
        paused_workflow="expense",
        resumed_workflow="leave",
        lang="banglish",
    )
    low = msg.lower()
    assert "pause" in low
    assert "leave" in low
    assert "expense continue" in low or "expense" in low
