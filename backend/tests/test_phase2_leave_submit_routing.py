"""Phase 2 — submit-stage routing, post-cancel summary, gatekeeper, expense interrupt."""

from __future__ import annotations

from unittest.mock import patch

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.intent_rules import is_bare_rejection, is_cancel_workflow_message
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.schemas import PlanOp, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision, PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
)
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.test_plan_builder import _ctx


LEAVE_NARRATIVE = (
    "Hi, amar ekta leave apply korte hobe. Amar ma onek osustho, tai take hospital e niye jete hobe. "
    "Ei karone ami 15 July 2026 theke 18 July 2026 porjonto office e aste parbo na. "
    "Eta Annual Leave hisebe apply korte chai."
)


def _leave_submit_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-07-15",
                    "end_date": "2026-07-18",
                    "reason": "family emergency",
                },
            )
        },
        pending_confirmation="submit",
    )


def test_bare_rejection_and_cancel_patterns():
    assert is_bare_rejection("no")
    assert is_bare_rejection("lagbe nah")
    assert is_cancel_workflow_message("cancel", workflow_id="leave")
    assert is_cancel_workflow_message("batil koro", workflow_id="leave")


def test_plan_builder_submit_no_declines_to_resolve():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.REVIEW.value,
        confidence=0.9,
        reasoning="User declined submit confirmation.",
    )
    plan = PlanBuilder.build(
        _ctx(user_message="no"),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.RESOLVE_SUBMIT_CONFIRMATION


def test_plan_builder_submit_summary_shows_review():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.REVIEW.value,
        confidence=0.88,
    )
    plan = PlanBuilder.build(
        _ctx(user_message="leave summery dekhao"),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_SHOW_REVIEW


def test_plan_builder_submit_cancel():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.CANCEL.value,
        confidence=0.92,
    )
    plan = PlanBuilder.build(
        _ctx(user_message="cancel"),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_CANCEL


def test_no_at_submit_clears_pending_and_shows_review():
    memory = _leave_submit_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.9,
        reasoning="decline",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    msg, meta = handle_with_rules_understanding(
        pipeline,
        "no",
        memory=memory,
        pq_decision=pq,
        trace_id="phase2-no-submit",
        route_source="active",
    )
    assert memory.pending_confirmation is None
    assert "yes" not in msg.lower() or "submit" in msg.lower()
    assert "family" in msg.lower() or "annual" in msg.lower()
    assert meta.get("rules_applied") == ["SUBMIT_DECLINED"] or "SUBMIT_DECLINED" in (meta.get("rules_applied") or [])


def test_summary_at_submit_shows_full_review():
    memory = _leave_submit_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.86,
        reasoning="summary",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    msg, _meta = handle_with_rules_understanding(
        pipeline,
        "leave summery dekhao",
        memory=memory,
        pq_decision=pq,
        trace_id="phase2-summary-submit",
        route_source="active",
    )
    assert "annual" in msg.lower() or "july" in msg.lower()
    assert "family" in msg.lower() or "15" in msg


def test_cancel_at_submit_abandons_draft():
    memory = _leave_submit_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CANCEL_WORKFLOW,
        confidence=0.9,
        reasoning="cancel",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    msg, meta = handle_with_rules_understanding(
        pipeline,
        "cancel",
        memory=memory,
        pq_decision=pq,
        trace_id="phase2-cancel-submit",
        route_source="active",
    )
    assert memory.active_workflow is None
    assert meta.get("outcome") == "CANCELLED"
    assert "cancel" in msg.lower()


def test_post_cancel_summary_no_submit_error():
    memory = SessionMemory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.86,
        reasoning="summary after cancel",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, meta = handle_with_rules_understanding(
        pipeline,
        "leave summery dekhao",
        memory=memory,
        pq_decision=pq,
        trace_id="phase2-post-cancel-summary",
    )
    assert "active leave" in msg.lower() or "no " in msg.lower() or "draft" in msg.lower()
    assert meta.get("outcome") == "INFORMATIONAL"


def test_gatekeeper_active_leave_narrative_resend():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={"leave_type": "annual"},
            )
        },
        pending_question=None,
    )
    layer = AIUnderstandingLayer()
    llm_weak = UnderstandingResult(
        goal="Chat",
        workflow="none",
        action=UnderstandingAction.NONE.value,
        confidence=0.7,
        is_greeting=True,
        reasoning="Conversational.",
        source="llm",
    )
    with patch.object(layer, "_understand_llm", return_value=llm_weak):
        with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
            mock_cls.return_value.is_configured.return_value = True
            result = layer.understand(
                LEAVE_NARRATIVE,
                memory=memory,
                conversation_history=[],
                trace_id="phase2-gatekeeper-resend",
            )
    assert result.source == "rules_gatekeeper"
    assert result.workflow == "leave"
    assert result.action in (UnderstandingAction.START.value, UnderstandingAction.COLLECT.value)


def test_expense_interrupt_during_submit_pending():
    engine = PendingQuestionEngine()
    memory = _leave_submit_memory()
    understanding = UnderstandingResult(
        goal="Expense",
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.95,
        reasoning="Expense during submit.",
        source="llm",
        interrupt_workflow="expense",
    )
    decision = engine.classify(
        "expense",
        memory=memory,
        conversation_history=[],
        trace_id="phase2-expense-interrupt",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "expense"


def test_gatekeeper_expense_interrupt_during_leave_submit():
    """Expense narrative during leave submit must not crash gatekeeper (_gatekeeper_copy scope)."""
    memory = _leave_submit_memory()
    layer = AIUnderstandingLayer()
    expense_msg = (
        "Aj office jawar somoy bus e 120 taka lagse. Dupure lunch korlam 280 taka. "
        "Bikale ekta snack kheyechi 70 taka."
    )
    llm_weak = UnderstandingResult(
        goal="Chat",
        workflow="leave",
        action=UnderstandingAction.NONE.value,
        confidence=0.5,
        reasoning="LLM unavailable.",
        source="llm",
    )
    with patch.object(layer, "_understand_llm", return_value=llm_weak):
        with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
            mock_cls.return_value.is_configured.return_value = True
            result = layer.understand(
                expense_msg,
                memory=memory,
                conversation_history=[],
                trace_id="phase2-gatekeeper-expense-interrupt",
            )
    assert result.source == "rules_gatekeeper"
    assert result.workflow == "expense"
    assert result.interrupt_workflow == "expense"
