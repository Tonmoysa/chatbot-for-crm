"""Expense draft must survive policy interrupts and resume on navigation."""

from chat.services.expense_policy_session_fix import apply as apply_fix
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.intent_rules import should_resume_suspended_expense
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import PlanOp, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
    reduce_suspend_active_workflow,
)


def _expense_memory_with_items() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-26",
                    "items": [{"category": "bus", "amount": 100.0, "status": "complete"}],
                },
            ),
        },
    )


def test_should_resume_expense_when_draft_stored_without_suspend():
    memory = _expense_memory_with_items()
    memory.active_workflow = None
    assert should_resume_suspended_expense(
        message="expense e jao",
        active_workflow_id=None,
        suspended_workflows=[],
        memory=memory,
    )


def test_policy_suspend_preserves_expense_draft():
    memory = _expense_memory_with_items()
    reduce_suspend_active_workflow(memory)
    draft = (memory.workflow_drafts or {}).get("expense") or (memory.workflow_drafts or {}).get("default")
    assert draft is not None
    assert list((draft.fields or {}).get("items") or [])
    assert memory.suspended_workflows
    assert memory.active_workflow is None


def test_pause_on_policy_op():
    apply_fix()
    memory = _expense_memory_with_items()
    state = StatePatchBuffer(memory)
    WorkflowPipeline._pause_active_workflow_for_interrupt(state)
    state.flush()
    assert memory.active_workflow is None
    assert memory.suspended_workflows


def test_expense_e_jao_shortcut_after_policy_context():
    apply_fix()
    from chat.services.plan_shortcut_router import detect_plan_shortcut

    memory = _expense_memory_with_items()
    reduce_suspend_active_workflow(memory)
    decision = detect_plan_shortcut("expense e jao", memory=memory, conversation_history=[])
    assert decision is not None
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "expense"


def test_show_review_uses_expense_target_not_policy_workflow():
    apply_fix()
    memory = _expense_memory_with_items()
    reduce_suspend_active_workflow(memory)
    pipeline = WorkflowPipeline()
    u = UnderstandingResult(
        goal="Policy question",
        workflow="policy",
        action=UnderstandingAction.QUERY.value,
        confidence=1.0,
        reasoning="misread",
        source="llm_session_context",
    )
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.SHOW_REVIEW,
        confidence=1.0,
        reasoning="show expense",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    from chat.services.session_memory import build_turn_context

    ctx = build_turn_context(
        message="expense er summery daw",
        memory=memory,
        conversation_history=[],
        trace_id="test-show-expense",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    state = StatePatchBuffer(memory)
    result = pipeline._run_workflow_plan_op(
        PlanOp.WORKFLOW_SHOW_REVIEW,
        ctx=ctx,
        message="expense er summery daw",
        memory=memory,
        understanding=u,
        pq_decision=pq,
        conversation_history=[],
        trace_id="test-show-expense",
        lang="banglish",
        company_id="c1",
        employee_id="e1",
        session_id="s1",
        idempotency_key="",
        state=state,
    )
    assert result is not None
    msg, _meta = result
    assert "No active draft" not in msg
    assert "100" in msg or "bus" in msg.lower() or "expense" in msg.lower()
