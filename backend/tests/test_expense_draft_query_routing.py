"""Expense draft lookup, submit routing, and post-submit lock regressions."""

from __future__ import annotations

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.intent_rules import (
    is_expense_draft_query,
    is_strong_new_workflow_message,
    parse_submit_workflow,
    should_resume_suspended_expense,
)
from chat.services.platform.pipeline import PlanBuilder
from chat.services.platform.schemas import PlanOp, TurnContext, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, SessionMemory, SuspendedWorkflow, WorkflowDraft


def _leave_with_suspended_expense() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={"leave_type": "annual", "start_date": "2026-06-24"},
            ),
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 140.0},
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
            )
        ],
    )


def test_expense_draft_query_banglish_phrases():
    for phrase in (
        "amar expense koi",
        "where is my expense",
        "where is my expense?",
        "expense kothay",
        "expense er summery ta daw",
    ):
        assert is_expense_draft_query(phrase), phrase


def test_expense_submit_not_treated_as_new_workflow():
    assert parse_submit_workflow("expense submit koro", active_workflow_id="expense") == "expense"
    assert not is_strong_new_workflow_message("expense submit koro")


def test_understand_expense_submit_active_expense():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [{"category": "bus", "amount": 120.0}],
                },
            )
        },
    )
    layer = AIUnderstandingLayer()
    u = layer._understand_rules("expense submit koro", memory=memory, pending_kind=None)
    assert u.workflow == "expense"
    assert u.action == UnderstandingAction.SUBMIT.value


def test_understand_amar_expense_koi_resumes_suspended_draft():
    memory = _leave_with_suspended_expense()
    layer = AIUnderstandingLayer()
    u = layer._understand_rules("amar expense koi", memory=memory, pending_kind=None)
    assert u.workflow == "expense"
    assert u.action == UnderstandingAction.REVIEW.value
    assert u.interrupt_workflow == "expense"
    assert should_resume_suspended_expense(
        message="amar expense koi",
        active_workflow_id="leave",
        suspended_workflows=memory.suspended_workflows,
    )


def test_plan_builder_locked_expense_blocks_modify_allows_new():
    ctx = TurnContext(
        trace_id="t1",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
        user_message="lunch 200 taka",
        conversation_history=(),
        document_text=None,
        idempotency_key="",
        user_language="en",
        reply_language="en",
        today_iso="2026-06-24",
        turn_count_at_start=1,
        memory_schema_version=1,
        active_workflow_id="expense",
        active_workflow_stage="submitted",
        draft_id="default",
        pending_question_field=None,
        pending_question_prompt=None,
        pending_question_workflow_id=None,
        pending_confirmation=None,
        draft_snapshot={
            "workflow_id": "expense",
            "locked": True,
            "submitted_request_id": "EX-1",
            "fields": {"items": [{"category": "bus", "amount": 120.0}]},
        },
        suspended_workflows=(),
        conversation_facts={},
        has_active_workflow=True,
        has_pending_question=False,
        has_pending_confirmation=False,
        draft_locked=True,
        wizard_active=False,
    )
    modify_plan = PlanBuilder.build(
        ctx,
        TurnDecision(
            pq=None,
            understanding=UnderstandingResult(
                workflow="expense",
                action=UnderstandingAction.MODIFY.value,
                confidence=0.9,
            ),
            route_source="active",
        ),
    )
    assert modify_plan is not None
    assert modify_plan.primary_op == PlanOp.LOCKED_RESPONSE

    new_plan = PlanBuilder.build(
        TurnContext(**{**ctx.__dict__, "user_message": "bus 90 taka"}),
        TurnDecision(
            pq=None,
            understanding=UnderstandingResult(
                workflow="expense",
                action=UnderstandingAction.START.value,
                confidence=0.9,
            ),
            route_source="active",
        ),
    )
    assert new_plan is not None
    assert new_plan.primary_op == PlanOp.WORKFLOW_NEW
