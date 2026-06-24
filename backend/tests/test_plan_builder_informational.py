"""Phase 4 — informational PlanBuilder + execute_planned_turn tests."""

from unittest.mock import patch

import pytest
from chat.services.pending_question_engine import (
    MessageIntentKind,
    PendingQuestionDecision,
    PendingQuestionEngine,
    informational_priority_decision,
)
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.schemas import PlanOp, TurnContext, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, SessionMemory


def _ctx(**overrides) -> TurnContext:
    base = dict(
        trace_id="t1",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
        user_message="hello",
        conversation_history=(),
        document_text=None,
        idempotency_key="",
        user_language="en",
        reply_language="en",
        today_iso="2026-06-21",
        turn_count_at_start=0,
        memory_schema_version=1,
        active_workflow_id=None,
        active_workflow_stage=None,
        draft_id=None,
        pending_question_field=None,
        pending_question_prompt=None,
        pending_question_workflow_id=None,
        pending_confirmation=None,
        draft_snapshot=None,
        suspended_workflows=(),
        conversation_facts={},
        has_active_workflow=False,
        has_pending_question=False,
        has_pending_confirmation=False,
        draft_locked=False,
        wizard_active=False,
    )
    base.update(overrides)
    return TurnContext(**base)


def _pq(kind: MessageIntentKind, **kwargs) -> PendingQuestionDecision:
    return PendingQuestionDecision(
        kind=kind,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=False,
        **kwargs,
    )


def test_plan_builder_policy_kind():
    plan = PlanBuilder.build(
        _ctx(user_message="what is the leave policy?"),
        TurnDecision(
            pq=_pq(MessageIntentKind.ASK_POLICY),
            understanding=UnderstandingResult(
                workflow="none",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.9,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.workflow_id == "informational"
    assert plan.primary_op == PlanOp.REPLY_POLICY


def test_plan_builder_status_kind():
    plan = PlanBuilder.build(
        _ctx(user_message="status of REQ-123"),
        TurnDecision(
            pq=_pq(MessageIntentKind.ASK_STATUS),
            understanding=UnderstandingResult(
                workflow="none",
                action=UnderstandingAction.STATUS.value,
                confidence=0.9,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.REPLY_STATUS


def test_plan_builder_today_date_kind():
    plan = PlanBuilder.build(
        _ctx(user_message="What is today's date?"),
        TurnDecision(
            pq=_pq(MessageIntentKind.ASK_TODAY_DATE),
            understanding=UnderstandingResult(
                workflow="none",
                action=UnderstandingAction.QUERY.value,
                confidence=1.0,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.REPLY_TODAY_DATE


def test_plan_builder_oos_kind():
    plan = PlanBuilder.build(
        _ctx(user_message="write python code for me"),
        TurnDecision(
            pq=_pq(MessageIntentKind.OUT_OF_SCOPE),
            understanding=UnderstandingResult(
                workflow="none",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.9,
                is_out_of_scope=True,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.REPLY_OOS


def test_plan_builder_greeting_kind():
    plan = PlanBuilder.build(
        _ctx(user_message="hi"),
        TurnDecision(
            pq=_pq(MessageIntentKind.NEW_WORKFLOW),
            understanding=UnderstandingResult(
                workflow="none",
                action=UnderstandingAction.START.value,
                confidence=0.92,
                is_greeting=True,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.REPLY_GREETING


def test_plan_builder_greeting_wins_during_active_leave():
    plan = PlanBuilder.build(
        _ctx(
            user_message="hi",
            active_workflow_id="leave",
            has_active_workflow=True,
        ),
        TurnDecision(
            pq=_pq(MessageIntentKind.NEW_WORKFLOW),
            understanding=UnderstandingResult(
                workflow="leave",
                action=UnderstandingAction.NONE.value,
                confidence=0.92,
                is_greeting=True,
            ),
            route_source="active",
        ),
    )
    assert plan is not None
    assert plan.workflow_id == "informational"
    assert plan.primary_op == PlanOp.REPLY_GREETING


def test_plan_builder_informational_not_selected_for_leave():
    plan = PlanBuilder.build(
        _ctx(
            user_message="annual leave tomorrow",
            active_workflow_id="leave",
            has_active_workflow=True,
            draft_snapshot={"workflow_id": "leave", "fields": {}},
        ),
        TurnDecision(
            pq=_pq(MessageIntentKind.NEW_WORKFLOW, target_workflow="leave"),
            understanding=UnderstandingResult(
                workflow="leave",
                action=UnderstandingAction.START.value,
                confidence=0.9,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.workflow_id == "leave"


def test_informational_priority_policy_during_leave_wizard():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
    )
    decision = informational_priority_decision(
        "what is sick leave policy?",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_classify_policy_wins_over_leave_understanding():
    engine = PendingQuestionEngine()
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
    )
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.8,
        reasoning="collect leave",
    )
    decision = engine.classify(
        "what is sick leave policy?",
        memory=memory,
        conversation_history=[],
        trace_id="policy-wizard",
        understanding=u,
    )
    assert decision.kind == MessageIntentKind.ASK_POLICY


@pytest.mark.django_db
def test_execute_planned_turn_policy_op():
    memory = SessionMemory()
    ctx = _ctx(user_message="what is sick leave policy?", company_id="co1")
    u = UnderstandingResult(
        workflow="none",
        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
        confidence=0.9,
    )
    pq = _pq(MessageIntentKind.ASK_POLICY)
    pipeline = WorkflowPipeline()

    with patch("knowledge_base.services.rag_pipeline.try_hr_policy_rag") as rag_mock:
        rag_mock.return_value = {"text": "Sick leave requires a note after 2 days."}
        with patch("chat.services.translator.align_policy_answer_language") as align_mock:
            align_mock.side_effect = lambda text, **_: text
            result = pipeline.execute_planned_turn(
                "what is sick leave policy?",
                memory=memory,
                understanding=u,
                pq_decision=pq,
                conversation_history=[],
                trace_id="info-plan-1",
                turn_context=ctx,
                company_id="co1",
            )

    assert result is not None
    msg, envelope = result
    assert "Sick leave" in msg
    assert envelope.get("execution_plan", {}).get("primary_op") == PlanOp.REPLY_POLICY.value


@pytest.mark.django_db
def test_execute_planned_turn_today_date_op():
    memory = SessionMemory()
    ctx = _ctx(user_message="What is today's date?", today_iso="2026-06-21")
    u = PendingQuestionEngine.synthetic_understanding_for_shortcut(
        _pq(MessageIntentKind.ASK_TODAY_DATE),
    )
    pq = _pq(MessageIntentKind.ASK_TODAY_DATE)
    pipeline = WorkflowPipeline()

    result = pipeline.execute_planned_turn(
        "What is today's date?",
        memory=memory,
        understanding=u,
        pq_decision=pq,
        conversation_history=[],
        trace_id="info-today-1",
        turn_context=ctx,
    )

    assert result is not None
    msg, envelope = result
    assert "2026-06-21" in msg
    assert envelope.get("execution_plan", {}).get("primary_op") == PlanOp.REPLY_TODAY_DATE.value
