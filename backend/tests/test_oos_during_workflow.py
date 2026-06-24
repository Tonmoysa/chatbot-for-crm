"""Off-HR topics during active leave/expense must not show workflow summaries."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.intent_rules import is_off_hr_topic_message, is_programming_question
from chat.services.platform.pipeline import PlanBuilder, PlanOp, WorkflowPipeline
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
    build_turn_context,
)
from chat.services.platform.schemas import TurnDecision, UnderstandingAction, UnderstandingResult
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def _expense_submit_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 140.0},
                    ],
                },
            ),
            "leave-draft": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "lwop",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "family emergency",
                },
            ),
        },
        suspended_workflows=[
            SuspendedWorkflow(
                workflow_id="leave",
                stage="confirm_submit",
                draft_id="leave-draft",
                suspended_at_turn=2,
            )
        ],
        pending_confirmation="submit",
    )


def test_what_is_json_is_programming_and_off_hr():
    assert is_programming_question("what is json?")
    assert is_off_hr_topic_message("what is json?", memory=_expense_submit_memory())


def test_lunch_modify_is_not_off_hr_during_expense():
    memory = _expense_submit_memory()
    assert not is_off_hr_topic_message("lunch 140 taka hobe", memory=memory)


def test_rules_understanding_oos_during_expense_submit():
    with llm_disabled():
        result = AIUnderstandingLayer().understand(
            "what is json?",
            memory=_expense_submit_memory(),
            conversation_history=[],
            trace_id="oos-json",
        )
    assert result.is_out_of_scope
    assert result.workflow == "none"


def test_pending_engine_oos_during_expense_submit():
    memory = _expense_submit_memory()
    decision = PendingQuestionEngine().classify(
        "what is json?",
        memory=memory,
        understanding=UnderstandingResult(
            workflow="expense",
            action=UnderstandingAction.REVIEW.value,
            confidence=1.0,
            entities={"expense_intent": "show_list"},
            source="llm",
        ),
        conversation_history=[],
        trace_id="oos-pqe",
    )
    assert decision.kind == MessageIntentKind.OUT_OF_SCOPE


def test_pending_engine_rules_path_oos_before_understanding():
    memory = _expense_submit_memory()
    decision = PendingQuestionEngine().classify(
        "what is json?",
        memory=memory,
        understanding=UnderstandingResult(
            workflow="expense",
            action=UnderstandingAction.REVIEW.value,
            confidence=1.0,
            source="llm",
        ),
        conversation_history=[],
        trace_id="oos-pqe-rules",
    )
    assert decision.kind == MessageIntentKind.OUT_OF_SCOPE


def test_plan_builder_oos_not_expense_summary():
    u = UnderstandingResult(
        workflow="none",
        action=UnderstandingAction.NONE.value,
        confidence=0.93,
        is_out_of_scope=True,
        source="rules",
    )
    plan = PlanBuilder.build(
        build_turn_context(
            message="what is json?",
            memory=_expense_submit_memory(),
            conversation_history=[],
            trace_id="oos-plan",
            session_id="s1",
            company_id="c1",
            employee_id="e1",
            idempotency_key="",
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.REJECT_OOS


def test_pipeline_oos_message_not_expense_list():
    memory = _expense_submit_memory()
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.OUT_OF_SCOPE,
        confidence=0.93,
        reasoning="off hr",
        source="rules",
        blocks_new_workflow=False,
    )
    with llm_disabled():
        msg, meta = handle_with_rules_understanding(
            WorkflowPipeline(),
            "what is json?",
            memory=memory,
            pq_decision=pq,
            trace_id="oos-pipeline",
            route_source="informational",
        )
    low = msg.lower()
    assert "expense summary" not in low
    assert "current expenses" not in low
    assert meta.get("outcome") == "INFORMATIONAL"
    assert "hr" in low or "scope" in low or "company" in low
