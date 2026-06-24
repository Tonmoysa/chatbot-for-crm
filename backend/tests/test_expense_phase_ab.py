"""Phase A+B — expense pipeline wiring + LLM resilience hooks."""

from __future__ import annotations

from unittest.mock import patch

from chat.services.platform.intent_rules import is_pure_expense_navigation
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.schemas import PlanOp, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    WorkflowDraft,
)
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.test_plan_builder import _ctx


COMPOUND_EXPENSE = (
    "Aj office jawar somoy bus e 120 taka lagse. Dupure lunch korlam 280 taka. "
    "Bikale ekta snack kheyechi 70 taka. Ferar somoy metro te 90 taka lagse "
    "Mirpur theke Agargaon porjonto. Ar ekta 150 taka expense hoise but category mone nei ekhon."
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
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "family emergency",
                },
            )
        },
        pending_confirmation="submit",
    )


def test_pure_expense_navigation_excludes_compound_message():
    assert not is_pure_expense_navigation(COMPOUND_EXPENSE)
    assert is_pure_expense_navigation("yes expense workflow te jao")


def test_plan_builder_expense_collect_without_field_updates():
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.55,
        field_updates=[],
        entities={"expense_intent": "conversation"},
    )
    plan = PlanBuilder.build(
        _ctx(user_message=COMPOUND_EXPENSE, active_workflow_id="expense"),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_COLLECT


def test_leave_submit_compound_expense_switch_adds_items():
    from chat.services.platform.ai_understanding import AIUnderstandingLayer
    from chat.services.session_memory import build_turn_context
    from tests.helpers.yaml_scenario_runner import llm_disabled

    memory = _leave_submit_memory()
    engine = PendingQuestionEngine()
    layer = AIUnderstandingLayer()
    with mock_expense_llm(), llm_disabled():
        u = layer.understand(
            COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-ab-switch-items",
        )
        pq = engine.classify(
            COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-ab-switch-items",
            understanding=u,
        )
        pipeline = WorkflowPipeline()
        turn_context = build_turn_context(
            message=COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-ab-switch-items",
            session_id="test-session",
            company_id="test-company",
            employee_id="test-employee",
            idempotency_key="",
        )
        pipeline.execute_workflow_turn(
            COMPOUND_EXPENSE,
            memory=memory,
            understanding=u,
            pq_decision=pq,
            conversation_history=[],
            trace_id="phase-ab-switch-items",
            turn_context=turn_context,
            route_source="active",
        )
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"
    draft = memory.active_draft()
    assert draft is not None
    items = draft.fields.get("items") or []
    amounts = {float(i.get("amount") or 0) for i in items}
    assert 120.0 in amounts
    assert len(items) >= 4


def test_active_expense_compound_message_collects_items():
    from chat.services.platform.ai_understanding import AIUnderstandingLayer
    from chat.services.session_memory import build_turn_context
    from tests.helpers.yaml_scenario_runner import llm_disabled

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(workflow_id="expense", fields={"items": []}),
        },
    )
    engine = PendingQuestionEngine()
    layer = AIUnderstandingLayer()
    pipeline = WorkflowPipeline()
    with mock_expense_llm(), llm_disabled():
        u = layer.understand(
            COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-ab-active-collect",
        )
        pq = engine.classify(
            COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-ab-active-collect",
            understanding=u,
        )
        turn_context = build_turn_context(
            message=COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-ab-active-collect",
            session_id="test-session",
            company_id="test-company",
            employee_id="test-employee",
            idempotency_key="",
        )
        msg, meta = pipeline.execute_workflow_turn(
            COMPOUND_EXPENSE,
            memory=memory,
            understanding=u,
            pq_decision=pq,
            conversation_history=[],
            trace_id="phase-ab-active-collect",
            turn_context=turn_context,
            route_source="active",
        )
    draft = memory.active_draft()
    items = (draft.fields if draft else {}).get("items") or []
    assert len(items) >= 4
    amounts = {float(i.get("amount") or 0) for i in items}
    assert 120.0 in amounts


def test_expense_llm_scoped_circuit_does_not_block_other_scope():
    from chat.services.llm_client import LLMClient, _circuit_key, clear_llm_trace_state
    from chat.services.llm_client import _RATE_LIMIT_TRIPPED

    clear_llm_trace_state("trace-1")
    _RATE_LIMIT_TRIPPED.add(_circuit_key("trace-1", "understanding"))
    client = LLMClient()
    with patch.object(client, "is_configured", return_value=True):
        with patch.object(client, "_complete", return_value=None) as mock_complete:
            out = client.chat_json(
                system_prompt="sys",
                user_prompt="user",
                trace_id="trace-1",
                scope="expense-draft",
            )
    assert out is None
    mock_complete.assert_called()
    clear_llm_trace_state("trace-1")
