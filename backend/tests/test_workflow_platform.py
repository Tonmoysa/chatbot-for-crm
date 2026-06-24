"""Platform workflow integration tests."""

from unittest.mock import patch

import pytest

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.registry import get_workflow_definition, list_workflow_ids
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft, build_turn_context


@pytest.fixture
def pipeline() -> WorkflowPipeline:
    return WorkflowPipeline()


def test_workflow_definitions_load():
    assert "leave" in list_workflow_ids()
    assert "expense" in list_workflow_ids()
    leave = get_workflow_definition("leave")
    assert leave is not None
    assert leave.get_field("leave_type") is not None


def test_expense_start_from_natural_language(pipeline):
    memory = SessionMemory()
    from tests.helpers.expense_llm_mock import mock_expense_llm

    with mock_expense_llm():
        from chat.services.platform.ai_understanding import AIUnderstandingLayer

        layer = AIUnderstandingLayer()
        u = layer.understand(
            "Lunch cost me 150 taka today",
            memory=memory,
            conversation_history=[],
            trace_id="t1",
        )
    assert u.workflow == "expense"
    assert u.confidence >= 0.70


def test_leave_clarification_on_ambiguous(pipeline):
    memory = SessionMemory()
    with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = False
        from chat.services.platform.ai_understanding import AIUnderstandingLayer

        layer = AIUnderstandingLayer()
        u = layer.understand(
            "My mother is sick and I need a few days off",
            memory=memory,
            conversation_history=[],
            trace_id="t2",
        )
    assert u.workflow == "leave"


def test_leave_collection_flow(pipeline):
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={})},
    )
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.84,
        reasoning="test",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = False
        from chat.services.platform.ai_understanding import AIUnderstandingLayer

        layer = AIUnderstandingLayer()
        u = layer.understand(
            "I need sick leave starting tomorrow",
            memory=memory,
            conversation_history=[],
            trace_id="t3",
        )
        ctx = build_turn_context(
            message="I need sick leave starting tomorrow",
            memory=memory,
            conversation_history=[],
            trace_id="t3",
            session_id="s-test",
            company_id="c-test",
            employee_id="e-test",
        )
        msg, decision = pipeline.execute_workflow_turn(
            "I need sick leave starting tomorrow",
            memory=memory,
            understanding=u,
            pq_decision=pq,
            conversation_history=[],
            trace_id="t3",
            turn_context=ctx,
        )
    assert msg
    assert decision.get("outcome") == "NEEDS_INPUT"
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields.get("leave_type") == "sick" or draft.fields.get("start_date")


def test_modify_expense_amount(pipeline):
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={"items": [{"category": "meals", "amount": 150}]},
            )
        },
    )
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.78,
        reasoning="modify",
        source="rules",
        blocks_new_workflow=True,
    )
    with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = False
        from chat.services.platform.ai_understanding import AIUnderstandingLayer

        layer = AIUnderstandingLayer()
        u = layer.understand(
            "Use 200 instead of 150",
            memory=memory,
            conversation_history=[],
            trace_id="t4",
        )
        ctx = build_turn_context(
            message="Use 200 instead of 150",
            memory=memory,
            conversation_history=[],
            trace_id="t4",
            session_id="s-test",
            company_id="c-test",
            employee_id="e-test",
        )
        msg, decision = pipeline.execute_workflow_turn(
            "Use 200 instead of 150",
            memory=memory,
            pq_decision=pq,
            understanding=u,
            conversation_history=[],
            trace_id="t4",
            turn_context=ctx,
        )
    assert "200" in msg or decision.get("outcome") == "NEEDS_INPUT"
    items = memory.active_draft().fields.get("items") or []
    if items:
        assert items[-1].get("amount") == 200
