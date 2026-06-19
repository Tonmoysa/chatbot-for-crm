"""Tests for Pending Question Engine precedence and guardrails."""

from unittest.mock import patch

import pytest

from chat.services.pending_question_engine import (
    MessageIntentKind,
    PendingQuestionEngine,
)
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)


@pytest.fixture
def engine() -> PendingQuestionEngine:
    with patch("chat.services.pending_question_engine.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = False
        yield PendingQuestionEngine()


def _memory_with_pending(*, field: str = "start_date", workflow_id: str = "leave") -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id=workflow_id, stage="collecting"),
        pending_question=PendingQuestion(
            field=field,
            prompt="When does your leave start?",
            workflow_id=workflow_id,
        ),
        workflow_drafts={
            "default": WorkflowDraft(workflow_id=workflow_id, fields={}),
        },
    )


def test_pending_question_blocks_new_workflow_for_ambiguous_message(engine):
    decision = engine.classify(
        "tomorrow",
        memory=_memory_with_pending(),
        conversation_history=["Assistant: When does your leave start?"],
        trace_id="test-1",
    )
    assert decision.kind == MessageIntentKind.ANSWER_PENDING
    assert decision.blocks_new_workflow is True
    assert decision.confidence >= 0.55
    assert decision.field_value == "tomorrow"


def test_policy_query_wins_over_pending_slot(engine):
    decision = engine.classify(
        "What is the reimbursement policy?",
        memory=_memory_with_pending(field="reason"),
        conversation_history=[],
        trace_id="test-2",
    )
    assert decision.kind == MessageIntentKind.ASK_POLICY
    assert decision.blocks_new_workflow is True


def test_status_query_during_pending_question(engine):
    decision = engine.classify(
        "What is the status of MOCK-12345?",
        memory=_memory_with_pending(),
        conversation_history=[],
        trace_id="test-3",
    )
    assert decision.kind == MessageIntentKind.ASK_STATUS


def test_modify_with_active_draft(engine):
    memory = _memory_with_pending()
    memory.pending_question = None
    memory.workflow_drafts["default"].fields = {"amount": 150}
    decision = engine.classify(
        "Use 200 instead of 150",
        memory=memory,
        conversation_history=[],
        trace_id="test-4",
    )
    assert decision.kind == MessageIntentKind.MODIFY_DATA


def test_strong_new_workflow_can_override_pending(engine):
    decision = engine.classify(
        "I want to apply for a new leave starting Monday",
        memory=_memory_with_pending(field="reason"),
        conversation_history=[],
        trace_id="test-5",
    )
    assert decision.kind == MessageIntentKind.NEW_WORKFLOW
    assert decision.blocks_new_workflow is False


def test_no_pending_defers_to_new_workflow(engine):
    decision = engine.classify(
        "hello",
        memory=SessionMemory(),
        conversation_history=[],
        trace_id="test-6",
    )
    assert decision.kind == MessageIntentKind.NEW_WORKFLOW
    assert decision.blocks_new_workflow is False


def test_out_of_scope_during_wizard(engine):
    decision = engine.classify(
        "When is Eid this year?",
        memory=_memory_with_pending(),
        conversation_history=[],
        trace_id="test-7",
    )
    assert decision.kind == MessageIntentKind.OUT_OF_SCOPE


def test_expense_interrupt_via_understanding(engine):
    from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult

    memory = _memory_with_pending(field="start_date")
    understanding = UnderstandingResult(
        goal="Expense interrupt",
        workflow="expense",
        action=UnderstandingAction.START.value,
        confidence=0.95,
        reasoning="Expense claim during active leave workflow.",
        source="rules",
    )
    decision = engine.classify(
        "amar ajke expense hoyeche 100 taka for bus",
        memory=memory,
        conversation_history=[],
        trace_id="test-understanding-interrupt",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "expense"
    assert decision.blocks_new_workflow is True


def test_decision_includes_reasoning_and_confidence(engine):
    decision = engine.classify(
        "sick leave",
        memory=_memory_with_pending(field="leave_type"),
        conversation_history=[],
        trace_id="test-8",
    )
    log = decision.to_log_dict()
    assert "reasoning" in log
    assert "confidence" in log
    assert log["kind"] == MessageIntentKind.ANSWER_PENDING.value
