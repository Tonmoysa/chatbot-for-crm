"""Tests for Pending Question Engine precedence and guardrails."""

from unittest.mock import patch

import pytest

from chat.services.platform.ai_understanding import AIUnderstandingLayer
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


def _classify(
    engine: PendingQuestionEngine,
    message: str,
    *,
    memory: SessionMemory,
    conversation_history: list[str] | None = None,
    trace_id: str = "test",
    understanding=None,
):
    """Phase 3 — classify requires Understanding SSOT."""
    if understanding is None:
        layer = AIUnderstandingLayer()
        understanding = layer.understand(
            message,
            memory=memory,
            conversation_history=conversation_history or [],
            trace_id=trace_id,
        )
    return engine.classify(
        message,
        memory=memory,
        conversation_history=conversation_history or [],
        trace_id=trace_id,
        understanding=understanding,
    )


@pytest.fixture
def engine() -> PendingQuestionEngine:
    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm:
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
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
    decision = _classify(
        engine,
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
    decision = _classify(
        engine,
        "What is the reimbursement policy?",
        memory=_memory_with_pending(field="reason"),
        trace_id="test-2",
    )
    assert decision.kind == MessageIntentKind.ASK_POLICY
    assert decision.blocks_new_workflow is True


def test_leave_narrative_with_required_word_not_policy(engine):
    msg = (
        "Hi, amar ekta leave apply korte hobe. Agami 2 September 2026 theke 4 September 2026 "
        "porjonto ami office e aste parbo na. Amar basay kichu urgent family issue hoyeche. "
        "Amar mone hoy shob required information diyechi. Please amar leave request ta "
        "prepare kore review dekhao."
    )
    decision = _classify(engine, msg, memory=SessionMemory(), trace_id="test-leave-required")
    assert decision.kind != MessageIntentKind.ASK_POLICY


def test_status_query_during_pending_question(engine):
    decision = _classify(
        engine,
        "What is the status of MOCK-12345?",
        memory=_memory_with_pending(),
        trace_id="test-3",
    )
    assert decision.kind == MessageIntentKind.ASK_STATUS


def test_modify_with_active_draft(engine):
    memory = _memory_with_pending()
    memory.pending_question = None
    memory.workflow_drafts["default"].fields = {"amount": 150}
    decision = _classify(
        engine,
        "Use 200 instead of 150",
        memory=memory,
        trace_id="test-4",
    )
    assert decision.kind == MessageIntentKind.MODIFY_DATA


def test_strong_new_workflow_can_override_pending(engine):
    decision = _classify(
        engine,
        "I want to apply for a new leave starting Monday",
        memory=_memory_with_pending(field="reason"),
        trace_id="test-5",
    )
    assert decision.kind == MessageIntentKind.NEW_WORKFLOW
    assert decision.blocks_new_workflow is False


def test_no_pending_defers_to_new_workflow(engine):
    decision = _classify(
        engine,
        "hello",
        memory=SessionMemory(),
        trace_id="test-6",
    )
    assert decision.kind == MessageIntentKind.NEW_WORKFLOW
    assert decision.blocks_new_workflow is False


def test_out_of_scope_during_wizard(engine):
    decision = _classify(
        engine,
        "When is Eid this year?",
        memory=_memory_with_pending(),
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


def test_expense_route_answer_does_not_resume_leave(engine):
    from chat.services.platform.schemas import FieldUpdate, UnderstandingAction, UnderstandingResult
    from chat.services.session_memory import SuspendedWorkflow

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        suspended_workflows=[
            SuspendedWorkflow(workflow_id="leave", stage="confirm_submit", draft_id="leave-draft"),
        ],
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Where did you travel from and to for the first expense?",
            workflow_id="expense",
            item_index=0,
        ),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "items": [
                        {"category": "bus", "amount": 120, "missing_fields": ["route"]},
                    ],
                },
            ),
            "leave-draft": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "family emergency",
                },
            ),
        },
    )
    # Simulate misclassified UNDERSTAND output (workflow=leave during expense route answer).
    understanding = UnderstandingResult(
        goal="Answer route",
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.9,
        answers_pending_field=True,
        field_updates=[
            FieldUpdate(
                field="items",
                value={"from_location": "Mirpur", "to_location": "Motijheel"},
                item_index=0,
                action="update",
            ),
        ],
        reasoning="LLM wrongly tagged leave workflow for a route-only reply.",
        source="llm",
    )
    decision = engine.classify(
        "mirpur to motejheel",
        memory=memory,
        conversation_history=[],
        trace_id="test-expense-route-not-leave",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.ANSWER_PENDING
    assert decision.target_workflow == "expense"
    assert decision.kind != MessageIntentKind.SWITCH_WORKFLOW


def test_decision_includes_reasoning_and_confidence(engine):
    decision = _classify(
        engine,
        "sick leave",
        memory=_memory_with_pending(field="leave_type"),
        trace_id="test-8",
    )
    log = decision.to_log_dict()
    assert "reasoning" in log
    assert "confidence" in log
    assert log["kind"] == MessageIntentKind.ANSWER_PENDING.value
