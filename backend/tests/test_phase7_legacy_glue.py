"""Phase 7/8 — legacy path gating and platform glue."""

from __future__ import annotations

from chat.services.orchestrator import ChatOrchestrator
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    is_expense_platform_scenario,
    is_leave_platform_scenario,
)


def test_is_leave_platform_scenario_active_workflow():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting")
    assert is_leave_platform_scenario(memory) is True


def test_is_leave_platform_scenario_pending_question():
    memory = SessionMemory()
    memory.pending_question = PendingQuestion(
        field="leave_type",
        prompt="Which leave type?",
        workflow_id="leave",
        asked_at_turn=1,
    )
    assert is_leave_platform_scenario(memory) is True


def test_is_leave_platform_scenario_understanding_only():
    memory = SessionMemory()
    u = UnderstandingResult(workflow="leave", action=UnderstandingAction.START.value)
    assert is_leave_platform_scenario(memory, understanding=u) is True


def test_legacy_path_blocked_for_leave():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting")
    u = UnderstandingResult(workflow="leave", action=UnderstandingAction.COLLECT.value)
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="answer pending",
        source="test",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    assert ChatOrchestrator._legacy_path_allowed(memory, u, pq) is False


def test_legacy_path_blocked_for_expense_when_new_arch():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="expense", stage="collecting")
    u = UnderstandingResult(workflow="expense", action=UnderstandingAction.COLLECT.value)
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="answer pending",
        source="test",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    assert is_expense_platform_scenario(memory, understanding=u) is True
    assert ChatOrchestrator._legacy_path_allowed(memory, u, pq) is False
