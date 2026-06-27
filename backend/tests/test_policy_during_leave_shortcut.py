"""Policy queries during active leave must shortcut before leave domain LLM."""

from chat.services.plan_shortcut_router import (
    detect_plan_shortcut,
    synthetic_understanding_for_shortcut,
)
from chat.services.pending_question_engine import MessageIntentKind
from chat.services.platform.schemas import UnderstandingAction
from chat.services.session_memory import ActiveWorkflow, PendingQuestion, SessionMemory


def _leave_with_pending_type():
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
        pending_question=PendingQuestion(
            field="leave_type",
            prompt="Kon dhoroner chuti?",
            workflow_id="leave",
            asked_at_turn=1,
        ),
    )


def test_detect_plan_shortcut_policy_during_leave():
    memory = _leave_with_pending_type()
    decision = detect_plan_shortcut(
        "sick leave policy ki?",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_detect_plan_shortcut_banglish_policy_during_leave():
    memory = _leave_with_pending_type()
    decision = detect_plan_shortcut(
        "leave policy ta bolo amake",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_detect_plan_shortcut_policy_no_active_workflow():
    """Policy query with only suspended drafts — must not hit LLM OOS gate."""
    from chat.services.pending_question_engine import informational_priority_decision
    from chat.services.session_memory import SessionMemory

    memory = SessionMemory(
        suspended_workflows=[
            {"workflow_id": "expense", "stage": "collecting", "draft_id": "expense", "suspended_at_turn": 6},
            {"workflow_id": "leave", "stage": "collecting", "draft_id": "leave", "suspended_at_turn": 10},
        ],
    )
    decision = informational_priority_decision(
        "leave policy ta amake bolo",
        memory=memory,
        conversation_history=[],
        include_policy_status=True,
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_detect_plan_shortcut_policy_word_order_variant():
    memory = _leave_with_pending_type()
    decision = detect_plan_shortcut(
        "leave policy ta amake bolo",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_synthetic_understanding_for_policy_shortcut_leave():
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ASK_POLICY,
        confidence=0.9,
        reasoning="Policy or rules query detected.",
        source="rules",
        blocks_new_workflow=True,
    )
    u = synthetic_understanding_for_shortcut(pq)
    assert u.workflow == "none"
    assert u.action == UnderstandingAction.QUERY.value
