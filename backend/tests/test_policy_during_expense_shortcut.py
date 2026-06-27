"""Policy queries during active expense must shortcut before expense domain LLM."""

from chat.services.plan_shortcut_router import (
    detect_plan_shortcut,
    synthetic_understanding_for_shortcut,
)
from chat.services.pending_question_engine import MessageIntentKind
from chat.services.platform.schemas import UnderstandingAction


def _expense_with_pending_route():
    from chat.services.session_memory import ActiveWorkflow, PendingQuestion, SessionMemory

    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting", draft_id="default"),
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Bus route?",
            workflow_id="expense",
            asked_at_turn=1,
            item_index=0,
        ),
    )


def test_detect_plan_shortcut_policy_during_expense():
    memory = _expense_with_pending_route()
    decision = detect_plan_shortcut(
        "leave policy ta bolo amake",
        memory=memory,
        conversation_history=[],
    )
    assert decision is not None
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_synthetic_understanding_for_policy_shortcut():
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
    assert "policy" in (u.goal or "").lower()


def test_is_informational_interrupt_message():
    from chat.services._policy_interrupt import is_informational_interrupt_message

    assert is_informational_interrupt_message("leave policy ta bolo amake")
    assert not is_informational_interrupt_message("mirpur theke uttora")
