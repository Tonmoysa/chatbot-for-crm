"""Bridge Pending Question Engine decisions to the workflow platform."""

from __future__ import annotations

from typing import Any

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
    save_session_memory,
)

_pipeline = WorkflowPipeline()


def handle_pending_decision(
    decision: PendingQuestionDecision,
    *,
    message: str,
    memory: SessionMemory,
    conversation_history: list[str],
    trace_id: str,
    understanding: UnderstandingResult | None = None,
    company_id: str = "",
    employee_id: str = "",
    session_id: str = "",
    idempotency_key: str = "",
) -> tuple[str, dict[str, Any]] | None:
    """Route PQ decisions through the platform layer stack."""
    result = _pipeline.execute_turn(
        message,
        memory=memory,
        pq_decision=decision,
        understanding=understanding,
        conversation_history=conversation_history,
        trace_id=trace_id,
        company_id=company_id,
        employee_id=employee_id,
        session_id=session_id,
        idempotency_key=idempotency_key,
    )
    if result:
        msg, envelope = result
        envelope["pending_question_decision"] = decision.to_log_dict()
        envelope.setdefault("rules_applied", []).append("PENDING_QUESTION_ENGINE")
        envelope["rules_applied"].append(decision.kind.value.upper())
        return msg, envelope
    return None


def build_pending_response(
    decision: PendingQuestionDecision,
    *,
    memory: SessionMemory,
    user_message: str,
    conversation_history: list[str] | None = None,
    trace_id: str = "",
    understanding: UnderstandingResult | None = None,
    company_id: str = "",
    employee_id: str = "",
    session_id: str = "",
    idempotency_key: str = "",
) -> tuple[str, dict[str, Any]]:
    """Build assistant message + decision envelope for pending-question routing."""
    handled = handle_pending_decision(
        decision,
        message=user_message,
        memory=memory,
        conversation_history=conversation_history or [],
        trace_id=trace_id,
        understanding=understanding,
        company_id=company_id,
        employee_id=employee_id,
        session_id=session_id,
        idempotency_key=idempotency_key,
    )
    if handled:
        return handled
    return "", {}


def set_pending_question_for_demo(
    session,
    *,
    workflow_id: str,
    field: str,
    prompt: str,
) -> SessionMemory:
    """Helper for tests to arm pending_question."""
    memory = SessionMemory.from_workflow_state(getattr(session, "workflow_state", None) or {})
    memory.active_workflow = ActiveWorkflow(id=workflow_id, stage="collecting")
    memory.workflow_drafts.setdefault(
        "default",
        WorkflowDraft(workflow_id=workflow_id),
    )
    memory.pending_question = PendingQuestion(
        field=field,
        prompt=prompt,
        workflow_id=workflow_id,
        asked_at_turn=memory.turn_count,
    )
    save_session_memory(session, memory)
    return memory
