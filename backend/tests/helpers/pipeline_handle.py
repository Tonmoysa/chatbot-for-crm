"""Test helper: call WorkflowPipeline.execute_workflow_turn with rules understanding."""

from __future__ import annotations

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.pending_question_engine import PendingQuestionDecision
from chat.services.session_memory import SessionMemory, build_turn_context
from tests.helpers.yaml_scenario_runner import llm_disabled


def handle_with_rules_understanding(
    pipeline: WorkflowPipeline,
    message: str,
    *,
    memory: SessionMemory,
    pq_decision: PendingQuestionDecision,
    conversation_history: list[str] | None = None,
    trace_id: str = "",
    **kwargs,
):
    with llm_disabled():
        layer = AIUnderstandingLayer()
        understanding = layer.understand(
            message,
            memory=memory,
            conversation_history=conversation_history or [],
            trace_id=trace_id,
        )
        turn_context = kwargs.pop("turn_context", None)
        if turn_context is None:
            turn_context = build_turn_context(
                message=message,
                memory=memory,
                conversation_history=conversation_history or [],
                trace_id=trace_id or "test",
                session_id=str(kwargs.get("session_id") or "test-session"),
                company_id=str(kwargs.get("company_id") or "test-company"),
                employee_id=str(kwargs.get("employee_id") or "test-employee"),
                idempotency_key=str(kwargs.get("idempotency_key") or ""),
            )
        return pipeline.execute_workflow_turn(
            message,
            memory=memory,
            understanding=understanding,
            pq_decision=pq_decision,
            conversation_history=conversation_history or [],
            trace_id=trace_id,
            turn_context=turn_context,
            route_source=kwargs.pop("route_source", "pending"),
            **kwargs,
        )
