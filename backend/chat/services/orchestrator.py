"""
Central chat orchestrator — thin coordinator (Phase 2).

Builds TurnContext, runs Understanding once, delegates all routing to Decision Core.
"""

from __future__ import annotations

from typing import Any

from chat.constants import INTENT_UNKNOWN
from chat.services.llm_client import clear_llm_trace_state
from chat.services.observability import (
    begin_turn_trace,
    finish_turn_trace,
    log_step,
    log_turn_context_layer,
    patch_turn_trace,
    snapshot_workflow_state,
)
from chat.services.pending_question_engine import PendingQuestionEngine
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import UnderstandingResult
from chat.services.session_memory import SessionMemory, build_turn_context
from chat.services.session_store import SessionStore


class ChatOrchestrator:
    """User input → TurnContext → Understanding → Decision Core → response."""

    def __init__(self) -> None:
        self.session_store = SessionStore()
        self.decision_core = PendingQuestionEngine()
        self.understanding_layer = AIUnderstandingLayer()
        self.workflow_pipeline = WorkflowPipeline()

    def run_chat(
        self,
        *,
        message: str,
        session_id: str | None,
        company_id: str,
        employee_id: str,
        trace_id: str,
        document_text: str | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        clear_llm_trace_state(trace_id)

        bundle = self.session_store.open(
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id or "",
        )
        session = bundle.session
        session_memory = bundle.memory
        context_lines = bundle.transcript_lines

        turn_context = build_turn_context(
            message=message,
            memory=session_memory,
            conversation_history=context_lines,
            trace_id=trace_id,
            session_id=session.session_id,
            company_id=company_id,
            employee_id=employee_id,
            document_text=document_text,
            idempotency_key=idempotency_key,
        )
        from chat.services.platform.turn_semantics import infer_pending_kind

        pending_kind = infer_pending_kind(session_memory)
        begin_turn_trace(
            trace_id,
            user_message=message,
            state_before=snapshot_workflow_state(session_memory),
            session_id=session.session_id,
            company_id=company_id,
            employee_id=employee_id,
        )
        log_turn_context_layer(trace_id, "orchestrator", turn_context)

        plan_shortcut = self.decision_core.detect_plan_shortcut(
            message,
            memory=session_memory,
            conversation_history=context_lines,
        )
        if plan_shortcut is not None:
            turn_understanding = self.decision_core.synthetic_understanding_for_shortcut(plan_shortcut)
            patch_turn_trace(trace_id, understanding=turn_understanding.to_dict())
            log_turn_context_layer(trace_id, "understanding", turn_context)
            return self.decision_core.decide_and_execute_turn(
                message,
                memory=session_memory,
                conversation_history=context_lines,
                trace_id=trace_id,
                understanding=turn_understanding,
                turn_context=turn_context,
                session=session,
                company_id=company_id,
                employee_id=employee_id,
                document_text=document_text,
                idempotency_key=idempotency_key,
                orchestrator=self,
                pq_override=plan_shortcut,
            )

        turn_understanding = self.understanding_layer.understand(
            message,
            memory=session_memory,
            conversation_history=context_lines,
            trace_id=trace_id,
            pending_kind=pending_kind,
        )
        understanding_pre_patches = [
            {
                "op": "merge_last_entities",
                "value": {
                    **dict(turn_understanding.entities or {}),
                    "turn_understanding": turn_understanding.to_dict(),
                },
            }
        ]
        patch_turn_trace(trace_id, understanding=turn_understanding.to_dict())
        log_turn_context_layer(trace_id, "understanding", turn_context)

        return self.decision_core.decide_and_execute_turn(
            message,
            memory=session_memory,
            conversation_history=context_lines,
            trace_id=trace_id,
            understanding=turn_understanding,
            turn_context=turn_context,
            session=session,
            company_id=company_id,
            employee_id=employee_id,
            document_text=document_text,
            idempotency_key=idempotency_key,
            orchestrator=self,
            pre_patches=understanding_pre_patches,
        )

    @staticmethod
    def _legacy_path_allowed(
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq,
    ) -> bool:
        """Legacy orchestrator routing removed (Phase 10). Kept for regression tests."""
        _ = (memory, understanding, pq)
        return False

    def _complete_workflow_turn(
        self,
        session,
        memory: SessionMemory,
        message: str,
        msg: str,
        decision: dict[str, Any],
        trace_id: str,
        understanding: UnderstandingResult,
    ) -> dict[str, Any]:
        intent = self._workflow_intent(understanding, memory)
        return self._complete_turn(
            session,
            memory,
            message,
            msg,
            trace_id,
            {
                "intent": intent,
                "entities": memory.last_entities,
                "decision": decision,
                "response": {
                    "message": msg,
                    "status": "success" if decision.get("outcome") != "ERROR" else "error",
                    "request_id": decision.get("request_id", ""),
                },
                "status": "success",
            },
        )

    def _complete_turn(
        self,
        session,
        memory: SessionMemory,
        message: str,
        msg: str,
        trace_id: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        from chat.services.translator import resolve_reply_language

        stored = (memory.last_entities or {}).get("reply_language")
        stored_lang = stored if isinstance(stored, str) else None
        reply_lang = resolve_reply_language(message, stored_lang)
        memory.last_entities = {**(memory.last_entities or {}), "reply_language": reply_lang}
        self.session_store.commit_turn(
            session,
            memory,
            user_message=message,
            assistant_message=msg,
        )
        return self._finalize(
            session,
            message,
            msg,
            trace_id,
            envelope,
            session_memory=memory,
        )

    def _finalize(
        self,
        session,
        user_message: str,
        assistant_message: str,
        trace_id: str,
        envelope: dict[str, Any],
        *,
        session_memory: SessionMemory | None = None,
    ) -> dict[str, Any]:
        _ = (session, user_message, assistant_message)
        finish_turn_trace(
            trace_id,
            state_after=snapshot_workflow_state(session_memory) if session_memory else {},
            assistant_message=assistant_message,
            envelope=envelope,
        )
        envelope["trace_id"] = trace_id
        envelope["_session_id"] = session.session_id
        log_step(trace_id, "chat_complete", {"intent": envelope.get("intent")})
        return envelope

    @staticmethod
    def _workflow_intent(understanding: UnderstandingResult, memory: SessionMemory) -> str:
        if memory.active_workflow:
            return memory.active_workflow.id.upper()
        if understanding.workflow:
            return understanding.workflow.upper()
        return INTENT_UNKNOWN
