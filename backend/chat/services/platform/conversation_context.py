"""LLM session-context resolver — bind short replies to the last bot question."""

from __future__ import annotations

import json
from typing import Any

from chat.services.platform.llm_prompts import SESSION_CONTEXT_REPLY_SYSTEM
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.platform.turn_semantics import last_assistant_message, understanding_session_context
from chat.services.session_memory import SessionMemory


def _session_payload(
    message: str,
    memory: SessionMemory,
    conversation_history: list[str] | None,
) -> dict[str, Any]:
    draft = memory.active_draft()
    return {
        "user_message": message,
        "active_workflow": memory.active_workflow.to_dict() if memory.active_workflow else None,
        "pending_confirmation": memory.pending_confirmation,
        "pending_question": memory.pending_question.to_dict() if memory.pending_question else None,
        "draft_locked": bool(draft and draft.locked),
        "draft_workflow_id": draft.workflow_id if draft else None,
        "suspended_workflows": [
            {"workflow_id": sw.workflow_id, "stage": sw.stage}
            for sw in memory.suspended_workflows
        ],
        "last_assistant_message": last_assistant_message(conversation_history),
        "conversation_history": list(conversation_history or [])[-4:],
        "last_entities": {
            k: memory.last_entities.get(k)
            for k in (
                "switch_pending_message",
                "last_expense_clarify_message",
                "expense_turn",
                "expense_intent",
            )
            if isinstance(memory.last_entities, dict) and k in memory.last_entities
        },
        **understanding_session_context(memory, conversation_history),
    }


def resolve_session_context_turn(
    message: str,
    memory: SessionMemory,
    conversation_history: list[str] | None,
    *,
    trace_id: str = "",
) -> UnderstandingResult | None:
    """Use LLM to interpret yes/no/ha and other short replies against session context."""
    from chat.services.platform.turn_semantics import should_skip_session_context_llm

    if should_skip_session_context_llm(message, memory):
        return None

    from chat.services.llm_client import LLMClient

    client = LLMClient()
    if not client.is_configured():
        return None

    raw = (message or "").strip()
    if not raw:
        return None

    payload = _session_payload(raw, memory, conversation_history)
    parsed = client.chat_json(
        system_prompt=SESSION_CONTEXT_REPLY_SYSTEM,
        user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
        trace_id=trace_id or "",
        scope="session-context",
    )
    if not isinstance(parsed, dict):
        return None

    resolution = str(parsed.get("resolution") or "none").strip().lower()
    confidence = float(parsed.get("confidence") or 0.0)
    if resolution in ("none", "") or confidence < 0.55:
        return None

    target = str(parsed.get("target_workflow") or "").strip().lower()
    reasoning = str(parsed.get("reasoning") or "Session context LLM.")

    from chat.services.platform.hr_assistant_scope import resolve_hr_assistant_scope

    scope_oos = resolve_hr_assistant_scope(
        raw,
        memory,
        conversation_history=conversation_history,
        trace_id=trace_id or "",
    )

    if resolution == "confirm_switch":
        pending = memory.pending_confirmation or ""
        parts = pending.split(":")
        to_wf = parts[2] if len(parts) == 3 else (target or "expense")
        return UnderstandingResult(
            goal=f"Switch to {to_wf}",
            workflow=to_wf,
            action=UnderstandingAction.SWITCH.value,
            confidence=max(confidence, 0.88),
            interrupt_workflow=to_wf,
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "decline_switch":
        aw = memory.active_workflow.id if memory.active_workflow else "leave"
        return UnderstandingResult(
            goal=f"Continue {aw}",
            workflow=aw,
            action=UnderstandingAction.COLLECT.value,
            confidence=max(confidence, 0.85),
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "confirm_expense_start":
        if scope_oos is not None:
            return scope_oos
        return UnderstandingResult(
            goal="Start expense",
            workflow="expense",
            action=UnderstandingAction.START.value,
            confidence=max(confidence, 0.9),
            interrupt_workflow="expense",
            entities={"expense_intent": "add", "session_confirmed_expense_start": True},
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "confirm_leave_submit":
        if scope_oos is not None:
            return scope_oos
        return UnderstandingResult(
            goal="Confirm submit",
            workflow="leave",
            action=UnderstandingAction.CONFIRM.value,
            confidence=max(confidence, 0.92),
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "decline_leave_submit":
        return UnderstandingResult(
            goal="Decline submit",
            workflow="leave",
            action=UnderstandingAction.REVIEW.value,
            confidence=max(confidence, 0.9),
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "resume_suspended":
        wf = target or "expense"
        return UnderstandingResult(
            goal=f"Resume {wf}",
            workflow=wf,
            action=UnderstandingAction.SWITCH.value,
            confidence=max(confidence, 0.88),
            interrupt_workflow=wf,
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "policy_query":
        return UnderstandingResult(
            goal="Policy question",
            workflow="policy",
            action=UnderstandingAction.QUERY.value,
            confidence=max(confidence, 0.88),
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "out_of_scope":
        from chat.services.platform.workflow_show import (
            resolve_workflow_show_target,
            session_has_workflow_context,
        )

        if session_has_workflow_context(memory):
            active_id = (memory.active_workflow.id if memory.active_workflow else "").strip().lower()
            show_wf = resolve_workflow_show_target(
                raw,
                memory,
                active_workflow_id=active_id,
                trace_id=trace_id or "",
                conversation_history=conversation_history,
            )
            if show_wf in ("leave", "expense"):
                return None
        return UnderstandingResult(
            goal="Out of scope",
            workflow="none",
            action=UnderstandingAction.NONE.value,
            confidence=max(confidence, 0.9),
            is_out_of_scope=True,
            reasoning=reasoning,
            source="llm_session_context",
        )

    if resolution == "continue_current":
        aw = memory.active_workflow.id if memory.active_workflow else "none"
        return UnderstandingResult(
            goal=f"Continue {aw}",
            workflow=aw,
            action=UnderstandingAction.COLLECT.value,
            confidence=max(confidence, 0.8),
            reasoning=reasoning,
            source="llm_session_context",
        )

    return None
