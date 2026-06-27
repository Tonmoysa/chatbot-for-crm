"""LLM scope gate — leave, expense, policy, greetings vs off-topic chat."""

from __future__ import annotations

import json
from typing import Any

from chat.services.platform.llm_prompts import HR_ASSISTANT_SCOPE_SYSTEM
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.platform.turn_semantics import understanding_session_context
from chat.services.session_memory import SessionMemory


def _scope_payload(
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
        "draft_fields": dict((draft.fields if draft else {}) or {}),
        "draft_locked": bool(draft and draft.locked),
        **understanding_session_context(memory, conversation_history),
    }


def resolve_hr_assistant_scope(
    message: str,
    memory: SessionMemory,
    *,
    conversation_history: list[str] | None = None,
    trace_id: str = "",
) -> UnderstandingResult | None:
    """Return OOS understanding when message is outside HR assistant scope; else None."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return None

    from chat.services.platform.field_extractors.expense import is_expense_pending_field_value_answer

    if is_expense_pending_field_value_answer(raw, memory):
        return None

    from chat.services.platform.field_extractors.expense import is_expense_review_edit_turn

    if is_expense_review_edit_turn(raw, memory):
        return None

    from chat.services._policy_interrupt import is_informational_interrupt_message

    if is_informational_interrupt_message(raw):
        return None

    from chat.services.llm_client import LLMClient

    client = LLMClient()
    if not client.is_configured():
        return None

    parsed = client.chat_json(
        system_prompt=HR_ASSISTANT_SCOPE_SYSTEM,
        user_prompt=json.dumps(
            _scope_payload(raw, memory, conversation_history),
            ensure_ascii=False,
            default=str,
        ),
        trace_id=trace_id or "",
        scope="hr-assistant-scope",
    )
    if not isinstance(parsed, dict):
        return None

    in_scope = parsed.get("in_scope")
    if in_scope is None:
        in_scope = parsed.get("inScope")
    confidence = float(parsed.get("confidence") or 0.0)
    category = str(parsed.get("category") or "").strip().lower()
    reasoning = str(parsed.get("reasoning") or "Off-topic for HR assistant.").strip()

    if in_scope is True or confidence < 0.6:
        return None
    if in_scope is False and confidence >= 0.6:
        if category in ("greeting", "leave", "expense", "policy", "workflow_nav", "workflow"):
            return None
        return UnderstandingResult(
            goal="Out of scope",
            workflow="none",
            action=UnderstandingAction.NONE.value,
            confidence=max(confidence, 0.88),
            is_out_of_scope=True,
            reasoning=reasoning,
            source="llm_hr_scope",
        )
    return None
