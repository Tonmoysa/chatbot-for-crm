"""Turn semantics: pending-slot vs navigation/meta (answers_pending_field SSOT)."""

from __future__ import annotations

import re
from typing import Any

from chat.services.platform.banglish_normalize import normalize_banglish_message
from chat.services.platform.intent_rules import (
    is_bare_confirmation,
    is_bare_rejection,
    is_cancel_workflow_message,
    is_greeting_or_chitchat,
    is_modify_request,
    is_workflow_show_request,
)
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import SessionMemory

_META_COMPLAINT_RE = re.compile(
    r"(?:"
    r"\b(?:keno|kano|why|how\s+come)\b|"
    r"\b(?:tumi|apni|you|bot)\b.{0,50}\b(?:bujh|understand|context|modify|korcho|korchho|bolcho)\b|"
    r"\bcontext\b.{0,25}\b(?:bujh|nai|na|paro\s*nah)\b|"
    r"(?:reason|date|type)\s*(?:ta\s+)?(?:keno|kano)\b|"
    r"\bbujh(?:i|te)?\s+nai\b|"
    r"\bbujhi\s+nai\b"
    r")",
    re.I | re.UNICODE,
)

_PROCESS_QUESTION_RE = re.compile(
    r"(?:"
    r"^\s*(?:ar\s+)?ki\s+lagbe\s*\??\s*$|"
    r"^\s*(?:aaro|ar)\s+ki\s+(?:lagbe|dorkar|proyojon)\s*\??\s*$|"
    r"^\s*what\s+else\s+(?:do\s+you\s+)?need\s*\??\s*$|"
    r"^\s*(?:ki\s+ki|koto\s+ta)\s+(?:lagbe|dorkar|proyojon)\s*\??\s*$|"
    r"^\s*(?:submit|review)\s+(?:er\s+)?(?:jonno|for)\s+(?:ar\s+)?ki\s+lagbe\s*\??\s*$"
    r")",
    re.I | re.UNICODE,
)

_INTERNAL_REASONING_MARKERS = re.compile(
    r"(?:"
    r"user\s+is\s+asking\s+for\s+clarification|"
    r"likely\s+due\s+to\s+the\s+assistant|"
    r"deterministic\s+rules\s+override"
    r")",
    re.I,
)


def last_assistant_message(conversation_history: list[str] | tuple[str, ...] | None) -> str | None:
    for line in reversed(list(conversation_history or ())):
        if line.startswith("Assistant:"):
            return line[len("Assistant:") :].strip()
    return None


def recent_user_messages(
    conversation_history: list[str] | tuple[str, ...] | None,
    *,
    limit: int = 3,
) -> list[str]:
    """Last N user lines from orchestrator history (prefix parsing only)."""
    out: list[str] = []
    for line in reversed(list(conversation_history or ())):
        if line.startswith("User:"):
            text = line[len("User:") :].strip()
            if text:
                out.append(text)
            if len(out) >= limit:
                break
    return list(reversed(out))


def expense_conversation_payload(
    conversation_history: list[str] | tuple[str, ...] | None,
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Conversation context for expense draft interpreter (Phase 1)."""
    history = list(conversation_history or ())[-limit:]
    return {
        "conversation_history": history,
        "recent_user_messages": recent_user_messages(history, limit=3),
        "last_assistant_message": last_assistant_message(history),
    }


def infer_pending_kind(memory: SessionMemory) -> str | None:
    """Hint for Understanding layer — maps session state to pending routing mode."""
    if (memory.pending_confirmation or "") == "submit":
        return "submit_confirm"
    pq = memory.pending_question
    if not pq:
        return None
    return "answer_pending"


def wizard_semantics_active(memory: SessionMemory) -> bool:
    """Active draft, pending slot, or submit confirm — prefer semantic (LLM) routing."""
    if memory.pending_confirmation:
        return True
    if memory.pending_question:
        return True
    if memory.active_workflow and memory.active_draft():
        return True
    return False


def is_workflow_meta_complaint(message: str) -> bool:
    """User questions bot behavior / context — not a slot answer."""
    raw = normalize_banglish_message(message)
    if not raw or is_greeting_or_chitchat(raw):
        return False
    return bool(_META_COMPLAINT_RE.search(raw))


def is_process_question(message: str) -> bool:
    """User asks what else is needed — not a field value."""
    raw = normalize_banglish_message(message)
    if not raw:
        return False
    return bool(_PROCESS_QUESTION_RE.search(raw))


def is_internal_reasoning_text(text: str) -> bool:
    """Block LLM/gatekeeper reasoning from user-facing copy."""
    return bool(_INTERNAL_REASONING_MARKERS.search(text or ""))


def _looks_like_slot_answer(message: str, *, field: str, workflow_id: str) -> bool:
    raw = normalize_banglish_message(message)
    if not raw:
        return False
    if is_workflow_show_request(raw, workflow_id=workflow_id):
        return False
    if is_workflow_meta_complaint(raw) or is_process_question(raw):
        return False
    if is_modify_request(raw):
        return False
    if is_cancel_workflow_message(raw, workflow_id=workflow_id):
        return False
    if is_bare_confirmation(raw) or is_bare_rejection(raw):
        if (field or "") == "reason":
            return False
        return field in ("leave_type", "day_scope", "half_day_period", "medical_document")
    if field in ("start_date", "end_date"):
        return len(raw.split()) <= 25
    if field == "leave_type":
        low = raw.lower().strip()
        if low in ("annual", "sick", "lwop", "unpaid"):
            return True
        return len(raw.split()) <= 6
    if field in ("reason", "day_scope", "half_day_period"):
        return len(raw.split()) <= 25 and not is_process_question(raw)
    return True


def enrich_answers_pending_field(
    message: str,
    memory: SessionMemory,
    result: UnderstandingResult,
) -> UnderstandingResult:
    """Set answers_pending_field; align action with navigation/meta/process signals."""
    message = normalize_banglish_message(message)
    pq = memory.pending_question
    aw = memory.active_workflow
    active_id = aw.id if aw else ""

    if memory.pending_confirmation == "submit":
        result.answers_pending_field = False
        if is_bare_confirmation(message) and result.action not in (
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SUBMIT.value,
            UnderstandingAction.CANCEL.value,
        ):
            result.action = UnderstandingAction.CONFIRM.value
            result.workflow = active_id or result.workflow or "leave"
        return result

    if not pq:
        return result

    wf_id = (pq.workflow_id or active_id or "").strip().lower()
    explicit = result.answers_pending_field

    expense_intent = str((result.entities or {}).get("expense_intent") or "").lower()
    if expense_intent in ("fix_mistake", "anti_summary") and result.field_updates:
        result.answers_pending_field = False
        return result

    if wf_id == "expense" and expense_intent in (
        "anti_summary",
        "clarify_modify",
        "clarify_delete",
        "fix_mistake",
    ):
        result.answers_pending_field = False
        if expense_intent == "anti_summary":
            result.entities = {**(result.entities or {}), "meta_complaint": True, "anti_summary": True}
        if not result.field_updates:
            result.action = UnderstandingAction.CLARIFICATION_NEEDED.value
            result.workflow = "expense"
        return result

    if wf_id == "expense":
        from chat.services.platform.field_extractors.expense import is_expense_anti_summary_request

        if is_expense_anti_summary_request(message):
            result.answers_pending_field = False
            result.field_updates = []
            result.action = UnderstandingAction.CLARIFICATION_NEEDED.value
            result.workflow = "expense"
            result.entities = {
                **(result.entities or {}),
                "meta_complaint": True,
                "anti_summary": True,
                "expense_intent": "anti_summary",
            }
            return result

    if is_workflow_show_request(message, workflow_id=wf_id or None):
        result.answers_pending_field = False
        result.field_updates = []
        if result.action not in (
            UnderstandingAction.REVIEW.value,
            UnderstandingAction.MODIFY.value,
            UnderstandingAction.CANCEL.value,
        ):
            result.action = UnderstandingAction.REVIEW.value
            result.workflow = wf_id or result.workflow or active_id
        return result

    if is_workflow_meta_complaint(message):
        result.answers_pending_field = False
        result.field_updates = []
        result.action = UnderstandingAction.CLARIFICATION_NEEDED.value
        result.workflow = active_id or result.workflow
        result.entities = {**(result.entities or {}), "meta_complaint": True}
        if not result.reasoning:
            result.reasoning = "User is questioning bot behavior or draft state."
        return result

    if active_id == "expense" or (pq and pq.workflow_id == "expense"):
        from chat.services.platform.field_extractors.expense import (
            is_expense_anti_summary_request,
            is_expense_collect_complaint,
        )

        if is_expense_collect_complaint(message) and not result.field_updates:
            result.answers_pending_field = False
            result.field_updates = []
            result.action = UnderstandingAction.CLARIFICATION_NEEDED.value
            result.workflow = "expense"
            result.entities = {**(result.entities or {}), "meta_complaint": True}
            if is_expense_anti_summary_request(message):
                result.entities["anti_summary"] = True
            if not result.reasoning:
                result.reasoning = "Expense user frustration or complaint."
            return result

    if is_process_question(message):
        result.answers_pending_field = False
        result.field_updates = []
        result.action = UnderstandingAction.CLARIFICATION_NEEDED.value
        result.workflow = active_id or result.workflow
        result.entities = {**(result.entities or {}), "process_question": True}
        return result

    if result.action in (
        UnderstandingAction.REVIEW.value,
        UnderstandingAction.MODIFY.value,
        UnderstandingAction.DELETE.value,
        UnderstandingAction.CANCEL.value,
        UnderstandingAction.SUBMIT.value,
        UnderstandingAction.SWITCH.value,
    ):
        result.answers_pending_field = False
        return result

    if explicit is not None:
        if explicit is False and result.field_updates:
            result.field_updates = [
                u
                for u in result.field_updates
                if str(u.field) != (pq.field or "")
            ]
        return result

    if result.action in (
        UnderstandingAction.COLLECT.value,
        UnderstandingAction.CONFIRM.value,
    ):
        result.answers_pending_field = _looks_like_slot_answer(
            message, field=pq.field, workflow_id=wf_id
        )
        if result.answers_pending_field is False:
            result.field_updates = [
                u for u in (result.field_updates or []) if str(u.field) != pq.field
            ]
        return result

    if result.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
        result.answers_pending_field = False
        return result

    result.answers_pending_field = _looks_like_slot_answer(
        message, field=pq.field, workflow_id=wf_id
    )
    if result.answers_pending_field is False:
        result.field_updates = [
            u for u in (result.field_updates or []) if str(u.field) != pq.field
        ]
    return result


def understanding_session_context(
    memory: SessionMemory,
    conversation_history: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Extra JSON context for LLM understanding."""
    pq = memory.pending_question
    draft = memory.active_draft()
    return {
        "pending_kind": infer_pending_kind(memory),
        "wizard_active": wizard_semantics_active(memory),
        "last_assistant_message": last_assistant_message(conversation_history),
        "pending_field": pq.field if pq else None,
        "pending_field_help": (
            f"User was asked for '{pq.field}': {pq.prompt}" if pq else None
        ),
        "draft_field_names": list((draft.fields or {}).keys()) if draft else [],
        "draft_fields": dict(draft.fields) if draft else {},
    }
