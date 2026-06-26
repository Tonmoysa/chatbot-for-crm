"""Cross-workflow cancel routing — LLM with rules fallback."""

from __future__ import annotations

from typing import Any

_CANCEL_SEMANTICS_CACHE: dict[str, str | None] = {}


def clear_workflow_cancel_cache(trace_id: str = "") -> None:
    key = (trace_id or "").strip()
    if not key:
        _CANCEL_SEMANTICS_CACHE.clear()
        return
    _CANCEL_SEMANTICS_CACHE.pop(key, None)


def message_might_request_workflow_cancel(message: str) -> bool:
    """Cheap pre-filter before LLM cancel routing."""
    import re

    raw = (message or "").strip()
    if not raw:
        return False
    return bool(re.search(r"\b(cancel|batil|bandho|abort|discard|stop)\b", raw, re.I))


def _contextual_cancel_phrase(message: str) -> bool:
    import re

    raw = (message or "").strip()
    return bool(
        re.match(
            r"^\s*(?:cancel|batil|bandho|abort|discard|stop)"
            r"(?:\s+(?:it|this|that|the\s+request|my\s+request))?\s*\.?$",
            raw,
            re.I,
        )
    )


def _last_assistant_lower(conversation_history: list[str] | None) -> str:
    from chat.services.platform.turn_semantics import last_assistant_message

    return (last_assistant_message(conversation_history) or "").lower()


def _rules_workflow_cancel_target(
    message: str,
    memory=None,
    *,
    conversation_history: list[str] | None = None,
) -> str | None:
    from chat.services.platform.intent_rules import is_cancel_workflow_message
    from chat.services.platform.workflow_show import session_has_workflow_context, session_leave_draft

    raw = (message or "").strip()
    if not raw or not message_might_request_workflow_cancel(raw):
        return None
    if memory and not session_has_workflow_context(memory):
        return None

    for wf in ("leave", "expense"):
        if is_cancel_workflow_message(raw, workflow_id=wf):
            return wf

    if not _contextual_cancel_phrase(raw):
        return None

    last_bot = _last_assistant_lower(conversation_history)
    if any(tok in last_bot for tok in ("chuti saransho", "leave summary", "chutir dhoron", "ছুটি সারাংশ")):
        return "leave"
    if any(tok in last_bot for tok in ("expense", "khoroch", "submit korar age", "lunch")):
        if "leave" not in last_bot or "expense" in last_bot:
            return "expense"

    if memory:
        last_action = str(getattr(memory, "last_action", "") or "")
        entities = getattr(memory, "last_entities", None) or {}
        if last_action == "field_collected":
            for ev in reversed(list(getattr(memory, "events_tail", None) or [])):
                if isinstance(ev, dict) and ev.get("event_type") == "field_collected":
                    if ev.get("workflow_id") == "leave":
                        return "leave"
                    if ev.get("workflow_id") == "expense":
                        return "expense"
                    break
        turn_u = entities.get("turn_understanding") if isinstance(entities, dict) else None
        if isinstance(turn_u, dict):
            show_tgt = str((turn_u.get("entities") or {}).get("show_workflow_target") or "").lower()
            if show_tgt in ("leave", "expense"):
                return show_tgt
            wf = str(turn_u.get("workflow") or "").lower()
            if wf in ("leave", "expense"):
                return wf

        suspended = [
            str(sw.get("workflow_id") or "").strip().lower()
            for sw in (memory.suspended_workflows or [])
            if isinstance(sw, dict)
        ]
        leave_pending = bool(session_leave_draft(memory))
        if leave_pending and suspended.count("leave") >= 1 and suspended.count("expense") == 0:
            return "leave"
        if suspended == ["leave"]:
            return "leave"
        if suspended == ["expense"]:
            return "expense"
        if leave_pending and "expense" not in suspended:
            return "leave"

    return None


def resolve_workflow_cancel_target(
    message: str,
    memory=None,
    *,
    active_workflow_id: str = "",
    trace_id: str = "",
    conversation_history: list[str] | None = None,
) -> str | None:
    """Return leave|expense when user wants to cancel that workflow draft."""
    if not message_might_request_workflow_cancel(message):
        return None

    key = (trace_id or "").strip()
    if key and key in _CANCEL_SEMANTICS_CACHE:
        return _CANCEL_SEMANTICS_CACHE[key]

    rules = _rules_workflow_cancel_target(
        message,
        memory,
        conversation_history=conversation_history,
    )

    active = (active_workflow_id or "").strip().lower()
    if memory and getattr(memory, "active_workflow", None):
        active = active or str(memory.active_workflow.id or "").strip().lower()

    from chat.services.platform.workflow_show import session_has_workflow_context

    if not rules and not session_has_workflow_context(memory):
        if key:
            _CANCEL_SEMANTICS_CACHE[key] = None
        return None

    payload: dict[str, Any] = {
        "message": (message or "").strip(),
        "active_workflow": active or None,
        "suspended_workflows": [
            {"workflow_id": sw.get("workflow_id"), "stage": sw.get("stage")}
            for sw in (getattr(memory, "suspended_workflows", None) or [])
            if isinstance(sw, dict)
        ],
        "last_assistant_message": _last_assistant_lower(conversation_history),
        "last_action": getattr(memory, "last_action", None) if memory else None,
        "recent_user_messages": list(conversation_history or [])[-4:],
    }

    llm_target: str | None = None
    try:
        import json

        from chat.services.llm_client import LLMClient
        from chat.services.platform.llm_prompts import WORKFLOW_CANCEL_TARGET_SYSTEM

        parsed = LLMClient().chat_json(
            system_prompt=WORKFLOW_CANCEL_TARGET_SYSTEM,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            trace_id=trace_id or "",
            scope="workflow-cancel",
        )
        if isinstance(parsed, dict):
            if not parsed.get("is_cancel"):
                llm_target = None
            else:
                raw_target = str(parsed.get("target_workflow") or "").strip().lower()
                if raw_target in ("leave", "expense"):
                    llm_target = raw_target
                elif raw_target == "active" and active in ("leave", "expense"):
                    llm_target = active
    except Exception:
        llm_target = None

    out = llm_target or rules
    if key:
        _CANCEL_SEMANTICS_CACHE[key] = out
    return out
