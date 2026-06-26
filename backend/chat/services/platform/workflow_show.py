"""Cross-workflow summary / show routing — LLM with rules fallback."""

from __future__ import annotations

from typing import Any

_SHOW_SEMANTICS_CACHE: dict[str, str | None] = {}


def clear_workflow_show_cache(trace_id: str = "") -> None:
    key = (trace_id or "").strip()
    if not key:
        _SHOW_SEMANTICS_CACHE.clear()
        return
    _SHOW_SEMANTICS_CACHE.pop(key, None)


def session_has_workflow_context(memory) -> bool:
    """Session has active, suspended, or draft leave/expense state worth LLM show routing."""
    if not memory:
        return False
    if getattr(memory, "active_workflow", None):
        return True
    if memory.suspended_workflows:
        return True
    if session_leave_draft(memory):
        return True
    for draft_id in ("expense", "leave", "default"):
        draft = (memory.workflow_drafts or {}).get(draft_id)
        if draft and getattr(draft, "workflow_id", None) in ("leave", "expense"):
            if (getattr(draft, "fields", None) or {}) or getattr(draft, "locked", False):
                return True
    return False


def session_leave_draft(memory) -> Any | None:
    """Pending or in-progress leave draft — prefer the most complete / active draft."""
    from chat.services.platform.field_engine import leave_draft_in_progress

    if not memory:
        return None

    def _score(draft: Any) -> int:
        fields = dict(getattr(draft, "fields", None) or {})
        score = 0
        for key, weight in (
            ("leave_type", 4),
            ("day_scope", 4),
            ("start_date", 3),
            ("end_date", 2),
            ("reason", 1),
        ):
            if fields.get(key) not in (None, ""):
                score += weight
        if fields.get("reason_skipped"):
            score += 1
        score += int(getattr(draft, "version", 0) or 0) // 10
        return score

    candidates: list[Any] = []
    drafts = memory.workflow_drafts or {}
    aw = memory.active_workflow
    active_did = str(aw.draft_id or "") if aw and aw.id == "leave" else ""

    for draft_id, draft in drafts.items():
        if not draft or getattr(draft, "workflow_id", None) != "leave":
            continue
        if leave_draft_in_progress(draft) or getattr(draft, "locked", False):
            candidates.append(draft)

    for sw in memory.suspended_workflows or []:
        if not isinstance(sw, dict) or sw.get("workflow_id") != "leave":
            continue
        did = str(sw.get("draft_id") or "leave")
        draft = drafts.get(did)
        if draft and getattr(draft, "workflow_id", None) == "leave":
            if draft not in candidates:
                candidates.append(draft)

    if not candidates:
        return None

    def _sort_key(draft: Any) -> tuple[int, int]:
        did = ""
        for k, v in drafts.items():
            if v is draft:
                did = str(k)
                break
        active_boost = 10 if active_did and did == active_did else 0
        return (_score(draft) + active_boost, int(getattr(draft, "version", 0) or 0))

    return max(candidates, key=_sort_key)


def _rules_workflow_show_target(message: str, *, active_workflow_id: str = "") -> str | None:
    import re

    from chat.services.platform.intent_rules import (
        is_expense_draft_query,
        is_resume_workflow_request,
        is_summary_request,
        is_workflow_show_request,
    )

    raw = (message or "").strip()
    if not raw:
        return None
    low = raw.lower()
    active = (active_workflow_id or "").strip().lower()

    leave_named = bool(re.search(r"\b(leave|chuti|chhuti|ছুটি)\b", low, re.I))
    expense_named = bool(
        is_expense_draft_query(raw)
        or re.search(r"\b(expense|expenses|khoroch|kharcha|claim)\b", low, re.I)
    )

    if leave_named and (
        is_summary_request(raw)
        or is_resume_workflow_request(raw, workflow_id="leave")
        or is_workflow_show_request(raw, workflow_id="leave")
    ):
        return "leave"

    if expense_named and (
        is_expense_draft_query(raw)
        or is_summary_request(raw)
        or is_workflow_show_request(raw, workflow_id="expense")
    ):
        return "expense"

    if active and is_workflow_show_request(raw, workflow_id=active):
        return active

    if is_summary_request(raw) and active in ("leave", "expense"):
        return active

    return None


def resolve_workflow_show_target(
    message: str,
    memory=None,
    *,
    active_workflow_id: str = "",
    trace_id: str = "",
    conversation_history: list[str] | None = None,
) -> str | None:
    """Return leave|expense when user asks to view that workflow; None if not a show request."""
    from chat.services.platform.workflow_cancel import (
        message_might_request_workflow_cancel,
        resolve_workflow_cancel_target,
    )

    if message_might_request_workflow_cancel(message):
        if resolve_workflow_cancel_target(
            message,
            memory,
            active_workflow_id=active_workflow_id,
            trace_id=trace_id,
            conversation_history=conversation_history,
        ):
            return None

    key = (trace_id or "").strip()
    if key and key in _SHOW_SEMANTICS_CACHE:
        return _SHOW_SEMANTICS_CACHE[key]

    rules = _rules_workflow_show_target(message, active_workflow_id=active_workflow_id)
    if not rules and not _message_might_be_show_request(message):
        if key:
            _SHOW_SEMANTICS_CACHE[key] = None
        return None

    active = (active_workflow_id or "").strip().lower()
    if memory and getattr(memory, "active_workflow", None):
        active = active or str(memory.active_workflow.id or "").strip().lower()

    payload: dict[str, Any] = {
        "message": (message or "").strip(),
        "active_workflow": active or None,
        "has_pending_leave_draft": bool(session_leave_draft(memory)),
        "has_suspended_leave": any(
            isinstance(sw, dict) and sw.get("workflow_id") == "leave"
            for sw in (getattr(memory, "suspended_workflows", None) or [])
        ),
        "submitted_leave_count": len(
            list((getattr(memory, "conversation_facts", None) or {}).get("submitted_leave_ranges") or [])
        ),
        "recent_user_messages": list(conversation_history or [])[-4:],
    }

    llm_target: str | None = None
    try:
        import json

        from chat.services.llm_client import LLMClient
        from chat.services.platform.llm_prompts import WORKFLOW_SHOW_TARGET_SYSTEM

        parsed = LLMClient().chat_json(
            system_prompt=WORKFLOW_SHOW_TARGET_SYSTEM,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            trace_id=trace_id or "",
            scope="workflow-show",
        )
        if isinstance(parsed, dict):
            raw_target = str(parsed.get("target_workflow") or "").strip().lower()
            if raw_target in ("leave", "expense"):
                llm_target = raw_target
            elif raw_target == "active" and active in ("leave", "expense"):
                llm_target = active
    except Exception:
        llm_target = None

    out = rules if rules in ("leave", "expense") else (llm_target or rules)
    if key:
        _SHOW_SEMANTICS_CACHE[key] = out
    return out


def _message_might_be_show_request(message: str) -> bool:
    from chat.services.platform.intent_rules import (
        is_expense_draft_query,
        is_resume_workflow_request,
        is_summary_request,
    )

    raw = (message or "").strip()
    if not raw:
        return False
    return bool(
        is_summary_request(raw)
        or is_expense_draft_query(raw)
        or is_resume_workflow_request(raw, workflow_id=None)
    )
