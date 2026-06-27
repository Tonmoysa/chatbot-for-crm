"""Structured pipeline logging and full-turn trace (Phase 10)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from django.conf import settings

logger = logging.getLogger("hr_chatbot")

TURN_TRACE_SCHEMA_VERSION = "turn_trace_v1"

_REDACT = re.compile(
    r"(api[_-]?key|password|token|authorization)\s*[:=]\s*[^\s,}\"]+",
    re.I,
)

_turn_traces: dict[str, dict[str, Any]] = {}


def is_full_turn_observability_enabled() -> bool:
    return getattr(settings, "FULL_TURN_OBSERVABILITY", True)


def _safe_message(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    t = _REDACT.sub(r"\1=<redacted>", text)
    if len(t) > max_len:
        t = t[:max_len] + "…"
    return t


def log_turn_context_layer(trace_id: str, layer: str, ctx: Any) -> None:
    """Log immutable turn snapshot fields at layer boundaries (Phase 1)."""
    draft = getattr(ctx, "draft_snapshot", None) or {}
    log_step(
        trace_id,
        "turn_context_layer",
        {
            "layer": layer,
            "trace_id": getattr(ctx, "trace_id", trace_id),
            "active_workflow_id": getattr(ctx, "active_workflow_id", None),
            "pending_question_field": getattr(ctx, "pending_question_field", None),
            "pending_confirmation": getattr(ctx, "pending_confirmation", None),
            "reply_language": getattr(ctx, "reply_language", None),
            "draft_version": draft.get("version") if isinstance(draft, dict) else None,
        },
    )


def log_step(trace_id: str, step: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"step": step, **(extra or {})}
    if not settings.DEBUG:
        if "user_message" in payload:
            payload["user_message"] = _safe_message(str(payload["user_message"]))
        if "assistant_message" in payload:
            payload["assistant_message"] = _safe_message(str(payload["assistant_message"]))
    logger.info(
        "pipeline_step trace_id=%s %s",
        trace_id,
        json.dumps(payload, default=str),
        extra={"trace_id": trace_id},
    )


def log_expense_draft_turn(
    trace_id: str,
    *,
    message: str,
    turn: dict[str, Any],
    llm_used: bool,
    wizard_fallback: bool = False,
) -> None:
    """Phase D — expense interpreter outcome for replay / regression."""
    patches = list(turn.get("item_patches") or [])
    delete_n = len(turn.get("delete_indices") or [])
    intent = str(turn.get("intent") or "").lower()
    has_actionable = bool(patches or delete_n) and intent in (
        "fix_mistake",
        "answer_pending",
        "add",
        "update",
        "modify_review",
        "delete",
        "correct",
    )
    log_step(
        trace_id,
        "expense_draft_turn",
        {
            "intent": intent,
            "patch_count": len(patches) + delete_n,
            "llm_used": llm_used,
            "wizard_fallback": wizard_fallback,
            "llm_degraded": bool(turn.get("llm_degraded")),
            "message_len": len(message or ""),
            "has_actionable_patches": has_actionable,
        },
    )


def classify_leave_field_apply_mode(message: str, *, memory: Any) -> str:
    """modify vs collect — for observability and audit."""
    from chat.services.platform.field_extractors.leave import (
        _llm_client_configured,
        is_leave_review_mode,
    )

    if memory and is_leave_review_mode(memory):
        if not _llm_client_configured():
            return "legacy_review_fallback"
        return "semantic_review"
    pq = getattr(memory, "pending_question", None)
    if pq and getattr(pq, "workflow_id", "") == "leave":
        if not _llm_client_configured():
            return "collect_deterministic"
        return "collect_slot"
    pending = getattr(memory, "pending_confirmation", None)
    if pending == "submit":
        return "submit_review"
    return "collect"


def log_field_updates_applied(
    trace_id: str,
    *,
    workflow_id: str,
    draft_id: str,
    updates: list[dict[str, Any]],
    before_fields: dict[str, Any],
    after_fields: dict[str, Any],
    apply_mode: str,
    message: str = "",
) -> None:
    """Log targeted field mutations — which fields changed and why (Phase 4)."""
    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(before_fields) | set(after_fields)):
        old = before_fields.get(key)
        new = after_fields.get(key)
        if old != new:
            changed[key] = {"before": old, "after": new}

    log_step(
        trace_id or "no-trace",
        "field_updates_applied",
        {
            "workflow_id": workflow_id,
            "draft_id": draft_id,
            "apply_mode": apply_mode,
            "fields_requested": [u.get("field") for u in updates if u.get("field")],
            "fields_changed": changed,
            "user_message": message,
        },
    )


def snapshot_workflow_state(memory: Any) -> dict[str, Any]:
    """Compact workflow snapshot for turn_before / turn_after logs."""
    snap = dict(memory.to_workflow_state())
    events = snap.pop("events", [])
    snap["events_tail"] = events[-3:] if events else []
    snap["events_count"] = len(events)
    drafts = snap.get("workflow_drafts") or {}
    if isinstance(drafts, dict):
        snap["draft_field_keys"] = {
            draft_id: sorted((draft or {}).get("fields") or {})
            for draft_id, draft in drafts.items()
        }
    return snap


def diff_workflow_state(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Highlight workflow fields that changed between turn start and end."""
    if not before or not after:
        return {}

    delta: dict[str, Any] = {}
    for key in (
        "active_workflow",
        "pending_question",
        "pending_confirmation",
        "turn_count",
        "last_action",
    ):
        b_val = before.get(key)
        a_val = after.get(key)
        if b_val != a_val:
            delta[key] = {"before": b_val, "after": a_val}

    b_drafts = before.get("workflow_drafts") or {}
    a_drafts = after.get("workflow_drafts") or {}
    draft_delta: dict[str, Any] = {}
    for draft_id in sorted(set(b_drafts) | set(a_drafts)):
        b_fields = (b_drafts.get(draft_id) or {}).get("fields") or {}
        a_fields = (a_drafts.get(draft_id) or {}).get("fields") or {}
        if b_fields != a_fields:
            draft_delta[draft_id] = {"before": b_fields, "after": a_fields}
    if draft_delta:
        delta["draft_fields"] = draft_delta

    b_suspended = before.get("suspended_workflows") or []
    a_suspended = after.get("suspended_workflows") or []
    if b_suspended != a_suspended:
        delta["suspended_workflows"] = {
            "before_count": len(b_suspended),
            "after_count": len(a_suspended),
        }
    return delta


def begin_turn_trace(
    trace_id: str,
    *,
    user_message: str,
    state_before: dict[str, Any],
    session_id: str = "",
    company_id: str = "",
    employee_id: str = "",
) -> None:
    if not is_full_turn_observability_enabled():
        return
    _turn_traces[trace_id] = {
        "turn_trace_schema_version": TURN_TRACE_SCHEMA_VERSION,
        "session_id": session_id,
        "company_id": company_id,
        "employee_id": employee_id,
        "user_message": user_message,
        "state_before": state_before,
    }
    log_step(
        trace_id,
        "turn_begin",
        {
            "user_message": user_message,
            "state_before": state_before,
            "session_id": session_id,
        },
    )


def patch_turn_trace(trace_id: str, **fields: Any) -> None:
    if not is_full_turn_observability_enabled():
        return
    trace = _turn_traces.get(trace_id)
    if trace is None:
        return
    trace.update(fields)


def _response_summary(envelope: dict[str, Any] | None) -> dict[str, Any]:
    if not envelope:
        return {}
    decision = envelope.get("decision") or {}
    response = envelope.get("response") or {}
    message = str(response.get("message") or "")
    return {
        "intent": envelope.get("intent"),
        "status": envelope.get("status"),
        "decision_outcome": decision.get("outcome"),
        "rules_applied": decision.get("rules_applied"),
        "request_id": response.get("request_id"),
        "response_status": response.get("status"),
        "message_preview": message[:240],
        "message_len": len(message),
    }


def finish_turn_trace(
    trace_id: str,
    *,
    state_after: dict[str, Any],
    assistant_message: str = "",
    envelope: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Emit consolidated turn_complete log — context through response."""
    if not is_full_turn_observability_enabled():
        return None

    trace = _turn_traces.pop(trace_id, {})
    state_before = trace.get("state_before") or {}
    record: dict[str, Any] = {
        **trace,
        "state_after": state_after,
        "state_delta": diff_workflow_state(state_before, state_after),
        "assistant_message": assistant_message,
        "response": _response_summary(envelope),
    }
    if envelope:
        record["envelope_intent"] = envelope.get("intent")
        record["envelope_status"] = envelope.get("status")

    log_step(trace_id, "turn_complete", record)
    return record


def format_turn_replay(record: dict[str, Any]) -> str:
    """Human-readable replay from a turn_complete log record (Phase 10 gate)."""
    lines: list[str] = []
    lines.append(f"Turn replay ({record.get('turn_trace_schema_version', 'unknown')})")
    lines.append(f"User: {record.get('user_message', '')}")

    context = record.get("context")
    if context:
        lines.append(
            "Context: "
            f"workflow={context.get('active_workflow_id')} "
            f"stage={context.get('active_workflow_stage')} "
            f"pending={context.get('pending_question_field')} "
            f"confirm={context.get('pending_confirmation')}"
        )

    understanding = record.get("understanding") or {}
    if understanding:
        lines.append(
            "Understanding: "
            f"{understanding.get('workflow')}/{understanding.get('action')} "
            f"conf={understanding.get('confidence')} "
            f"source={understanding.get('source')}"
        )
        if understanding.get("reasoning"):
            lines.append(f"  reason: {understanding['reasoning']}")

    pq = record.get("pq_decision") or {}
    if pq:
        lines.append(
            "Decision: "
            f"{pq.get('kind')} conf={pq.get('confidence')} source={pq.get('source')}"
        )
        if pq.get("reasoning"):
            lines.append(f"  reason: {pq['reasoning']}")

    plan = record.get("execution_plan")
    if plan:
        lines.append(
            "Plan: "
            f"{plan.get('workflow_id')} ops={plan.get('ops')} reason={plan.get('reason')}"
        )
    elif record.get("plan_skipped"):
        lines.append(f"Plan: skipped ({record.get('plan_skip_reason', 'none')})")

    delta = record.get("state_delta") or {}
    if delta:
        lines.append(f"State delta keys: {', '.join(sorted(delta))}")
        for key, change in delta.items():
            if key == "draft_fields":
                for draft_id, fields in (change or {}).items():
                    lines.append(f"  draft {draft_id}: {fields.get('before')} -> {fields.get('after')}")
            elif isinstance(change, dict) and "before" in change and "after" in change:
                lines.append(f"  {key}: {change['before']} -> {change['after']}")

    response = record.get("response") or {}
    if response:
        lines.append(
            "Response: "
            f"outcome={response.get('decision_outcome')} "
            f"intent={response.get('intent')} "
            f"rules={response.get('rules_applied')}"
        )
        preview = response.get("message_preview")
        if preview:
            lines.append(f"  preview: {preview}")

    assistant = record.get("assistant_message")
    if assistant and not (response or {}).get("message_preview"):
        lines.append(f"Assistant: {str(assistant)[:240]}")

    return "\n".join(lines)


def replay_turn_from_log(record: dict[str, Any]) -> dict[str, Any]:
    """Structured replay payload for debugging failed tests from logs."""
    return {
        "schema_version": record.get("turn_trace_schema_version", TURN_TRACE_SCHEMA_VERSION),
        "user_message": record.get("user_message"),
        "context": record.get("context"),
        "understanding": record.get("understanding"),
        "pq_decision": record.get("pq_decision"),
        "execution_plan": record.get("execution_plan"),
        "turn_decision": record.get("turn_decision"),
        "state_delta": record.get("state_delta"),
        "response": record.get("response"),
        "replay_text": format_turn_replay(record),
    }
