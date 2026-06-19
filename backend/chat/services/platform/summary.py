"""Workflow draft summaries with totals."""

from __future__ import annotations

from typing import Any

from chat.services.platform.field_extractors import format_iso_date_display
from chat.services.session_memory import SessionMemory, WorkflowDraft


def _items(draft: WorkflowDraft) -> list[dict[str, Any]]:
    return list(draft.fields.get("items") or draft.line_items or [])


def expense_total(draft: WorkflowDraft) -> float:
    total = 0.0
    for item in _items(draft):
        try:
            total += float(item.get("amount") or 0)
        except (TypeError, ValueError):
            continue
    return total


def format_expense_summary(draft: WorkflowDraft, *, lang: str = "en", include_status: bool = True) -> str:
    lines = ["**Expense Summary**", ""]
    if include_status:
        status = "Submitted" if draft.locked else "Pending (not submitted)"
        lines.append(f"Status: **{status}**")
        if draft.submitted_request_id:
            lines.append(f"Reference: **`{draft.submitted_request_id}`**")
        lines.append("")

    items = _items(draft)
    if not items:
        lines.append("_No expense items yet._")
    else:
        lines.append("**Items:**")
        for i, item in enumerate(items, 1):
            cat = item.get("category", "?")
            amt = item.get("amount", "?")
            desc = item.get("description") or ""
            lines.append(f"  {i}. [{cat}] {amt} taka — {desc}")
        lines.append("")
        lines.append(f"**Total: {expense_total(draft):.0f} taka**")

    route_from = draft.fields.get("from_location")
    route_to = draft.fields.get("to_location")
    if route_from or route_to:
        lines.append(f"Route: {route_from or '?'} → {route_to or '?'}")
    incurred = draft.fields.get("incurred_date")
    if incurred:
        lines.append(f"Date: {incurred}")
    return "\n".join(lines)


def format_leave_summary(draft: WorkflowDraft, *, lang: str = "en") -> str:
    lines = ["**Leave Summary**", ""]
    status = "Submitted" if draft.locked else "Pending (not submitted)"
    lines.append(f"Status: **{status}**")
    if draft.submitted_request_id:
        lines.append(f"Reference: **`{draft.submitted_request_id}`**")
    lines.append("")
    for key in ("leave_type", "start_date", "end_date", "day_scope", "half_day_period", "reason"):
        val = draft.fields.get(key)
        if val not in (None, ""):
            if key in ("start_date", "end_date"):
                display = format_iso_date_display(str(val))
            elif key == "day_scope":
                display = str(val).replace("_", " ")
            elif key == "reason":
                text = str(val).strip()
                display = text if len(text) <= 120 else text[:117] + "..."
            else:
                display = val
            lines.append(f"- **{key.replace('_', ' ')}**: {display}")
    missing = []
    for key in ("leave_type", "start_date", "day_scope"):
        if not draft.fields.get(key):
            missing.append(key.replace("_", " "))
    if missing and not draft.locked:
        lines.append("")
        lines.append(f"_Still needed: {', '.join(missing)}_")
    return "\n".join(lines)


def format_session_context(memory: SessionMemory, *, lang: str = "en") -> str:
    """Explain current session state when user says 'ha' without pending confirmation."""
    wf = memory.active_workflow
    if not wf:
        if lang == "bn":
            return "এখন কোনো active workflow নেই। Leave বা expense শুরু করতে বলুন।"
        return "There is no active workflow. Say if you'd like to start leave or expense."

    draft = memory.active_draft()
    if not draft:
        return "No draft found for the active workflow."

    if wf.id == "expense":
        base = format_expense_summary(draft, lang=lang)
        if memory.pending_confirmation == "submit":
            extra = "\n\n_Awaiting your **yes** to submit this expense._"
        else:
            extra = "\n\n_Tell me if you want to **submit**, **review**, or **modify** something._"
        return base + extra

    if wf.id == "leave":
        base = format_leave_summary(draft, lang=lang)
        if memory.pending_confirmation == "submit":
            extra = "\n\n_Awaiting your **yes** to submit this leave request._"
        else:
            extra = "\n\n_Tell me if you want to **submit**, add missing info, or **cancel**._"
        return base + extra

    return f"Active workflow: **{wf.id}** (stage: {wf.stage})."
