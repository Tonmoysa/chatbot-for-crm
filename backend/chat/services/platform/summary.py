"""Workflow draft summaries with totals."""

from __future__ import annotations

from typing import Any

from chat.services.platform.field_extractors import format_iso_date_display
from chat.services.platform.field_extractors.expense import (
    category_display_name,
    is_travel_category,
    is_valid_expense_route,
    normalize_expense_category,
    sync_expense_draft_fields,
)
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


def _format_item_line(item: dict[str, Any], *, index: int | None = None) -> str:
    prefix = f"{index}. " if index is not None else ""
    cat = category_display_name(normalize_expense_category(item.get("category")) or "?")
    amt = item.get("amount", "?")
    line = f"{prefix}**{cat}** — {amt} taka"
    if is_travel_category(item.get("category")) and is_valid_expense_route(
        item.get("from_location"), item.get("to_location")
    ):
        frm = item.get("from_location")
        to = item.get("to_location")
        line += f" ({frm} → {to})"
    return line


def format_expense_missing_section(
    draft: WorkflowDraft,
    *,
    lang: str = "en",
    focus_only: bool = True,
) -> str:
    from chat.services.platform.field_extractors.expense import build_pending_queue, sync_expense_draft_fields

    fields = sync_expense_draft_fields(dict(draft.fields or {}))
    items = fields.get("items") or []
    queue = build_pending_queue(items)
    if not queue:
        return ""
    if focus_only:
        entry = queue[0]
        label = {
            "category": "Category",
            "amount": "Amount",
            "route": "Route",
        }.get(entry.field, entry.field.replace("_", " ").title())
        header = "**Still needed**" if lang == "en" else "**অপেক্ষমাণ**"
        return f"{header}\n- {label}"
    header = "**Pending Information**" if lang == "en" else "**অপেক্ষমাণ তথ্য**"
    lines = [header, ""]
    grouped: dict[int, list[str]] = {}
    for entry in queue:
        label = {
            "category": "Category",
            "amount": "Amount",
            "route": "Route",
        }.get(entry.field, entry.field.replace("_", " ").title())
        grouped.setdefault(entry.item_index, []).append(label)
    for idx in sorted(grouped):
        lines.append(f"Expense {idx + 1}")
        lines.append("Missing:" if lang == "en" else "অনুপস্থিত:")
        for label in grouped[idx]:
            lines.append(f"- {label}")
        lines.append("")
    return "\n".join(lines).strip()


def format_expense_summary(
    draft: WorkflowDraft,
    *,
    lang: str = "en",
    include_status: bool = True,
    memory: SessionMemory | None = None,
) -> str:
    lines: list[str] = ["**Expense Summary**", ""]

    submitted_rows = []
    if memory is not None:
        submitted_rows = list((memory.conversation_facts or {}).get("submitted_expenses") or [])

    if submitted_rows:
        lines.append("**Submitted Expenses**")
        for row in submitted_rows:
            if not isinstance(row, dict):
                continue
            rid = row.get("request_id") or ""
            header = f"Reference: `{rid}`" if rid else "Submitted claim"
            lines.append(header)
            for i, item in enumerate(row.get("items") or [], 1):
                if isinstance(item, dict):
                    lines.append(f"  {_format_item_line(item, index=i)}")
            lines.append("")

    lines.append("**Current Expenses**")
    items = _items(draft)
    if draft.locked:
        lines.append("_No open draft — start a new expense when ready._")
    elif not items:
        lines.append("_No expense items yet._")
    else:
        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                lines.append(_format_item_line(item, index=i))
        lines.append("")
        lines.append(f"**Total: {expense_total(draft):.0f} taka**")

    if include_status and not draft.locked:
        lines.append("")
        lines.append("Status: **Pending (not submitted)**")
    elif draft.locked and draft.submitted_request_id:
        lines.append("")
        lines.append(f"Status: **Submitted** — Reference: `{draft.submitted_request_id}`")

    incurred = draft.fields.get("incurred_date")
    if incurred and not draft.locked:
        lines.append(f"Date: {format_iso_date_display(str(incurred))}")

    missing = format_expense_missing_section(draft, lang=lang, focus_only=False)
    if missing and not draft.locked:
        lines.extend(["", missing])

    return "\n".join(lines)


def format_expense_status_report(
    memory: SessionMemory,
    *,
    lang: str = "en",
    focus_draft: WorkflowDraft | None = None,
) -> str:
    """Submitted + open pending expense drafts — full session expense picture."""
    submitted_rows = list((memory.conversation_facts or {}).get("submitted_expenses") or [])
    open_drafts: list[WorkflowDraft] = []
    seen: set[str] = set()
    for draft_id, draft in (memory.workflow_drafts or {}).items():
        if not draft or draft.workflow_id != "expense":
            continue
        if draft.locked or draft.status == "submitted":
            continue
        items = list((draft.fields or {}).get("items") or [])
        if not items:
            continue
        key = str(draft_id)
        if key in seen:
            continue
        seen.add(key)
        open_drafts.append(draft)

    if focus_draft and not focus_draft.locked and focus_draft not in open_drafts:
        items = list((focus_draft.fields or {}).get("items") or [])
        if items:
            open_drafts.insert(0, focus_draft)

    if not submitted_rows and not open_drafts:
        if lang == "bn":
            return (
                "Apnar kono **expense** shuru hoyni ebong kono submit kora claim o nei.\n\n"
                "Expense add korte category ar amount bolen — jemon `lunch 120 taka`."
            )
        if lang == "banglish":
            return (
                "Apnar kono **expense** ekhono shuru hoyni, ar kono submit kora claim o nei.\n\n"
                "Expense add korte category ar amount bolen — jemon `lunch 120 taka`."
            )
        return (
            "You have not started any **expense** yet, and there are no submitted claims.\n\n"
            "To add an expense, tell me the category and amount — e.g. `lunch 120 taka`."
        )

    lines: list[str] = []
    if lang == "bn":
        lines.append("**Apnar Expense**")
    elif lang == "banglish":
        lines.append("**Apnar Expense**")
    else:
        lines.append("**Your Expenses**")
    lines.append("")

    if submitted_rows:
        header = "**Submit kora expense**" if lang in ("bn", "banglish") else "**Submitted expenses**"
        lines.append(header)
        for row in submitted_rows:
            if not isinstance(row, dict):
                continue
            rid = row.get("request_id") or ""
            if rid:
                lines.append(f"- Reference: `{rid}`")
            for i, item in enumerate(row.get("items") or [], 1):
                if isinstance(item, dict):
                    lines.append(f"  {_format_item_line(item, index=i)}")
        lines.append("")

    if open_drafts:
        header = "**Pending expense (submit hoyni)**" if lang in ("bn", "banglish") else "**Pending expenses (not submitted)**"
        lines.append(header)
        for draft in open_drafts:
            fields = sync_expense_draft_fields(dict(draft.fields or {}))
            items = list(fields.get("items") or [])
            for i, item in enumerate(items, 1):
                if isinstance(item, dict):
                    lines.append(_format_item_line(item, index=i))
            if items:
                lines.append(f"  **Total: {expense_total(draft):.0f} taka**")
            incurred = fields.get("incurred_date")
            if incurred:
                lines.append(f"  Date: {format_iso_date_display(str(incurred))}")
            lines.append("")

    if submitted_rows and not open_drafts:
        extra = (
            "_Kono open pending expense draft nei — notun expense add korte parben._"
            if lang in ("bn", "banglish")
            else "_No open pending draft — you can start a new expense anytime._"
        )
        lines.append(extra)

    return "\n".join(lines).strip()


def format_expense_collect_recap(
    draft: WorkflowDraft,
    *,
    lang: str = "en",
    update_notes: list[str] | None = None,
    include_focus_question: str | None = None,
) -> str:
    """Compact list + pending block shown after each expense collect turn."""
    fields = sync_expense_draft_fields(dict(draft.fields or {}))
    items = fields.get("items") or []
    lines: list[str] = []
    for note in update_notes or []:
        if note:
            lines.append(f"✓ {note}")
    if lines:
        lines.append("")
    header = "**Current Expenses**" if lang == "en" else "**বর্তমান খরচ**"
    lines.append(header)
    if not items:
        lines.append("_No expense items yet._" if lang == "en" else "_এখনো কোনো খরচ নেই।_")
    else:
        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                lines.append(_format_item_line(item, index=i))
        lines.append("")
        lines.append(f"**Total: {expense_total(draft):.0f} taka**")
    missing = format_expense_missing_section(draft, lang=lang, focus_only=True)
    if missing:
        lines.extend(["", missing])
    if include_focus_question:
        lines.extend(["", include_focus_question])
    return "\n".join(lines).strip()


def format_leave_summary(draft: WorkflowDraft, *, lang: str = "en", include_status: bool = True) -> str:
    from chat.services.platform.response_composer import ResponseComposer

    return ResponseComposer().leave_summary(draft, lang=lang, include_status=include_status)


def format_session_context(memory: SessionMemory, *, lang: str = "en") -> str:
    """Explain current session state when user says 'ha' without pending confirmation."""
    from chat.services.platform.response_composer import localized

    wf = memory.active_workflow
    if not wf:
        if lang == "bn":
            return "এখন কোনো active workflow নেই। Leave বা expense শুরু করতে বলুন।"
        return "There is no active workflow. Say if you'd like to start leave or expense."

    draft = memory.active_draft()
    if not draft:
        return "No draft found for the active workflow."

    if wf.id == "expense":
        base = format_expense_summary(draft, lang=lang, memory=memory)
        if memory.pending_confirmation == "submit":
            extra = localized(
                lang,
                en="\n\n_Awaiting your **yes** to submit this expense. You can also **modify** or **cancel**._",
                bn="\n\n_Expense submit করতে **ha** বলুন। **modify** বা **cancel**-ও করতে পারেন।_",
                banglish="\n\n_Expense submit korte **ha** bolen. **modify** ba **cancel** o korte paren._",
            )
        else:
            extra = localized(
                lang,
                en="\n\n_Tell me if you want to **submit**, **review**, or **modify** something._",
                bn="\n\n_**submit**, **review**, বা **modify** করতে বলুন।_",
                banglish="\n\n_**submit**, **review**, ba **modify** korte bolen._",
            )
        base = base + extra
    elif wf.id == "leave":
        base = format_leave_summary(draft, lang=lang)
        if memory.pending_confirmation == "submit":
            extra = localized(
                lang,
                en="\n\n_Awaiting your **yes** to submit this leave request. You can also **modify** or **cancel**._",
                bn="\n\n_Leave submit করতে **ha** বলুন। **modify** বা **cancel**-ও করতে পারেন।_",
                banglish="\n\n_Leave submit korte **ha** bolen. **modify** ba **cancel** o korte paren._",
            )
        else:
            extra = localized(
                lang,
                en="\n\n_Tell me if you want to **submit**, add missing info, or **cancel**._",
                bn="\n\n_**submit**, missing info দিন, বা **cancel** করুন।_",
                banglish="\n\n_**submit**, missing info din, ba **cancel** korun._",
            )
        base = base + extra
    else:
        return f"Active workflow: **{wf.id}** (stage: {wf.stage})."

    suspended = memory.suspended_workflows or []
    if suspended:
        sw = suspended[-1]
        sw_id = sw.workflow_id
        if sw_id == "expense":
            pause_hint = localized(
                lang,
                en=f"\n\n_(Your **expense** request is paused — say **expense continue** to resume.)_",
                bn=f"\n\n_(আপনার **expense** request pause আছে — resume করতে **expense continue** বলুন।)_",
                banglish=f"\n\n_(Apnar **expense** request pause ache — resume korte **expense continue** bolen.)_",
            )
        elif sw_id == "leave":
            pause_hint = localized(
                lang,
                en=f"\n\n_(Your **leave** request is paused — say **leave continue** to resume.)_",
                bn=f"\n\n_(আপনার **leave** request pause আছে — resume করতে **leave continue** বলুন।)_",
                banglish=f"\n\n_(Apnar **leave** request pause ache — resume korte **leave continue** bolen.)_",
            )
        else:
            pause_hint = ""
        base = base + pause_hint
    return base
