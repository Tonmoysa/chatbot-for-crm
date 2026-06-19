"""Universal field engine — config-driven collect, update, next question."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from chat.services.platform.field_extractors.leave import sanitize_leave_type_value
from chat.services.platform.field_extractors import (
    extract_expense_items,
    extract_leave_fields,
    format_iso_date_display,
    parse_leave_field,
    parse_relative_date,
    parse_route,
)
from chat.services.platform.schemas import FieldDefinition, FieldUpdate, WorkflowDefinition
from chat.services.session_memory import PendingQuestion, SessionMemory, WorkflowDraft


class FieldEngine:
    def field_is_active(self, field: FieldDefinition, draft: WorkflowDraft) -> bool:
        cond = field.conditional or {}
        if not cond:
            return True
        when_field = cond.get("field")
        if when_field:
            expected = cond.get("equals")
            actual = draft.fields.get(when_field)
            if actual != expected:
                return False
        if cond.get("any_item_category"):
            cat = cond["any_item_category"]
            items = draft.fields.get("items") or draft.line_items or []
            if not any(isinstance(i, dict) and i.get("category") == cat for i in items):
                return False
        if cond.get("min_days"):
            if not self._leave_days_gte(draft, int(cond["min_days"])):
                return False
        return True

    def missing_fields(self, draft: WorkflowDraft, definition: WorkflowDefinition) -> list[str]:
        missing: list[str] = []
        for f in definition.fields:
            if not self.field_is_active(f, draft):
                continue
            if f.field_type == "line_items":
                items = draft.fields.get("items") or draft.line_items or []
                if f.required and not items:
                    missing.append(f.name)
                continue
            if f.required and draft.fields.get(f.name) in (None, ""):
                missing.append(f.name)
            if f.optional:
                continue
            if not f.required and f.name not in draft.fields:
                if f.conditional and self.field_is_active(f, draft):
                    if draft.fields.get(f.name) in (None, ""):
                        missing.append(f.name)
            elif f.conditional and self.field_is_active(f, draft):
                if draft.fields.get(f.name) in (None, "", False, "false", "False"):
                    missing.append(f.name)
        return missing

    def apply_updates(self, draft: WorkflowDraft, updates: list[FieldUpdate], *, message: str = "") -> None:
        for upd in updates:
            if upd.field == "leave_type" and message:
                clean = sanitize_leave_type_value(message, str(upd.value or ""))
                if not clean:
                    continue
                upd = FieldUpdate(field="leave_type", value=clean, action=upd.action, item_index=upd.item_index)
            if upd.field == "medical_document" and upd.value in (False, "false", "False", "no", "nai", "nei"):
                continue
            self._apply_one(draft, upd)
        draft.version += 1

    def _apply_one(self, draft: WorkflowDraft, upd: FieldUpdate) -> None:
        if upd.action == "delete":
            draft.fields.pop(upd.field, None)
            return
        if upd.field == "items":
            items = list(draft.fields.get("items") or draft.line_items or [])
            if upd.action == "append" and isinstance(upd.value, dict):
                items.append(dict(upd.value))
            elif upd.action in ("update", "update_last") and isinstance(upd.value, dict):
                if upd.action == "update_last" and items:
                    items[-1].update(upd.value)
                elif upd.item_index is not None and 0 <= upd.item_index < len(items):
                    items[upd.item_index].update(upd.value)
            elif upd.item_index is not None and 0 <= upd.item_index < len(items):
                if upd.action == "delete":
                    items.pop(upd.item_index)
                elif isinstance(upd.value, dict):
                    items[upd.item_index].update(upd.value)
                else:
                    items[upd.item_index] = upd.value
            draft.fields["items"] = items
            draft.line_items = items
            return
        draft.fields[upd.field] = upd.value

    def next_question(
        self,
        memory: SessionMemory,
        draft: WorkflowDraft,
        definition: WorkflowDefinition,
        *,
        lang: str = "en",
    ) -> PendingQuestion | None:
        # Travel route takes priority when bus/travel items exist
        items = draft.fields.get("items") or draft.line_items or []
        has_travel = any(isinstance(i, dict) and i.get("category") == "travel" for i in items)
        if has_travel:
            if not draft.fields.get("from_location"):
                return PendingQuestion(
                    field="from_location",
                    prompt="Where did you travel from?" if lang != "bn" else "কোথা থেকে যাত্রা?",
                    workflow_id=definition.workflow_id,
                    asked_at_turn=memory.turn_count,
                )
            if not draft.fields.get("to_location"):
                return PendingQuestion(
                    field="to_location",
                    prompt="Where did you travel to?" if lang != "bn" else "কোথায় গিয়েছেন?",
                    workflow_id=definition.workflow_id,
                    asked_at_turn=memory.turn_count,
                )

        for name in self.missing_fields(draft, definition):
            fdef = definition.get_field(name)
            if not fdef:
                continue
            prompt = fdef.prompt_bn if lang == "bn" else fdef.prompt_en
            if not prompt:
                prompt = f"Please provide {name.replace('_', ' ')}."
            return PendingQuestion(
                field=name,
                prompt=prompt,
                workflow_id=definition.workflow_id,
                asked_at_turn=memory.turn_count,
            )
        return None

    def build_review(self, draft: WorkflowDraft, definition: WorkflowDefinition) -> str:
        lines = [f"**{definition.name} — Review**", ""]

        def _fmt(name: str, val: Any) -> str:
            if name in ("start_date", "end_date", "incurred_date") and val:
                return format_iso_date_display(str(val))
            if name == "day_scope" and val:
                return str(val).replace("_", " ")
            if name == "reason" and val:
                text = str(val).strip()
                return text if len(text) <= 120 else text[:117] + "..."
            return str(val)

        for f in definition.fields:
            if not self.field_is_active(f, draft):
                continue
            val = draft.fields.get(f.name)
            if f.field_type == "line_items":
                items = draft.fields.get("items") or draft.line_items or []
                if not items:
                    continue
                lines.append(f"- **{f.name}**:")
                for i, item in enumerate(items, 1):
                    lines.append(f"  {i}. {item}")
            elif val not in (None, ""):
                lines.append(f"- **{f.name.replace('_', ' ')}**: {_fmt(f.name, val)}")
        lines.append("")
        lines.append("_Reply **yes** to submit, or tell me what to change._")
        return "\n".join(lines)

    def parse_pending_field(self, workflow_id: str, field: str, message: str) -> Any:
        """Deterministic single-field parse — no intent classification."""
        wf = (workflow_id or "").strip().lower()
        if wf == "leave":
            return parse_leave_field(message, field)
        if wf == "expense" and field == "incurred_date":
            return parse_relative_date(message) or None
        if wf == "expense" and field in ("from_location", "to_location", "route"):
            route = parse_route(message)
            if route and field == "from_location":
                return route[0]
            if route and field == "to_location":
                return route[1]
            return message.strip() if message.strip() else None
        return message.strip() if message.strip() else None

    def extract_workflow_fields(self, workflow_id: str, message: str) -> dict[str, Any]:
        """Extract all parseable fields for a workflow from natural language."""
        wf = (workflow_id or "").strip().lower()
        if wf == "leave":
            return extract_leave_fields(message)
        if wf == "expense":
            out: dict[str, Any] = {}
            items = extract_expense_items(message)
            if items:
                out["items"] = items
            d = parse_relative_date(message)
            if d:
                out["incurred_date"] = d
            route = parse_route(message)
            if route:
                out["from_location"], out["to_location"] = route
            return out
        return {}

    @staticmethod
    def _leave_days_gte(draft: WorkflowDraft, min_days: int) -> bool:
        start = draft.fields.get("start_date")
        end = draft.fields.get("end_date") or start
        if not start:
            return False
        try:
            s = date.fromisoformat(str(start)[:10])
            e = date.fromisoformat(str(end)[:10])
            return (e - s).days + 1 >= min_days
        except ValueError:
            return False
