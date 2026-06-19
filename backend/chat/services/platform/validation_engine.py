"""Validation engine — runs rules from workflow definitions."""

from __future__ import annotations

from datetime import date
from typing import Any

from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.schemas import ValidationRule, WorkflowDefinition
from chat.services.session_memory import WorkflowDraft


class ValidationEngine:
    def __init__(self) -> None:
        self.fields = FieldEngine()

    def validate(
        self,
        draft: WorkflowDraft,
        definition: WorkflowDefinition,
        *,
        lang: str = "en",
    ) -> list[str]:
        errors: list[str] = []
        for rule in definition.validation_rules:
            msg = self._check_rule(draft, definition, rule, lang=lang)
            if msg:
                errors.append(msg)
        return errors

    def _check_rule(
        self,
        draft: WorkflowDraft,
        definition: WorkflowDefinition,
        rule: ValidationRule,
        *,
        lang: str,
    ) -> str | None:
        p = rule.params
        rtype = rule.rule_type

        if rtype == "required":
            field = p.get("field")
            if field and draft.fields.get(field) in (None, ""):
                return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "conditional_required":
            when_field = p.get("when_field")
            expected = p.get("equals")
            require_field = p.get("require_field")
            if draft.fields.get(when_field) == expected:
                if p.get("min_days") and not self.fields._leave_days_gte(draft, int(p["min_days"])):
                    return None
                val = draft.fields.get(require_field)
                if val in (None, "", False, "false", "False"):
                    return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "date_gte":
            a = draft.fields.get(p.get("field"))
            b = draft.fields.get(p.get("gte_field"))
            if a and b:
                try:
                    if date.fromisoformat(str(a)[:10]) < date.fromisoformat(str(b)[:10]):
                        return rule.message_bn if lang == "bn" else rule.message_en
                except ValueError:
                    pass

        if rtype == "date_not_future":
            field = p.get("field")
            val = draft.fields.get(field)
            if val:
                try:
                    if date.fromisoformat(str(val)[:10]) > date.today():
                        return rule.message_bn if lang == "bn" else rule.message_en
                except ValueError:
                    pass

        if rtype == "min_line_items":
            items = draft.fields.get(p.get("field")) or []
            if len(items) < int(p.get("min", 1)):
                return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "line_item_amount_gt":
            items = draft.fields.get(p.get("field")) or []
            amt_field = p.get("amount_field", "amount")
            minimum = float(p.get("min", 0))
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    if float(item.get(amt_field) or 0) <= minimum:
                        return rule.message_bn if lang == "bn" else rule.message_en
                except (TypeError, ValueError):
                    return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "travel_route_required":
            items = draft.fields.get(p.get("items_field", "items")) or []
            cat = p.get("category", "travel")
            if any(isinstance(i, dict) and i.get("category") == cat for i in items):
                if not draft.fields.get(p.get("from_field")) or not draft.fields.get(p.get("to_field")):
                    return rule.message_bn if lang == "bn" else rule.message_en

        return None

    def all_valid(self, draft: WorkflowDraft, definition: WorkflowDefinition, *, lang: str = "en") -> bool:
        return not self.validate(draft, definition, lang=lang)
