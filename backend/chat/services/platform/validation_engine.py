"""Validation engine — runs rules from workflow definitions."""

from __future__ import annotations

from datetime import date
from typing import Any

from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.response_composer import leave_validation_message
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
        collect_mode: bool = False,
    ) -> list[str]:
        errors: list[str] = []
        for rule in definition.validation_rules:
            msg = self._check_rule(
                draft,
                definition,
                rule,
                lang=lang,
                collect_mode=collect_mode,
            )
            if msg:
                errors.append(msg)
        if definition.workflow_id == "leave":
            from chat.services.platform.field_extractors.leave import is_garbage_leave_reason

            reason = draft.fields.get("reason")
            if reason and is_garbage_leave_reason(str(reason)):
                if lang == "bn":
                    errors.append(
                        "ছুটির কারণ স্পষ্ট করুন — command-style বার্তা reason হিসেবে রাখা যাবে না।"
                    )
                else:
                    errors.append(
                        "Please provide a clear leave reason (not a change command)."
                    )
        return errors

    @staticmethod
    def _expense_items_for_validation(
        draft: WorkflowDraft,
        definition: WorkflowDefinition,
        *,
        collect_mode: bool,
        items_field: str = "items",
    ) -> list[dict[str, Any]]:
        items = draft.fields.get(items_field) or []
        if not collect_mode or definition.workflow_id != "expense":
            return [i for i in items if isinstance(i, dict)]
        from chat.services.platform.field_extractors.expense import compute_item_missing_fields

        complete: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            missing = list(item.get("missing_fields") or compute_item_missing_fields(item))
            if not missing:
                complete.append(item)
        return complete

    def _check_rule(
        self,
        draft: WorkflowDraft,
        definition: WorkflowDefinition,
        rule: ValidationRule,
        *,
        lang: str,
        collect_mode: bool = False,
    ) -> str | None:
        p = rule.params
        rtype = rule.rule_type

        if rtype == "required":
            field = p.get("field")
            if field and draft.fields.get(field) in (None, ""):
                return self._rule_message(definition, rule, lang=lang)

        if rtype == "conditional_required":
            when_field = p.get("when_field")
            expected = p.get("equals")
            require_field = p.get("require_field")
            if draft.fields.get(when_field) == expected:
                if p.get("min_days") and not self.fields._leave_days_gte(draft, int(p["min_days"])):
                    return None
                val = draft.fields.get(require_field)
                if val in (None, "", False, "false", "False"):
                    return self._rule_message(definition, rule, lang=lang)

        if rtype == "date_gte":
            a = draft.fields.get(p.get("field"))
            b = draft.fields.get(p.get("gte_field"))
            if a and b:
                try:
                    if date.fromisoformat(str(a)[:10]) < date.fromisoformat(str(b)[:10]):
                        return self._rule_message(definition, rule, lang=lang)
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

        if rtype == "date_must_equal_today":
            field = p.get("field")
            val = draft.fields.get(field)
            if val:
                try:
                    if date.fromisoformat(str(val)[:10]) != date.today():
                        return rule.message_bn if lang == "bn" else rule.message_en
                except ValueError:
                    return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "min_line_items":
            items = draft.fields.get(p.get("field")) or []
            if len(items) < int(p.get("min", 1)):
                return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "line_item_amount_gt":
            items = self._expense_items_for_validation(
                draft,
                definition,
                collect_mode=collect_mode,
                items_field=p.get("field", "items"),
            )
            amt_field = p.get("amount_field", "amount")
            minimum = float(p.get("min", 0))
            for item in items:
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

        if rtype == "item_travel_route_required":
            from chat.services.platform.field_extractors.expense import (
                TRAVEL_CATEGORIES,
                is_valid_expense_route,
                normalize_expense_category,
            )

            items = self._expense_items_for_validation(
                draft,
                definition,
                collect_mode=collect_mode,
                items_field=p.get("items_field", "items"),
            )
            travel_cats = set(p.get("travel_categories") or TRAVEL_CATEGORIES)
            for item in items:
                cat = normalize_expense_category(item.get("category"))
                if cat in travel_cats:
                    if not is_valid_expense_route(item.get("from_location"), item.get("to_location")):
                        return rule.message_bn if lang == "bn" else rule.message_en

        if rtype == "supported_expense_categories":
            from chat.services.platform.field_extractors.expense import (
                SUPPORTED_CATEGORIES,
                normalize_expense_category,
            )

            allowed = set(p.get("allowed") or SUPPORTED_CATEGORIES)
            items = self._expense_items_for_validation(
                draft,
                definition,
                collect_mode=collect_mode,
                items_field=p.get("items_field", "items"),
            )
            for item in items:
                raw_cat = item.get("category")
                if raw_cat in (None, ""):
                    continue
                if normalize_expense_category(raw_cat) not in allowed:
                    return rule.message_bn if lang == "bn" else rule.message_en

        return None

    @staticmethod
    def _rule_message(definition: WorkflowDefinition, rule: ValidationRule, *, lang: str) -> str:
        if definition.workflow_id == "leave" and rule.rule_id:
            return leave_validation_message(rule.rule_id, lang=lang)
        return rule.message_bn if lang == "bn" else rule.message_en

    def all_valid(self, draft: WorkflowDraft, definition: WorkflowDefinition, *, lang: str = "en") -> bool:
        return not self.validate(draft, definition, lang=lang)
