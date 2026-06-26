"""Universal field engine — config-driven collect, update, next question."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from chat.services.platform.field_extractors.leave import (
    is_garbage_leave_reason_value,
    is_medical_document_skip_message,
    is_reason_skip_message,
    is_temporal_or_request_like_leave_reason,
    merge_leave_field_dicts,
    normalize_leave_type_value,
    sanitize_leave_type_value,
)
from chat.services.platform.field_extractors import (
    format_iso_date_display,
)
from chat.services.platform.intent_rules import is_leave_message
from chat.services.platform.response_composer import leave_field_prompt
from chat.services.platform.schemas import (
    FieldDefinition,
    FieldUpdate,
    UnderstandingAction,
    UnderstandingResult,
    WorkflowDefinition,
)
from chat.services.session_memory import PendingQuestion, SessionMemory, WorkflowDraft


LEAVE_COLLECT_ORDER = (
    "leave_type",
    "day_scope",
    "half_day_period",
    "start_date",
    "end_date",
    "reason",
    "medical_document",
)


def leave_draft_in_progress(draft: WorkflowDraft | None) -> bool:
    """True when an in-progress leave draft has meaningful fields filled."""
    if not draft or draft.workflow_id != "leave" or draft.locked:
        return False
    fields = draft.fields or {}
    return bool(fields.get("leave_type") or fields.get("start_date") or fields.get("day_scope"))


def is_duplicate_leave_attempt(
    message: str,
    understanding: UnderstandingResult,
    draft: WorkflowDraft,
) -> bool:
    """SSOT — user is starting/overlapping leave while another draft is in progress."""
    if draft.locked:
        return False
    u = understanding
    if u.action == UnderstandingAction.START.value and u.workflow == "leave":
        return True
    if not is_leave_message(message):
        return False
    new_date = None
    for upd in u.field_updates or []:
        if upd.field == "start_date" and upd.value:
            new_date = upd.value
            break
    if new_date and draft.fields.get("start_date") and new_date != draft.fields.get("start_date"):
        return True
    if re.search(r"\bleave\s+chai\b", message.lower()) and draft.fields.get("start_date"):
        return True
    if u.action == UnderstandingAction.START.value:
        return True
    return False


def duplicate_leave_arm_entities(
    memory: SessionMemory,
    understanding: UnderstandingResult,
) -> dict[str, Any]:
    """Payload for ``arm_duplicate_leave`` reducer op."""
    entities = dict(memory.last_entities or {})
    entities["duplicate_leave_updates"] = [
        {
            "field": upd.field,
            "value": upd.value,
            "action": upd.action,
            "item_index": upd.item_index,
        }
        for upd in (understanding.field_updates or [])
    ]
    return entities


def _medical_value_invalid(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "na", "nai", "nei", "n/a"):
        return True
    return False


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
        if definition.workflow_id == "leave":
            return self._leave_missing_fields(draft, definition)
        if definition.workflow_id == "expense":
            return self._expense_missing_fields(draft, definition)
        return self._generic_missing_fields(draft, definition)

    def _expense_missing_fields(self, draft: WorkflowDraft, definition: WorkflowDefinition) -> list[str]:
        from chat.services.platform.field_extractors.expense import (
            expense_draft_missing_fields,
            sync_expense_draft,
        )

        sync_expense_draft(draft)
        return expense_draft_missing_fields(draft)

    def _leave_missing_fields(self, draft: WorkflowDraft, definition: WorkflowDefinition) -> list[str]:
        missing: list[str] = []
        for name in LEAVE_COLLECT_ORDER:
            fdef = definition.get_field(name)
            if not fdef or not self.field_is_active(fdef, draft):
                continue
            if name == "reason":
                if draft.fields.get("reason") or draft.fields.get("reason_skipped"):
                    continue
                missing.append("reason")
                continue
            if name == "medical_document":
                if draft.fields.get("medical_document") or draft.fields.get("medical_document_skipped"):
                    continue
                missing.append(name)
                continue
            if name == "end_date" and not fdef.required:
                continue
            if fdef.required and draft.fields.get(name) in (None, ""):
                missing.append(name)
            elif fdef.conditional and self.field_is_active(fdef, draft):
                if draft.fields.get(name) in (None, "", False, "false", "False"):
                    missing.append(name)
        return missing

    def _generic_missing_fields(self, draft: WorkflowDraft, definition: WorkflowDefinition) -> list[str]:
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

    def apply_updates(
        self,
        draft: WorkflowDraft,
        updates: list[FieldUpdate],
        *,
        message: str = "",
        context: str = "",
    ) -> None:
        for upd in updates:
            if upd.field == "leave_type" and message:
                clean = sanitize_leave_type_value(
                    message,
                    str(upd.value or ""),
                )
                if not clean:
                    continue
                upd = FieldUpdate(field="leave_type", value=clean, action=upd.action, item_index=upd.item_index)
            if upd.field == "medical_document" and upd.value in (False, "false", "False", "no", "nai", "nei"):
                continue
            self._apply_one(draft, upd)
        draft.version += 1

    def _apply_one(self, draft: WorkflowDraft, upd: FieldUpdate) -> None:
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
            if draft.workflow_id == "expense":
                from chat.services.platform.field_extractors.expense import sync_expense_draft

                sync_expense_draft(draft)
            return
        if upd.action == "delete":
            draft.fields.pop(upd.field, None)
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
        if definition.workflow_id == "expense":
            from chat.services.platform.field_extractors.expense import next_pending_question, sync_expense_draft

            sync_expense_draft(draft)
            return next_pending_question(memory, lang=lang)

        for name in self.missing_fields(draft, definition):
            fdef = definition.get_field(name)
            if not fdef:
                continue
            if definition.workflow_id == "leave":
                prompt = leave_field_prompt(name, lang=lang)
            else:
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

    def build_review(self, draft: WorkflowDraft, definition: WorkflowDefinition, *, lang: str = "en") -> str:
        """Formal pre-submit review with submit CTA — see ``response_composer`` module doc for vs ``format_*_summary``."""
        if definition.workflow_id == "leave":
            from chat.services.platform.response_composer import ResponseComposer

            return ResponseComposer().leave_review(draft, definition, lang=lang)
        if definition.workflow_id == "expense":
            from chat.services.platform.response_composer import ResponseComposer

            return ResponseComposer().expense_review(draft, definition, lang=lang)

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

    def parse_pending_field(
        self,
        workflow_id: str,
        field: str,
        message: str,
        *,
        memory: SessionMemory | None = None,
    ) -> Any:
        """Deterministic single-field parse — no intent classification."""
        wf = (workflow_id or "").strip().lower()
        draft = memory.active_draft() if memory else None
        if wf == "leave" or wf == "expense":
            return None
        return message.strip() if message.strip() else None

    def leave_field_updates_from_message(self, message: str, *, memory: SessionMemory | None = None) -> list[FieldUpdate]:
        """Deterministic leave fields → FieldUpdate list (single extraction path)."""
        if memory is not None:
            return self._explicit_leave_field_updates(message, memory=memory)
        return [
            FieldUpdate(field=k, value=v)
            for k, v in self.extract_workflow_fields("leave", message).items()
        ]

    def _explicit_leave_field_updates(
        self,
        message: str,
        *,
        memory: SessionMemory,
        trace_id: str = "",
    ) -> list[FieldUpdate]:
        """Whitelist-only leave extraction — explicit user signals; no narrative inference."""
        from chat.services.platform.field_extractors.leave import (
            collect_slot_field_updates,
            is_leave_review_mode,
            review_field_updates_from_message,
        )

        draft = memory.active_draft()
        pq = memory.pending_question

        if is_leave_review_mode(memory):
            updates = review_field_updates_from_message(
                message, memory, trace_id=trace_id
            )
            if updates:
                return updates
            if pq and pq.workflow_id == "leave" and pq.field:
                if pq.field == "reason" and is_reason_skip_message(message):
                    from chat.services.platform.field_extractors.leave import infer_leave_reason_from_history

                    backfill = infer_leave_reason_from_history(memory, trace_id=trace_id)
                    if backfill:
                        return [FieldUpdate(field="reason", value=backfill, action="set")]
                    return [FieldUpdate(field="reason_skipped", value=True, action="set")]
                if pq.field == "medical_document" and is_medical_document_skip_message(message):
                    return [FieldUpdate(field="medical_document_skipped", value=True, action="set")]
            return []

        if pq and pq.workflow_id == "leave" and pq.field:
            if pq.field == "reason" and is_reason_skip_message(message):
                from chat.services.platform.field_extractors.leave import infer_leave_reason_from_history

                backfill = infer_leave_reason_from_history(memory, trace_id=trace_id)
                if backfill:
                    return [FieldUpdate(field="reason", value=backfill, action="set")]
                return [FieldUpdate(field="reason_skipped", value=True, action="set")]
            if pq.field == "medical_document" and is_medical_document_skip_message(message):
                return [FieldUpdate(field="medical_document_skipped", value=True, action="set")]
            return collect_slot_field_updates(message, memory, trace_id=trace_id)

        return []

    @staticmethod
    def _field_updates_to_dict(updates: list[FieldUpdate]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for upd in updates or []:
            if upd.field and upd.value not in (None, ""):
                out[str(upd.field)] = upd.value
        return out

    @staticmethod
    def _dict_to_field_updates(fields: dict[str, Any]) -> list[FieldUpdate]:
        return [
            FieldUpdate(field=str(k), value=v, action="set")
            for k, v in (fields or {}).items()
            if v not in (None, "")
        ]

    def ground_leave_understanding(
        self,
        message: str,
        result: UnderstandingResult,
        *,
        memory: SessionMemory,
        trace_id: str = "",
    ) -> UnderstandingResult:
        """Ground leave fields — LLM extraction with validation/coercion only."""
        from chat.services.platform.banglish_normalize import normalize_banglish_message
        from chat.services.platform.intent_rules import is_bare_confirmation

        message = normalize_banglish_message(message)

        if result.source == "llm_leave":
            return result

        if (memory.last_entities or {}).get("leave_start_clarify") and is_bare_confirmation(message):
            if memory.active_workflow and memory.active_draft():
                return result
            if memory.pending_confirmation == "submit":
                return result
            result.field_updates = []
            return result

        from chat.services.platform.schemas import UnderstandingAction

        if result.action in (
            UnderstandingAction.REVIEW.value,
            UnderstandingAction.CANCEL.value,
            UnderstandingAction.SUBMIT.value,
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SWITCH.value,
            UnderstandingAction.STATUS.value,
            UnderstandingAction.QUERY.value,
            UnderstandingAction.CLARIFICATION_NEEDED.value,
            UnderstandingAction.NONE.value,
        ):
            result.field_updates = []
            return result

        from chat.services.platform.field_extractors.leave import (
            is_leave_review_mode,
            merge_leave_field_dicts,
            normalize_leave_type_value,
            interpret_leave_review_message,
            review_delta_to_field_updates,
            sanitize_leave_review_updates,
            sanitize_leave_type_value,
        )
        from chat.services.platform.schemas import UnderstandingAction

        if is_leave_review_mode(memory):
            if result.action in (
                UnderstandingAction.MODIFY.value,
                UnderstandingAction.COLLECT.value,
            ):
                sanitized = sanitize_leave_review_updates(
                    list(result.field_updates or []),
                    message,
                    memory=memory,
                )
                if not sanitized:
                    delta = interpret_leave_review_message(
                        message, memory, trace_id=trace_id
                    )
                    if delta:
                        sanitized = sanitize_leave_review_updates(
                            review_delta_to_field_updates(delta),
                            message,
                            memory=memory,
                        )
                result.field_updates = sanitized
                if sanitized:
                    result.action = UnderstandingAction.MODIFY.value
                return result
            return result

        if result.workflow not in ("leave", "") and not (
            memory.active_workflow and memory.active_workflow.id == "leave"
        ):
            return result

        from chat.services.platform.intent_rules import is_leave_message

        leave_active = result.workflow in ("leave", "") or (
            memory.active_workflow and memory.active_workflow.id == "leave"
        )
        should_extract = leave_active and (
            result.action
            in (
                UnderstandingAction.START.value,
                UnderstandingAction.COLLECT.value,
                UnderstandingAction.CLARIFICATION_NEEDED.value,
            )
            or is_leave_message(message)
        )

        understanding_fields = self._field_updates_to_dict(result.field_updates or [])
        extracted_fields: dict[str, Any] = {}
        if should_extract and not is_leave_review_mode(memory):
            from chat.services.platform.field_extractors.leave import extract_leave_fields_via_llm

            extracted_fields = extract_leave_fields_via_llm(
                message, memory, trace_id=trace_id
            )
        combined = dict(understanding_fields)
        for key, val in extracted_fields.items():
            if val in (None, ""):
                continue
            if key == "reason" or not combined.get(key):
                combined[key] = val
        merged = merge_leave_field_dicts({}, combined, message, memory=memory)
        if merged.get("reason") and (
            is_garbage_leave_reason_value(str(merged["reason"]))
            or is_temporal_or_request_like_leave_reason(str(merged["reason"]))
        ):
            merged.pop("reason", None)
        from chat.services.platform.field_extractors.leave import remember_leave_narrative_seed

        remember_leave_narrative_seed(memory, message)
        requested = (result.entities or {}).get("requested_leave_type")
        if requested:
            canonical = normalize_leave_type_value(requested)
            entities = dict(result.entities or {})
            if canonical:
                merged["leave_type"] = canonical
                entities.pop("requested_leave_type", None)
                result.entities = entities or None
            else:
                entities["requested_leave_type"] = str(requested).strip()
                result.entities = entities
                merged.pop("leave_type", None)
        elif merged.get("leave_type"):
            clean = sanitize_leave_type_value(
                message,
                normalize_leave_type_value(merged.get("leave_type")) or merged.get("leave_type"),
            )
            if clean:
                merged["leave_type"] = clean
            else:
                merged.pop("leave_type", None)

        pq = memory.pending_question
        if (
            pq
            and pq.workflow_id == "leave"
            and pq.field
            and not is_leave_review_mode(memory)
            and result.answers_pending_field is not False
        ):
            from chat.services.platform.field_extractors.leave import (
                _pending_collect_allowed_fields,
                collect_slot_field_updates,
            )

            allowed = _pending_collect_allowed_fields(pq.field)
            merged = {k: v for k, v in merged.items() if k in allowed}
            if not merged:
                slot = collect_slot_field_updates(
                    message,
                    memory,
                    trace_id=trace_id,
                    understanding_updates=result.field_updates,
                )
                result.field_updates = slot
                if slot:
                    result.action = UnderstandingAction.COLLECT.value
                return result

        result.field_updates = self._dict_to_field_updates(merged)
        return result

    def ground_expense_understanding(
        self,
        message: str,
        result: UnderstandingResult,
        *,
        memory: SessionMemory,
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> UnderstandingResult:
        """Ground expense via draft editor LLM — intent + patches."""
        from chat.services.platform.banglish_normalize import normalize_banglish_message
        from chat.services.platform.field_extractors.expense import (
            expense_turn_to_field_updates,
        )
        from chat.services.platform.schemas import UnderstandingAction

        message = normalize_banglish_message(message)

        if result.source == "llm_expense":
            return result

        from chat.services.llm_client import expense_llm_done, peek_expense_turn_cache

        if expense_llm_done(trace_id or ""):
            cached = peek_expense_turn_cache(trace_id or "")
            if cached or (result.entities or {}).get("expense_turn"):
                if result.workflow == "expense" and result.field_updates:
                    return result
                if cached and result.workflow in ("expense", ""):
                    from chat.services.platform.field_extractors.expense import (
                        expense_entities_for_turn,
                        expense_turn_to_field_updates,
                    )

                    turn, updates = expense_turn_to_field_updates(
                        message,
                        memory,
                        trace_id=trace_id,
                        conversation_history=conversation_history,
                        expense_turn=cached,
                    )
                    if updates or str(turn.get("intent") or "") not in ("conversation", "llm_unavailable", ""):
                        entities = expense_entities_for_turn(
                            dict(result.entities or {}),
                            turn,
                            expense_intent=str(turn.get("intent") or ""),
                            action=str(result.action or ""),
                        )
                        result.entities = entities
                        if updates:
                            result.field_updates = updates
                        result.workflow = "expense"
                        return result

        from chat.services.llm_client import mark_expense_llm_done

        if (
            result.source == "llm"
            and result.workflow == "expense"
            and result.field_updates
            and any(
                getattr(u, "field", None) in ("items", "incurred_date")
                for u in result.field_updates
            )
        ):
            entities = dict(result.entities or {})
            if not entities.get("expense_intent"):
                if any(
                    getattr(u, "action", None) == "append" and getattr(u, "field", None) == "items"
                    for u in result.field_updates
                ):
                    entities["expense_intent"] = "add"
            entities["expense_domain_llm"] = True
            result.entities = entities
            if (trace_id or "").strip():
                mark_expense_llm_done(trace_id)
            return result

        from chat.services.platform.intent_rules import (
            is_clearly_off_hr_question,
            is_off_hr_topic_message,
            is_programming_question,
        )

        if (
            result.is_out_of_scope
            or is_programming_question(message)
            or is_clearly_off_hr_question(message)
            or is_off_hr_topic_message(message, memory=memory)
        ):
            return result

        aw = memory.active_workflow if memory else None
        goal_low = (result.goal or "").strip().lower()
        wf = (result.workflow or "").strip().lower()
        if aw and aw.id == "expense" and (
            wf == "leave"
            or result.interrupt_workflow == "leave"
            or "leave" in goal_low
        ):
            return result

        from chat.services.platform.intent_rules import is_greeting_or_chitchat

        if is_greeting_or_chitchat(message):
            return result

        from chat.services.platform.turn_semantics import is_expense_review_request

        if is_expense_review_request(message, result):
            return result

        if result.source == "rules":
            entities = result.entities or {}
            expense_intent = str(entities.get("expense_intent") or "").lower()
            if entities.get("expense_turn") is not None and expense_intent not in (
                "",
                "conversation",
            ):
                return result
            if expense_intent and expense_intent != "conversation" and result.field_updates:
                return result
            if result.field_updates:
                return result

        if result.action in (
            UnderstandingAction.CANCEL.value,
            UnderstandingAction.SUBMIT.value,
            UnderstandingAction.SWITCH.value,
            UnderstandingAction.STATUS.value,
            UnderstandingAction.QUERY.value,
            UnderstandingAction.NONE.value,
        ):
            return result

        if not (
            result.workflow in ("expense", "")
            or (memory.active_workflow and memory.active_workflow.id == "expense")
        ):
            return result

        turn, updates = expense_turn_to_field_updates(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            expense_turn=None,
        )
        intent = str(turn.get("intent") or "").lower()
        if intent == "conversation" and turn.get("llm_degraded"):
            return result
        if turn.get("off_topic"):
            result.is_out_of_scope = True
            result.workflow = "none"
            result.action = UnderstandingAction.NONE.value
            result.field_updates = []
            entities = dict(result.entities or {})
            entities.pop("expense_intent", None)
            result.entities = entities
            return result
        from chat.services.platform.field_extractors.expense import expense_entities_for_turn

        entities = expense_entities_for_turn(
            dict(result.entities or {}),
            turn,
            expense_intent=intent,
            action=str(result.action or ""),
        )
        if turn.get("llm_degraded"):
            entities["expense_llm_degraded"] = True
        if turn.get("wizard_fallback"):
            entities["expense_wizard_fallback"] = True
        result.entities = entities

        intent_to_action = {
            "add": UnderstandingAction.COLLECT.value,
            "update": UnderstandingAction.MODIFY.value,
            "correct": UnderstandingAction.MODIFY.value,
            "delete": UnderstandingAction.DELETE.value,
            "answer_pending": UnderstandingAction.COLLECT.value,
            "fix_mistake": UnderstandingAction.MODIFY.value,
            "anti_summary": UnderstandingAction.CLARIFICATION_NEEDED.value,
            "modify_review": UnderstandingAction.MODIFY.value,
            "show_summary": UnderstandingAction.REVIEW.value,
            "show_list": UnderstandingAction.REVIEW.value,
            "show_total": UnderstandingAction.REVIEW.value,
            "submit": UnderstandingAction.SUBMIT.value,
            "confirm": UnderstandingAction.CONFIRM.value,
            "cancel": UnderstandingAction.CANCEL.value,
            "continue": UnderstandingAction.COLLECT.value,
            "conversation": UnderstandingAction.CLARIFICATION_NEEDED.value,
            "clarify_modify": UnderstandingAction.CLARIFICATION_NEEDED.value,
            "clarify_delete": UnderstandingAction.CLARIFICATION_NEEDED.value,
            "llm_unavailable": UnderstandingAction.CLARIFICATION_NEEDED.value,
        }
        if intent in intent_to_action:
            result.action = intent_to_action[intent]

        pq = memory.pending_question
        if (
            pq
            and pq.workflow_id == "expense"
            and intent in ("answer_pending", "add", "update", "correct")
        ):
            result.answers_pending_field = True
            if intent == "answer_pending":
                result.action = UnderstandingAction.COLLECT.value

        if updates:
            result.field_updates = updates
        elif intent in ("add", "answer_pending", "update", "correct"):
            result.action = UnderstandingAction.COLLECT.value

        if intent in ("show_summary", "show_list", "show_total"):
            result.action = UnderstandingAction.REVIEW.value
            result.field_updates = []

        if memory.active_workflow and memory.active_workflow.id == "expense":
            if result.interrupt_workflow == "leave":
                result.workflow = "leave"
                return result
            result.workflow = "expense"
            pq = memory.pending_question
            if pq and pq.workflow_id == "expense":
                if intent in ("answer_pending", "add", "update", "correct") or updates:
                    result.answers_pending_field = True
                    result.action = UnderstandingAction.COLLECT.value
                if intent in ("show_summary", "show_list", "show_total"):
                    result.action = UnderstandingAction.REVIEW.value
                    result.answers_pending_field = False

        return result

    def expense_field_updates_from_message(
        self,
        message: str,
        *,
        memory: SessionMemory,
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> list[FieldUpdate]:
        from chat.services.platform.field_extractors.expense import expense_field_updates_from_message

        return expense_field_updates_from_message(
            message,
            memory=memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )

    def extract_workflow_fields(
        self,
        workflow_id: str,
        message: str,
        *,
        memory: SessionMemory | None = None,
    ) -> dict[str, Any]:
        """Extract workflow fields via LLM (leave) or deterministic helpers (expense)."""
        wf = (workflow_id or "").strip().lower()
        if wf == "leave":
            from chat.services.platform.field_extractors.leave import extract_leave_fields_via_llm

            return extract_leave_fields_via_llm(message, memory)
        if wf == "expense":
            from chat.services.platform.field_extractors.expense import expense_fields_from_message

            return expense_fields_from_message(message, memory, trace_id="")
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


def serialize_field_updates(updates: list[FieldUpdate]) -> list[dict[str, Any]]:
    return [
        {
            "field": upd.field,
            "value": upd.value,
            "item_index": upd.item_index,
            "action": upd.action,
        }
        for upd in updates
    ]


def deserialize_field_updates(raw: list[dict[str, Any]] | None) -> list[FieldUpdate]:
    out: list[FieldUpdate] = []
    for item in raw or []:
        field = str(item.get("field") or "").strip()
        if not field:
            continue
        out.append(
            FieldUpdate(
                field=field,
                value=item.get("value"),
                item_index=item.get("item_index"),
                action=str(item.get("action") or "set"),
            )
        )
    return out
