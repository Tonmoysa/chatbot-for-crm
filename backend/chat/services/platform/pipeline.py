"""Main workflow pipeline — ties platform modules together."""

from __future__ import annotations

import re
from typing import Any

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.modification_engine import ModificationEngine
from chat.services.platform.field_extractors.leave import is_medical_document_unavailable
from chat.services.platform.field_extractors import parse_modify_request, parse_relative_date
from chat.services.platform.intent_rules import (
    is_bare_confirmation,
    is_expense_message,
    is_greeting_or_chitchat,
    is_leave_message,
    is_programming_question,
    is_vague_delete,
    parse_submit_workflow,
)
from chat.services.platform.registry import get_workflow_definition
from chat.services.platform.response_composer import ResponseComposer
from chat.services.platform.review_engine import ReviewEngine
from chat.services.platform.schemas import FieldUpdate, UnderstandingAction, UnderstandingResult, WorkflowStage
from chat.services.platform.submission_engine import SubmissionEngine
from chat.services.platform.summary import (
    expense_total,
    format_expense_summary,
    format_leave_summary,
    format_session_context,
)
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.session_memory import PendingQuestion, SessionMemory, WorkflowDraft
from chat.services.translator import detect_user_language


class WorkflowPipeline:
    """Platform layer stack: Workflow Manager → Field Engine → Validation → Review → Submission → Composer."""

    def __init__(self) -> None:
        self.manager = WorkflowManager()
        self.understanding = AIUnderstandingLayer()
        self.fields = FieldEngine()
        self.validator = ValidationEngine()
        self.modifier = ModificationEngine(self.fields)
        self.review = ReviewEngine(self.fields, self.validator, self.manager)
        self.submission = SubmissionEngine(self.validator, self.manager)
        self.composer = ResponseComposer()

    def execute_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Run one workflow turn through the platform layers (after PQ + AI Understanding).

        Workflow Manager → Field Engine → Validation → Review → Submission → Response Composer
        """
        return self.handle(
            message,
            memory=memory,
            pq_decision=pq_decision,
            understanding=understanding,
            conversation_history=conversation_history,
            trace_id=trace_id,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )

    def handle(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        lang = detect_user_language(message)
        kind = pq_decision.kind
        u = understanding or self.understanding.understand(
            message, memory=memory, conversation_history=conversation_history, trace_id=trace_id,
        )

        if u.is_out_of_scope:
            self._pause_for_oos(memory)
            return self._reject_oos(lang)

        if self.manager.is_locked(memory):
            return self._locked_response(memory, lang)

        if memory.pending_confirmation and memory.pending_confirmation.startswith("switch:"):
            return self._resolve_workflow_switch(message, memory, lang)

        if memory.pending_confirmation == "duplicate_leave":
            return self._resolve_duplicate_leave(message, memory, lang, understanding=u)

        submit_result = self._resolve_submit_confirmation(
            message,
            memory,
            u,
            lang=lang,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        if submit_result is not None:
            return submit_result

        if kind == MessageIntentKind.ANSWER_PENDING:
            dup = self._maybe_duplicate_leave_confirm(message, memory, u, lang)
            if dup:
                return dup
            switch = self._maybe_workflow_switch_confirm(message, memory, lang, understanding=u)
            if switch:
                return switch
            return self._handle_collect(
                message, memory=memory, pq_decision=pq_decision, understanding=u,
                conversation_history=conversation_history, trace_id=trace_id, lang=lang,
            )

        if kind == MessageIntentKind.MODIFY_DATA:
            return self._handle_modify(
                message, memory=memory,
                conversation_history=conversation_history, trace_id=trace_id, lang=lang,
            )

        if kind == MessageIntentKind.DELETE_DATA:
            return self._handle_delete(
                message, memory=memory,
                conversation_history=conversation_history, trace_id=trace_id, lang=lang,
            )

        if kind == MessageIntentKind.SWITCH_WORKFLOW:
            switch = self._maybe_workflow_switch_confirm(message, memory, lang, understanding=u)
            if switch:
                return switch
            return self._handle_switch(
                message, memory=memory, pq_decision=pq_decision, understanding=u,
                conversation_history=conversation_history, trace_id=trace_id, lang=lang,
            )

        if kind == MessageIntentKind.NEW_WORKFLOW and not pq_decision.blocks_new_workflow:
            dup = self._maybe_duplicate_leave_confirm(message, memory, u, lang)
            if dup:
                return dup
            return self._handle_new(
                message, memory=memory, understanding=u,
                conversation_history=conversation_history, trace_id=trace_id, lang=lang,
                company_id=company_id, employee_id=employee_id, session_id=session_id,
                idempotency_key=idempotency_key,
            )

        if kind == MessageIntentKind.CLARIFICATION_NEEDED:
            draft = memory.active_draft()
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": u.reasoning,
                "understanding": u.to_dict(),
            }

        return None

    def try_handle_active_workflow(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        lang = detect_user_language(message)
        u = understanding or self.understanding.understand(
            message, memory=memory, conversation_history=conversation_history, trace_id=trace_id,
        )

        if u.is_out_of_scope:
            self._pause_for_oos(memory)
            return self._reject_oos(lang)

        if self.manager.is_locked(memory):
            return self._locked_response(memory, lang)

        # Pending workflow switch (leave ↔ expense)
        if memory.pending_confirmation and memory.pending_confirmation.startswith("switch:"):
            return self._resolve_workflow_switch(message, memory, lang)

        if memory.pending_confirmation == "duplicate_leave":
            return self._resolve_duplicate_leave(message, memory, lang, understanding=u)

        submit_result = self._resolve_submit_confirmation(
            message,
            memory,
            u,
            lang=lang,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        if submit_result is not None:
            return submit_result

        # Pending modify confirmation
        if memory.pending_confirmation and memory.pending_confirmation.startswith("modify:"):
            if u.action == UnderstandingAction.CONFIRM.value or is_bare_confirmation(message):
                return self._apply_pending_modify(memory, lang)
            memory.pending_confirmation = None

        submit_wf = parse_submit_workflow(message) if u.source != "llm" else None
        if u.action == UnderstandingAction.SUBMIT.value:
            submit_wf = u.workflow or submit_wf
        if submit_wf:
            defn = get_workflow_definition(submit_wf)
            if defn:
                if not memory.active_workflow or memory.active_workflow.id != submit_wf:
                    self.manager.start_workflow(memory, submit_wf)
                return self._request_submit(memory, defn, lang=lang)

        if u.action == UnderstandingAction.REVIEW.value:
            defn = self._active_defn(memory)
            if defn:
                draft = memory.active_draft()
                if defn.workflow_id == "expense" and draft:
                    msg = format_expense_summary(draft, lang=lang)
                    if re.search(r"total", message.lower()):
                        msg += f"\n\n**Total: {expense_total(draft):.0f} taka**"
                    return msg, {"outcome": "INFORMATIONAL", "rules_applied": ["SUMMARY"]}
                if defn.workflow_id == "leave" and draft:
                    return format_leave_summary(draft, lang=lang), {"outcome": "INFORMATIONAL", "rules_applied": ["SUMMARY"]}

        if u.action == UnderstandingAction.CONFIRM.value:
            if memory.pending_confirmation == "submit":
                defn = self._active_defn(memory)
                if defn:
                    return self._confirm_submit(
                        memory, defn, lang=lang,
                        company_id=company_id, employee_id=employee_id,
                        session_id=session_id, idempotency_key=idempotency_key,
                    )
            return format_session_context(memory, lang=lang), {
                "outcome": "INFORMATIONAL",
                "reason": "No pending confirmation — showing session context.",
                "rules_applied": ["SESSION_CONTEXT"],
            }

        if u.action == UnderstandingAction.SUBMIT.value:
            defn = get_workflow_definition(u.workflow) or self._active_defn(memory)
            if defn:
                return self._request_submit(memory, defn, lang=lang)

        if u.action in (UnderstandingAction.START.value, UnderstandingAction.COLLECT.value) and u.field_updates:
            defn = get_workflow_definition(u.workflow) or self._active_defn(memory)
            if defn and memory.active_workflow:
                draft = memory.active_draft()
                if draft:
                    memory.pending_confirmation = None
                    self.fields.apply_updates(draft, u.field_updates)
                    self._default_expense_date(draft)
                    prefix = self._item_prefix(u, lang)
                    return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        return None

    def _handle_new(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
    ) -> tuple[str, dict[str, Any]]:
        u = understanding
        memory.last_entities = {**dict(memory.last_entities or {}), "turn_understanding": u.to_dict()}

        if u.is_out_of_scope:
            self._pause_for_oos(memory)
            return self._reject_oos(lang)

        if u.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
            draft = memory.active_draft()
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": u.reasoning,
                "understanding": u.to_dict(),
            }

        wf_id = (u.workflow or "").strip().lower()
        if wf_id in ("none", "policy", "status", ""):
            return self.composer.clarification(u, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        defn = get_workflow_definition(wf_id)
        if not defn:
            return self.composer.clarification(u, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        draft = memory.active_draft()
        if u.action in (UnderstandingAction.SUBMIT.value, UnderstandingAction.CONFIRM.value):
            if draft and memory.active_workflow and memory.active_workflow.id == wf_id:
                return self._maybe_submit(
                    memory, defn, u, lang=lang,
                    company_id=company_id, employee_id=employee_id,
                    session_id=session_id, idempotency_key=idempotency_key,
                )
            # Misclassified submit on first message — start collecting instead.
            u = UnderstandingResult(
                goal=u.goal,
                workflow=wf_id,
                action=UnderstandingAction.START.value,
                confidence=u.confidence,
                field_updates=u.field_updates,
                reasoning=u.reasoning,
                source=u.source,
            )

        if u.action == UnderstandingAction.REVIEW.value:
            return self._show_review(memory, defn, lang=lang)

        if not memory.active_workflow or memory.active_workflow.id != wf_id:
            self.manager.start_workflow(memory, wf_id)
        draft = memory.active_draft() or self.manager.start_workflow(memory, wf_id)

        blocked = self._block_submitted_leave_overlap(memory, u, lang)
        if blocked:
            return blocked

        if u.field_updates:
            self.fields.apply_updates(draft, u.field_updates, message=message)
            self._default_expense_date(draft)
            memory.pending_confirmation = None
            self.manager.events.emit(memory, "field_collected", wf_id, {})

        prefix = self._item_prefix(u, lang) or self.composer.workflow_started(defn.name, lang=lang)

        # Vague travel/category — ask category before route
        items = draft.fields.get("items") or []
        last = items[-1] if items else {}
        if last and not last.get("category") and last.get("amount"):
            return self.composer.category_clarify(float(last["amount"]), lang=lang), {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["CATEGORY_CLARIFY"],
            }

        return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

    def _handle_collect(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        u = understanding
        wf_id = memory.active_workflow.id if memory.active_workflow else u.workflow
        defn = self.manager.ensure_definition(wf_id)
        draft = memory.active_draft() or self.manager.start_workflow(memory, wf_id)

        if u.action == UnderstandingAction.REVIEW.value:
            if defn.workflow_id == "expense" and draft:
                return format_expense_summary(draft, lang=lang), {"outcome": "INFORMATIONAL", "rules_applied": ["SUMMARY"]}
            if defn.workflow_id == "leave" and draft:
                return format_leave_summary(draft, lang=lang), {"outcome": "INFORMATIONAL", "rules_applied": ["SUMMARY"]}

        saved_field = memory.pending_question.field if memory.pending_question else "field"

        if (
            memory.pending_question
            and memory.pending_question.field == "medical_document"
            and is_medical_document_unavailable(message)
        ):
            return self._handle_medical_document_unavailable(memory, defn, lang=lang)

        if memory.pending_question and memory.pending_question.field == "delete_which_item":
            m = re.search(r"\b(\d+)\b", message)
            if m:
                idx = int(m.group(1)) - 1
                from chat.services.platform.schemas import TargetRef, UnderstandingResult

                self.modifier.apply_understanding(
                    memory,
                    UnderstandingResult(
                        targets=[TargetRef(field="items", item_index=idx)],
                    ),
                )
                memory.pending_question = None
                prefix = f"Removed item {idx + 1}." if lang != "bn" else f"Entry {idx + 1} মুছে ফেলা হয়েছে।"
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        if u.field_updates:
            self.fields.apply_updates(draft, u.field_updates, message=message)
            saved_field = u.field_updates[0].field
        elif u.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": u.reasoning,
                "understanding": u.to_dict(),
            }

        if not u.field_updates and memory.pending_question and u.source != "llm":
            pending_field = memory.pending_question.field
            wf_id_for_field = memory.pending_question.workflow_id or wf_id
            parsed_val = self.fields.parse_pending_field(wf_id_for_field, pending_field, message)
            if parsed_val is not None:
                self.fields.apply_updates(
                    draft,
                    [FieldUpdate(field=pending_field, value=parsed_val)],
                    message=message,
                )
                saved_field = pending_field
            else:
                pq = memory.pending_question
                still_need = pq.prompt if pq else "Please provide the requested information."
                if lang == "bn":
                    still_need = f"এখনও **{pending_field.replace('_', ' ')}** দরকার।\n\n{still_need}"
                else:
                    still_need = f"I still need your **{pending_field.replace('_', ' ')}**.\n\n{still_need}"
                return still_need, {
                    "outcome": "NEEDS_INPUT",
                    "rules_applied": ["SLOT_MISSING"],
                }
        elif not u.field_updates and memory.pending_question and u.source == "llm":
            pq = memory.pending_question
            still_need = pq.prompt if pq else "Please provide the requested information."
            return still_need, {"outcome": "NEEDS_INPUT", "rules_applied": ["LLM_SLOT_MISSING"]}

        # Category answer for vague expense
        if saved_field in ("category", "category_clarify") or (
            memory.pending_question and memory.pending_question.field == "category_clarify"
        ):
            cat = detect_expense_category_from_reply(message)
            items = draft.fields.get("items") or []
            if items and cat:
                items[-1]["category"] = cat
                draft.fields["items"] = items

        memory.pending_question = None
        memory.pending_confirmation = None
        self._default_expense_date(draft)
        self.manager.events.emit(memory, "field_collected", wf_id, {"field": saved_field})

        prefix = self.composer.field_saved(str(saved_field), lang=lang)
        if u.field_updates and u.field_updates[0].field == "items":
            val = u.field_updates[0].value
            if isinstance(val, dict):
                prefix = self.composer.item_added(val, lang=lang)

        return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

    def _handle_modify(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        u = self.understanding.understand(
            message, memory=memory, conversation_history=conversation_history,
            trace_id=trace_id, pending_kind="modify_data",
        )
        draft = memory.active_draft()
        if u.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "understanding": u.to_dict(),
            }

        if u.field_updates and draft:
            upd = u.field_updates[0]
            items = draft.fields.get("items") or []
            idx = upd.item_index if upd.item_index is not None else 0
            parsed = parse_modify_request(message, items)
            needs_confirm = parsed.get("needs_confirm") if parsed else len(items) > 1
            if 0 <= idx < len(items):
                old_amt = float(items[idx].get("amount") or 0)
                new_amt = float((upd.value or {}).get("amount") or 0)
                label = str(u.entities.get("modify_label") or parsed.get("label") if parsed else f"item {idx + 1}")
                if needs_confirm or (parsed and parsed.get("needs_confirm")):
                    memory.pending_confirmation = f"modify:{idx}:{new_amt}"
                    return self.composer.modify_confirm(
                        label=label, old=old_amt, new=new_amt, draft=draft, lang=lang
                    ), {"outcome": "NEEDS_INPUT", "awaiting_confirmation": True}
                self.fields.apply_updates(draft, u.field_updates)
                prefix = f"Updated **{label}** to **{new_amt:.0f} taka**."
                defn = self.manager.ensure_definition(draft.workflow_id)
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        return self.composer.clarification(u, lang=lang, draft=draft), {"outcome": "NEEDS_CLARIFICATION"}

    def _handle_delete(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        u = self.understanding.understand(
            message, memory=memory, conversation_history=conversation_history,
            trace_id=trace_id, pending_kind="delete_data",
        )
        draft = memory.active_draft()
        if not draft:
            return "No draft to delete from.", {"outcome": "NEEDS_CLARIFICATION"}

        if u.action == UnderstandingAction.CLARIFICATION_NEEDED.value or is_vague_delete(message):
            memory.pending_question = PendingQuestion(
                field="delete_which_item",
                prompt="Which entry number to delete?",
                workflow_id=draft.workflow_id,
                asked_at_turn=memory.turn_count,
            )
            return self.composer.delete_pick(draft, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        applied = self.modifier.apply_understanding(memory, u)
        defn = self.manager.ensure_definition(draft.workflow_id)
        if applied:
            return self._continue_collection(
                memory, defn, lang=lang,
                prefix=f"Removed **{applied[0]}**." if lang != "bn" else f"**{applied[0]}** মুছে ফেলা হয়েছে।",
            )
        return self.composer.delete_pick(draft, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

    def _handle_switch(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
    ) -> tuple[str, dict[str, Any]] | None:
        u = understanding
        target = (pq_decision.target_workflow or u.interrupt_workflow or u.workflow or "").strip().lower()
        if not target or not get_workflow_definition(target):
            return "Which workflow?", {"outcome": "NEEDS_CLARIFICATION"}
        self.manager.switch_to(memory, target)
        defn = self.manager.ensure_definition(target)
        draft = memory.active_draft() or self.manager.start_workflow(memory, target)
        if u.field_updates:
            self.fields.apply_updates(draft, u.field_updates)
        prefix = f"Switched to **{target}**." if lang != "bn" else f"**{target}** workflow-এ গেলাম।"
        return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

    def _continue_collection(
        self,
        memory: SessionMemory,
        defn,
        *,
        lang: str,
        prefix: str = "",
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft:
            return prefix, {"outcome": "NEEDS_INPUT"}

        self._default_expense_date(draft)
        missing = self.fields.missing_fields(draft, defn)
        errors = self.validator.validate(draft, defn, lang=lang)

        if not missing and not errors:
            review_text, _ = self.review.prepare_review(memory, defn)
            memory.pending_question = None
            memory.pending_confirmation = "submit"
            self.manager.set_stage(memory, WorkflowStage.CONFIRM_SUBMIT.value)
            msg = f"{prefix}\n\n{review_text or ''}".strip()
            return msg, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["REVIEW_READY"],
                "stage": WorkflowStage.REVIEW.value,
                "awaiting_confirmation": True,
            }

        pq = self.fields.next_question(memory, draft, defn, lang=lang)
        memory.pending_question = pq
        parts = [p for p in [prefix, errors[0] if errors else ""] if p]
        if pq:
            parts.append(pq.prompt)
        msg = "\n\n".join(parts).strip()
        return msg, {"outcome": "NEEDS_INPUT", "missing_fields": missing, "rules_applied": ["FIELD_COLLECTION"]}

    def _request_submit(self, memory: SessionMemory, defn, *, lang: str) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft:
            wf = defn.workflow_id
            if lang == "bn":
                return f"কোনো open **{wf}** draft নেই।", {"outcome": "NEEDS_CLARIFICATION"}
            return f"No open **{wf}** draft to submit.", {"outcome": "NEEDS_CLARIFICATION"}

        missing = self.fields.missing_fields(draft, defn)
        if missing:
            pq = self.fields.next_question(memory, draft, defn, lang=lang)
            memory.pending_question = pq
            msg = self.composer.missing_for_submit(missing, lang=lang)
            if pq:
                msg += f"\n\n{pq.prompt}"
            return msg, {"outcome": "NEEDS_INPUT", "missing_fields": missing}

        errors = self.validator.validate(draft, defn, lang=lang)
        if errors:
            return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}

        review_text, _ = self.review.prepare_review(memory, defn)
        memory.pending_confirmation = "submit"
        return self.composer.submit_confirm(review_text or "", lang=lang), {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
        }

    def _confirm_submit(
        self,
        memory: SessionMemory,
        defn,
        *,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
    ) -> tuple[str, dict[str, Any]]:
        missing = self.fields.missing_fields(memory.active_draft(), defn) if memory.active_draft() else ["draft"]
        if missing:
            memory.pending_confirmation = None
            return self.composer.missing_for_submit(missing, lang=lang), {"outcome": "NEEDS_INPUT"}

        msg, meta = self.submission.confirm_and_submit(
            memory, defn,
            company_id=company_id, employee_id=employee_id,
            session_id=session_id, idempotency_key=idempotency_key,
        )
        outcome = "SUBMITTED" if meta.get("submitted") else "NEEDS_INPUT"
        return msg, {"outcome": outcome, **meta}

    def _resolve_submit_confirmation(
        self,
        message: str,
        memory: SessionMemory,
        u: UnderstandingResult,
        *,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
    ) -> tuple[str, dict[str, Any]] | None:
        if memory.pending_confirmation != "submit":
            return None
        defn = self._active_defn(memory)
        if not defn:
            return None
        if is_bare_confirmation(message) or u.action in (
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SUBMIT.value,
        ):
            return self._confirm_submit(
                memory, defn, lang=lang,
                company_id=company_id, employee_id=employee_id,
                session_id=session_id, idempotency_key=idempotency_key,
            )
        if u.action in (
            UnderstandingAction.MODIFY.value,
            UnderstandingAction.DELETE.value,
        ) or (u.field_updates and u.action == UnderstandingAction.COLLECT.value):
            memory.pending_confirmation = None
            self.manager.set_stage(memory, WorkflowStage.COLLECTING.value)
            return None
        review_text, _ = self.review.prepare_review(memory, defn)
        return self.composer.submit_confirm(review_text or "", lang=lang), {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
            "rules_applied": ["SUBMIT_CONFIRM_RETRY"],
        }

    def _block_submitted_leave_overlap(
        self, memory: SessionMemory, u: UnderstandingResult, lang: str
    ) -> tuple[str, dict[str, Any]] | None:
        from chat.services.platform.field_extractors.leave import find_submitted_leave_overlap

        wf = (u.workflow or "").strip().lower()
        if wf not in ("leave", "") and not (
            memory.active_workflow and memory.active_workflow.id == "leave"
        ):
            return None
        fields = {upd.field: upd.value for upd in (u.field_updates or [])}
        draft = memory.active_draft()
        start = fields.get("start_date") or (draft.fields.get("start_date") if draft else None)
        end = fields.get("end_date") or (draft.fields.get("end_date") if draft else None) or start
        if not start:
            return None
        hit = find_submitted_leave_overlap(memory, str(start), str(end) if end else None)
        if not hit:
            return None
        msg = self.composer.submitted_leave_overlap(hit, lang=lang)
        return msg, {
            "outcome": "NEEDS_CLARIFICATION",
            "rules_applied": ["SUBMITTED_LEAVE_DATE_OVERLAP"],
        }

    def _handle_medical_document_unavailable(
        self, memory: SessionMemory, defn, *, lang: str
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if draft:
            draft.fields.pop("medical_document", None)
            if draft.fields.get("leave_type") == "sick":
                draft.fields.pop("leave_type", None)
            draft.version += 1
        memory.pending_question = None
        prefix = self.composer.medical_document_unavailable(lang=lang)
        return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

    def _apply_pending_modify(self, memory: SessionMemory, lang: str) -> tuple[str, dict[str, Any]]:
        raw = memory.pending_confirmation or ""
        parts = raw.split(":")
        memory.pending_confirmation = None
        if len(parts) != 3:
            return format_session_context(memory, lang=lang), {"outcome": "INFORMATIONAL"}
        idx = int(parts[1])
        new_amt = float(parts[2])
        draft = memory.active_draft()
        if not draft:
            return "No draft.", {"outcome": "NEEDS_CLARIFICATION"}
        items = list(draft.fields.get("items") or [])
        if 0 <= idx < len(items):
            items[idx]["amount"] = new_amt
            draft.fields["items"] = items
            draft.version += 1
        defn = self.manager.ensure_definition(draft.workflow_id)
        prefix = f"Updated item {idx + 1} to **{new_amt:.0f} taka**." if lang != "bn" else f"Item {idx + 1} **{new_amt:.0f} taka** করা হয়েছে।"
        return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

    def _maybe_submit(self, memory, defn, u, *, lang, company_id, employee_id, session_id, idempotency_key):
        if u.action == UnderstandingAction.CONFIRM.value:
            return self._confirm_submit(
                memory, defn, lang=lang,
                company_id=company_id, employee_id=employee_id,
                session_id=session_id, idempotency_key=idempotency_key,
            )
        return self._request_submit(memory, defn, lang=lang)

    def _show_review(self, memory: SessionMemory, defn, *, lang: str) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if defn.workflow_id == "expense" and draft:
            return format_expense_summary(draft, lang=lang), {"outcome": "INFORMATIONAL"}
        if defn.workflow_id == "leave" and draft:
            return format_leave_summary(draft, lang=lang), {"outcome": "INFORMATIONAL"}
        review_text, errors = self.review.prepare_review(memory, defn)
        if errors:
            return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}
        return review_text or "", {"outcome": "NEEDS_INPUT"}

    def _locked_response(self, memory: SessionMemory, lang: str) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        rid = draft.submitted_request_id if draft else ""
        msg = self.composer.locked_message(lang=lang)
        if rid:
            msg += f"\n\nReference: **`{rid}`**"
        return msg, {"outcome": "INFORMATIONAL", "rules_applied": ["POST_SUBMIT_LOCK"]}

    @staticmethod
    def _pause_for_oos(memory: SessionMemory) -> None:
        draft = memory.active_draft()
        if memory.active_workflow and (not draft or not draft.locked):
            WorkflowManager().suspend_active(memory)

    def _maybe_workflow_switch_confirm(
        self, message: str, memory: SessionMemory, lang: str, *, understanding: UnderstandingResult | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        if memory.pending_confirmation and memory.pending_confirmation.startswith("switch:"):
            return None
        interrupt = self.manager.detect_interrupt(message, memory, understanding=understanding)
        if not interrupt:
            return None
        self.manager.arm_switch_confirm(
            memory,
            from_workflow=interrupt.from_workflow,
            to_workflow=interrupt.to_workflow,
            pending_message=interrupt.pending_message,
        )
        msg = self.manager.switch_confirm_message(
            interrupt.from_workflow, interrupt.to_workflow, lang=lang
        )
        return msg, {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
            "rules_applied": ["WORKFLOW_SWITCH_CONFIRM"],
        }

    def _resolve_workflow_switch(
        self, message: str, memory: SessionMemory, lang: str
    ) -> tuple[str, dict[str, Any]]:
        raw = memory.pending_confirmation or ""
        parts = raw.split(":")
        if len(parts) != 3:
            memory.pending_confirmation = None
            return format_session_context(memory, lang=lang), {"outcome": "INFORMATIONAL"}

        from_wf, to_wf = parts[1], parts[2]
        choice = self.manager.parse_switch_reply(message, from_wf, to_wf)

        if choice == "continue":
            self.manager.clear_switch_confirm(memory)
            defn = self.manager.ensure_definition(from_wf)
            if not memory.active_workflow:
                self.manager.switch_to(memory, from_wf)
            prefix = (
                f"**{from_wf}** request চালিয়ে যাচ্ছি।"
                if lang == "bn"
                else f"Continuing your **{from_wf}** request."
            )
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        if choice == "switch":
            pending_msg = self.manager.clear_switch_confirm(memory)
            self.manager.switch_to(memory, to_wf)
            defn = self.manager.ensure_definition(to_wf)
            if pending_msg:
                u = self.understanding._start_expense(pending_msg, memory)
                draft = memory.active_draft()
                if draft and u.field_updates:
                    self.fields.apply_updates(draft, u.field_updates)
                    self._default_expense_date(draft)
                prefix = self.composer.workflow_started(defn.name, lang=lang)
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix)
            prefix = f"Switched to **{to_wf}**." if lang != "bn" else f"**{to_wf}** workflow-এ গেলাম।"
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        msg = self.manager.switch_retry_message(from_wf, to_wf, lang=lang)
        return msg, {"outcome": "NEEDS_INPUT", "awaiting_confirmation": True}

    def _reject_oos(self, lang: str) -> tuple[str, dict[str, Any]]:
        if lang == "bn":
            msg = "এটি company HR assistant-এর scope-এর বাইরে। আমি leave, expense ও company policy নিয়ে সাহায্য করি।"
        else:
            msg = "That's outside my scope as an HR assistant. I help with leave, expense, and company policies."
        return msg, {"outcome": "INFORMATIONAL", "rules_applied": ["OUT_OF_SCOPE_REJECT"]}

    @staticmethod
    def _default_expense_date(draft) -> None:
        if draft.workflow_id != "expense":
            return
        if not draft.fields.get("incurred_date"):
            from datetime import date
            draft.fields["incurred_date"] = date.today().isoformat()

    @staticmethod
    def _item_prefix(u, lang: str) -> str:
        for upd in u.field_updates or []:
            if upd.field == "items" and isinstance(upd.value, dict) and upd.action == "append":
                c = ResponseComposer()
                return c.item_added(upd.value, lang=lang)
        return ""

    @staticmethod
    def _active_defn(memory: SessionMemory):
        if memory.active_workflow:
            return get_workflow_definition(memory.active_workflow.id)
        return None

    @staticmethod
    def _leave_draft_in_progress(draft: WorkflowDraft | None) -> bool:
        if not draft or draft.workflow_id != "leave":
            return False
        fields = draft.fields or {}
        return bool(fields.get("leave_type") or fields.get("start_date") or fields.get("day_scope"))

    @staticmethod
    def _is_duplicate_leave_attempt(
        message: str, u: UnderstandingResult, draft: WorkflowDraft
    ) -> bool:
        from chat.services.platform.intent_rules import is_leave_message

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

    def _maybe_duplicate_leave_confirm(
        self,
        message: str,
        memory: SessionMemory,
        u: UnderstandingResult,
        lang: str,
    ) -> tuple[str, dict[str, Any]] | None:
        if memory.pending_confirmation == "duplicate_leave":
            return None
        draft = memory.active_draft()
        if not draft or not memory.active_workflow or memory.active_workflow.id != "leave":
            return None
        if not self._leave_draft_in_progress(draft):
            return None
        if not self._is_duplicate_leave_attempt(message, u, draft):
            return None
        entities = dict(memory.last_entities or {})
        entities["duplicate_leave_updates"] = [
            {"field": upd.field, "value": upd.value, "action": upd.action, "item_index": upd.item_index}
            for upd in (u.field_updates or [])
        ]
        memory.last_entities = entities
        memory.pending_confirmation = "duplicate_leave"
        msg = self.composer.duplicate_leave_prompt(draft, lang=lang)
        return msg, {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
            "rules_applied": ["DUPLICATE_LEAVE_CHOICE"],
        }

    @staticmethod
    def parse_duplicate_leave_reply(message: str) -> str | None:
        """Return ``continue``, ``new``, or None."""
        low = (message or "").lower().strip()
        if re.search(r"\b(continue|chaliye|current|same|ager|jotokhon|thik\s*ache)\b", low):
            return "continue"
        if re.search(r"\b(cancel|new|notun|fresh|start over|yes|ha+h)\b", low):
            return "new"
        return None

    def _resolve_duplicate_leave(
        self,
        message: str,
        memory: SessionMemory,
        lang: str,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> tuple[str, dict[str, Any]]:
        choice = self.parse_duplicate_leave_reply(message)
        entities = dict(memory.last_entities or {})
        pending_updates = entities.pop("duplicate_leave_updates", [])
        memory.last_entities = entities

        draft = memory.active_draft()
        defn = self.manager.ensure_definition("leave")

        if choice == "continue":
            memory.pending_confirmation = None
            prefix = (
                "**leave** request চালিয়ে যাচ্ছি।"
                if lang == "bn"
                else "Continuing your **leave** request."
            )
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        if choice == "new":
            memory.pending_confirmation = None
            if draft:
                draft.fields = {}
                draft.version += 1
            memory.pending_question = None
            u = understanding
            if u and u.field_updates:
                self.fields.apply_updates(draft, u.field_updates)
            elif pending_updates:
                updates = [
                    FieldUpdate(
                        field=pu["field"],
                        value=pu["value"],
                        action=pu.get("action") or "set",
                        item_index=pu.get("item_index"),
                    )
                    for pu in pending_updates
                    if pu.get("field")
                ]
                if updates:
                    self.fields.apply_updates(draft, updates)
            elif message.strip():
                u2 = self.understanding.understand(
                    message, memory=memory, conversation_history=[], trace_id="dup-leave",
                )
                if u2.field_updates:
                    self.fields.apply_updates(draft, u2.field_updates)
            prefix = (
                "নতুন **leave request** শুরু হলো।"
                if lang == "bn"
                else "Started a fresh **leave request**."
            )
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix)

        msg = self.composer.duplicate_leave_prompt(draft, lang=lang) if draft else (
            "Reply **continue** or **new**." if lang != "bn" else "**continue** বা **new** বলুন।"
        )
        memory.pending_confirmation = "duplicate_leave"
        return msg, {"outcome": "NEEDS_INPUT", "awaiting_confirmation": True}


def detect_expense_category_from_reply(message: str) -> str:
    low = (message or "").lower()
    if re.search(r"\b(bus|train|travel|transport)\b", low):
        return "travel"
    if re.search(r"\b(lunch|meal|dinner|snack|nasta|nasto)\b", low):
        return "meals"
    return "meals"
