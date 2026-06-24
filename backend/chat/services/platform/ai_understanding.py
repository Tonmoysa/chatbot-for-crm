"""AI Understanding Layer — single structured output contract (Phase 3 SSOT)."""

from __future__ import annotations

import json
import re
from typing import Any

from chat.services.platform.banglish_normalize import normalize_banglish_message
from chat.services.observability import log_step
from chat.services.platform.confidence import apply_confidence_guard
from chat.services.platform.field_extractors import (
    is_vague_amount_modify,
    parse_amount,
    parse_delete_request,
    parse_modify_request,
    parse_relative_date,
    parse_route,
)
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.field_extractors.leave import is_reason_skip_message
from chat.services.platform.intent_rules import (
    infer_new_workflow_target,
    infer_switch_target,
    is_bare_confirmation,
    is_bare_rejection,
    is_cancel_workflow_message,
    is_expense_message,
    is_greeting_or_chitchat,
    is_leave_message,
    is_modify_request,
    is_delete_request,
    is_programming_question,
    is_status_query,
    is_resume_workflow_request,
    is_same_workflow_navigation,
    is_strong_new_workflow_message,
    is_summary_request,
    is_switch_request,
    is_total_request,
    is_vague_delete,
    is_workflow_show_request,
    is_workflow_interrupt_expense,
    is_workflow_application_message,
    expense_navigation_kind,
    is_expense_draft_query,
    parse_submit_workflow,
    should_resume_suspended_expense,
)
from chat.services.platform.llm_prompts import UNDERSTAND_SYSTEM
from chat.services.platform.registry import list_workflow_ids
from chat.services.platform.schemas import (
    FieldUpdate,
    TargetRef,
    UnderstandingAction,
    UnderstandingResult,
)
from chat.services.session_memory import SessionMemory
from chat.services.llm_client import LLMClient


_SYSTEM = """You interpret user messages for an HR conversational workflow platform.
Return ONLY valid JSON with goal, workflow, action, confidence, entities, field_updates, targets, reasoning.
See platform rules for natural language leave/expense, modify, delete, submit, summary."""

_GATEKEEPER_WEAK_ACTIONS = frozenset(
    {
        UnderstandingAction.NONE.value,
        UnderstandingAction.CLARIFICATION_NEEDED.value,
    }
)


class AIUnderstandingLayer:
    def __init__(self) -> None:
        self.fields = FieldEngine()

    def understand(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        llm: LLMClient | None = None,
        pending_kind: str | None = None,
    ) -> UnderstandingResult:
        raw = normalize_banglish_message((message or "").strip())

        client = llm or LLMClient()

        if client.is_configured():
            domain = self._try_domain_workflow_understanding(
                raw,
                memory=memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
            if domain is not None:
                from chat.services.platform.turn_semantics import enrich_answers_pending_field

                result = enrich_answers_pending_field(message, memory, domain)
                guarded = apply_confidence_guard(result)
                self._log(trace_id, raw, memory, guarded)
                return guarded

        from chat.services.platform.intent_rules import (
            is_clearly_off_hr_question,
            is_off_hr_topic_message,
            is_programming_question,
            is_workflow_turn_message,
        )

        active_id = memory.active_workflow.id if memory.active_workflow else ""
        if is_programming_question(raw):
            oos = True
        elif active_id and is_workflow_turn_message(raw, memory=memory):
            oos = False
        elif is_clearly_off_hr_question(raw) or is_off_hr_topic_message(raw, memory=memory):
            oos = True
        else:
            oos = False

        if oos:
            result = UnderstandingResult(
                goal="Out of scope",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.93,
                is_out_of_scope=True,
                reasoning="General / programming question outside HR assistant scope.",
                source="rules",
            )
            guarded = apply_confidence_guard(result)
            self._log(trace_id, raw, memory, guarded)
            return guarded

        if client.is_configured():
            result = self._understand_llm(
                raw,
                memory=memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
                client=client,
                pending_kind=pending_kind,
            )
            if result is not None:
                result = self._apply_deterministic_gatekeeper(
                    raw,
                    memory=memory,
                    llm_result=result,
                    pending_kind=pending_kind,
                    trace_id=trace_id,
                    conversation_history=conversation_history,
                )
                guarded = apply_confidence_guard(result)
                self._log(trace_id, raw, memory, guarded)
                return guarded

        result = apply_confidence_guard(
            self._sanitize_leave_result(
                raw,
                self._understand_rules(
                    raw,
                    memory=memory,
                    pending_kind=pending_kind,
                    conversation_history=conversation_history,
                ),
                memory=memory,
                trace_id=trace_id,
                conversation_history=conversation_history,
            )
        )
        self._log(trace_id, raw, memory, result)
        return result

    def _log(self, trace_id: str, message: str, memory: SessionMemory, result: UnderstandingResult) -> None:
        log_step(
            trace_id,
            "understanding_completed",
            {
                "user_message": message,
                "understanding": result.to_dict(),
                "workflow_before": memory.active_workflow.to_dict() if memory.active_workflow else None,
            },
        )

    def _needs_global_understand_llm(
        self,
        message: str,
        memory: SessionMemory,
        *,
        active_id: str,
    ) -> bool:
        """True when we must run the fat UNDERSTAND LLM (switch / new workflow / cross-flow)."""
        from chat.services.platform.intent_rules import (
            infer_new_workflow_target,
            infer_switch_target,
            is_greeting_or_chitchat,
            is_leave_message,
            is_leave_navigation_from_expense,
            is_resume_workflow_request,
            is_strong_new_workflow_message,
            is_switch_request,
            is_workflow_interrupt_expense,
            should_resume_suspended_expense,
        )

        if not active_id:
            return True
        if is_greeting_or_chitchat(message) and not memory.active_draft():
            return True
        if is_switch_request(message):
            return True
        if is_resume_workflow_request(message, workflow_id=active_id):
            return True
        if should_resume_suspended_expense(
            message=message,
            active_workflow_id=active_id,
            suspended_workflows=memory.suspended_workflows,
        ):
            return True
        if active_id == "leave" and is_workflow_interrupt_expense(message, active_workflow="leave"):
            return True
        if active_id == "expense" and is_leave_navigation_from_expense(message):
            return True
        if is_strong_new_workflow_message(message):
            target = (infer_switch_target(message) or infer_new_workflow_target(message) or "").strip().lower()
            if target and target != active_id:
                return True
        if is_leave_message(message) and active_id != "leave":
            return True
        return False

    def _try_domain_workflow_understanding(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
    ) -> UnderstandingResult | None:
        """One domain LLM per turn when a workflow is already active (leave-like path)."""
        active_id = memory.active_workflow.id if memory.active_workflow else ""
        if not active_id or self._needs_global_understand_llm(message, memory, active_id=active_id):
            return None
        if active_id == "expense":
            return self._expense_domain_understanding(
                message,
                memory,
                active_id,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
        if active_id == "leave":
            return self._leave_domain_understanding(
                message,
                memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
        return None

    def _expense_domain_understanding(
        self,
        message: str,
        memory: SessionMemory,
        active_id: str,
        *,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
    ) -> UnderstandingResult:
        result = self._expense_collect(
            message,
            memory,
            active_id,
            conversation_history=conversation_history,
            trace_id=trace_id,
        )
        llm_primary = result.source == "llm_expense"
        entities = dict(result.entities or {})
        entities["expense_domain_llm"] = llm_primary
        result.entities = entities
        result.reasoning = (
            "Expense domain LLM (single call)."
            if llm_primary
            else "Expense rules fallback after domain LLM."
        )
        return result

    def _leave_domain_understanding(
        self,
        message: str,
        memory: SessionMemory,
        *,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
    ) -> UnderstandingResult:
        from chat.services.platform.field_extractors.leave import (
            collect_slot_field_updates,
            is_leave_review_mode,
            review_field_updates_from_message,
        )
        from chat.services.platform.schemas import FieldUpdate

        if is_leave_review_mode(memory):
            updates = review_field_updates_from_message(
                message, memory, trace_id=trace_id
            )
            if updates:
                return UnderstandingResult(
                    goal="Modify leave draft",
                    workflow="leave",
                    action=UnderstandingAction.MODIFY.value,
                    confidence=0.9,
                    field_updates=updates,
                    reasoning="Leave review domain LLM.",
                    source="llm_leave",
                )
            return UnderstandingResult(
                goal="Ambiguous leave modify",
                workflow="leave",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.55,
                reasoning="Which leave field should I change — reason, date, or leave type?",
                source="llm_leave",
            )

        pq = memory.pending_question
        if pq and pq.workflow_id == "leave" and pq.field:
            updates = collect_slot_field_updates(
                message, memory, trace_id=trace_id
            )
            if updates:
                return UnderstandingResult(
                    goal="Answer leave slot",
                    workflow="leave",
                    action=UnderstandingAction.COLLECT.value,
                    confidence=0.88,
                    field_updates=updates,
                    answers_pending_field=True,
                    reasoning="Leave collect-slot domain LLM.",
                    source="llm_leave",
                )

        from chat.services.platform.field_extractors.leave import extract_leave_fields_via_llm

        fields = extract_leave_fields_via_llm(
            message, memory, trace_id=trace_id
        )
        updates = [
            FieldUpdate(field=str(k), value=v, action="set")
            for k, v in (fields or {}).items()
            if v not in (None, "")
        ]
        return UnderstandingResult(
            goal="Update leave",
            workflow="leave",
            action=UnderstandingAction.COLLECT.value if updates else UnderstandingAction.CLARIFICATION_NEEDED.value,
            confidence=0.85 if updates else 0.5,
            field_updates=updates,
            reasoning="Leave collect domain LLM.",
            source="llm_leave" if updates else "rules",
        )

    def _slim_understand_context(
        self,
        memory: SessionMemory,
        conversation_history: list[str],
        pending_kind: str | None,
    ) -> dict[str, Any]:
        from datetime import date

        draft = memory.active_draft()
        draft_snapshot: Any = None
        if draft:
            if draft.workflow_id == "expense":
                from chat.services.platform.field_extractors.expense import draft_context_payload

                draft_snapshot = draft_context_payload(memory, compact=True)
            else:
                draft_snapshot = dict(draft.fields or {})
        from chat.services.platform.turn_semantics import understanding_session_context

        return {
            "today_iso": date.today().isoformat(),
            "workflows_available": list_workflow_ids(),
            "active_workflow": memory.active_workflow.to_dict() if memory.active_workflow else None,
            "pending_question": memory.pending_question.to_dict() if memory.pending_question else None,
            "workflow_draft": draft_snapshot,
            "pending_confirmation": memory.pending_confirmation,
            "suspended_workflows": [
                {"workflow_id": sw.workflow_id, "stage": sw.stage}
                for sw in memory.suspended_workflows
            ],
            "pending_kind_hint": pending_kind,
            "conversation_history": list(conversation_history or [])[-3:],
            **understanding_session_context(memory, conversation_history),
        }

    def _understand_llm(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        client: LLMClient,
        pending_kind: str | None,
    ) -> UnderstandingResult | None:
        from datetime import date

        draft = memory.active_draft()
        from chat.services.platform.turn_semantics import understanding_session_context

        context = self._slim_understand_context(memory, conversation_history, pending_kind)
        parsed = client.chat_json(
            system_prompt=UNDERSTAND_SYSTEM,
            user_prompt=(
                "Session context (JSON):\n"
                f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
                f"User message:\n{message}"
            ),
            trace_id=trace_id,
            scope="understanding",
        )
        if not isinstance(parsed, dict):
            return None
        result = self._parse_result(parsed, source="llm")
        return self._sanitize_leave_result(
            message,
            result,
            memory=memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )

    def _sanitize_leave_result(
        self,
        message: str,
        result: UnderstandingResult,
        *,
        memory: SessionMemory,
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> UnderstandingResult:
        from chat.services.platform.turn_semantics import enrich_answers_pending_field

        if result.source in ("llm_expense", "llm_leave"):
            return enrich_answers_pending_field(message, memory, result)
        if (result.entities or {}).get("expense_domain_llm"):
            return enrich_answers_pending_field(message, memory, result)

        grounded = self.fields.ground_leave_understanding(
            message, result, memory=memory, trace_id=trace_id
        )
        grounded = self.fields.ground_expense_understanding(
            message,
            grounded,
            memory=memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
        return enrich_answers_pending_field(message, memory, grounded)

    def _rules_snapshot(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pending_kind: str | None,
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> UnderstandingResult:
        return self._sanitize_leave_result(
            message,
            self._understand_rules(
                message,
                memory=memory,
                pending_kind=pending_kind,
                conversation_history=conversation_history,
            ),
            memory=memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )

    def _apply_deterministic_gatekeeper(
        self,
        message: str,
        *,
        memory: SessionMemory,
        llm_result: UnderstandingResult,
        pending_kind: str | None,
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> UnderstandingResult:
        """Phase 1 — deterministic rules override LLM when workflow signals are obvious."""
        rules = self._rules_snapshot(
            message,
            memory=memory,
            pending_kind=pending_kind,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
        active_id = memory.active_workflow.id if memory.active_workflow else ""

        def _gatekeeper_copy(result: UnderstandingResult) -> UnderstandingResult:
            result.source = "rules_gatekeeper"
            note = "Deterministic rules override LLM classification."
            result.reasoning = f"{result.reasoning} {note}".strip() if result.reasoning else note
            return result

        from chat.services.platform.intent_rules import is_clearly_off_hr_question, is_off_hr_topic_message

        if (
            rules.is_out_of_scope
            or is_programming_question(message)
            or is_clearly_off_hr_question(message)
            or is_off_hr_topic_message(message, memory=memory)
        ):
            if not rules.is_out_of_scope and is_programming_question(message):
                rules = UnderstandingResult(
                    goal="Programming question",
                    workflow="none",
                    action=UnderstandingAction.NONE.value,
                    confidence=0.95,
                    is_out_of_scope=True,
                    reasoning="Programming languages are out of scope for HR assistant.",
                    source="rules",
                )
            return _gatekeeper_copy(rules)

        if (
            memory.pending_confirmation == "submit"
            and active_id == "leave"
        ):
            if is_workflow_interrupt_expense(message, active_workflow="leave"):
                expense_result = self._start_expense(message, memory)
                expense_result.workflow = "expense"
                expense_result.interrupt_workflow = "expense"
                if expense_result.action in _GATEKEEPER_WEAK_ACTIONS:
                    expense_result.action = UnderstandingAction.START.value
                return _gatekeeper_copy(expense_result)

        if rules.action in (
            UnderstandingAction.STATUS.value,
            UnderstandingAction.QUERY.value,
        ):
            if rules.workflow in ("none", "", "policy"):
                return llm_result

        from chat.services.platform.turn_semantics import (
            is_process_question,
            wizard_semantics_active,
        )

        if wizard_semantics_active(memory) and is_process_question(message):
            proc = UnderstandingResult(
                goal="Process question",
                workflow=active_id or "leave",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.92,
                answers_pending_field=False,
                entities={"process_question": True},
                reasoning="User asks what else is needed.",
                source="rules_gatekeeper",
            )
            return proc

        draft = memory.active_draft()
        locked_leave = active_id == "leave" and bool(draft and draft.locked)
        from chat.services.platform.field_extractors.leave import is_leave_review_mode

        leave_start_signal = is_leave_message(message) or is_strong_new_workflow_message(message)
        if is_leave_review_mode(memory):
            leave_start_signal = False
        if (
            rules.workflow == "leave"
            and rules.action in (UnderstandingAction.START.value, UnderstandingAction.COLLECT.value)
            and leave_start_signal
            and not is_leave_review_mode(memory)
        ):
            llm_missed_leave = (
                llm_result.workflow != "leave"
                or llm_result.action in _GATEKEEPER_WEAK_ACTIONS
                or llm_result.is_greeting
            )
            if llm_missed_leave:
                return _gatekeeper_copy(rules)

        if locked_leave:
            from chat.services.platform.intent_rules import should_route_expense_after_submitted_leave

            if should_route_expense_after_submitted_leave(
                draft_locked=True,
                active_workflow_id=active_id,
                message=message,
                understanding=llm_result,
            ):
                expense_result = self._rules_snapshot(message, memory=memory, pending_kind=pending_kind, trace_id=trace_id)
                if expense_result.workflow != "expense":
                    expense_result = self._start_expense(message, memory)
                expense_result.workflow = "expense"
                expense_result.interrupt_workflow = "expense"
                if expense_result.action in _GATEKEEPER_WEAK_ACTIONS:
                    expense_result.action = UnderstandingAction.START.value
                expense_result.source = "rules_gatekeeper"
                return expense_result
            if (
                rules.action == UnderstandingAction.REVIEW.value
                and llm_result.action != UnderstandingAction.REVIEW.value
            ):
                return _gatekeeper_copy(rules)
            if (
                rules.action in (UnderstandingAction.START.value, UnderstandingAction.COLLECT.value)
                and leave_start_signal
                and llm_result.workflow != "leave"
            ):
                return _gatekeeper_copy(rules)

        if active_id == "leave":
            from chat.services.platform.field_extractors.leave import is_leave_review_mode

            if is_leave_review_mode(memory):
                if llm_result.action in (
                    UnderstandingAction.MODIFY.value,
                    UnderstandingAction.COLLECT.value,
                ) and llm_result.field_updates:
                    grounded = llm_result
                    if grounded.action == UnderstandingAction.COLLECT.value:
                        grounded.action = UnderstandingAction.MODIFY.value
                    return grounded
                if (
                    rules.action == UnderstandingAction.MODIFY.value
                    and (rules.field_updates or [])
                    and llm_result.action in _GATEKEEPER_WEAK_ACTIONS
                ):
                    return _gatekeeper_copy(rules)
                if (
                    llm_result.action == UnderstandingAction.CLARIFICATION_NEEDED.value
                    and rules.action == UnderstandingAction.CLARIFICATION_NEEDED.value
                    and "leave field" in (rules.reasoning or "").lower()
                ):
                    return llm_result

            if rules.action == UnderstandingAction.SUBMIT.value and llm_result.action != UnderstandingAction.SUBMIT.value:
                return _gatekeeper_copy(rules)
            if rules.action == UnderstandingAction.REVIEW.value and llm_result.action != UnderstandingAction.REVIEW.value:
                return _gatekeeper_copy(rules)
            if (
                memory.pending_question
                and rules.answers_pending_field is False
                and llm_result.answers_pending_field is not False
                and llm_result.action in (
                    UnderstandingAction.COLLECT.value,
                    UnderstandingAction.CONFIRM.value,
                )
            ):
                return _gatekeeper_copy(rules)
            if (
                rules.action == UnderstandingAction.CONFIRM.value
                and memory.pending_confirmation == "submit"
                and llm_result.action != UnderstandingAction.CONFIRM.value
            ):
                return _gatekeeper_copy(rules)

        if (
            rules.workflow == "leave"
            and rules.action == UnderstandingAction.REVIEW.value
            and llm_result.workflow != "leave"
        ):
            return _gatekeeper_copy(rules)

        if (
            rules.workflow == "expense"
            and rules.action == UnderstandingAction.REVIEW.value
            and (rules.interrupt_workflow or "").strip().lower() == "expense"
            and llm_result.workflow == "leave"
        ):
            return _gatekeeper_copy(rules)

        from chat.services.platform.intent_rules import is_expense_draft_query as _expense_draft_query

        if (
            _expense_draft_query(message)
            and rules.workflow == "expense"
            and rules.action == UnderstandingAction.REVIEW.value
            and llm_result.action in _GATEKEEPER_WEAK_ACTIONS
        ):
            return _gatekeeper_copy(rules)

        if (
            rules.workflow == "expense"
            and rules.action == UnderstandingAction.SUBMIT.value
            and llm_result.action != UnderstandingAction.SUBMIT.value
        ):
            return _gatekeeper_copy(rules)

        pq = memory.pending_question
        if pq and pq.workflow_id == "leave" and not is_leave_review_mode(memory):
            if llm_result.answers_pending_field and llm_result.field_updates:
                return llm_result
            rules_pending = [
                u for u in (rules.field_updates or []) if u.field == pq.field
            ]
            if rules_pending and not llm_result.field_updates:
                return _gatekeeper_copy(rules)

        if (
            pq
            and pq.workflow_id == "expense"
            and memory.active_workflow
            and memory.active_workflow.id == "expense"
        ):
            if llm_result.answers_pending_field and llm_result.field_updates:
                return llm_result
            if llm_result.workflow != "expense" and (
                llm_result.field_updates
                or llm_result.answers_pending_field is not False
            ):
                llm_result.workflow = "expense"
                llm_result.interrupt_workflow = None
                if llm_result.answers_pending_field is None:
                    llm_result.answers_pending_field = True
                return llm_result
            if (
                rules.workflow == "expense"
                and rules.action == UnderstandingAction.COLLECT.value
                and llm_result.workflow == "leave"
            ):
                return _gatekeeper_copy(rules)

        from chat.services.platform.field_extractors.expense import is_expense_review_mode
        from chat.services.platform.intent_rules import is_delete_request, is_modify_request

        if (
            active_id == "expense"
            and is_expense_review_mode(memory)
            and (is_modify_request(message) or is_delete_request(message))
            and llm_result.workflow != "expense"
        ):
            expense_rules = self._expense_collect(
                message,
                memory,
                active_id,
                conversation_history=conversation_history,
            )
            return _gatekeeper_copy(expense_rules)

        if wizard_semantics_active(memory) and llm_result.is_out_of_scope and not is_programming_question(message):
            llm_result.is_out_of_scope = False

        return llm_result

    def _understand_rules(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pending_kind: str | None,
        conversation_history: list[str] | None = None,
    ) -> UnderstandingResult:
        draft = memory.active_draft()
        active_id = memory.active_workflow.id if memory.active_workflow else ""

        if is_programming_question(message):
            return UnderstandingResult(
                goal="Programming question",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.95,
                is_out_of_scope=True,
                reasoning="Programming languages are out of scope for HR assistant.",
                source="rules",
            )

        if is_greeting_or_chitchat(message):
            return UnderstandingResult(
                goal="Greeting",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.92,
                is_greeting=True,
                reasoning="Greeting or chitchat.",
                source="rules",
            )

        from chat.services.platform.intent_rules import is_off_hr_topic_message

        if is_off_hr_topic_message(message, memory=memory):
            return UnderstandingResult(
                goal="Out of scope",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.9,
                is_out_of_scope=True,
                reasoning="General / programming question outside HR assistant scope.",
                source="rules",
            )

        from chat.services.policy_intent_helpers import (
            is_general_knowledge_out_of_scope,
            is_hr_assistant_in_scope,
            is_off_topic_for_hr_assistant,
        )

        from chat.services.platform.turn_semantics import wizard_semantics_active

        pq_active = memory.pending_question
        active_id = memory.active_workflow.id if memory.active_workflow else ""
        wizard_on = wizard_semantics_active(memory)

        if active_id and is_workflow_show_request(message, workflow_id=active_id):
            return UnderstandingResult(
                goal="Show summary",
                workflow=active_id,
                action=UnderstandingAction.REVIEW.value,
                confidence=0.92,
                reasoning="Workflow summary or navigation (active draft).",
                source="rules",
                answers_pending_field=False,
            )

        if not wizard_on and (
            is_general_knowledge_out_of_scope(message)
            or (
                is_off_topic_for_hr_assistant(message, wizard_active=bool(pq_active))
                and not is_hr_assistant_in_scope(message)
            )
        ):
            return UnderstandingResult(
                goal="Out of scope",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.88,
                is_out_of_scope=True,
                reasoning="General-knowledge / off-HR topic detected.",
                source="rules",
            )

        if is_status_query(message):
            return UnderstandingResult(
                goal="Request status",
                workflow="none",
                action=UnderstandingAction.STATUS.value,
                confidence=0.9,
                reasoning="Request reference or status lookup phrasing.",
                source="rules",
            )

        from chat.services.policy_intent_helpers import is_hr_today_date_query

        if is_hr_today_date_query(message):
            return UnderstandingResult(
                goal="Today's date",
                workflow="none",
                action=UnderstandingAction.QUERY.value,
                confidence=1.0,
                reasoning="Today's calendar date.",
                source="rules",
            )

        submit_wf = parse_submit_workflow(message, active_workflow_id=active_id or None)
        if submit_wf:
            return UnderstandingResult(
                goal=f"Submit {submit_wf}",
                workflow=submit_wf,
                action=UnderstandingAction.SUBMIT.value,
                confidence=0.9,
                reasoning=f"Submit command for {submit_wf}.",
                source="rules",
            )

        if is_strong_new_workflow_message(message):
            target = infer_new_workflow_target(message)
            return UnderstandingResult(
                goal=f"Start {target or 'workflow'}",
                workflow=target or "none",
                action=UnderstandingAction.START.value,
                confidence=0.84,
                reasoning="Explicit new leave/expense/WFH request phrasing.",
                source="rules",
            )

        if active_id and is_same_workflow_navigation(message, active_workflow_id=active_id):
            return UnderstandingResult(
                goal="Show workflow review",
                workflow=active_id,
                action=UnderstandingAction.REVIEW.value,
                confidence=0.9,
                reasoning="Show or resume active workflow draft.",
                source="rules",
            )

        if active_id == "expense" and is_expense_draft_query(message):
            return UnderstandingResult(
                goal="Show expense draft",
                workflow="expense",
                action=UnderstandingAction.REVIEW.value,
                confidence=0.92,
                entities={
                    "expense_navigation": expense_navigation_kind(message),
                    "expense_intent": "show_summary",
                },
                reasoning="User asks to see their expense draft.",
                source="rules",
            )

        if is_switch_request(message, active_workflow_id=active_id or None):
            target = infer_switch_target(message)
            return UnderstandingResult(
                goal=f"Switch to {target or 'workflow'}",
                workflow=target or active_id or "none",
                action=UnderstandingAction.SWITCH.value,
                confidence=0.82,
                interrupt_workflow=target,
                reasoning="Explicit workflow switch/resume phrasing.",
                source="rules",
            )

        if memory.pending_confirmation == "submit" and active_id:
            if active_id == "leave" and is_workflow_interrupt_expense(message, active_workflow="leave"):
                result = self._start_expense(message, memory)
                result.workflow = "expense"
                result.interrupt_workflow = "expense"
                if result.action in _GATEKEEPER_WEAK_ACTIONS:
                    result.action = UnderstandingAction.START.value
                result.source = "rules"
                return result

            if active_id == "expense":
                from chat.services.platform.field_extractors.expense import is_expense_review_mode
                from chat.services.platform.field_extractors.modify import looks_like_expense_item_delete

                if is_expense_review_mode(memory):
                    if is_delete_request(message) or looks_like_expense_item_delete(message):
                        draft = memory.active_draft()
                        if draft:
                            return self._delete(message, draft, active_id)
                    if is_modify_request(message):
                        return self._expense_collect(
                            message,
                            memory,
                            active_id,
                            conversation_history=conversation_history,
                        )

            from chat.services.platform.field_extractors.leave import (
                is_leave_review_mode,
                review_field_updates_from_message,
            )

            if is_bare_rejection(message):
                return UnderstandingResult(
                    goal="Decline submit",
                    workflow=active_id,
                    action=UnderstandingAction.REVIEW.value,
                    confidence=0.9,
                    reasoning="User declined submit confirmation.",
                    source="rules",
                )
            if is_bare_confirmation(message):
                return UnderstandingResult(
                    goal="Confirm submit",
                    workflow=active_id,
                    action=UnderstandingAction.CONFIRM.value,
                    confidence=0.92,
                    reasoning="User confirmed submit.",
                    source="rules",
                )
            if is_resume_workflow_request(message, workflow_id=active_id):
                return UnderstandingResult(
                    goal="Show leave review",
                    workflow=active_id,
                    action=UnderstandingAction.REVIEW.value,
                    confidence=0.88,
                    reasoning="Return to review screen.",
                    source="rules",
                )
            if active_id == "leave" and is_leave_review_mode(memory):
                draft = memory.active_draft()
                if draft:
                    updates = review_field_updates_from_message(message, memory)
                    if updates:
                        return UnderstandingResult(
                            goal="Modify leave draft",
                            workflow=active_id or draft.workflow_id,
                            action=UnderstandingAction.MODIFY.value,
                            confidence=0.88,
                            field_updates=updates,
                            reasoning="Leave draft modification at review.",
                            source="rules",
                        )
                return UnderstandingResult(
                    goal="Ambiguous leave modify",
                    workflow=active_id or "leave",
                    action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                    confidence=0.55,
                    reasoning="Which leave field should I change — reason, date, or leave type?",
                    source="rules",
                )

        if active_id and is_resume_workflow_request(message, workflow_id=active_id):
            return UnderstandingResult(
                goal="Show leave review",
                workflow=active_id,
                action=UnderstandingAction.REVIEW.value,
                confidence=0.88,
                reasoning="Return to review screen.",
                source="rules",
            )

        if is_workflow_show_request(message, workflow_id=active_id or None):
            if should_resume_suspended_expense(
                message=message,
                active_workflow_id=active_id,
                suspended_workflows=memory.suspended_workflows,
            ):
                nav = expense_navigation_kind(message)
                return UnderstandingResult(
                    goal="Show suspended expense",
                    workflow="expense",
                    action=UnderstandingAction.REVIEW.value,
                    confidence=0.93,
                    interrupt_workflow="expense",
                    entities={"expense_navigation": nav, "expense_intent": "show_summary"},
                    reasoning="Expense summary while another workflow is active.",
                    source="rules",
                )
            wf = active_id or ("expense" if is_expense_message(message) else "leave")
            if "expense" in message.lower():
                wf = "expense"
            elif re.search(r"\b(leave|chuti|chhuti)\b", message.lower()):
                wf = "leave"
            return UnderstandingResult(
                goal="Show summary",
                workflow=wf or active_id,
                action=UnderstandingAction.REVIEW.value,
                confidence=0.9,
                reasoning="Workflow summary or navigation requested.",
                source="rules",
            )

        if is_bare_confirmation(message):
            if memory.pending_confirmation == "submit":
                return UnderstandingResult(
                    goal="Confirm",
                    workflow=active_id,
                    action=UnderstandingAction.CONFIRM.value,
                    confidence=0.92,
                    answers_pending_field=False,
                    reasoning="Confirm during submit review.",
                    source="rules",
                )
            if (memory.last_entities or {}).get("leave_start_clarify"):
                return UnderstandingResult(
                    goal="Start leave",
                    workflow="leave",
                    action=UnderstandingAction.START.value,
                    confidence=0.88,
                    field_updates=[],
                    reasoning="User confirmed starting leave after clarify prompt.",
                    source="rules",
                )
            if memory.active_workflow and memory.active_draft():
                return UnderstandingResult(
                    goal="Confirm",
                    workflow=active_id,
                    action=UnderstandingAction.CONFIRM.value,
                    confidence=0.92,
                    answers_pending_field=False,
                    reasoning="Confirm during active workflow or submit review.",
                    source="rules",
                )
            return UnderstandingResult(
                goal="Confirm",
                workflow=active_id,
                action=UnderstandingAction.CONFIRM.value,
                confidence=0.9,
                reasoning="Bare confirmation detected.",
                source="rules",
            )

        pq_slot = memory.pending_question
        if (
            pq_slot
            and pq_slot.workflow_id == "leave"
            and pq_slot.field == "reason"
            and is_reason_skip_message(message)
        ):
            from chat.services.platform.schemas import FieldUpdate

            return UnderstandingResult(
                goal="Skip optional reason",
                workflow="leave",
                action=UnderstandingAction.COLLECT.value,
                confidence=0.95,
                field_updates=[FieldUpdate(field="reason_skipped", value=True, action="set")],
                answers_pending_field=True,
                reasoning="User skipped optional leave reason.",
                source="rules",
            )

        from chat.services.policy_intent_helpers import is_policy_kb_query, is_rules_query

        if not is_workflow_application_message(message) and (
            is_rules_query(message) or is_policy_kb_query(message)
        ):
            return UnderstandingResult(
                goal="Policy query",
                workflow="none",
                action=UnderstandingAction.QUERY.value,
                confidence=0.88,
                reasoning="Policy or rules query detected.",
                source="rules",
            )

        if active_id and is_cancel_workflow_message(message, workflow_id=active_id):
            return UnderstandingResult(
                goal="Cancel workflow",
                workflow=active_id,
                action=UnderstandingAction.CANCEL.value,
                confidence=0.92,
                reasoning="User cancelled active workflow.",
                source="rules",
            )

        if not active_id and is_bare_rejection(message):
            return UnderstandingResult(
                goal="No active workflow",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.85,
                reasoning="Decline with no active workflow.",
                source="rules",
            )

        if active_id and draft and is_delete_request(message):
            return self._delete(message, draft, active_id)

        if active_id == "expense" and draft:
            from chat.services.platform.field_extractors.expense import is_expense_review_mode
            from chat.services.platform.intent_rules import is_expense_add_request, is_off_hr_topic_message

            if (
                not is_off_hr_topic_message(message, memory=memory)
                and (is_expense_review_mode(memory) or is_expense_add_request(message))
            ):
                return self._expense_collect(
                    message, memory, active_id, conversation_history=conversation_history
                )

        if active_id and draft and is_modify_request(message):
            if active_id == "leave" or draft.workflow_id == "leave":
                from chat.services.platform.field_extractors.leave import (
                    is_leave_review_mode,
                    review_field_updates_from_message,
                )
                from chat.services.platform.field_extractors.modify import parse_modify_request

                if is_leave_review_mode(memory):
                    updates = review_field_updates_from_message(message, memory)
                    if updates:
                        return UnderstandingResult(
                            goal="Modify leave draft",
                            workflow=active_id or draft.workflow_id,
                            action=UnderstandingAction.MODIFY.value,
                            confidence=0.88,
                            field_updates=updates,
                            reasoning="Leave draft modification.",
                            source="rules",
                        )
                if draft.fields.get("items") or parse_modify_request(
                    message, list(draft.fields.get("items") or [])
                ):
                    return self._modify(message, draft, active_id)
            elif draft.fields.get("items") or active_id == "expense":
                return self._modify(message, draft, active_id)
            return UnderstandingResult(
                goal="Modify field",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.MODIFY.value,
                confidence=0.78,
                reasoning="Modify/correct phrasing with active draft.",
                source="rules",
            )

        if active_id and is_vague_delete(message):
            return self._delete(message, draft, active_id)

        if active_id and draft and is_vague_amount_modify(message):
            return self._modify(message, draft, active_id)

        if pending_kind == "answer_pending" and memory.pending_question:
            return self._answer_pending(
                message, memory, conversation_history=conversation_history
            )

        if pending_kind == "modify_data" and draft:
            return self._modify(message, draft, active_id)

        if pending_kind == "delete_data" and draft:
            return self._delete(message, draft, active_id)

        if should_resume_suspended_expense(
            message=message,
            active_workflow_id=active_id,
            suspended_workflows=memory.suspended_workflows,
        ):
            nav = expense_navigation_kind(message)
            action = (
                UnderstandingAction.COLLECT.value
                if nav == "continue"
                else UnderstandingAction.REVIEW.value
            )
            return UnderstandingResult(
                goal="Resume suspended expense",
                workflow="expense",
                action=action,
                confidence=0.94,
                interrupt_workflow="expense",
                entities={
                    "expense_navigation": nav,
                    "expense_intent": "show_summary" if nav == "summary" else "continue",
                },
                reasoning="User navigates to suspended expense draft.",
                source="rules",
            )

        if active_id == "leave" and draft and draft.locked:
            from chat.services.platform.intent_rules import (
                find_submitted_leave_overlap_from_message,
                is_summary_request,
                message_has_new_leave_date_range,
                should_route_expense_after_submitted_leave,
            )

            if should_route_expense_after_submitted_leave(
                draft_locked=True,
                active_workflow_id=active_id,
                message=message,
            ):
                result = self._start_expense(message, memory)
                result.workflow = "expense"
                result.interrupt_workflow = "expense"
                result.action = UnderstandingAction.START.value
                return result

            submitted = list((memory.conversation_facts or {}).get("submitted_leave_ranges") or [])
            if is_leave_message(message) or is_strong_new_workflow_message(message):
                overlap = find_submitted_leave_overlap_from_message(message, submitted)
                if overlap:
                    return UnderstandingResult(
                        goal="Overlapping submitted leave dates",
                        workflow="leave",
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.92,
                        entities={"submitted_leave_overlap": overlap},
                        reasoning="Leave dates overlap an already submitted request.",
                        source="rules",
                    )
                if message_has_new_leave_date_range(message, submitted_ranges=submitted):
                    result = self._start_leave(message, memory)
                    if result.action == UnderstandingAction.REVIEW.value:
                        result.action = UnderstandingAction.START.value
                    return result
                if is_summary_request(message):
                    return UnderstandingResult(
                        goal="Leave summary",
                        workflow="leave",
                        action=UnderstandingAction.REVIEW.value,
                        confidence=0.9,
                        reasoning="Summary of submitted leave.",
                        source="rules",
                    )

        if active_id and is_expense_message(message) and not is_leave_message(message):
            if is_expense_draft_query(message):
                if active_id == "expense":
                    return UnderstandingResult(
                        goal="Show expense draft",
                        workflow="expense",
                        action=UnderstandingAction.REVIEW.value,
                        confidence=0.9,
                        entities={
                            "expense_navigation": expense_navigation_kind(message),
                            "expense_intent": "show_summary",
                        },
                        reasoning="User asks to see their expense draft.",
                        source="rules",
                    )
            else:
                return self._expense_collect(
                    message, memory, active_id, conversation_history=conversation_history
                )

        if active_id == "leave" and (is_leave_message(message) or memory.pending_question):
            from chat.services.platform.field_extractors.leave import (
                is_leave_review_mode,
                review_field_updates_from_message,
            )

            if is_leave_review_mode(memory) or memory.pending_confirmation == "submit":
                updates = review_field_updates_from_message(message, memory)
                if updates:
                    draft = memory.active_draft()
                    return UnderstandingResult(
                        goal="Modify leave draft",
                        workflow="leave",
                        action=UnderstandingAction.MODIFY.value,
                        confidence=0.88,
                        field_updates=updates,
                        reasoning="Leave draft modification at review.",
                        source="rules",
                    )
                if is_leave_review_mode(memory):
                    return UnderstandingResult(
                        goal="Ambiguous leave modify",
                        workflow="leave",
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.55,
                        reasoning="Which leave field should I change — reason, date, or leave type?",
                        source="rules",
                    )
            return self._leave_collect(message, memory)

        if is_expense_message(message):
            return self._start_expense(message, memory)

        if is_leave_message(message):
            return self._start_leave(message, memory)

        if active_id:
            return UnderstandingResult(
                goal="Continue workflow",
                workflow=active_id,
                action=UnderstandingAction.COLLECT.value,
                confidence=0.5,
                reasoning="Active workflow but message unclear.",
                source="rules",
            )

        return UnderstandingResult(
            goal="Unclear",
            workflow="none",
            action=UnderstandingAction.CLARIFICATION_NEEDED.value,
            confidence=0.4,
            reasoning="Could not determine intent.",
            source="rules",
        )

    def _answer_pending(
        self,
        message: str,
        memory: SessionMemory,
        *,
        conversation_history: list[str] | None = None,
    ) -> UnderstandingResult:
        pq = memory.pending_question
        wf = pq.workflow_id if pq else ""
        active_id = memory.active_workflow.id if memory.active_workflow else wf

        if pq and pq.field == "delete_which_item":
            m = re.search(r"\b(\d+)\b", message)
            if m:
                idx = int(m.group(1)) - 1
                return UnderstandingResult(
                    goal="Delete item",
                    workflow=active_id,
                    action=UnderstandingAction.DELETE.value,
                    confidence=0.85,
                    targets=[TargetRef(field="items", item_index=idx)],
                    reasoning=f"Delete item #{idx + 1}.",
                    source="rules",
                )

        if pq and pq.workflow_id == "expense" and pq.field in (
            "item_route",
            "item_category",
            "item_amount",
            "items",
            "incurred_date",
        ):
            result = self._expense_collect(
                message, memory, active_id, conversation_history=conversation_history
            )
            result.answers_pending_field = True
            return result

        if pq and pq.field in ("from_location", "to_location", "route", "item_route"):
            if is_expense_message(message):
                return self._expense_collect(
                    message, memory, active_id, conversation_history=conversation_history
                )
            route = parse_route(message)
            updates = []
            if route:
                updates = [
                    FieldUpdate(field="from_location", value=route[0]),
                    FieldUpdate(field="to_location", value=route[1]),
                ]
            elif pq.field == "from_location" and not parse_amount(message):
                updates = [FieldUpdate(field="from_location", value=message.strip())]
            elif pq.field == "to_location" and not parse_amount(message):
                updates = [FieldUpdate(field="to_location", value=message.strip())]
            else:
                return UnderstandingResult(
                    goal="Could not parse route",
                    workflow=active_id,
                    action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                    confidence=0.7,
                    reasoning="Please provide a travel route (from X to Y).",
                    source="rules",
                )
            return UnderstandingResult(
                goal="Provide route",
                workflow=active_id,
                action=UnderstandingAction.COLLECT.value,
                confidence=0.8,
                field_updates=updates,
                reasoning="Route information provided.",
                source="rules",
            )

        if wf == "leave" or active_id == "leave":
            if is_workflow_interrupt_expense(message, active_workflow="leave"):
                return UnderstandingResult(
                    goal="Expense interrupt during leave",
                    workflow="expense",
                    action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                    confidence=0.9,
                    reasoning="Expense claim detected while leave collection is active.",
                    source="rules",
                )

            if pq:
                parsed = self.fields.parse_pending_field("leave", pq.field, message, memory=memory)
                if parsed is None and (
                    is_workflow_interrupt_expense(message, active_workflow="leave")
                    or is_expense_message(message)
                ):
                    return UnderstandingResult(
                        goal="Expense interrupt during leave",
                        workflow="expense",
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.9,
                        reasoning="Expense claim detected while leave collection is active.",
                        source="rules",
                    )
                if parsed is None:
                    from chat.services.platform.field_extractors.leave import (
                        collect_slot_field_updates,
                    )

                    slot_updates = collect_slot_field_updates(message, memory)
                    if slot_updates:
                        return UnderstandingResult(
                            goal=f"Answer {pq.field}",
                            workflow="leave",
                            action=UnderstandingAction.COLLECT.value,
                            confidence=0.84,
                            answers_pending_field=True,
                            field_updates=slot_updates,
                            reasoning=f"Semantic collect for pending field '{pq.field}'.",
                            source="rules",
                        )
                    return UnderstandingResult(
                        goal=f"Could not parse {pq.field}",
                        workflow="leave",
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.85,
                        reasoning=f"Message does not answer pending field '{pq.field}'.",
                        source="rules",
                    )
                updates = [FieldUpdate(field=pq.field, value=parsed)]
            else:
                updates = self.fields.leave_field_updates_from_message(message, memory=memory)

            return UnderstandingResult(
                goal=f"Answer {pq.field if pq else 'field'}",
                workflow="leave",
                action=UnderstandingAction.COLLECT.value,
                confidence=0.82,
                field_updates=updates,
                reasoning="Leave field answer.",
                source="rules",
            )

        val = message.strip()
        if pq and pq.field == "incurred_date":
            val = parse_relative_date(message) or val
        return UnderstandingResult(
            goal=f"Answer {pq.field if pq else 'field'}",
            workflow=active_id,
            action=UnderstandingAction.COLLECT.value,
            confidence=0.82,
            field_updates=[FieldUpdate(field=pq.field, value=val)] if pq else [],
            reasoning="Pending question answer.",
            source="rules",
        )

    def _leave_modify(self, message: str, draft, active_id: str, *, memory=None) -> UnderstandingResult:
        from chat.services.platform.field_extractors.leave import review_field_updates_from_message

        updates = review_field_updates_from_message(message, memory) if memory else []
        if not updates:
            return UnderstandingResult(
                goal="Ambiguous leave modify",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.55,
                reasoning="Which leave field should I change — reason, date, or leave type?",
                source="rules",
            )
        return UnderstandingResult(
            goal="Modify leave draft",
            workflow=active_id or draft.workflow_id,
            action=UnderstandingAction.MODIFY.value,
            confidence=0.88,
            field_updates=updates,
            reasoning="Leave draft modification.",
            source="rules",
        )

    def _modify(self, message: str, draft, active_id: str) -> UnderstandingResult:
        items = list(draft.fields.get("items") or [])
        if is_vague_amount_modify(message):
            return UnderstandingResult(
                goal="Ambiguous modify",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.55,
                reasoning="Multiple amounts exist — which item should be updated?",
                source="rules",
            )
        parsed = parse_modify_request(message, items)
        if not parsed:
            return UnderstandingResult(
                goal="Ambiguous modify",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.5,
                reasoning="Could not identify which field or item to modify.",
                source="rules",
            )
        idx = parsed["item_index"]
        return UnderstandingResult(
            goal="Modify item",
            workflow=active_id or draft.workflow_id,
            action=UnderstandingAction.MODIFY.value,
            confidence=0.85,
            field_updates=[
                FieldUpdate(
                    field="items",
                    value={"amount": parsed["amount"]},
                    item_index=idx,
                    action="update",
                )
            ],
            entities={"modify_label": parsed.get("label"), "modify_index": idx, "modify_amount": parsed["amount"]},
            reasoning=f"Modify {parsed.get('label', 'item')} amount to {parsed['amount']}.",
            source="rules",
        )

    def _delete(self, message: str, draft, active_id: str) -> UnderstandingResult:
        if is_vague_delete(message):
            return UnderstandingResult(
                goal="Pick delete target",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.6,
                reasoning="Delete requested without specifying which entry.",
                source="rules",
            )
        items = list(draft.fields.get("items") or [])
        if items and (active_id == "expense" or draft.workflow_id == "expense"):
            parsed = parse_delete_request(message, items)
            if parsed:
                if parsed.get("needs_clarify"):
                    label = parsed.get("label") or parsed.get("category") or "item"
                    return UnderstandingResult(
                        goal="Pick delete target",
                        workflow=active_id or draft.workflow_id,
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.65,
                        reasoning=f"Multiple {label} lines — which one should I delete?",
                        source="rules",
                    )
                idx = int(parsed["item_index"])
                return UnderstandingResult(
                    goal="Delete item",
                    workflow=active_id or draft.workflow_id,
                    action=UnderstandingAction.DELETE.value,
                    confidence=0.9,
                    targets=[TargetRef(field="items", item_index=idx)],
                    reasoning=f"Delete {parsed.get('label', 'item')}.",
                    source="rules",
                )
        m = re.search(r"\b(\d+)\b", message)
        if m:
            idx = int(m.group(1)) - 1
            return UnderstandingResult(
                goal="Delete item",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.DELETE.value,
                confidence=0.85,
                targets=[TargetRef(field="items", item_index=idx)],
                source="rules",
            )
        if re.search(r"\bsecond\b", message.lower()) and len(items) > 1:
            return UnderstandingResult(
                goal="Delete second item",
                workflow=active_id or draft.workflow_id,
                action=UnderstandingAction.DELETE.value,
                confidence=0.85,
                targets=[TargetRef(field="items", item_index=1)],
                source="rules",
            )
        return UnderstandingResult(
            goal="Delete unclear",
            workflow=active_id or draft.workflow_id,
            action=UnderstandingAction.CLARIFICATION_NEEDED.value,
            confidence=0.5,
            reasoning="Which entry should I delete?",
            source="rules",
        )

    def _expense_collect(
        self,
        message: str,
        memory: SessionMemory,
        active_id: str,
        *,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
    ) -> UnderstandingResult:
        from chat.services.platform.field_extractors.expense import (
            expense_field_updates_from_message,
            expense_turn_to_field_updates,
            filter_expense_updates_for_review,
            is_expense_review_mode,
        )

        turn, updates = expense_turn_to_field_updates(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
        if is_expense_review_mode(memory) and updates:
            updates = filter_expense_updates_for_review(
                updates,
                message,
                memory=memory,
                trace_id=trace_id,
            )
        llm_primary = bool(turn.get("llm_used")) and not turn.get("wizard_fallback")
        source = "llm_expense" if llm_primary else "rules"
        if turn.get("off_topic"):
            return UnderstandingResult(
                goal="Out of scope",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.9,
                is_out_of_scope=True,
                reasoning="General question outside HR assistant scope.",
                source="rules",
            )
        intent = str(turn.get("intent") or "").lower()
        if updates or intent in (
            "add",
            "update",
            "delete",
            "modify_review",
            "answer_pending",
            "fix_mistake",
            "anti_summary",
            "show_summary",
            "show_list",
            "show_total",
            "confirm",
            "cancel",
            "clarify_modify",
            "clarify_delete",
        ):
            action = UnderstandingAction.COLLECT.value
            if intent == "delete":
                action = UnderstandingAction.DELETE.value
            elif intent in ("modify_review", "update"):
                action = UnderstandingAction.MODIFY.value
            elif intent in ("clarify_modify", "clarify_delete"):
                action = UnderstandingAction.CLARIFICATION_NEEDED.value
            elif intent == "fix_mistake":
                action = (
                    UnderstandingAction.MODIFY.value
                    if memory.pending_confirmation == "submit"
                    else UnderstandingAction.COLLECT.value
                )
            elif intent in ("show_summary", "show_list", "show_total"):
                action = UnderstandingAction.REVIEW.value
            elif intent in ("confirm", "submit"):
                action = UnderstandingAction.CONFIRM.value
            elif intent == "cancel":
                action = UnderstandingAction.CANCEL.value
            elif intent == "anti_summary":
                action = UnderstandingAction.CLARIFICATION_NEEDED.value
            entities = {"expense_intent": intent, "expense_turn": turn}
            if turn.get("llm_degraded"):
                entities["expense_llm_degraded"] = True
            if turn.get("wizard_fallback"):
                entities["expense_wizard_fallback"] = True
            elif llm_primary:
                entities.pop("expense_llm_degraded", None)
                entities.pop("expense_wizard_fallback", None)
            if intent == "anti_summary":
                entities["anti_summary"] = True
                entities["meta_complaint"] = True
            return UnderstandingResult(
                goal="Update expense draft",
                workflow="expense",
                action=action,
                confidence=0.88,
                field_updates=updates,
                entities=entities,
                reasoning="Expense draft editor.",
                source=source,
            )
        updates = expense_field_updates_from_message(
            message, memory=memory, conversation_history=conversation_history
        )
        if updates:
            return UnderstandingResult(
                goal="Add expense",
                workflow="expense",
                action=UnderstandingAction.COLLECT.value,
                confidence=0.85,
                field_updates=updates,
                entities={"expense_intent": "add"},
                interrupt_workflow="expense" if active_id and active_id != "expense" else None,
                reasoning="Expense line item added to active draft.",
                source="rules",
            )
        return UnderstandingResult(
            goal="Add expense",
            workflow="expense",
            action=UnderstandingAction.COLLECT.value,
            confidence=0.55,
            field_updates=[],
            entities={
                "expense_intent": intent or "conversation",
                **({"expense_llm_degraded": True} if turn.get("llm_degraded") else {}),
            },
            interrupt_workflow="expense" if active_id and active_id != "expense" else None,
            reasoning="Expense draft turn with no extractable patch.",
            source="rules",
        )

    def _start_expense(self, message: str, memory: SessionMemory) -> UnderstandingResult:
        from chat.services.platform.field_extractors.expense import expense_turn_to_field_updates

        turn, updates = expense_turn_to_field_updates(message, memory)
        if not updates:
            return UnderstandingResult(
                goal="Start expense",
                workflow="expense",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.5,
                entities={"expense_intent": str(turn.get("intent") or "conversation")},
                reasoning="Expense intent but no extractable items.",
                source="rules",
            )
        return UnderstandingResult(
            goal="Start expense",
            workflow="expense",
            action=UnderstandingAction.START.value,
            confidence=0.88,
            field_updates=updates,
            entities={"expense_intent": str(turn.get("intent") or "add")},
            reasoning="New expense from unified draft interpreter.",
            source="rules",
        )

    def _leave_collect(self, message: str, memory: SessionMemory) -> UnderstandingResult:
        draft = memory.active_draft()
        if draft and (draft.workflow_id == "leave" or (memory.active_workflow and memory.active_workflow.id == "leave")):
            from chat.services.platform.field_extractors.leave import (
                is_leave_review_mode,
                review_field_updates_from_message,
            )

            if is_leave_review_mode(memory):
                updates = review_field_updates_from_message(message, memory)
                if updates:
                    return UnderstandingResult(
                        goal="Modify leave draft",
                        workflow="leave",
                        action=UnderstandingAction.MODIFY.value,
                        confidence=0.88,
                        field_updates=updates,
                        reasoning="Leave draft modification at review.",
                        source="rules",
                    )
                return UnderstandingResult(
                    goal="Ambiguous leave modify",
                    workflow="leave",
                    action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                    confidence=0.55,
                    reasoning="Which leave field should I change — reason, date, or leave type?",
                    source="rules",
                )
        updates = self.fields.leave_field_updates_from_message(message, memory=memory)
        return UnderstandingResult(
            goal="Update leave",
            workflow="leave",
            action=UnderstandingAction.COLLECT.value,
            confidence=0.85,
            field_updates=updates,
            reasoning="Leave information collected.",
            source="rules",
        )

    def _start_leave(self, message: str, memory: SessionMemory) -> UnderstandingResult:
        seed = (message or "").strip().lower()
        if seed in ("leave", "chuti", "chhuti", "ছুটি"):
            updates = []
        else:
            updates = self.fields.leave_field_updates_from_message(message, memory=memory)
        return UnderstandingResult(
            goal="Start leave",
            workflow="leave",
            action=UnderstandingAction.START.value,
            confidence=0.88 if updates else 0.65,
            field_updates=updates,
            reasoning="New leave request from natural language.",
            source="rules",
        )

    def _parse_result(self, parsed: dict[str, Any], *, source: str) -> UnderstandingResult:
        updates = []
        for u in parsed.get("field_updates") or []:
            if isinstance(u, dict) and u.get("field"):
                updates.append(
                    FieldUpdate(
                        field=str(u["field"]),
                        value=u.get("value"),
                        item_index=u.get("item_index"),
                        action=str(u.get("action") or "set"),
                    )
                )
        targets = [
            TargetRef(field=str(t["field"]), item_index=t.get("item_index"))
            for t in (parsed.get("targets") or [])
            if isinstance(t, dict) and t.get("field")
        ]
        conf = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
        interrupt = parsed.get("interrupt_workflow")
        interrupt_wf = str(interrupt).strip().lower() if interrupt not in (None, "", "null", "none") else None
        if interrupt_wf not in ("leave", "expense"):
            interrupt_wf = None
        apf = parsed.get("answers_pending_field")
        if apf is None or apf == "null":
            answers_pending_field = None
        else:
            answers_pending_field = bool(apf)
        return UnderstandingResult(
            goal=str(parsed.get("goal") or ""),
            workflow=str(parsed.get("workflow") or "none"),
            action=str(parsed.get("action") or UnderstandingAction.NONE.value),
            confidence=conf,
            entities=dict(parsed.get("entities") or {}),
            field_updates=updates,
            targets=targets,
            missing_fields=list(parsed.get("missing_fields") or []),
            is_out_of_scope=bool(parsed.get("is_out_of_scope")),
            is_greeting=bool(parsed.get("is_greeting")),
            interrupt_workflow=interrupt_wf,
            reasoning=str(parsed.get("reasoning") or ""),
            source=source,
            answers_pending_field=answers_pending_field,
        )
