"""AI Understanding Layer — single structured output contract."""

from __future__ import annotations

import json
import re
from typing import Any

from chat.services.llm_client import LLMClient
from chat.services.observability import log_step
from chat.services.platform.confidence import apply_confidence_guard
from chat.services.platform.field_extractors import (
    extract_expense_item,
    extract_expense_items,
    is_vague_amount_modify,
    parse_amount,
    parse_modify_request,
    parse_relative_date,
    parse_route,
)
from chat.services.platform.field_extractors.leave import (
    infer_leave_type_from_text,
    merge_deterministic_leave_dates,
    sanitize_leave_type_value,
    text_has_third_party_sick_signal,
)
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.intent_rules import (
    expense_signal_strength,
    is_bare_confirmation,
    is_expense_message,
    is_greeting_or_chitchat,
    is_leave_message,
    is_programming_question,
    is_summary_request,
    is_total_request,
    is_vague_delete,
    is_workflow_interrupt_expense,
    parse_submit_workflow,
)
from chat.services.platform.llm_prompts import UNDERSTAND_SYSTEM
from chat.services.platform.registry import get_workflow_definition, list_workflow_ids
from chat.services.platform.schemas import (
    FieldUpdate,
    TargetRef,
    UnderstandingAction,
    UnderstandingResult,
)
from chat.services.session_memory import SessionMemory


def _medical_value_invalid(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "na", "nai", "nei", "n/a"):
        return True
    return False


_SYSTEM = """You interpret user messages for an HR conversational workflow platform.
Return ONLY valid JSON with goal, workflow, action, confidence, entities, field_updates, targets, reasoning.
See platform rules for natural language leave/expense, modify, delete, submit, summary."""

_INTENT_SYSTEM = """You classify user messages for an HR conversational workflow platform.

Return ONLY valid JSON:
{
  "workflow": "leave|expense|none",
  "action": "start|collect|modify|delete|review|submit|confirm|clarification_needed|none",
  "confidence": number 0-1,
  "is_out_of_scope": boolean,
  "reasoning": "one or two sentences"
}

RULES:
- Classification ONLY — do not extract field values or amounts.
- If pending_question is set and the message plausibly answers that slot, use action=collect with the active workflow.
- If user sends expense/reimbursement content while leave workflow is active, use workflow=expense action=start with high confidence.
- Programming trivia, general knowledge → is_out_of_scope=true.
- summary/review/total requests → action=review.
"""


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
        raw = (message or "").strip()
        client = llm or LLMClient()

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
                guarded = apply_confidence_guard(result)
                self._log(trace_id, raw, memory, guarded)
                return guarded

        result = apply_confidence_guard(
            self._sanitize_leave_result(
                raw,
                self._understand_rules(raw, memory=memory, pending_kind=pending_kind),
                memory=memory,
            )
        )
        self._log(trace_id, raw, memory, result)
        return result

    def classify_intent(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        llm: LLMClient | None = None,
    ) -> UnderstandingResult:
        """Alias for full understand() — single LLM call for intent + fields."""
        return self.understand(
            message,
            memory=memory,
            conversation_history=conversation_history,
            trace_id=trace_id,
            llm=llm,
        )

    def _log_intent(
        self, trace_id: str, message: str, memory: SessionMemory, result: UnderstandingResult
    ) -> None:
        log_step(
            trace_id,
            "intent_classification",
            {
                "user_message": message,
                "understanding": result.to_dict(),
                "active_workflow": memory.active_workflow.to_dict() if memory.active_workflow else None,
                "pending_question": (
                    memory.pending_question.to_dict() if memory.pending_question else None
                ),
            },
        )

    def _classify_intent_llm(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        client: LLMClient,
    ) -> UnderstandingResult | None:
        draft = memory.active_draft()
        context = {
            "workflows_available": list_workflow_ids(),
            "active_workflow": memory.active_workflow.to_dict() if memory.active_workflow else None,
            "pending_question": memory.pending_question.to_dict() if memory.pending_question else None,
            "workflow_draft": draft.to_dict() if draft else None,
            "conversation_history": conversation_history[-8:],
        }
        parsed = client.chat_json(
            system_prompt=_INTENT_SYSTEM,
            user_prompt=(
                "Session context (JSON):\n"
                f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
                f"User message:\n{message}"
            ),
            trace_id=trace_id,
        )
        if not isinstance(parsed, dict):
            return None
        wf_raw = str(parsed.get("workflow") or "none").strip().lower()
        action_raw = str(parsed.get("action") or UnderstandingAction.NONE.value).strip().lower()

        wf = wf_raw.split("|")[0].strip() if "|" in wf_raw else wf_raw
        action = action_raw.split("|")[0].strip() if "|" in action_raw else action_raw

        if wf not in ("leave", "expense", "none"):
            if "expense" in wf_raw:
                wf = "expense"
            elif "leave" in wf_raw:
                wf = "leave"
            else:
                return None

        allowed_actions = {a.value for a in UnderstandingAction}
        if action not in allowed_actions:
            if "review" in action_raw or "summary" in action_raw:
                action = UnderstandingAction.REVIEW.value
            elif "start" in action_raw or "collect" in action_raw:
                action = UnderstandingAction.START.value
            else:
                return None

        conf = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
        result = UnderstandingResult(
            goal="LLM intent classification",
            workflow=wf,
            action=action,
            confidence=conf,
            is_out_of_scope=bool(parsed.get("is_out_of_scope")),
            reasoning=str(parsed.get("reasoning") or "LLM intent classification."),
            source="llm",
        )
        if is_greeting_or_chitchat(message):
            result.is_out_of_scope = False
            result.workflow = "none"
            result.action = UnderstandingAction.NONE.value
            result.goal = "Greeting"
        return result

    def _classify_intent_rules(self, message: str, *, memory: SessionMemory) -> UnderstandingResult:
        active_id = memory.active_workflow.id if memory.active_workflow else ""

        if is_greeting_or_chitchat(message):
            return UnderstandingResult(
                goal="Greeting",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=0.95,
                reasoning="User greeting — no workflow action.",
                source="rules",
            )

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

        submit_wf = parse_submit_workflow(message)
        if submit_wf:
            return UnderstandingResult(
                goal=f"Submit {submit_wf}",
                workflow=submit_wf,
                action=UnderstandingAction.SUBMIT.value,
                confidence=0.9,
                reasoning=f"Submit command for {submit_wf}.",
                source="rules",
            )

        if is_summary_request(message) or is_total_request(message):
            wf = active_id
            if "expense" in message.lower():
                wf = "expense"
            elif "leave" in message.lower():
                wf = "leave"
            return UnderstandingResult(
                goal="Show summary",
                workflow=wf or "expense",
                action=UnderstandingAction.REVIEW.value,
                confidence=0.88,
                reasoning="Summary or total requested.",
                source="rules",
            )

        if is_bare_confirmation(message):
            return UnderstandingResult(
                goal="Confirm",
                workflow=active_id,
                action=UnderstandingAction.CONFIRM.value,
                confidence=0.9,
                reasoning="Bare confirmation detected.",
                source="rules",
            )

        if is_vague_delete(message):
            return UnderstandingResult(
                goal="Delete",
                workflow=active_id,
                action=UnderstandingAction.DELETE.value,
                confidence=0.75,
                reasoning="Vague delete phrasing.",
                source="rules",
            )

        if memory.pending_question and is_workflow_interrupt_expense(message, active_workflow=active_id):
            return UnderstandingResult(
                goal="Expense interrupt",
                workflow="expense",
                action=UnderstandingAction.START.value,
                confidence=expense_signal_strength(message),
                reasoning="Expense claim detected during active leave workflow.",
                source="rules",
            )

        if is_expense_message(message) and not is_leave_message(message):
            return UnderstandingResult(
                goal="Expense intent",
                workflow="expense",
                action=UnderstandingAction.START.value if not active_id else UnderstandingAction.COLLECT.value,
                confidence=expense_signal_strength(message) or 0.85,
                reasoning="Expense claim detected.",
                source="rules",
            )

        if is_leave_message(message):
            return UnderstandingResult(
                goal="Leave intent",
                workflow="leave",
                action=UnderstandingAction.START.value if not active_id else UnderstandingAction.COLLECT.value,
                confidence=0.88,
                reasoning="Leave request detected.",
                source="rules",
            )

        if memory.pending_question:
            pq_wf = memory.pending_question.workflow_id or active_id
            return UnderstandingResult(
                goal=f"Answer {memory.pending_question.field}",
                workflow=pq_wf,
                action=UnderstandingAction.COLLECT.value,
                confidence=0.72,
                reasoning="Message may answer pending slot.",
                source="rules",
            )

        if active_id:
            return UnderstandingResult(
                goal="Continue workflow",
                workflow=active_id,
                action=UnderstandingAction.COLLECT.value,
                confidence=0.5,
                reasoning="Active workflow but intent unclear.",
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
        context = {
            "today_iso": date.today().isoformat(),
            "workflows_available": list_workflow_ids(),
            "active_workflow": memory.active_workflow.to_dict() if memory.active_workflow else None,
            "pending_question": memory.pending_question.to_dict() if memory.pending_question else None,
            "workflow_draft": draft.to_dict() if draft else None,
            "pending_confirmation": memory.pending_confirmation,
            "suspended_workflows": [s.to_dict() for s in memory.suspended_workflows],
            "pending_kind_hint": pending_kind,
            "conversation_history": conversation_history[-8:],
        }
        parsed = client.chat_json(
            system_prompt=UNDERSTAND_SYSTEM,
            user_prompt=(
                "Session context (JSON):\n"
                f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
                f"User message:\n{message}"
            ),
            trace_id=trace_id,
        )
        if not isinstance(parsed, dict):
            return None
        result = self._parse_result(parsed, source="llm")
        return self._sanitize_leave_result(message, result, memory=memory)

    @staticmethod
    def _sanitize_leave_result(
        message: str,
        result: UnderstandingResult,
        *,
        memory: SessionMemory,
    ) -> UnderstandingResult:
        """Ground leave field_updates — family illness ≠ sick leave; prefer parsed dates."""
        if result.workflow not in ("leave", "") and not (
            memory.active_workflow and memory.active_workflow.id == "leave"
        ):
            return result

        pq = memory.pending_question
        combined = message
        if pq and pq.field == "leave_type":
            # Direct slot answer — only sanitize explicit sick misuse.
            updates = []
            for upd in result.field_updates:
                if upd.field == "leave_type":
                    clean = sanitize_leave_type_value(message, str(upd.value or ""))
                    if clean:
                        updates.append(FieldUpdate(field="leave_type", value=clean, action=upd.action))
                elif upd.field == "medical_document" and _medical_value_invalid(upd.value):
                    continue
                else:
                    updates.append(upd)
            result.field_updates = updates
            return result

        # Merge deterministic dates when narrative contains explicit calendar dates.
        det = merge_deterministic_leave_dates({}, message)
        updates: list[FieldUpdate] = []
        seen: set[str] = set()
        for upd in result.field_updates:
            if upd.field == "leave_type":
                clean = sanitize_leave_type_value(message, str(upd.value or ""))
                if not clean:
                    continue
                updates.append(FieldUpdate(field="leave_type", value=clean, action=upd.action))
                seen.add("leave_type")
            elif upd.field == "medical_document" and _medical_value_invalid(upd.value):
                continue
            elif upd.field in det and det[upd.field]:
                updates.append(FieldUpdate(field=upd.field, value=det[upd.field], action=upd.action))
                seen.add(upd.field)
            else:
                updates.append(upd)
                seen.add(upd.field)

        for key in ("start_date", "end_date"):
            if key in det and key not in seen:
                updates.append(FieldUpdate(field=key, value=det[key], action="set"))

        # Rules/LLM may omit leave_type but infer sick wrongly via reason-only narrative.
        if "leave_type" not in seen and text_has_third_party_sick_signal(message):
            inferred = infer_leave_type_from_text(message)
            if inferred and inferred != "sick":
                updates.append(FieldUpdate(field="leave_type", value=inferred, action="set"))

        # Ensure reason captured for long family narratives.
        if text_has_third_party_sick_signal(message) and "reason" not in seen:
            updates.append(FieldUpdate(field="reason", value=message.strip()[:500], action="set"))

        result.field_updates = updates
        if text_has_third_party_sick_signal(message) and any(
            u.field == "leave_type" and u.value == "sick" for u in updates
        ):
            result.field_updates = [u for u in updates if not (u.field == "leave_type" and u.value == "sick")]
        return result

    def _understand_rules(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pending_kind: str | None,
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

        submit_wf = parse_submit_workflow(message)
        if submit_wf:
            return UnderstandingResult(
                goal=f"Submit {submit_wf}",
                workflow=submit_wf,
                action=UnderstandingAction.SUBMIT.value,
                confidence=0.9,
                reasoning=f"Submit command for {submit_wf}.",
                source="rules",
            )

        if is_summary_request(message) or is_total_request(message):
            wf = active_id or ("expense" if is_expense_message(message) else "leave" if is_leave_message(message) else active_id)
            if "expense" in message.lower() or "expense" in (message or ""):
                wf = "expense"
            elif "leave" in message.lower() or "leave" in (message or ""):
                wf = "leave"
            return UnderstandingResult(
                goal="Show summary",
                workflow=wf or active_id,
                action=UnderstandingAction.REVIEW.value,
                confidence=0.88,
                reasoning="Summary or total requested.",
                source="rules",
            )

        if is_bare_confirmation(message):
            return UnderstandingResult(
                goal="Confirm",
                workflow=active_id,
                action=UnderstandingAction.CONFIRM.value,
                confidence=0.9,
                reasoning="Bare confirmation detected.",
                source="rules",
            )

        from chat.services.policy_intent_helpers import is_policy_kb_query, is_rules_query

        if is_rules_query(message) or is_policy_kb_query(message):
            return UnderstandingResult(
                goal="Policy query",
                workflow="none",
                action=UnderstandingAction.QUERY.value,
                confidence=0.88,
                reasoning="Policy or rules query detected.",
                source="rules",
            )

        if active_id and is_vague_delete(message):
            return self._delete(message, draft, active_id)

        if active_id and draft and is_vague_amount_modify(message):
            return self._modify(message, draft, active_id)

        if pending_kind == "answer_pending" and memory.pending_question:
            return self._answer_pending(message, memory)

        if pending_kind == "modify_data" and draft:
            return self._modify(message, draft, active_id)

        if pending_kind == "delete_data" and draft:
            return self._delete(message, draft, active_id)

        if active_id and is_expense_message(message) and not is_leave_message(message):
            return self._expense_collect(message, memory, active_id)

        if active_id == "leave" and (is_leave_message(message) or memory.pending_question):
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

    def _answer_pending(self, message: str, memory: SessionMemory) -> UnderstandingResult:
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

        if pq and pq.field in ("from_location", "to_location", "route"):
            if is_expense_message(message):
                return self._expense_collect(message, memory, active_id)
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
                parsed = self.fields.parse_pending_field("leave", pq.field, message)
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
                fields = self.fields.extract_workflow_fields("leave", message)
                updates = [FieldUpdate(field=k, value=v) for k, v in fields.items()]

            draft = memory.active_draft()
            if draft and updates:
                for upd in updates:
                    if upd.field == "start_date" and draft.fields.get("start_date"):
                        if upd.value != draft.fields.get("start_date") and is_leave_message(message):
                            return UnderstandingResult(
                                goal="Blocked duplicate leave",
                                workflow="leave",
                                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                                confidence=0.92,
                                reasoning="An open leave request already exists — submit or cancel it first.",
                                source="rules",
                            )
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

    def _expense_collect(self, message: str, memory: SessionMemory, active_id: str) -> UnderstandingResult:
        items = extract_expense_items(message)
        updates: list[FieldUpdate] = []
        for item in items:
            updates.append(FieldUpdate(field="items", value=item, action="append"))
        if not memory.active_draft().fields.get("incurred_date"):
            d = parse_relative_date(message)
            if d:
                updates.append(FieldUpdate(field="incurred_date", value=d))
        route = parse_route(message)
        if route:
            updates.extend([
                FieldUpdate(field="from_location", value=route[0]),
                FieldUpdate(field="to_location", value=route[1]),
            ])
        return UnderstandingResult(
            goal="Add expense",
            workflow="expense",
            action=UnderstandingAction.COLLECT.value,
            confidence=0.85,
            field_updates=updates,
            reasoning="Expense line item added to active draft.",
            source="rules",
        )

    def _leave_collect(self, message: str, memory: SessionMemory) -> UnderstandingResult:
        draft = memory.active_draft()
        fields = self.fields.extract_workflow_fields("leave", message)
        if draft and fields.get("start_date") and draft.fields.get("start_date"):
            if fields["start_date"] != draft.fields.get("start_date") and is_leave_message(message):
                return UnderstandingResult(
                    goal="Blocked duplicate leave",
                    workflow="leave",
                    action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                    confidence=0.92,
                    reasoning="An open leave request already exists — submit or cancel it first.",
                    source="rules",
                )
        updates = [FieldUpdate(field=k, value=v) for k, v in fields.items()]
        return UnderstandingResult(
            goal="Update leave",
            workflow="leave",
            action=UnderstandingAction.COLLECT.value,
            confidence=0.85,
            field_updates=updates,
            reasoning="Leave information collected.",
            source="rules",
        )

    def _start_expense(self, message: str, memory: SessionMemory) -> UnderstandingResult:
        items = extract_expense_items(message)
        if not items:
            return UnderstandingResult(
                goal="Start expense",
                workflow="expense",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.5,
                reasoning="Expense intent but no amount found.",
                source="rules",
            )
        updates = [FieldUpdate(field="items", value=item, action="append") for item in items]
        d = parse_relative_date(message)
        if d:
            updates.append(FieldUpdate(field="incurred_date", value=d))
        route = parse_route(message)
        if route:
            updates.extend([
                FieldUpdate(field="from_location", value=route[0]),
                FieldUpdate(field="to_location", value=route[1]),
            ])
        elif items[0].get("category") == "travel" and not route:
            if re.search(r"office jete|khoroch hoise|lagse\b", message.lower()):
                return UnderstandingResult(
                    goal="Clarify expense category",
                    workflow="expense",
                    action=UnderstandingAction.START.value,
                    confidence=0.72,
                    field_updates=[FieldUpdate(
                        field="items",
                        value={"amount": items[0]["amount"], "description": message.strip()},
                        action="append",
                    )],
                    reasoning="Expense amount found but category/route unclear.",
                    source="rules",
                )
        return UnderstandingResult(
            goal="Start expense",
            workflow="expense",
            action=UnderstandingAction.START.value,
            confidence=0.88,
            field_updates=updates,
            reasoning="New expense from natural language.",
            source="rules",
        )

    def _start_leave(self, message: str, memory: SessionMemory) -> UnderstandingResult:
        if memory.active_workflow and memory.active_workflow.id == "leave":
            draft = memory.active_draft()
            if draft and not draft.locked and draft.fields.get("start_date"):
                new_fields = self.fields.extract_workflow_fields("leave", message)
                new_date = new_fields.get("start_date")
                if new_date and new_date != draft.fields.get("start_date"):
                    return UnderstandingResult(
                        goal="Blocked duplicate leave",
                        workflow="leave",
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.9,
                        reasoning="An open leave request already exists — submit or cancel it first.",
                        source="rules",
                    )
                if re.search(r"\bleave\s+chai\b", message.lower()) and draft.fields:
                    return UnderstandingResult(
                        goal="Blocked duplicate leave",
                        workflow="leave",
                        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                        confidence=0.9,
                        reasoning="A leave draft is already in progress.",
                        source="rules",
                    )

        fields = self.fields.extract_workflow_fields("leave", message)
        updates = [FieldUpdate(field=k, value=v) for k, v in fields.items()]
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
        )
