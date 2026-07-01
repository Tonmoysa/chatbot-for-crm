"""
Decision Core (Pending Question Engine) — authoritative turn routing.

Phase 3: maps UnderstandingResult → PendingQuestionDecision; no legacy re-interpretation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from chat.constants import (
    INTENT_HR_POLICY,
    INTENT_REQUEST_STATUS,
    INTENT_UNKNOWN,
)
from chat.services.platform.intent_rules import (
    infer_new_workflow_target,
    is_leave_navigation_from_expense,
    is_off_hr_topic_message,
    is_status_query,
    is_strong_new_workflow_message,
    is_workflow_application_message,
    is_workflow_show_request,
    suspended_workflow_ids,
)
from chat.services.platform.turn_semantics import is_process_question, is_workflow_meta_complaint
from chat.services.policy_intent_helpers import (
    is_hr_today_date_query,
    is_policy_kb_query,
    is_rules_query,
)
from chat.services.llm_client import LLMClient
from chat.services.observability import log_step, log_turn_context_layer
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import SessionMemory
from chat.services.translator import is_translation_request


class MessageIntentKind(str, Enum):
    ANSWER_PENDING = "answer_pending"
    MODIFY_DATA = "modify_data"
    DELETE_DATA = "delete_data"
    SWITCH_WORKFLOW = "switch_workflow"
    ASK_POLICY = "ask_policy"
    ASK_STATUS = "ask_status"
    ASK_TODAY_DATE = "ask_today_date"
    ASK_TRANSLATION = "ask_translation"
    OUT_OF_SCOPE = "out_of_scope"
    NEW_WORKFLOW = "new_workflow"
    CANCEL_WORKFLOW = "cancel_workflow"
    CLARIFICATION_NEEDED = "clarification_needed"
    SHOW_REVIEW = "show_review"


# Kinds that must win over starting a new workflow when pending_question is active.
_INTERRUPT_KINDS = frozenset(
    {
        MessageIntentKind.MODIFY_DATA,
        MessageIntentKind.DELETE_DATA,
        MessageIntentKind.SWITCH_WORKFLOW,
        MessageIntentKind.CANCEL_WORKFLOW,
        MessageIntentKind.CLARIFICATION_NEEDED,
        MessageIntentKind.SHOW_REVIEW,
        MessageIntentKind.ASK_POLICY,
        MessageIntentKind.ASK_STATUS,
        MessageIntentKind.ASK_TODAY_DATE,
        MessageIntentKind.ASK_TRANSLATION,
        MessageIntentKind.OUT_OF_SCOPE,
    }
)

# Strong new-workflow phrasing — SSOT in intent_rules (Phase 3).

_INFORMATIONAL_PRIORITY_KINDS = frozenset(
    {
        MessageIntentKind.ASK_POLICY,
        MessageIntentKind.ASK_STATUS,
        MessageIntentKind.ASK_TODAY_DATE,
        MessageIntentKind.ASK_TRANSLATION,
        MessageIntentKind.OUT_OF_SCOPE,
    }
)

_PLATFORM_PQ_KINDS = frozenset(
    {
        MessageIntentKind.ANSWER_PENDING,
        MessageIntentKind.MODIFY_DATA,
        MessageIntentKind.DELETE_DATA,
        MessageIntentKind.SWITCH_WORKFLOW,
        MessageIntentKind.CLARIFICATION_NEEDED,
        MessageIntentKind.SHOW_REVIEW,
    }
)


def intent_priority_decision(
    message: str,
    *,
    memory: SessionMemory,
    understanding: UnderstandingResult,
) -> "PendingQuestionDecision | None":
    """Structured understanding beats open pending_question (SSOT priority)."""
    from chat.services.platform.field_extractors.expense import (
        expense_turn_has_targeted_patches,
        memory_has_expense_draft,
        message_has_new_expense_items,
        message_requests_submit_after_edit,
        pending_expense_edit_active,
        resolve_pending_expense_edit_turn,
    )
    from chat.services.platform.intent_rules import parse_submit_workflow

    aw = memory.active_workflow
    pq = memory.pending_question
    conf = understanding.confidence
    src = understanding.source or "understanding"
    expense_intent = str((understanding.entities or {}).get("expense_intent") or "").lower()
    expense_turn = (understanding.entities or {}).get("expense_turn")
    wf = understanding.workflow or (aw.id if aw else "")

    from chat.services.platform.turn_semantics import (
        cross_workflow_switch_target,
        is_expense_review_request,
    )

    cancel_target = str((understanding.entities or {}).get("cancel_workflow_target") or "").lower()
    if not cancel_target:
        from chat.services.platform.workflow_cancel import resolve_workflow_cancel_target

        cancel_target = (
            resolve_workflow_cancel_target(
                message,
                memory,
                active_workflow_id=aw.id if aw else "",
            )
            or ""
        )
    if cancel_target in ("leave", "expense") or understanding.action == UnderstandingAction.CANCEL.value:
        wf = cancel_target or understanding.workflow or (aw.id if aw else None)
        return PendingQuestionDecision(
            kind=MessageIntentKind.CANCEL_WORKFLOW,
            confidence=max(conf, 0.92),
            reasoning=understanding.reasoning or f"Cancel {wf} workflow.",
            source=src,
            blocks_new_workflow=True,
            target_workflow=wf if wf not in ("none", "") else None,
            extracted_entities={"interrupts_pending": bool(pq)},
        )

    show_target = str((understanding.entities or {}).get("show_workflow_target") or "").lower()
    if not show_target:
        from chat.services.platform.workflow_show import resolve_workflow_show_target

        show_target = (
            resolve_workflow_show_target(
                message,
                memory,
                active_workflow_id=aw.id if aw else "",
            )
            or ""
        )
    if show_target in ("leave", "expense"):
        return PendingQuestionDecision(
            kind=MessageIntentKind.SHOW_REVIEW,
            confidence=max(conf, 0.92),
            reasoning=understanding.reasoning or f"Show {show_target} summary.",
            source=src,
            blocks_new_workflow=True,
            target_workflow=show_target,
            extracted_entities={"interrupts_pending": bool(pq)},
        )

    if is_expense_review_request(message, understanding):
        return PendingQuestionDecision(
            kind=MessageIntentKind.SHOW_REVIEW,
            confidence=max(conf, 0.9),
            reasoning=understanding.reasoning or "Show expense list or summary.",
            source=src,
            blocks_new_workflow=True,
            target_workflow="expense",
            extracted_entities={"interrupts_pending": bool(pq)},
        )

    def _decision(kind: MessageIntentKind, *, reasoning: str, target: str | None = None):
        return PendingQuestionDecision(
            kind=kind,
            confidence=max(conf, 0.88),
            reasoning=reasoning,
            source=src,
            blocks_new_workflow=True,
            target_workflow=target or (aw.id if aw else wf or None),
            extracted_entities={"interrupts_pending": bool(pq)},
        )

    def _new_expense_with_updates(*, reasoning: str) -> PendingQuestionDecision:
        return PendingQuestionDecision(
            kind=MessageIntentKind.NEW_WORKFLOW,
            confidence=max(conf, 0.88),
            reasoning=reasoning,
            source=src,
            blocks_new_workflow=False,
            target_workflow="expense",
            extracted_entities={"interrupts_pending": bool(pq)},
        )

    def _route_expense_mutation(*, reasoning: str, target: str | None = None):
        wf_target = (target or wf or "expense").strip().lower()
        if wf_target == "expense" and has_patches and not memory_has_expense_draft(memory):
            return _new_expense_with_updates(reasoning=reasoning or "New expense claim with line items.")
        return _decision(
            MessageIntentKind.MODIFY_DATA,
            reasoning=reasoning,
            target=target,
        )

    from chat.services.platform.intent_rules import is_bare_confirmation

    if (
        pq
        and pq.workflow_id == "expense"
        and is_bare_confirmation(message)
        and pq.field in ("item_route", "item_category", "item_amount", "route")
    ):
        return _decision(
            MessageIntentKind.CLARIFICATION_NEEDED,
            reasoning=f"Pending {pq.field} — need a field value, not yes/no.",
        )

    switch_target = cross_workflow_switch_target(understanding, aw.id if aw else None)
    if (
        switch_target
        and aw
        and conf >= 0.65
        and not is_expense_review_request(message, understanding)
    ):
        return PendingQuestionDecision(
            kind=MessageIntentKind.SWITCH_WORKFLOW,
            confidence=max(conf, 0.85),
            reasoning=understanding.reasoning or f"Switch to {switch_target}.",
            source=src,
            blocks_new_workflow=True,
            target_workflow=switch_target,
            extracted_entities={"interrupts_pending": bool(pq)},
        )

    if understanding.action == UnderstandingAction.DELETE.value or expense_intent in (
        "delete",
        "clarify_delete",
    ):
        if expense_intent == "clarify_delete" and not understanding.field_updates:
            if not expense_turn_has_targeted_patches(expense_turn, memory):
                return _decision(
                    MessageIntentKind.CLARIFICATION_NEEDED,
                    reasoning=understanding.reasoning or "Clarify which expense to delete.",
                )
        return _decision(
            MessageIntentKind.DELETE_DATA,
            reasoning=understanding.reasoning or "Delete expense line.",
        )

    has_patches = bool(
        understanding.field_updates
        or expense_turn_has_targeted_patches(expense_turn, memory)
    )
    if message_has_new_expense_items(message) and pq and pq.workflow_id == "expense":
        from chat.services.platform.field_extractors.expense import _should_compound_add_over_mutation

        expense_turn = (understanding.entities or {}).get("expense_turn")
        if _should_compound_add_over_mutation(
            expense_turn if isinstance(expense_turn, dict) else None,
            message,
            memory,
        ):
            return _decision(
                MessageIntentKind.MODIFY_DATA,
                reasoning=understanding.reasoning or "Add new expense line while collecting.",
                target=aw.id if aw else "expense",
            )

    if understanding.action == UnderstandingAction.MODIFY.value or expense_intent in (
        "modify_review",
        "update",
        "correct",
    ):
        if has_patches and not (
            pq
            and pq.workflow_id == "expense"
            and is_bare_confirmation(message)
        ):
            return _route_expense_mutation(
                reasoning=understanding.reasoning or "Modify expense line.",
            )

    item_field_updates = [
        u
        for u in (understanding.field_updates or [])
        if getattr(u, "field", None) == "items"
    ]

    if expense_intent == "add" and understanding.field_updates:
        if message_has_new_expense_items(message) and item_field_updates:
            return _route_expense_mutation(
                reasoning=understanding.reasoning or "Compound expense overrides pending slot.",
            )
        if not pq and (not aw or aw.id != "expense"):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=max(conf, 0.88),
                reasoning=understanding.reasoning or "New expense claim with line items.",
                source=src,
                blocks_new_workflow=False,
                target_workflow="expense",
                extracted_entities={"interrupts_pending": bool(pq)},
            )
        if not item_field_updates:
            return _decision(
                MessageIntentKind.ANSWER_PENDING,
                reasoning=understanding.reasoning or "Add expense line.",
                target=aw.id if aw else "expense",
            )
        return _route_expense_mutation(
            reasoning=understanding.reasoning or "Add expense line.",
            target=aw.id if aw else "expense",
        )

    if understanding.action in (
        UnderstandingAction.START.value,
        UnderstandingAction.COLLECT.value,
    ) and has_patches and str(wf or "").lower() == "expense" and not memory_has_expense_draft(memory):
        return _new_expense_with_updates(
            reasoning=understanding.reasoning or "Start expense with line items.",
        )

    if understanding.action == UnderstandingAction.REVIEW.value or expense_intent in (
        "show_summary",
        "show_list",
        "show_total",
    ):
        return _decision(
            MessageIntentKind.SHOW_REVIEW,
            reasoning=understanding.reasoning or "Show expense draft.",
        )

    active_wf = aw.id if aw else "expense"
    wants_submit = (
        understanding.action == UnderstandingAction.SUBMIT.value
        or expense_intent in ("confirm", "submit")
        or message_requests_submit_after_edit(message, active_workflow_id=active_wf)
        or (
            message_has_new_expense_items(message)
            and parse_submit_workflow(message, active_workflow_id=active_wf)
        )
    )
    if wants_submit:
        if message_has_new_expense_items(message) and has_patches:
            return _route_expense_mutation(
                reasoning=understanding.reasoning or "Apply expense items before submit.",
            )
        return _decision(
            MessageIntentKind.ANSWER_PENDING,
            reasoning=understanding.reasoning or "Confirm submit.",
        )

    if pending_expense_edit_active(memory):
        resolved = resolve_pending_expense_edit_turn(message, memory)
        if resolved and (
            resolved.get("item_patches") or resolved.get("delete_indices")
        ):
            resolved_intent = str(resolved.get("intent") or "").lower()
            if resolved_intent == "delete":
                return _decision(
                    MessageIntentKind.DELETE_DATA,
                    reasoning="Resolve pending edit clarification (delete).",
                )
            return _decision(
                MessageIntentKind.MODIFY_DATA,
                reasoning="Resolve pending edit clarification (modify).",
            )

    if expense_intent in ("clarify_modify", "clarify_delete"):
        return _decision(
            MessageIntentKind.CLARIFICATION_NEEDED,
            reasoning=understanding.reasoning or "Expense edit clarification.",
        )

    if pq and understanding.answers_pending_field is True and understanding.field_updates:
        return PendingQuestionDecision(
            kind=MessageIntentKind.ANSWER_PENDING,
            confidence=max(conf, 0.85),
            reasoning=understanding.reasoning or f"Answer pending field '{pq.field}'.",
            source=src,
            blocks_new_workflow=True,
            field_value=message.strip(),
            target_workflow=pq.workflow_id or (aw.id if aw else None),
            extracted_entities={"interrupts_pending": False},
        )

    return None


def _expense_pending_slot_answer(
    message: str,
    memory: SessionMemory,
    understanding: UnderstandingResult,
) -> bool:
    """True when user is answering an expense pending slot — must not switch to leave."""
    from chat.services.platform.intent_rules import is_resume_workflow_request, is_switch_request
    from chat.services.platform.field_extractors.expense import (
        is_expense_draft_mutation_message,
        is_expense_pending_field_value_answer,
        resolve_pending_expense_edit_turn,
    )

    pq = memory.pending_question
    aw = memory.active_workflow
    if not pq or pq.workflow_id != "expense":
        return False
    if not aw or aw.id != "expense":
        return False
    if is_expense_draft_mutation_message(message, memory):
        return False
    low = (message or "").lower()
    if is_resume_workflow_request(message, workflow_id="leave") or (
        is_switch_request(message, active_workflow_id="expense") and "leave" in low
    ):
        return False
    if resolve_pending_expense_edit_turn(message, memory):
        return False
    pending_edit = (memory.last_entities or {}).get("expense_pending_edit")
    if pending_edit:
        return False
    last_intent = str((memory.last_entities or {}).get("expense_intent") or "")
    if last_intent == "clarify_delete" and re.search(r"\d", message or ""):
        return False
    if understanding.answers_pending_field is False:
        return False
    expense_intent = str((understanding.entities or {}).get("expense_intent") or "").lower()
    if expense_intent in (
        "update",
        "modify_review",
        "correct",
        "fix_mistake",
        "delete",
        "clarify_delete",
        "show_summary",
        "show_list",
        "show_total",
    ):
        return False
    if understanding.action in (
        UnderstandingAction.MODIFY.value,
        UnderstandingAction.DELETE.value,
        UnderstandingAction.REVIEW.value,
    ):
        return False
    if is_expense_pending_field_value_answer(message, memory):
        return True
    if understanding.answers_pending_field is True:
        return True
    if any(u.field == "items" for u in (understanding.field_updates or [])):
        return is_expense_pending_field_value_answer(message, memory)
    intent = str((understanding.entities or {}).get("expense_intent") or "")
    if intent in ("answer_pending", "add", "update", "correct"):
        return is_expense_pending_field_value_answer(message, memory)
    if understanding.action in (
        UnderstandingAction.COLLECT.value,
        UnderstandingAction.CONFIRM.value,
    ):
        return is_expense_pending_field_value_answer(message, memory)
    return False

_PLAN_OP_EXECUTION_ROUTES: dict[str, str] = {
    "reply_policy": "policy",
    "reply_status": "status",
    "reply_oos": "out_of_scope",
    "reply_greeting": "greeting",
    "reply_conversational": "conversational_fallback",
    "reply_platform_clarify": "platform_clarify",
    "reply_general_help": "general_help",
    "reply_today_date": "today_date",
    "reply_translation": "translation",
}


def informational_priority_decision(
    message: str,
    *,
    memory: SessionMemory,
    conversation_history: list[str] | None = None,
    include_policy_status: bool = True,
    understanding: UnderstandingResult | None = None,
) -> PendingQuestionDecision | None:
    """Message-level informational signals that win over workflow slot interpretation."""
    raw = (message or "").strip()
    if not raw:
        return None

    pq = memory.pending_question
    blocks = bool(pq)

    from chat.services.platform.intent_rules import is_greeting_or_chitchat

    if understanding is not None and understanding.is_greeting:
        return PendingQuestionDecision(
            kind=MessageIntentKind.NEW_WORKFLOW,
            confidence=0.95,
            reasoning=understanding.reasoning or "Greeting — conversational reply.",
            source=understanding.source or "rules",
            blocks_new_workflow=False,
        )
    if is_greeting_or_chitchat(raw):
        return PendingQuestionDecision(
            kind=MessageIntentKind.NEW_WORKFLOW,
            confidence=0.95,
            reasoning="Greeting — conversational reply.",
            source="rules",
            blocks_new_workflow=False,
        )

    if is_hr_today_date_query(raw):
        return PendingQuestionDecision(
            kind=MessageIntentKind.ASK_TODAY_DATE,
            confidence=1.0,
            reasoning="Today's calendar date.",
            source="rules",
            blocks_new_workflow=False,
        )

    from chat.services.platform.workflow_cancel import resolve_workflow_cancel_target
    from chat.services.platform.workflow_show import session_has_workflow_context

    if session_has_workflow_context(memory):
        cancel_wf = resolve_workflow_cancel_target(
            raw,
            memory,
            conversation_history=conversation_history,
        )
        if cancel_wf in ("leave", "expense"):
            return PendingQuestionDecision(
                kind=MessageIntentKind.CANCEL_WORKFLOW,
                confidence=0.92,
                reasoning=f"Cancel {cancel_wf} draft (session-aware LLM routing).",
                source="llm_workflow_cancel",
                blocks_new_workflow=True,
                target_workflow=cancel_wf,
            )

    aw = memory.active_workflow
    from chat.services.platform.intent_rules import (
        expense_navigation_kind,
        should_resume_suspended_expense,
    )

    if should_resume_suspended_expense(
        message=raw,
        active_workflow_id=aw.id if aw else None,
        suspended_workflows=memory.suspended_workflows,
    ):
        nav = expense_navigation_kind(raw)
        return PendingQuestionDecision(
            kind=MessageIntentKind.SWITCH_WORKFLOW,
            confidence=0.94,
            reasoning="Navigate to suspended expense draft.",
            source="rules",
            blocks_new_workflow=True,
            target_workflow="expense",
            extracted_entities={"expense_navigation": nav},
        )

    from chat.services.platform.workflow_show import (
        _message_might_be_show_request,
        resolve_workflow_show_target,
    )

    if session_has_workflow_context(memory) and _message_might_be_show_request(raw):
        aw = memory.active_workflow
        show_wf = resolve_workflow_show_target(
            raw,
            memory,
            active_workflow_id=aw.id if aw else "",
            conversation_history=conversation_history,
        )
        if show_wf in ("leave", "expense"):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SHOW_REVIEW,
                confidence=0.92,
                reasoning=f"Show {show_wf} summary (session-aware LLM routing).",
                source="llm_workflow_show",
                blocks_new_workflow=True,
                target_workflow=show_wf,
            )

    from chat.services.platform.field_extractors.expense import is_expense_pending_field_value_answer
    from chat.services.platform.intent_rules import is_clearly_off_hr_question, is_off_hr_topic_message

    if pq and pq.workflow_id == "expense" and is_expense_pending_field_value_answer(raw, memory):
        return None

    if is_clearly_off_hr_question(raw) or is_off_hr_topic_message(raw, memory=memory):
        return PendingQuestionDecision(
            kind=MessageIntentKind.OUT_OF_SCOPE,
            confidence=0.93,
            reasoning="General / programming question outside HR assistant scope.",
            source="rules",
            blocks_new_workflow=False,
        )

    aw = memory.active_workflow
    from chat.services.platform.field_extractors.expense import is_expense_review_edit_turn

    if aw and aw.id == "expense" and is_expense_review_edit_turn(raw, memory, understanding):
        return None

    if include_policy_status:
        if is_status_query(raw):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_STATUS,
                confidence=0.9,
                reasoning="Request reference or status lookup phrasing.",
                source="rules",
                blocks_new_workflow=blocks,
            )
        if not is_workflow_application_message(raw) and (
            is_policy_kb_query(raw) or is_rules_query(raw)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_POLICY,
                confidence=0.9,
                reasoning="Policy or rules query detected.",
                source="rules",
                blocks_new_workflow=blocks,
            )

    from chat.services.platform.hr_assistant_scope import resolve_hr_assistant_scope

    scope_oos = resolve_hr_assistant_scope(
        raw,
        memory,
        conversation_history=conversation_history,
    )
    if scope_oos is not None and scope_oos.is_out_of_scope:
        return PendingQuestionDecision(
            kind=MessageIntentKind.OUT_OF_SCOPE,
            confidence=scope_oos.confidence,
            reasoning=scope_oos.reasoning or "Off-topic for HR assistant.",
            source=scope_oos.source or "llm_hr_scope",
            blocks_new_workflow=False,
        )

    translate_to = is_translation_request(raw)
    if translate_to and _assistant_text_for_translation(
        conversation_history or [],
        target_lang=translate_to,
    ):
        return PendingQuestionDecision(
            kind=MessageIntentKind.ASK_TRANSLATION,
            confidence=1.0,
            reasoning="Translated the previous assistant turn.",
            source="rules",
            blocks_new_workflow=False,
            field_value=translate_to,
        )

    if not include_policy_status:
        return None

    aw = memory.active_workflow
    from chat.services.platform.intent_rules import (
        expense_navigation_kind,
        should_resume_suspended_expense,
    )
    from chat.services.platform.field_extractors.expense import is_expense_review_edit_turn

    expense_review_edit = bool(
        aw and aw.id == "expense" and is_expense_review_edit_turn(raw, memory, understanding)
    )

    if understanding is not None:
        goal_low = (understanding.goal or "").lower()
        reasoning_low = (understanding.reasoning or "").lower()
        if understanding.action == UnderstandingAction.DELETE.value or (
            understanding.action == UnderstandingAction.MODIFY.value
            and ("delete" in goal_low or "delete" in reasoning_low)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.DELETE_DATA,
                confidence=max(understanding.confidence, 0.88),
                reasoning=understanding.reasoning or "Delete expense line.",
                source=understanding.source or "understanding",
                blocks_new_workflow=True,
                target_workflow=aw.id if aw else "expense",
            )
        if understanding.action == UnderstandingAction.MODIFY.value:
            from chat.services.platform.field_extractors.expense import (
                expense_turn_has_targeted_patches,
            )
            from chat.services.platform.turn_semantics import cross_workflow_switch_target

            switch_target = cross_workflow_switch_target(understanding, aw.id if aw else None)
            if switch_target and aw:
                return PendingQuestionDecision(
                    kind=MessageIntentKind.SWITCH_WORKFLOW,
                    confidence=max(understanding.confidence, 0.88),
                    reasoning=understanding.reasoning or f"Switch to {switch_target}.",
                    source=understanding.source or "understanding",
                    blocks_new_workflow=True,
                    target_workflow=switch_target,
                    extracted_entities={"interrupts_pending": bool(pq)},
                )

            expense_turn = (understanding.entities or {}).get("expense_turn")
            if (
                expense_review_edit
                or understanding.field_updates
                or expense_turn_has_targeted_patches(expense_turn, memory)
            ):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.MODIFY_DATA,
                    confidence=max(understanding.confidence, 0.88),
                    reasoning=understanding.reasoning or "Modify expense line.",
                    source=understanding.source or "understanding",
                    blocks_new_workflow=True,
                    target_workflow=aw.id if aw else "expense",
                    extracted_entities={"interrupts_pending": bool(pq)},
                )

    if (
        not expense_review_edit
        and aw
        and aw.id == "expense"
        and "leave" in suspended_workflow_ids(memory.suspended_workflows)
        and is_leave_navigation_from_expense(raw)
    ):
        return PendingQuestionDecision(
            kind=MessageIntentKind.SWITCH_WORKFLOW,
            confidence=0.94,
            reasoning="Return to suspended leave draft.",
            source="rules",
            blocks_new_workflow=True,
            target_workflow="leave",
        )

    if aw and not expense_review_edit and is_workflow_show_request(raw, workflow_id=aw.id):
        return PendingQuestionDecision(
            kind=MessageIntentKind.SHOW_REVIEW,
            confidence=0.92,
            reasoning="Show active workflow draft / summary.",
            source="rules",
            blocks_new_workflow=True,
            target_workflow=aw.id,
        )

    return None


def _assistant_text_for_translation(
    context_lines: list[str],
    *,
    target_lang: str,
) -> str | None:
    _ = target_lang
    for line in reversed(context_lines or []):
        if line.startswith("Assistant:"):
            return line[len("Assistant:") :].strip()
    return None


@dataclass
class PendingQuestionDecision:
    kind: MessageIntentKind
    confidence: float
    reasoning: str
    source: str
    blocks_new_workflow: bool
    field_value: str | None = None
    target_workflow: str | None = None
    extracted_entities: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "source": self.source,
            "blocks_new_workflow": self.blocks_new_workflow,
            "field_value": self.field_value,
            "target_workflow": self.target_workflow,
            "extracted_entities": self.extracted_entities,
        }


class PendingQuestionEngine:
    """Classifies each turn before intent detection / workflow selection."""

    @staticmethod
    def _workflow_pipeline():
        from chat.services.platform.pipeline import WorkflowPipeline

        return WorkflowPipeline()

    def classify(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        llm: LLMClient | None = None,
        understanding: UnderstandingResult | None = None,
    ) -> PendingQuestionDecision:
        raw = (message or "").strip()
        if not raw:
            decision = PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.2,
                reasoning="Empty message.",
                source="rules",
                blocks_new_workflow=False,
            )
            self._log(trace_id, raw, memory, decision, understanding=understanding)
            return decision

        from chat.services.platform.intent_rules import is_bare_confirmation

        if memory.pending_confirmation == "submit" and is_bare_confirmation(raw):
            aw = memory.active_workflow
            decision = PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=0.95,
                reasoning="Submit confirmation (yes/ha).",
                source="rules",
                blocks_new_workflow=True,
                target_workflow=aw.id if aw else None,
            )
            self._log(trace_id, raw, memory, decision, understanding=understanding)
            return decision

        if understanding is not None:
            priority = intent_priority_decision(
                raw,
                memory=memory,
                understanding=understanding,
            )
            if priority is not None:
                decision = self._apply_confidence_guard(
                    self._apply_pending_guardrails(
                        raw, memory, priority, understanding=understanding
                    )
                )
                self._log(trace_id, raw, memory, decision, understanding=understanding)
                return decision

            from chat.services.platform.turn_semantics import is_expense_review_request

            if (
                understanding.interrupt_workflow
                and memory.active_workflow
                and understanding.interrupt_workflow != memory.active_workflow.id
                and understanding.confidence >= 0.65
                and not is_expense_review_request(raw, understanding)
            ):
                decision = PendingQuestionDecision(
                    kind=MessageIntentKind.SWITCH_WORKFLOW,
                    confidence=max(understanding.confidence, 0.85),
                    reasoning=understanding.reasoning
                    or f"Switch to {understanding.interrupt_workflow}.",
                    source=understanding.source or "understanding",
                    blocks_new_workflow=True,
                    target_workflow=understanding.interrupt_workflow,
                )
                decision = self._apply_confidence_guard(
                    self._apply_pending_guardrails(
                        raw, memory, decision, understanding=understanding
                    )
                )
                self._log(trace_id, raw, memory, decision, understanding=understanding)
                return decision

        info_decision = informational_priority_decision(
            raw,
            memory=memory,
            conversation_history=conversation_history,
            include_policy_status=True,
            understanding=understanding,
        )
        if info_decision is not None:
            decision = self._apply_confidence_guard(
                self._apply_pending_guardrails(
                    raw, memory, info_decision, understanding=understanding
                )
            )
            self._log(trace_id, raw, memory, decision, understanding=understanding)
            return decision

        if understanding is None:
            raise TypeError(
                "classify() requires a pre-built UnderstandingResult (Phase 3 SSOT)"
            )

        u_decision = self._decide_from_understanding(raw, memory, understanding)
        if u_decision is not None:
            decision = self._apply_confidence_guard(
                self._apply_pending_guardrails(
                    raw, memory, u_decision, understanding=understanding
                )
            )
            self._log(trace_id, raw, memory, decision, understanding=understanding)
            return decision

        if understanding.source == "llm":
            fallback = PendingQuestionDecision(
                kind=MessageIntentKind.CLARIFICATION_NEEDED,
                confidence=understanding.confidence,
                reasoning=understanding.reasoning or "LLM understanding — needs clarification routing.",
                source="llm",
                blocks_new_workflow=bool(memory.pending_question),
            )
        else:
            fallback = self._fallback_decision_from_understanding(raw, memory, understanding)

        decision = self._apply_confidence_guard(
            self._apply_pending_guardrails(
                raw, memory, fallback, understanding=understanding
            )
        )
        self._log(trace_id, raw, memory, decision, understanding=understanding)
        return decision

    @staticmethod
    def _fallback_decision_from_understanding(
        message: str,
        memory: SessionMemory,
        understanding: UnderstandingResult,
    ) -> PendingQuestionDecision:
        """Map unhandled understanding output when rules path did not classify."""
        pq = memory.pending_question
        aw = memory.active_workflow
        src = understanding.source or "understanding"
        conf = understanding.confidence

        if understanding.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
            return PendingQuestionDecision(
                kind=MessageIntentKind.CLARIFICATION_NEEDED,
                confidence=conf,
                reasoning=understanding.reasoning or "Needs clarification.",
                source=src,
                blocks_new_workflow=bool(pq),
            )

        if pq and aw and is_workflow_show_request(message, workflow_id=aw.id):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SHOW_REVIEW,
                confidence=max(conf, 0.9),
                reasoning="Show workflow draft instead of treating as slot answer.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=aw.id,
            )

        if pq and understanding.action in (
            UnderstandingAction.COLLECT.value,
            UnderstandingAction.CONFIRM.value,
        ):
            if understanding.answers_pending_field is False:
                return PendingQuestionDecision(
                    kind=MessageIntentKind.CLARIFICATION_NEEDED,
                    confidence=max(conf, 0.75),
                    reasoning=understanding.reasoning or "Not a pending slot answer.",
                    source=src,
                    blocks_new_workflow=True,
                    target_workflow=pq.workflow_id or (
                        memory.active_workflow.id if memory.active_workflow else None
                    ),
                )
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(conf, 0.58),
                reasoning=understanding.reasoning or f"Pending question on '{pq.field}'.",
                source=src,
                blocks_new_workflow=True,
                field_value=message.strip(),
                target_workflow=pq.workflow_id or (
                    memory.active_workflow.id if memory.active_workflow else None
                ),
            )

        if pq:
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=0.58,
                reasoning=(
                    f"Pending question on '{pq.field}' — defaulting to slot answer "
                    "to avoid misrouting as new workflow."
                ),
                source=src,
                blocks_new_workflow=True,
                field_value=message.strip(),
                target_workflow=pq.workflow_id,
            )

        return PendingQuestionDecision(
            kind=MessageIntentKind.NEW_WORKFLOW,
            confidence=0.45,
            reasoning=understanding.reasoning or "No pending question; defer to workflow path.",
            source=src,
            blocks_new_workflow=False,
            target_workflow=understanding.workflow if understanding.workflow not in ("none", "") else None,
        )

    @staticmethod
    def _apply_confidence_guard(decision: PendingQuestionDecision) -> PendingQuestionDecision:
        """Block irreversible workflow routing when confidence is low."""
        if decision.kind in (
            MessageIntentKind.ASK_POLICY,
            MessageIntentKind.ASK_STATUS,
            MessageIntentKind.ASK_TODAY_DATE,
            MessageIntentKind.ASK_TRANSLATION,
            MessageIntentKind.OUT_OF_SCOPE,
            MessageIntentKind.ANSWER_PENDING,
            MessageIntentKind.CLARIFICATION_NEEDED,
            MessageIntentKind.SHOW_REVIEW,
            MessageIntentKind.NEW_WORKFLOW,
        ):
            return decision
        if decision.confidence >= 0.70:
            return decision
        if decision.kind in (
            MessageIntentKind.MODIFY_DATA,
            MessageIntentKind.DELETE_DATA,
            MessageIntentKind.SWITCH_WORKFLOW,
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.CLARIFICATION_NEEDED,
                confidence=decision.confidence,
                reasoning=decision.reasoning or "Low confidence — clarification required.",
                source=decision.source,
                blocks_new_workflow=True,
                target_workflow=decision.target_workflow,
            )
        return decision

    def _log(
        self,
        trace_id: str,
        message: str,
        memory: SessionMemory,
        decision: PendingQuestionDecision,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> None:
        pq = memory.pending_question
        payload: dict[str, Any] = {
            "user_message": message,
            "decision": decision.to_log_dict(),
            "active_workflow": (
                memory.active_workflow.to_dict() if memory.active_workflow else None
            ),
            "pending_question": pq.to_dict() if pq else None,
            "has_draft": memory.active_draft() is not None,
        }
        if understanding is not None:
            payload["turn_understanding"] = understanding.to_dict()
        log_step(trace_id, "decision_core", payload)
        log_step(trace_id, "pending_question_engine", payload)

    def decide_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        understanding: UnderstandingResult,
        session,
        company_id: str,
        employee_id: str,
        document_text: str | None,
        idempotency_key: str,
        turn_context,
        finalize: Callable[..., dict[str, Any]],
        run_policy_rag: Callable[..., tuple[Any, ...]],
    ) -> tuple[PendingQuestionDecision, dict[str, Any] | None]:
        """Deprecated — use ``decide_and_execute_turn()`` (Phase 10)."""
        import warnings

        warnings.warn(
            "PendingQuestionEngine.decide_turn() is deprecated; use decide_and_execute_turn().",
            DeprecationWarning,
            stacklevel=2,
        )
        _ = (document_text, run_policy_rag)
        decision = self.classify(
            message,
            memory=memory,
            conversation_history=conversation_history,
            trace_id=trace_id,
            understanding=understanding,
        )
        result = self._execute_planned_turn(
            message,
            memory=memory,
            understanding=understanding,
            pq_decision=decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=turn_context,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session.session_id,
            idempotency_key=idempotency_key,
            workflow_pipeline=self._workflow_pipeline(),
        )
        if result is None:
            return decision, None
        msg, plan_envelope = result
        routed = self._chat_envelope_from_plan(
            msg,
            plan_envelope,
            pq_decision=decision,
            understanding=understanding,
            memory=memory,
        )
        return decision, finalize(session, message, msg, trace_id, routed)

    @staticmethod
    def detect_plan_shortcut(
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
    ) -> PendingQuestionDecision | None:
        """Turns that skip Understanding — greeting, today date, translation follow-ups."""
        from chat.services.platform.intent_rules import is_greeting_or_chitchat

        raw = (message or "").strip()
        if is_greeting_or_chitchat(raw):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.95,
                reasoning="Greeting — conversational reply.",
                source="rules",
                blocks_new_workflow=False,
            )
        return informational_priority_decision(
            message,
            memory=memory,
            conversation_history=conversation_history,
            include_policy_status=False,
        )

    @staticmethod
    def synthetic_understanding_for_shortcut(pq: PendingQuestionDecision) -> UnderstandingResult:
        entities: dict[str, Any] = {}
        if pq.kind == MessageIntentKind.ASK_TRANSLATION and pq.field_value:
            entities["translation_target_lang"] = pq.field_value
        if pq.kind == MessageIntentKind.ASK_TODAY_DATE:
            entities["calendar_date_query"] = True
        if pq.kind == MessageIntentKind.CANCEL_WORKFLOW and pq.target_workflow:
            entities["cancel_workflow_target"] = pq.target_workflow
        if pq.kind == MessageIntentKind.SHOW_REVIEW and pq.target_workflow:
            entities["show_workflow_target"] = pq.target_workflow
        is_greeting = (
            pq.kind == MessageIntentKind.NEW_WORKFLOW
            and "greeting" in (pq.reasoning or "").lower()
        )
        if pq.kind == MessageIntentKind.CANCEL_WORKFLOW:
            wf = (pq.target_workflow or "leave").strip().lower()
            return UnderstandingResult(
                goal="Cancel workflow",
                workflow=wf,
                action=UnderstandingAction.CANCEL.value,
                confidence=pq.confidence,
                reasoning=pq.reasoning,
                source="plan_shortcut",
                entities=entities,
            )
        if pq.kind == MessageIntentKind.SHOW_REVIEW:
            wf = (pq.target_workflow or "leave").strip().lower()
            return UnderstandingResult(
                goal="Show summary",
                workflow=wf,
                action=UnderstandingAction.REVIEW.value,
                confidence=pq.confidence,
                reasoning=pq.reasoning,
                source="plan_shortcut",
                entities=entities,
            )
        if pq.kind == MessageIntentKind.SWITCH_WORKFLOW:
            wf = (pq.target_workflow or "expense").strip().lower()
            nav = str((pq.extracted_entities or {}).get("expense_navigation") or "").strip().lower()
            if nav:
                entities["expense_navigation"] = nav
            action = UnderstandingAction.REVIEW.value if nav == "summary" else UnderstandingAction.COLLECT.value
            return UnderstandingResult(
                goal=f"Switch to {wf}",
                workflow=wf,
                action=action,
                confidence=pq.confidence,
                reasoning=pq.reasoning,
                source="plan_shortcut",
                entities=entities,
                interrupt_workflow=wf,
            )
        if pq.kind == MessageIntentKind.OUT_OF_SCOPE:
            return UnderstandingResult(
                goal="Out of scope",
                workflow="none",
                action=UnderstandingAction.NONE.value,
                confidence=pq.confidence,
                reasoning=pq.reasoning or "Off-topic for HR assistant.",
                source=pq.source or "llm_hr_scope",
                is_out_of_scope=True,
                entities=entities,
            )
        return UnderstandingResult(
            goal="Greeting" if is_greeting else "",
            workflow="none",
            action=UnderstandingAction.NONE.value if is_greeting else UnderstandingAction.QUERY.value,
            confidence=pq.confidence,
            reasoning=pq.reasoning,
            source="plan_shortcut",
            entities=entities,
            is_greeting=is_greeting,
        )

    def decide_and_execute_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        understanding: UnderstandingResult,
        turn_context,
        session,
        company_id: str,
        employee_id: str,
        document_text: str | None,
        idempotency_key: str,
        orchestrator: Any,
        pq_override: PendingQuestionDecision | None = None,
        pre_patches: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Phase 4 — classify then PlanBuilder → execute_workflow_turn (single path)."""
        _ = document_text
        log_turn_context_layer(trace_id, "decision", turn_context)

        pq_decision = pq_override or self.classify(
            message,
            memory=memory,
            conversation_history=conversation_history,
            trace_id=trace_id,
            understanding=understanding,
        )

        result = self._execute_planned_turn(
            message,
            memory=memory,
            understanding=understanding,
            pq_decision=pq_decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=turn_context,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session.session_id,
            idempotency_key=idempotency_key,
            workflow_pipeline=orchestrator.workflow_pipeline,
            pre_patches=pre_patches,
        )
        if result is None:
            raise RuntimeError("execute_workflow_turn returned None — informational fallback should prevent this")
        msg, plan_envelope = result
        return self._complete_plan_turn(
            orchestrator,
            session,
            memory,
            message,
            msg,
            plan_envelope,
            trace_id=trace_id,
            pq_decision=pq_decision,
            understanding=understanding,
        )

    @staticmethod
    def _plan_route_params(
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> tuple[PendingQuestionDecision | None, str]:
        if understanding and understanding.is_greeting:
            return pq_decision, "pending"
        if pq_decision.kind in _INFORMATIONAL_PRIORITY_KINDS:
            return pq_decision, "pending"
        if pq_decision.kind in _PLATFORM_PQ_KINDS:
            return pq_decision, "pending"
        if pq_decision.kind == MessageIntentKind.SHOW_REVIEW:
            return pq_decision, "pending"
        if (
            pq_decision.kind == MessageIntentKind.NEW_WORKFLOW
            and "greeting" in (pq_decision.reasoning or "").lower()
        ):
            return pq_decision, "pending"
        if memory.active_workflow:
            return None, "active"
        return pq_decision, "pending"

    @staticmethod
    def _execute_planned_turn(
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision,
        conversation_history: list[str],
        trace_id: str,
        turn_context,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
        workflow_pipeline: Any,
        pre_patches: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        pq_exec, route_source = PendingQuestionEngine._plan_route_params(
            memory, pq_decision, understanding=understanding
        )
        result = workflow_pipeline.execute_workflow_turn(
            message,
            memory=memory,
            understanding=understanding,
            pq_decision=pq_exec,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=turn_context,
            route_source=route_source,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            pre_patches=pre_patches,
        )
        if not result:
            return None
        msg, envelope = result
        envelope["pending_question_decision"] = pq_decision.to_log_dict()
        envelope.setdefault("rules_applied", []).append("PENDING_QUESTION_ENGINE")
        envelope["rules_applied"].append(pq_decision.kind.value.upper())
        return msg, envelope

    def _complete_plan_turn(
        self,
        orchestrator: Any,
        session,
        memory: SessionMemory,
        message: str,
        msg: str,
        plan_envelope: dict[str, Any],
        *,
        trace_id: str,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult,
    ) -> dict[str, Any]:
        execution_plan = plan_envelope.get("execution_plan") or {}
        workflow_id = execution_plan.get("workflow_id", "")
        primary_op = execution_plan.get("primary_op", "")
        chat_envelope = self._chat_envelope_from_plan(
            msg,
            plan_envelope,
            pq_decision=pq_decision,
            understanding=understanding,
            memory=memory,
        )

        if workflow_id == "informational":
            execution_route = _PLAN_OP_EXECUTION_ROUTES.get(primary_op, "planned_informational")
            self._log_authoritative_decision(
                trace_id,
                pq_decision,
                execution_route=execution_route,
                workflow_route_source="",
            )
            return orchestrator._complete_turn(
                session,
                memory,
                message,
                msg,
                trace_id,
                chat_envelope,
            )

        self._log_authoritative_decision(
            trace_id,
            pq_decision,
            execution_route="workflow",
            workflow_route_source=self._workflow_route_source(memory),
        )
        return orchestrator._complete_workflow_turn(
            session,
            memory,
            message,
            msg,
            chat_envelope["decision"],
            trace_id,
            understanding,
        )

    @staticmethod
    def _chat_envelope_from_plan(
        msg: str,
        plan_envelope: dict[str, Any],
        *,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult,
        memory: SessionMemory,
    ) -> dict[str, Any]:
        execution_plan = plan_envelope.get("execution_plan") or {}
        primary_op = execution_plan.get("primary_op", "")
        workflow_id = execution_plan.get("workflow_id", "")
        decision = {
            key: value
            for key, value in plan_envelope.items()
            if key not in ("response_status", "execution_plan")
        }
        decision.setdefault("pending_question_decision", pq_decision.to_log_dict())
        resp_status = plan_envelope.get("response_status", "success")
        if plan_envelope.get("outcome") == "ERROR":
            resp_status = "error"
        intent = PendingQuestionEngine._intent_from_plan(primary_op, understanding, memory)
        entities: dict[str, Any] = memory.last_entities if workflow_id != "informational" else {}
        if primary_op == "reply_policy":
            entities = {}
            if memory.last_entities.get("document_text"):
                entities = {
                    "document_text": memory.last_entities["document_text"],
                    "document_read": memory.last_entities.get("document_read", True),
                }
        elif primary_op == "reply_today_date":
            from datetime import date

            entities = {"calendar_date": date.today().isoformat()}
        elif primary_op == "reply_translation" and pq_decision.field_value:
            entities = {"translation_target_lang": pq_decision.field_value}
        return {
            "intent": intent,
            "entities": entities,
            "decision": decision,
            "response": {
                "message": msg,
                "status": resp_status,
                "request_id": plan_envelope.get("request_id", ""),
            },
            "status": "success",
        }

    @staticmethod
    def _workflow_route_source(memory: SessionMemory) -> str:
        return "active" if memory.active_workflow else "pending"

    @staticmethod
    def _intent_from_plan(
        primary_op: str,
        understanding: UnderstandingResult,
        memory: SessionMemory,
    ) -> str:
        if primary_op == "reply_policy":
            return INTENT_HR_POLICY
        if primary_op == "reply_status":
            return INTENT_REQUEST_STATUS
        if primary_op == "reply_today_date":
            return INTENT_HR_POLICY
        if primary_op == "reply_translation":
            return INTENT_HR_POLICY
        if memory.active_workflow:
            return memory.active_workflow.id.upper()
        if understanding.workflow and understanding.workflow not in ("none", ""):
            return understanding.workflow.upper()
        return INTENT_UNKNOWN

    @staticmethod
    def _log_authoritative_decision(
        trace_id: str,
        decision: PendingQuestionDecision,
        *,
        execution_route: str,
        workflow_route_source: str,
    ) -> None:
        log_step(
            trace_id,
            "authoritative_decision",
            {
                "decision": decision.to_log_dict(),
                "execution_route": execution_route,
                "workflow_route_source": workflow_route_source,
                "confidence": round(decision.confidence, 3),
                "source": decision.source,
                "target_workflow": decision.target_workflow,
            },
        )

    @staticmethod
    def _decide_from_understanding(
        message: str,
        memory: SessionMemory,
        understanding: UnderstandingResult,
    ) -> PendingQuestionDecision | None:
        """Map AI Understanding Layer output to a pending-question decision."""
        pq = memory.pending_question
        aw = memory.active_workflow
        src = understanding.source or "understanding"
        conf = understanding.confidence
        from chat.services.platform.field_extractors.expense import is_expense_review_edit_turn

        expense_review_edit = bool(
            aw and aw.id == "expense" and is_expense_review_edit_turn(message, memory, understanding)
        )

        from chat.services.platform.field_extractors.expense import is_expense_pending_field_value_answer
        from chat.services.platform.intent_rules import is_clearly_off_hr_question, is_off_hr_topic_message

        if not (
            pq
            and pq.workflow_id == "expense"
            and is_expense_pending_field_value_answer(message, memory)
        ) and (
            is_clearly_off_hr_question(message)
            or is_off_hr_topic_message(message, memory=memory)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.OUT_OF_SCOPE,
                confidence=max(conf, 0.9),
                reasoning="General / programming question outside HR assistant scope.",
                source="rules",
                blocks_new_workflow=bool(pq),
            )

        if (
            aw
            and aw.id == "expense"
            and conf >= 0.65
            and (
                understanding.workflow == "leave"
                or understanding.interrupt_workflow == "leave"
                or "leave" in (understanding.goal or "").lower()
            )
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Leave intent during expense.",
                source=src,
                blocks_new_workflow=True,
                target_workflow="leave",
            )

        if memory.pending_confirmation == "submit" and understanding.action in (
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SUBMIT.value,
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(conf, 0.9),
                reasoning=understanding.reasoning or "Confirm leave/expense submit.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=aw.id if aw else understanding.workflow,
            )

        from chat.services.platform.intent_rules import (
            is_bare_rejection,
            is_cancel_workflow_message,
            is_workflow_interrupt_expense,
        )

        if memory.pending_confirmation == "submit" and is_bare_rejection(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.CLARIFICATION_NEEDED,
                confidence=0.9,
                reasoning="User declined submit — return to review.",
                source="rules",
                blocks_new_workflow=True,
                target_workflow=aw.id if aw else None,
            )

        if (
            aw
            and aw.id == "leave"
            and memory.pending_confirmation == "submit"
            and (
                is_workflow_interrupt_expense(message, active_workflow="leave")
                or understanding.interrupt_workflow == "expense"
                or (understanding.is_expense_intent() and conf >= 0.65)
            )
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.9),
                reasoning=understanding.reasoning or "Expense interrupt during leave submit review.",
                source=src,
                blocks_new_workflow=True,
                target_workflow="expense",
            )

        if aw and (
            understanding.action == UnderstandingAction.CANCEL.value
            or is_cancel_workflow_message(message, workflow_id=aw.id)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.CANCEL_WORKFLOW,
                confidence=max(conf, 0.9),
                reasoning=understanding.reasoning or "Cancel active workflow.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=aw.id,
            )

        cancel_target = str((understanding.entities or {}).get("cancel_workflow_target") or "").lower()
        if not cancel_target and not aw:
            from chat.services.platform.workflow_cancel import resolve_workflow_cancel_target

            cancel_target = (
                resolve_workflow_cancel_target(
                    message,
                    memory,
                    conversation_history=None,
                )
                or ""
            )
        if cancel_target in ("leave", "expense"):
            return PendingQuestionDecision(
                kind=MessageIntentKind.CANCEL_WORKFLOW,
                confidence=max(conf, 0.92),
                reasoning=understanding.reasoning or f"Cancel {cancel_target} workflow.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=cancel_target,
            )

        if understanding.is_greeting:
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.92,
                reasoning=understanding.reasoning or "Greeting — conversational reply.",
                source=src,
                blocks_new_workflow=False,
            )

        if understanding.action == UnderstandingAction.STATUS.value:
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_STATUS,
                confidence=max(conf, 0.9),
                reasoning=understanding.reasoning or "Request status (AI understanding).",
                source=src,
                blocks_new_workflow=bool(pq),
            )

        if understanding.action == UnderstandingAction.SWITCH.value:
            target = understanding.interrupt_workflow or understanding.workflow
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.82),
                reasoning=understanding.reasoning or f"Switch to {target}.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=target if target not in ("none", "") else None,
            )

        show_target = str((understanding.entities or {}).get("show_workflow_target") or "").lower()
        if not show_target:
            from chat.services.platform.workflow_show import resolve_workflow_show_target

            show_target = (
                resolve_workflow_show_target(
                    message,
                    memory,
                    active_workflow_id=aw.id if aw else "",
                )
                or ""
            )
        if show_target in ("leave", "expense") or understanding.action == UnderstandingAction.REVIEW.value:
            wf = show_target or understanding.workflow or (aw.id if aw else None)
            return PendingQuestionDecision(
                kind=MessageIntentKind.SHOW_REVIEW,
                confidence=max(conf, 0.86),
                reasoning=understanding.reasoning or "Summary/review (AI understanding).",
                source=src,
                blocks_new_workflow=True,
                target_workflow=wf if wf not in ("none", "") else None,
            )

        if understanding.is_out_of_scope or is_off_hr_topic_message(message, memory=memory):
            return PendingQuestionDecision(
                kind=MessageIntentKind.OUT_OF_SCOPE,
                confidence=max(conf, 0.88),
                reasoning=understanding.reasoning or "Out of scope (AI understanding).",
                source=src,
                blocks_new_workflow=bool(pq),
            )

        if pq and understanding.answers_pending_field is False:
            from chat.services.platform.field_extractors.expense import (
                expense_turn_has_targeted_patches,
            )

            leave_intent = str((understanding.entities or {}).get("leave_intent") or "").lower()
            if (
                leave_intent == "correct_field"
                and understanding.field_updates
                and aw
                and aw.id == "leave"
            ):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.MODIFY_DATA,
                    confidence=max(conf, 0.88),
                    reasoning=understanding.reasoning or "Correct prior leave field during collect.",
                    source=src,
                    blocks_new_workflow=True,
                    target_workflow="leave",
                )

            expense_turn = (understanding.entities or {}).get("expense_turn")
            if expense_turn_has_targeted_patches(expense_turn, memory):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.MODIFY_DATA,
                    confidence=max(conf, 0.88),
                    reasoning=understanding.reasoning or "Correct existing expense line.",
                    source=src,
                    blocks_new_workflow=True,
                    target_workflow=aw.id if aw else "expense",
                )
            if is_workflow_show_request(
                message,
                workflow_id=(aw.id if aw else pq.workflow_id),
            ) or understanding.action == UnderstandingAction.REVIEW.value:
                return PendingQuestionDecision(
                    kind=MessageIntentKind.SHOW_REVIEW,
                    confidence=max(conf, 0.88),
                    reasoning=understanding.reasoning or "Navigation — show draft, not slot answer.",
                    source=src,
                    blocks_new_workflow=True,
                    target_workflow=aw.id if aw else pq.workflow_id,
                )
            if (
                understanding.entities.get("meta_complaint")
                or understanding.entities.get("process_question")
                or is_workflow_meta_complaint(message)
                or is_process_question(message)
                or understanding.action == UnderstandingAction.CLARIFICATION_NEEDED.value
            ):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.CLARIFICATION_NEEDED,
                    confidence=max(conf, 0.8),
                    reasoning=understanding.reasoning or "Meta / contextual clarification.",
                    source=src,
                    blocks_new_workflow=True,
                    target_workflow=aw.id if aw else pq.workflow_id,
                )

        if understanding.action == UnderstandingAction.QUERY.value or (
            understanding.workflow in ("none", "") and "policy" in (understanding.goal or "").lower()
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_POLICY,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Policy query (AI understanding).",
                source=src,
                blocks_new_workflow=bool(pq),
            )

        if understanding.action == UnderstandingAction.MODIFY.value:
            goal_low = (understanding.goal or "").lower()
            reasoning_low = (understanding.reasoning or "").lower()
            if "delete" in goal_low or "delete" in reasoning_low:
                return PendingQuestionDecision(
                    kind=MessageIntentKind.DELETE_DATA,
                    confidence=max(conf, 0.88),
                    reasoning=understanding.reasoning or "Delete (AI understanding).",
                    source=src,
                    blocks_new_workflow=True,
                )
            return PendingQuestionDecision(
                kind=MessageIntentKind.MODIFY_DATA,
                confidence=conf,
                reasoning=understanding.reasoning or "Modify (AI understanding).",
                source=src,
                blocks_new_workflow=True,
            )

        if understanding.action == UnderstandingAction.DELETE.value:
            return PendingQuestionDecision(
                kind=MessageIntentKind.DELETE_DATA,
                confidence=conf,
                reasoning=understanding.reasoning or "Delete (AI understanding).",
                source=src,
                blocks_new_workflow=True,
            )

        from chat.services.platform.field_extractors.expense import (
            resolve_pending_expense_edit_turn,
        )

        resolved_edit = resolve_pending_expense_edit_turn(message, memory)
        if resolved_edit:
            resolved_intent = str(resolved_edit.get("intent") or "").lower()
            if resolved_intent == "delete":
                return PendingQuestionDecision(
                    kind=MessageIntentKind.DELETE_DATA,
                    confidence=max(conf, 0.88),
                    reasoning="Delete confirmation after clarify.",
                    source=src,
                    blocks_new_workflow=True,
                )
            if resolved_intent in ("modify_review", "update"):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.MODIFY_DATA,
                    confidence=max(conf, 0.88),
                    reasoning="Modify confirmation after clarify.",
                    source=src,
                    blocks_new_workflow=True,
                )

        if _expense_pending_slot_answer(message, memory, understanding):
            pq = memory.pending_question
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or f"Answer pending expense field '{pq.field if pq else 'field'}'.",
                source=src,
                blocks_new_workflow=True,
                field_value=message.strip(),
                target_workflow="expense",
            )

        if (
            not expense_review_edit
            and aw
            and understanding.interrupt_workflow
            and understanding.interrupt_workflow != aw.id
            and conf >= 0.65
            and not _expense_pending_slot_answer(message, memory, understanding)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or f"Switch to {understanding.interrupt_workflow}.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=understanding.interrupt_workflow,
            )

        if (
            pq
            and aw
            and aw.id == "leave"
            and understanding.is_expense_intent()
            and conf >= 0.7
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Expense intent during leave (AI understanding).",
                source=src,
                blocks_new_workflow=True,
                target_workflow="expense",
            )

        if (
            pq
            and understanding.is_expense_intent()
            and understanding.interrupts_active_workflow(aw.id if aw else None)
            and conf >= 0.7
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=conf,
                reasoning=understanding.reasoning or "Cross-workflow expense intent.",
                source=src,
                blocks_new_workflow=True,
                target_workflow="expense",
            )

        if not expense_review_edit and aw and aw.id == "expense" and is_leave_navigation_from_expense(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.9),
                reasoning=understanding.reasoning or "Return to leave from expense.",
                source="rules",
                blocks_new_workflow=True,
                target_workflow="leave",
            )

        if (
            not expense_review_edit
            and aw
            and understanding.is_leave_intent()
            and aw.id == "expense"
            and conf >= 0.7
            and not _expense_pending_slot_answer(message, memory, understanding)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Leave intent during expense.",
                source=src,
                blocks_new_workflow=True,
                target_workflow="leave",
            )

        if pq and understanding.action in (
            UnderstandingAction.COLLECT.value,
            UnderstandingAction.CONFIRM.value,
        ):
            if not expense_review_edit and aw and aw.id == "expense" and is_leave_navigation_from_expense(message):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.SWITCH_WORKFLOW,
                    confidence=max(conf, 0.9),
                    reasoning="Leave navigation — not answering expense pending slot.",
                    source="rules",
                    blocks_new_workflow=True,
                    target_workflow="leave",
                )
            if understanding.answers_pending_field is False:
                return PendingQuestionDecision(
                    kind=MessageIntentKind.CLARIFICATION_NEEDED,
                    confidence=max(conf, 0.78),
                    reasoning=understanding.reasoning or "LLM: not answering pending slot.",
                    source=src,
                    blocks_new_workflow=True,
                    target_workflow=pq.workflow_id or (aw.id if aw else None),
                )
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(conf, 0.72),
                reasoning=understanding.reasoning or f"Answer pending field '{pq.field}'.",
                source=src,
                blocks_new_workflow=True,
                field_value=message.strip(),
                target_workflow=pq.workflow_id or (aw.id if aw else None),
            )

        if understanding.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
        ) and understanding.workflow in ("leave", "expense"):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=max(conf, 0.8),
                reasoning=understanding.reasoning or f"Start/collect {understanding.workflow}.",
                source=src,
                blocks_new_workflow=bool(pq),
                target_workflow=understanding.workflow,
            )

        return None

    def _apply_pending_guardrails(
        self,
        message: str,
        memory: SessionMemory,
        decision: PendingQuestionDecision,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> PendingQuestionDecision:
        pq = memory.pending_question

        if decision.kind == MessageIntentKind.CANCEL_WORKFLOW:
            return decision

        strong_new = (
            decision.kind not in _INTERRUPT_KINDS
            and (
                is_strong_new_workflow_message(message)
                or (
                    understanding is not None
                    and understanding.action == UnderstandingAction.START.value
                    and understanding.confidence >= 0.84
                    and understanding.workflow not in ("none", "")
                )
            )
        )
        if strong_new:
            target = (
                understanding.workflow
                if understanding and understanding.workflow not in ("none", "")
                else infer_new_workflow_target(message)
            )
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=max(decision.confidence, 0.84),
                reasoning=(
                    "Explicit new leave/expense/WFH phrasing — overrides pending slot interpretation."
                    if pq
                    else "Explicit new workflow phrasing detected."
                ),
                source=decision.source,
                blocks_new_workflow=False,
                target_workflow=target,
            )

        if not pq:
            decision.blocks_new_workflow = decision.kind in _INTERRUPT_KINDS
            return decision

        if decision.kind == MessageIntentKind.NEW_WORKFLOW:
            decision.blocks_new_workflow = False
            return decision

        if decision.kind in _INTERRUPT_KINDS or decision.kind in (
            MessageIntentKind.ANSWER_PENDING,
            MessageIntentKind.SHOW_REVIEW,
        ):
            decision.blocks_new_workflow = True
            return decision

        if decision.confidence < 0.85:
            if is_workflow_show_request(
                message,
                workflow_id=(memory.active_workflow.id if memory.active_workflow else None),
            ):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.SHOW_REVIEW,
                    confidence=0.88,
                    reasoning="Workflow show/navigation overrides low-confidence slot routing.",
                    source=decision.source,
                    blocks_new_workflow=True,
                    target_workflow=memory.active_workflow.id if memory.active_workflow else pq.workflow_id,
                )
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(decision.confidence, 0.62),
                reasoning=(
                    f"Pending question on '{pq.field}' — message treated as slot answer "
                    f"instead of new workflow. Original: {decision.reasoning}"
                ),
                source=decision.source,
                blocks_new_workflow=True,
                field_value=decision.field_value or message.strip(),
                target_workflow=(
                    memory.active_workflow.id if memory.active_workflow else pq.workflow_id
                ),
            )

        decision.blocks_new_workflow = True
        return decision


def workflow_continuation_hint(memory: SessionMemory) -> str:
    """Deprecated shim — use ResponseComposer.workflow_continuation_hint (Phase 9)."""
    from chat.services.platform.response_composer import ResponseComposer

    return ResponseComposer().workflow_continuation_hint(memory)


# Phase 4 alias — single Decision Core entry point.
DecisionCore = PendingQuestionEngine
