"""Runtime patches for policy-during-expense routing (disk-safe install via apps.ready)."""

from __future__ import annotations

from typing import Any

from chat.services.pending_question_engine import (
    MessageIntentKind,
    PendingQuestionDecision,
    PendingQuestionEngine,
    informational_priority_decision,
)
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult


def detect_plan_shortcut(
    message: str,
    *,
    memory,
    conversation_history: list[str],
) -> PendingQuestionDecision | None:
    """Policy/status shortcuts must run before expense domain LLM."""
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
        include_policy_status=True,
    )


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
        action = (
            UnderstandingAction.REVIEW.value
            if nav == "summary"
            else UnderstandingAction.COLLECT.value
        )
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
    if pq.kind == MessageIntentKind.ASK_POLICY:
        return UnderstandingResult(
            goal="Policy query",
            workflow="none",
            action=UnderstandingAction.QUERY.value,
            confidence=pq.confidence,
            reasoning=pq.reasoning or "Policy or rules query detected.",
            source="plan_shortcut",
            entities=entities,
        )
    if pq.kind == MessageIntentKind.ASK_STATUS:
        return UnderstandingResult(
            goal="Request status",
            workflow="none",
            action=UnderstandingAction.STATUS.value,
            confidence=pq.confidence,
            reasoning=pq.reasoning or "Request reference or status lookup.",
            source="plan_shortcut",
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


def _patch_expense_interpreter_policy_guard() -> None:
    import chat.services.platform.field_extractors.expense as expense_mod

    original = expense_mod.interpret_expense_draft_turn

    def guarded(message, memory, *, trace_id="", conversation_history=None, fresh_claim=False):
        from chat.services._policy_interrupt import is_informational_interrupt_message
        from chat.services.platform.banglish_normalize import normalize_banglish_message

        raw = normalize_banglish_message((message or "").strip())
        if raw and is_informational_interrupt_message(raw):
            return expense_mod._log_and_return_expense_turn(
                trace_id,
                raw,
                {"intent": "conversation", "item_patches": []},
                llm_used=False,
            )
        return original(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            fresh_claim=fresh_claim,
        )

    expense_mod.interpret_expense_draft_turn = guarded


def _patch_leave_interpreter_policy_guard() -> None:
    import chat.services.platform.field_extractors.leave as leave_mod

    original = leave_mod.resolve_leave_collect_turn

    def guarded(message, memory, *, trace_id="", understanding_updates=None):
        from chat.services._policy_interrupt import is_informational_interrupt_message
        from chat.services.platform.banglish_normalize import normalize_banglish_message

        raw = normalize_banglish_message((message or "").strip())
        if raw and is_informational_interrupt_message(raw):
            return {
                "updates": {},
                "answers_pending_field": False,
                "is_correction": False,
            }
        return original(
            message,
            memory,
            trace_id=trace_id,
            understanding_updates=understanding_updates,
        )

    leave_mod.resolve_leave_collect_turn = guarded


def apply() -> None:
    PendingQuestionEngine.detect_plan_shortcut = staticmethod(detect_plan_shortcut)  # type: ignore[method-assign]
    PendingQuestionEngine.synthetic_understanding_for_shortcut = staticmethod(  # type: ignore[method-assign]
        synthetic_understanding_for_shortcut
    )
    _patch_expense_interpreter_policy_guard()
    _patch_leave_interpreter_policy_guard()
