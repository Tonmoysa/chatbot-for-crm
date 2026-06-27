"""Shared helpers for policy/status interrupts during active workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat.services.session_memory import SessionMemory


def should_pause_workflow_for_informational(memory: "SessionMemory | None") -> bool:
    """Policy/status/date interrupts pause in-progress leave/expense drafts (not submitted)."""
    if not memory or not memory.active_workflow:
        return False
    draft = memory.active_draft()
    if draft and draft.locked:
        return False
    return memory.active_workflow.id in ("leave", "expense")


def is_informational_interrupt_message(message: str) -> bool:
    """Policy/status/today — must not run expense/leave domain LLM."""
    from chat.services.platform.intent_rules import is_status_query, is_workflow_application_message
    from chat.services.policy_intent_helpers import (
        is_hr_today_date_query,
        is_policy_kb_query,
        is_rules_query,
    )

    if is_workflow_application_message(message):
        return False
    return (
        is_policy_kb_query(message)
        or is_rules_query(message)
        or is_status_query(message)
        or is_hr_today_date_query(message)
    )
