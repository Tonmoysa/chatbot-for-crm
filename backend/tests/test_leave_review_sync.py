"""Leave review — ack/review sync and complaint handling."""

from __future__ import annotations

from chat.services.platform.field_engine import deserialize_field_updates
from chat.services.platform.field_extractors.leave import (
    is_leave_complaint_reason_value,
    is_leave_review_complaint_or_question,
    review_field_updates_from_message,
)
from chat.services.platform.response_composer import ResponseComposer
from chat.services.platform.schemas import FieldUpdate
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    WorkflowDraft,
    reduce_apply_field_updates,
)


def _review_memory(*, reason: str = "family program") -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-06",
                    "end_date": "2026-08-09",
                    "reason": reason,
                },
            )
        },
        pending_confirmation="submit",
    )


def test_complaint_not_treated_as_reason_update():
    msg = "but ami toh update e 3 din dekhchi nah"
    assert is_leave_review_complaint_or_question(msg)
    assert is_leave_complaint_reason_value(msg)
    memory = _review_memory()
    assert review_field_updates_from_message(msg, memory) == []
    reduce_apply_field_updates(
        memory,
        "default",
        [{"field": "reason", "value": msg, "action": "set"}],
        message=msg,
        review_validated=True,
    )
    assert memory.active_draft().fields.get("reason") == "family program"


def test_prefix_only_reflects_applied_updates():
    memory = _review_memory()
    updates = [
        FieldUpdate(field="reason", value="choto boner biye", action="set"),
    ]
    applied_raw = reduce_apply_field_updates(
        memory,
        "default",
        [{"field": "reason", "value": "choto boner biye", "action": "set"}],
        message="reason choto boner biye hobe",
        review_validated=True,
    )
    applied = deserialize_field_updates(applied_raw)
    prefix = ResponseComposer().item_prefix_from_updates(applied, lang="en")
    assert "choto boner biye" in prefix.lower()
    assert memory.active_draft().fields.get("reason") == "choto boner biye"
    assert "family program" not in prefix.lower()

    # Stale planned update must not appear in prefix when apply blocked
    blocked = deserialize_field_updates(
        reduce_apply_field_updates(
            _review_memory(),
            "default",
            [{"field": "reason", "value": "choto boner biye", "action": "set"}],
            message="but ami toh update e 3 din dekhchi nah",
            review_validated=True,
        )
    )
    assert blocked == []
