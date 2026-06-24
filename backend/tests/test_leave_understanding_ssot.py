"""Step 3 — duplicate leave detection SSOT in field_engine."""

from chat.services.platform.field_engine import (
    duplicate_leave_arm_entities,
    is_duplicate_leave_attempt,
    leave_draft_in_progress,
)
from chat.services.platform.schemas import FieldUpdate, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import SessionMemory, WorkflowDraft


def _draft(**fields) -> WorkflowDraft:
    return WorkflowDraft(workflow_id="leave", fields=fields)


def test_leave_draft_in_progress_requires_meaningful_fields():
    assert not leave_draft_in_progress(_draft())
    assert leave_draft_in_progress(_draft(start_date="2026-08-15"))
    assert leave_draft_in_progress(_draft(leave_type="annual"))


def test_is_duplicate_leave_attempt_different_start_date():
    draft = _draft(start_date="2026-08-15")
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        field_updates=[FieldUpdate(field="start_date", value="2026-08-20")],
    )
    assert is_duplicate_leave_attempt("agami 20 august leave chai", u, draft)


def test_is_duplicate_leave_attempt_leave_chai_with_open_draft():
    draft = _draft(start_date="2026-08-15")
    u = UnderstandingResult(workflow="leave", action=UnderstandingAction.COLLECT.value)
    assert is_duplicate_leave_attempt("leave chai", u, draft)


def test_is_duplicate_leave_attempt_not_leave_message():
    draft = _draft(start_date="2026-08-15")
    u = UnderstandingResult(workflow="leave", action=UnderstandingAction.COLLECT.value)
    assert not is_duplicate_leave_attempt("full day", u, draft)


def test_duplicate_leave_arm_entities_serializes_updates():
    memory = SessionMemory(last_entities={"x": 1})
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        field_updates=[FieldUpdate(field="start_date", value="2026-08-20")],
    )
    entities = duplicate_leave_arm_entities(memory, u)
    assert entities["x"] == 1
    assert entities["duplicate_leave_updates"][0]["field"] == "start_date"
