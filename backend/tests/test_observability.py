"""Phase 10 — full turn observability and replay from logs."""

from __future__ import annotations

import json
import logging

import pytest
from chat.services.observability import (
    TURN_TRACE_SCHEMA_VERSION,
    begin_turn_trace,
    classify_leave_field_apply_mode,
    diff_workflow_state,
    finish_turn_trace,
    format_turn_replay,
    log_field_updates_applied,
    patch_turn_trace,
    replay_turn_from_log,
    snapshot_workflow_state,
)
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)


def _memory_with_leave_draft() -> SessionMemory:
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting", draft_id="default")
    memory.workflow_drafts["default"] = WorkflowDraft(
        workflow_id="leave",
        fields={"start_date": "2025-07-10", "leave_type": "annual"},
    )
    memory.pending_question = PendingQuestion(
        field="end_date",
        prompt="End date?",
        workflow_id="leave",
        asked_at_turn=2,
    )
    return memory


def test_snapshot_workflow_state_trims_events():
    memory = _memory_with_leave_draft()
    memory.events = [{"type": f"e{i}"} for i in range(5)]
    snap = snapshot_workflow_state(memory)
    assert snap["events_count"] == 5
    assert len(snap["events_tail"]) == 3
    assert "events" not in snap
    assert "default" in snap["draft_field_keys"]


def test_diff_workflow_state_detects_pending_and_draft_changes():
    before = snapshot_workflow_state(_memory_with_leave_draft())
    after_memory = _memory_with_leave_draft()
    after_memory.pending_question = None
    after_memory.workflow_drafts["default"].fields["end_date"] = "2025-07-12"
    after = snapshot_workflow_state(after_memory)

    delta = diff_workflow_state(before, after)
    assert "pending_question" in delta
    assert "draft_fields" in delta
    assert delta["draft_fields"]["default"]["after"]["end_date"] == "2025-07-12"


def test_turn_trace_lifecycle_emits_turn_complete(caplog):
    caplog.set_level(logging.INFO, logger="hr_chatbot")
    trace_id = "trace-test-1"
    memory = _memory_with_leave_draft()
    state_before = snapshot_workflow_state(memory)

    begin_turn_trace(
        trace_id,
        user_message="12 July",
        state_before=state_before,
        session_id="sess-1",
    )
    patch_turn_trace(
        trace_id,
        context={"active_workflow_id": "leave", "pending_question_field": "end_date"},
        understanding={"workflow": "leave", "action": "collect", "confidence": 0.82},
        pq_decision={"kind": "answer_pending", "confidence": 0.9},
        execution_plan={"workflow_id": "leave", "ops": ["leave_collect"], "reason": "answer pending"},
    )

    memory.workflow_drafts["default"].fields["end_date"] = "2025-07-12"
    memory.pending_question = None
    record = finish_turn_trace(
        trace_id,
        state_after=snapshot_workflow_state(memory),
        assistant_message="End date saved.",
        envelope={
            "intent": "LEAVE",
            "status": "success",
            "decision": {"outcome": "COLLECTING", "rules_applied": ["LEAVE_COLLECT"]},
            "response": {"message": "End date saved.", "status": "success", "request_id": ""},
        },
    )

    assert record is not None
    assert record["turn_trace_schema_version"] == TURN_TRACE_SCHEMA_VERSION
    assert record["state_delta"]["draft_fields"]["default"]["after"]["end_date"] == "2025-07-12"

    complete_logs = [
        json.loads(rec.message.split(" ", 2)[-1])
        for rec in caplog.records
        if "turn_complete" in rec.message
    ]
    assert len(complete_logs) == 1
    assert complete_logs[0]["step"] == "turn_complete"
    assert complete_logs[0]["execution_plan"]["ops"] == ["leave_collect"]


def test_classify_leave_field_apply_mode_submit_vs_modify():
    memory = _memory_with_leave_draft()
    memory.pending_confirmation = "submit"
    assert classify_leave_field_apply_mode("no", memory=memory) == "submit_review"
    assert classify_leave_field_apply_mode("reason ta wedding koro", memory=memory) == "modify"


def test_log_field_updates_applied_records_delta(caplog):
    caplog.set_level(logging.INFO, logger="hr_chatbot")
    log_field_updates_applied(
        "obs-apply-1",
        workflow_id="leave",
        draft_id="default",
        updates=[{"field": "end_date", "value": "2026-07-12"}],
        before_fields={"start_date": "2026-07-10", "end_date": None},
        after_fields={"start_date": "2026-07-10", "end_date": "2026-07-12"},
        apply_mode="collect",
        message="12 July",
    )
    assert any("field_updates_applied" in rec.message for rec in caplog.records)


def test_format_turn_replay_from_failed_scenario():
    record = {
        "turn_trace_schema_version": TURN_TRACE_SCHEMA_VERSION,
        "user_message": "আবার ১০ জুলাই ছুটি চাই",
        "context": {
            "active_workflow_id": "leave",
            "active_workflow_stage": "collecting",
            "pending_question_field": "end_date",
            "pending_confirmation": None,
        },
        "understanding": {
            "workflow": "leave",
            "action": "clarification_needed",
            "confidence": 0.92,
            "source": "rules",
            "reasoning": "An open leave request already exists — submit or cancel it first.",
        },
        "pq_decision": {
            "kind": "clarification_needed",
            "confidence": 0.92,
            "source": "understanding",
            "reasoning": "Duplicate leave blocked.",
        },
        "plan_skipped": True,
        "plan_skip_reason": "needs clarification",
        "state_delta": {"pending_confirmation": {"before": None, "after": "duplicate_leave"}},
        "response": {
            "decision_outcome": "NEEDS_CLARIFICATION",
            "intent": "LEAVE",
            "rules_applied": ["DUPLICATE_LEAVE"],
            "message_preview": "You already have an open leave request.",
        },
        "assistant_message": "You already have an open leave request.",
    }

    replay_text = format_turn_replay(record)
    assert "Duplicate leave blocked" in replay_text
    assert "clarification_needed" in replay_text
    assert "duplicate_leave" in replay_text

    structured = replay_turn_from_log(record)
    assert structured["replay_text"] == replay_text
    assert structured["understanding"]["action"] == "clarification_needed"
