"""Phase 4 — regression lock: reducer reason immunity, observability, transcript guards."""

from __future__ import annotations

import json
import logging

from chat.services.observability import classify_leave_field_apply_mode, log_field_updates_applied
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
    reduce_apply_field_updates,
)


def _review_leave_memory(*, reason: str = "Father unwell; Hospital/treatment visit") -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-06-29",
                    "end_date": "2026-07-02",
                    "reason": reason,
                },
            )
        },
        pending_confirmation="submit",
    )


def test_reducer_blocks_reason_without_modify_message():
    memory = _review_leave_memory()
    reduce_apply_field_updates(
        memory,
        "default",
        [
            {"field": "reason", "value": "overwrite attempt", "action": "set"},
            {"field": "start_date", "value": "2026-08-23", "action": "set"},
        ],
        message="random unclear text",
        trace_id="phase4-reducer-block-reason",
    )
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields.get("reason") == "Father unwell; Hospital/treatment visit"
    assert draft.fields.get("start_date") == "2026-08-23"


def test_reducer_allows_reason_with_explicit_modify():
    from tests.helpers.leave_llm_mock import mock_leave_llm

    memory = _review_leave_memory(reason="family program")
    with mock_leave_llm():
        reduce_apply_field_updates(
            memory,
            "default",
            [{"field": "reason", "value": "family wedding", "action": "set"}],
            message="reason ta family wedding koro",
            trace_id="phase4-reducer-allow-modify",
        )
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields.get("reason") == "family wedding"


def test_classify_leave_field_apply_mode():
    from unittest.mock import patch

    with patch(
        "chat.services.platform.field_extractors.leave._llm_client_configured",
        return_value=False,
    ):
        modify_memory = _review_leave_memory()
        assert classify_leave_field_apply_mode("reason ta family wedding koro", memory=modify_memory) == "legacy_review_fallback"

        collect_memory = SessionMemory(
            active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
            pending_question=PendingQuestion(
                field="end_date",
                prompt="End?",
                workflow_id="leave",
                asked_at_turn=1,
            ),
        )
        assert classify_leave_field_apply_mode("18 july", memory=collect_memory) == "collect_deterministic"

        submit_memory = _review_leave_memory()
        assert classify_leave_field_apply_mode("no", memory=submit_memory) == "legacy_review_fallback"


def test_field_updates_applied_emits_structured_log(caplog):
    caplog.set_level(logging.INFO, logger="hr_chatbot")
    before = {"start_date": "2026-06-29", "reason": "family program"}
    after = {"start_date": "2026-07-03", "reason": "family program"}
    log_field_updates_applied(
        "phase4-log-apply",
        workflow_id="leave",
        draft_id="default",
        updates=[{"field": "start_date", "value": "2026-07-03"}],
        before_fields=before,
        after_fields=after,
        apply_mode="modify",
        message="3 july koro",
    )
    logs = [
        json.loads(rec.message.split(" ", 2)[-1])
        for rec in caplog.records
        if "field_updates_applied" in rec.message
    ]
    assert len(logs) == 1
    payload = logs[0]
    assert payload["step"] == "field_updates_applied"
    assert payload["apply_mode"] == "modify"
    assert payload["fields_changed"]["start_date"]["after"] == "2026-07-03"
    assert "reason" not in payload["fields_changed"]


def test_reducer_logs_on_apply(caplog):
    from unittest.mock import patch

    caplog.set_level(logging.INFO, logger="hr_chatbot")
    memory = _review_leave_memory()
    with patch(
        "chat.services.platform.field_extractors.leave._llm_client_configured",
        return_value=False,
    ):
        reduce_apply_field_updates(
            memory,
            "default",
            [ {"field": "end_date", "value": "2026-07-07", "action": "set"} ],
            message="end date 7 july hobe",
            trace_id="phase4-reducer-log",
        )
    logs = [
        json.loads(rec.message.split(" ", 2)[-1])
        for rec in caplog.records
        if "field_updates_applied" in rec.message
    ]
    assert len(logs) == 1
    assert logs[0]["apply_mode"] == "legacy_review_fallback"
    assert logs[0]["fields_changed"]["end_date"]["after"] == "2026-07-07"
