"""TurnContext builder parity tests (Phase 2)."""

from chat.services.platform.schemas import TURN_CONTEXT_SCHEMA_VERSION
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
    assert_turn_context_parity,
    build_turn_context,
)


def test_build_turn_context_empty_memory():
    memory = SessionMemory()
    ctx = build_turn_context(
        message="leave chai",
        memory=memory,
        conversation_history=["User: hi"],
        trace_id="t1",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    assert_turn_context_parity(ctx, memory)
    assert ctx.context_schema_version == TURN_CONTEXT_SCHEMA_VERSION
    assert ctx.user_message == "leave chai"
    assert ctx.has_active_workflow is False
    assert ctx.wizard_active is False
    assert ctx.turn_count_at_start == 0


def test_build_turn_context_active_leave_with_pending_question():
    memory = SessionMemory(
        turn_count=3,
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={"leave_type": "annual", "start_date": "2026-06-20"},
                version=2,
            )
        },
        pending_question=PendingQuestion(
            field="end_date",
            prompt="When does your leave end?",
            workflow_id="leave",
            asked_at_turn=2,
        ),
        pending_confirmation="submit",
        suspended_workflows=[
            SuspendedWorkflow(workflow_id="expense", stage="collecting", draft_id="default", suspended_at_turn=1)
        ],
        conversation_facts={"submitted_leave_ranges": [{"start": "2026-01-01", "end": "2026-01-02"}]},
    )
    ctx = build_turn_context(
        message="yes",
        memory=memory,
        conversation_history=[],
        trace_id="t2",
        session_id="s2",
        company_id="c2",
        employee_id="e2",
    )
    assert_turn_context_parity(ctx, memory)
    assert ctx.active_workflow_id == "leave"
    assert ctx.pending_question_field == "end_date"
    assert ctx.pending_confirmation == "submit"
    assert ctx.draft_snapshot is not None
    assert ctx.draft_snapshot["fields"]["leave_type"] == "annual"
    assert ctx.draft_snapshot.get("version") == 2
    assert len(ctx.suspended_workflows) == 1
    assert ctx.has_pending_confirmation is True
    assert ctx.wizard_active is True
