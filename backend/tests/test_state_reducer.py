"""Phase 6 — state reducer tests."""

from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
    apply_state_patches,
    reduce_arm_duplicate_leave,
    reduce_cancel_active_workflow,
    reduce_clear_pending_confirmation,
    reduce_clear_pending_question,
    reduce_record_submitted_leave_range,
    reduce_set_pending_confirmation,
    reduce_set_pending_question,
    reduce_start_workflow,
    reduce_suspend_active_workflow,
)


def test_reduce_pending_question_roundtrip():
    memory = SessionMemory()
    pq = PendingQuestion(field="end_date", prompt="End?", workflow_id="leave", asked_at_turn=1)
    reduce_set_pending_question(memory, pq)
    assert memory.pending_question is not None
    assert memory.pending_question.field == "end_date"
    reduce_clear_pending_question(memory)
    assert memory.pending_question is None


def test_reduce_start_workflow_suspends_previous():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    draft = reduce_start_workflow(memory, "leave")
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "leave"
    assert draft.workflow_id == "leave"
    assert len(memory.suspended_workflows) == 1
    assert memory.suspended_workflows[0].workflow_id == "expense"
    assert memory.suspended_workflows[0].draft_id == "expense"
    assert memory.workflow_drafts["expense"].workflow_id == "expense"


def test_reduce_submit_confirmation_flow():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={"leave_type": "annual"})},
        pending_question=PendingQuestion(field="x", prompt="x", workflow_id="leave"),
    )
    reduce_set_pending_confirmation(memory, "submit")
    assert memory.pending_confirmation == "submit"
    reduce_clear_pending_confirmation(memory)
    assert memory.pending_confirmation is None


def test_reduce_record_submitted_leave_range():
    memory = SessionMemory()
    reduce_record_submitted_leave_range(
        memory,
        {"start_date": "2026-06-15", "end_date": "2026-06-16"},
        request_id="LV-1",
    )
    rows = memory.conversation_facts.get("submitted_leave_ranges") or []
    assert len(rows) == 1
    assert rows[0]["request_id"] == "LV-1"


def test_reduce_arm_duplicate_leave():
    memory = SessionMemory()
    reduce_arm_duplicate_leave(memory, {"duplicate_leave_updates": [{"field": "start_date", "value": "2026-07-01"}]})
    assert memory.pending_confirmation == "duplicate_leave"
    assert memory.last_entities["duplicate_leave_updates"][0]["field"] == "start_date"


def test_reduce_suspend_active_workflow():
    memory = SessionMemory(active_workflow=ActiveWorkflow(id="leave", stage="collecting"))
    reduce_suspend_active_workflow(memory)
    assert memory.active_workflow is None
    assert memory.pending_question is None
    assert len(memory.suspended_workflows) == 1


def test_apply_state_patches_start_workflow_and_stage():
    memory = SessionMemory()
    apply_state_patches(
        memory,
        [
            {"op": "start_workflow", "value": "leave"},
            {"op": "set_active_stage", "value": "confirm_submit"},
        ],
    )
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "leave"
    assert memory.active_workflow.stage == "confirm_submit"


def test_apply_state_patches_switch_and_lock():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={"leave_type": "annual"})},
    )
    apply_state_patches(memory, [{"op": "switch_to_workflow", "value": "expense"}])
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"

    apply_state_patches(memory, [{"op": "lock_submitted_draft", "request_id": "LV-99"}])
    draft = memory.active_draft()
    assert draft is not None
    assert draft.locked is True
    assert draft.submitted_request_id == "LV-99"
    assert memory.pending_confirmation is None


def test_apply_state_patches_field_updates():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={})},
    )
    apply_state_patches(
        memory,
        [
            {
                "op": "apply_field_updates",
                "draft_id": "default",
                "updates": [{"field": "leave_type", "value": "sick", "action": "set"}],
                "message": "",
            }
        ],
    )
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields.get("leave_type") == "sick"


def test_apply_state_patches_switch_clears_pending():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={"leave_type": "annual"})},
        pending_question=PendingQuestion(field="end_date", prompt="End?", workflow_id="leave", asked_at_turn=1),
        pending_confirmation="submit",
    )
    apply_state_patches(memory, [{"op": "switch_to_workflow", "value": "expense"}])
    assert memory.pending_question is None
    assert memory.pending_confirmation is None
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"


def test_state_patch_buffer_apply_field_updates_immediate():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={})},
    )
    from chat.services.platform.schemas import FieldUpdate
    from chat.services.session_memory import StatePatchBuffer

    buf = StatePatchBuffer(memory)
    buf.apply_field_updates([FieldUpdate(field="start_date", value="2026-07-01")])
    assert memory.active_draft().fields.get("start_date") == "2026-07-01"
    assert buf.patches == []


def test_apply_state_patches_arm_switch_confirm():
    memory = SessionMemory()
    apply_state_patches(
        memory,
        [
            {
                "op": "arm_switch_confirm",
                "from_workflow": "leave",
                "to_workflow": "expense",
                "pending_message": "lunch 200",
            }
        ],
    )
    assert memory.pending_confirmation == "switch:leave:expense"
    assert memory.last_entities.get("switch_pending_message") == "lunch 200"
    apply_state_patches(memory, [{"op": "clear_switch_confirm"}])
    assert memory.pending_confirmation is None
    assert "switch_pending_message" not in (memory.last_entities or {})


def test_reduce_cancel_active_workflow():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={"leave_type": "annual", "start_date": "2026-08-15"},
            )
        },
        pending_question=PendingQuestion(field="reason", prompt="Reason?", workflow_id="leave", asked_at_turn=1),
        pending_confirmation="submit",
    )
    reduce_cancel_active_workflow(memory)
    assert memory.active_workflow is None
    assert "default" not in memory.workflow_drafts
    assert memory.pending_question is None
    assert memory.pending_confirmation is None
    assert memory.last_action == "workflow_cancelled"
