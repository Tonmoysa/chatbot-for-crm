"""Submission engine — explicit confirmation required before CRM submit."""

from __future__ import annotations

from typing import Any

from chat.services.crm.base import CRMError
from chat.services.crm.factory import get_crm_adapter
from chat.services.platform.event_store import EventStore
from chat.services.platform.schemas import WorkflowStage
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.session_memory import (
    SessionMemory,
    StatePatchBuffer,
    apply_state_patches,
)


class SubmissionEngine:
    def __init__(
        self,
        validator: ValidationEngine | None = None,
        manager: WorkflowManager | None = None,
        events: EventStore | None = None,
    ) -> None:
        self.validator = validator or ValidationEngine()
        self.manager = manager or WorkflowManager(events)
        self.events = self.manager.events

    def request_submit(
        self,
        memory: SessionMemory,
        definition,
        *,
        state: StatePatchBuffer | None = None,
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft:
            return "No active draft to submit.", {"blocked": True}
        if draft.locked:
            return "This request is already submitted and locked.", {"blocked": True}
        errors = self.validator.validate(draft, definition)
        if errors:
            return errors[0], {"blocked": True, "errors": errors}
        patches = [
            {"op": "set_pending_confirmation", "value": "submit"},
            {"op": "set_active_stage", "value": WorkflowStage.CONFIRM_SUBMIT.value},
        ]
        if state is not None:
            for patch in patches:
                state.push(patch["op"], **{k: v for k, v in patch.items() if k != "op"})
        else:
            apply_state_patches(memory, patches)
        self.events.emit(memory, "submission_requested", definition.workflow_id, {})
        return (
            "Please confirm — reply **yes** to submit this request, or tell me what to change.",
            {"awaiting_confirmation": True},
        )

    def confirm_and_submit(
        self,
        memory: SessionMemory,
        definition,
        *,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str = "",
        state: StatePatchBuffer | None = None,
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft or draft.locked:
            return "Nothing to submit.", {"blocked": True}
        if memory.pending_confirmation != "submit":
            return "No submission awaiting confirmation.", {"blocked": True}
        errors = self.validator.validate(draft, definition)
        if errors:
            if state is not None:
                state.push("clear_pending_confirmation")
            else:
                apply_state_patches(memory, [{"op": "clear_pending_confirmation"}])
            return errors[0], {"blocked": True, "errors": errors}

        crm = get_crm_adapter()
        draft_fields = dict(draft.fields)
        if definition.workflow_id == "leave":
            from chat.services.platform.field_extractors.leave import leave_fields_for_submit

            draft_fields = leave_fields_for_submit(draft_fields)
        if definition.workflow_id == "expense":
            from chat.services.platform.field_extractors.expense import expense_fields_for_submit

            draft_fields = expense_fields_for_submit(draft_fields)
        entities = {
            "workflow_id": definition.workflow_id,
            "fields": draft_fields,
            "line_items": list(draft.line_items or draft.fields.get("items") or []),
        }
        decision = {"outcome": "SUBMITTED", "reason": f"{definition.name} submitted."}
        try:
            created = crm.create_request(
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
                intent=definition.crm_intent,
                entities=entities,
                decision=decision,
                idempotency_key=idempotency_key,
            )
        except CRMError as exc:
            return str(exc), {"error": True}

        request_id = str(created.get("request_id") or "")
        self.manager.lock_submitted(memory, request_id, state=state)
        if definition.workflow_id == "leave":
            from chat.services.platform.field_extractors.leave import leave_fields_for_submit

            leave_patch = {
                "op": "record_submitted_leave_range",
                "fields": leave_fields_for_submit(dict(draft.fields)),
                "request_id": request_id,
            }
            if state is not None:
                state.push(
                    "record_submitted_leave_range",
                    fields=leave_fields_for_submit(dict(draft.fields)),
                    request_id=request_id,
                )
            else:
                apply_state_patches(memory, [leave_patch])
        if definition.workflow_id == "expense":
            from chat.services.platform.field_extractors.expense import expense_fields_for_submit

            expense_patch = {
                "op": "record_submitted_expense",
                "fields": expense_fields_for_submit(dict(draft.fields)),
                "request_id": request_id,
            }
            if state is not None:
                state.push(
                    "record_submitted_expense",
                    fields=expense_fields_for_submit(dict(draft.fields)),
                    request_id=request_id,
                )
            else:
                apply_state_patches(memory, [expense_patch])
        return (
            f"Your **{definition.name}** has been submitted. Reference: **`{request_id}`**.",
            {"request_id": request_id, "submitted": True},
        )
