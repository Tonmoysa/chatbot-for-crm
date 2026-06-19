"""Submission engine — explicit confirmation required before CRM submit."""

from __future__ import annotations

from typing import Any

from chat.services.crm.base import CRMError
from chat.services.crm.factory import get_crm_adapter
from chat.services.platform.event_store import EventStore
from chat.services.platform.schemas import WorkflowStage
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.session_memory import SessionMemory


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

    def request_submit(self, memory: SessionMemory, definition) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft:
            return "No active draft to submit.", {"blocked": True}
        if draft.locked:
            return "This request is already submitted and locked.", {"blocked": True}
        errors = self.validator.validate(draft, definition)
        if errors:
            return errors[0], {"blocked": True, "errors": errors}
        memory.pending_confirmation = "submit"
        self.manager.set_stage(memory, WorkflowStage.CONFIRM_SUBMIT.value)
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
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft or draft.locked:
            return "Nothing to submit.", {"blocked": True}
        if memory.pending_confirmation != "submit":
            return "No submission awaiting confirmation.", {"blocked": True}
        errors = self.validator.validate(draft, definition)
        if errors:
            memory.pending_confirmation = None
            return errors[0], {"blocked": True, "errors": errors}

        crm = get_crm_adapter()
        entities = {
            "workflow_id": definition.workflow_id,
            "fields": dict(draft.fields),
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
        self.manager.lock_submitted(memory, request_id)
        if definition.workflow_id == "leave":
            from chat.services.platform.field_extractors.leave import record_submitted_leave_range

            record_submitted_leave_range(memory, draft.fields, request_id=request_id)
        return (
            f"Your **{definition.name}** has been submitted. Reference: **`{request_id}`**.",
            {"request_id": request_id, "submitted": True},
        )
