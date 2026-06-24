"""Review engine — generate review and manage review stage."""

from __future__ import annotations

from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.schemas import WorkflowStage
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.session_memory import SessionMemory


class ReviewEngine:
    def __init__(
        self,
        fields: FieldEngine | None = None,
        validator: ValidationEngine | None = None,
        manager: WorkflowManager | None = None,
    ) -> None:
        self.fields = fields or FieldEngine()
        self.validator = validator or ValidationEngine()
        self.manager = manager or WorkflowManager()

    def prepare_review(
        self, memory: SessionMemory, definition, *, lang: str = "en"
    ) -> tuple[str | None, list[str]]:
        draft = memory.active_draft()
        if not draft:
            return None, ["No active draft."]
        errors = self.validator.validate(draft, definition, lang=lang)
        if errors:
            return None, errors
        self.manager.set_stage(memory, WorkflowStage.REVIEW.value)
        self.manager.events.emit(memory, "review_requested", definition.workflow_id, {})
        return self.fields.build_review(draft, definition, lang=lang), []
