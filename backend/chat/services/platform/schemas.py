"""Platform data models — workflow definitions, drafts, events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowStage(str, Enum):
    COLLECTING = "collecting"
    REVIEW = "review"
    CONFIRM_SUBMIT = "confirm_submit"
    SUBMITTED = "submitted"


class UnderstandingAction(str, Enum):
    START = "start"
    COLLECT = "collect"
    MODIFY = "modify"
    DELETE = "delete"
    REVIEW = "review"
    SUBMIT = "submit"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    SWITCH = "switch"
    QUERY = "query"
    CLARIFICATION_NEEDED = "clarification_needed"
    NONE = "none"


@dataclass
class FieldDefinition:
    name: str
    field_type: str = "string"
    required: bool = False
    optional: bool = False
    enum_values: list[str] = field(default_factory=list)
    prompt_en: str = ""
    prompt_bn: str = ""
    conditional: dict[str, Any] = field(default_factory=dict)
    item_fields: list[FieldDefinition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FieldDefinition:
        items = [cls.from_dict(i) for i in (raw.get("item_fields") or [])]
        return cls(
            name=str(raw["name"]),
            field_type=str(raw.get("type") or "string"),
            required=bool(raw.get("required")),
            optional=bool(raw.get("optional")),
            enum_values=list(raw.get("enum") or []),
            prompt_en=str(raw.get("prompt_en") or ""),
            prompt_bn=str(raw.get("prompt_bn") or ""),
            conditional=dict(raw.get("conditional") or {}),
            item_fields=items,
        )


@dataclass
class ValidationRule:
    rule_id: str
    rule_type: str
    params: dict[str, Any] = field(default_factory=dict)
    message_en: str = ""
    message_bn: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ValidationRule:
        return cls(
            rule_id=str(raw.get("id") or raw.get("rule_id") or ""),
            rule_type=str(raw.get("type") or ""),
            params=dict(raw.get("params") or {}),
            message_en=str(raw.get("message_en") or ""),
            message_bn=str(raw.get("message_bn") or ""),
        )


@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    fields: list[FieldDefinition]
    validation_rules: list[ValidationRule] = field(default_factory=list)
    crm_intent: str = ""
    requires_review: bool = True
    requires_confirmation: bool = True

    def field_map(self) -> dict[str, FieldDefinition]:
        return {f.name: f for f in self.fields}

    def get_field(self, name: str) -> FieldDefinition | None:
        return self.field_map().get(name)


@dataclass
class FieldUpdate:
    field: str
    value: Any
    item_index: int | None = None
    action: str = "set"


@dataclass
class TargetRef:
    field: str
    item_index: int | None = None


@dataclass
class UnderstandingResult:
    goal: str = ""
    workflow: str = ""
    action: str = UnderstandingAction.NONE.value
    confidence: float = 0.0
    entities: dict[str, Any] = field(default_factory=dict)
    field_updates: list[FieldUpdate] = field(default_factory=list)
    targets: list[TargetRef] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    is_out_of_scope: bool = False
    is_greeting: bool = False
    interrupt_workflow: str | None = None
    reasoning: str = ""
    source: str = "rules"

    def is_expense_intent(self) -> bool:
        return self.workflow == "expense" and self.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
        )

    def is_leave_intent(self) -> bool:
        return self.workflow == "leave" and self.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
        )

    def interrupts_active_workflow(self, active_workflow_id: str | None) -> bool:
        if self.interrupt_workflow and active_workflow_id and self.interrupt_workflow != active_workflow_id:
            return True
        if not active_workflow_id or active_workflow_id == self.workflow:
            return False
        return self.is_expense_intent() or self.is_leave_intent()

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "workflow": self.workflow,
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "entities": self.entities,
            "field_updates": [
                {"field": u.field, "value": u.value, "item_index": u.item_index, "action": u.action}
                for u in self.field_updates
            ],
            "targets": [{"field": t.field, "item_index": t.item_index} for t in self.targets],
            "missing_fields": self.missing_fields,
            "is_out_of_scope": self.is_out_of_scope,
            "is_greeting": self.is_greeting,
            "interrupt_workflow": self.interrupt_workflow,
            "reasoning": self.reasoning,
            "source": self.source,
        }


@dataclass
class WorkflowEvent:
    event_type: str
    workflow_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "workflow_id": self.workflow_id,
            "payload": self.payload,
            "turn": self.turn,
        }
