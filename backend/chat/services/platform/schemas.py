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
    STATUS = "status"
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
    answers_pending_field: bool | None = None

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
            "answers_pending_field": self.answers_pending_field,
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


TURN_CONTEXT_SCHEMA_VERSION = "turn_context_v2"
EXECUTION_PLAN_SCHEMA_VERSION = "execution_plan_v1"


class PlanOp(str, Enum):
    """Single executable step for workflow turns (Phase 5 unified workflow ops + informational)."""

    NONE = "none"
    REJECT_OOS = "reject_oos"
    LOCKED_RESPONSE = "locked_response"
    SUBMITTED_LEAVE_OVERLAP = "submitted_leave_overlap"
    RESOLVE_WORKFLOW_SWITCH = "resolve_workflow_switch"
    RESOLVE_DUPLICATE_LEAVE = "resolve_duplicate_leave"
    RESOLVE_SUBMIT_CONFIRMATION = "resolve_submit_confirmation"
    APPLY_PENDING_MODIFY = "apply_pending_modify"
    MAYBE_DUPLICATE_LEAVE = "maybe_duplicate_leave"
    MAYBE_WORKFLOW_SWITCH = "maybe_workflow_switch"
    # Phase 5 — workflow-agnostic ops (workflow_id lives on ExecutionPlan)
    WORKFLOW_COLLECT = "workflow_collect"
    WORKFLOW_NEW = "workflow_new"
    WORKFLOW_MODIFY = "workflow_modify"
    WORKFLOW_DELETE = "workflow_delete"
    WORKFLOW_SWITCH = "workflow_switch"
    WORKFLOW_CLARIFICATION = "workflow_clarification"
    WORKFLOW_SHOW_REVIEW = "workflow_show_review"
    WORKFLOW_REQUEST_SUBMIT = "workflow_request_submit"
    WORKFLOW_CONFIRM_SUBMIT = "workflow_confirm_submit"
    WORKFLOW_APPLY_UPDATES = "workflow_apply_updates"
    WORKFLOW_CANCEL = "workflow_cancel"
    # Backward-compatible aliases (same handler as WORKFLOW_*)
    LEAVE_COLLECT = WORKFLOW_COLLECT
    LEAVE_NEW = WORKFLOW_NEW
    LEAVE_MODIFY = WORKFLOW_MODIFY
    LEAVE_DELETE = WORKFLOW_DELETE
    LEAVE_SWITCH = WORKFLOW_SWITCH
    LEAVE_CLARIFICATION = WORKFLOW_CLARIFICATION
    LEAVE_SHOW_REVIEW = WORKFLOW_SHOW_REVIEW
    LEAVE_REQUEST_SUBMIT = WORKFLOW_REQUEST_SUBMIT
    LEAVE_CONFIRM_SUBMIT = WORKFLOW_CONFIRM_SUBMIT
    LEAVE_APPLY_UPDATES = WORKFLOW_APPLY_UPDATES
    EXPENSE_COLLECT = WORKFLOW_COLLECT
    EXPENSE_NEW = WORKFLOW_NEW
    EXPENSE_MODIFY = WORKFLOW_MODIFY
    EXPENSE_DELETE = WORKFLOW_DELETE
    EXPENSE_SWITCH = WORKFLOW_SWITCH
    EXPENSE_CLARIFICATION = WORKFLOW_CLARIFICATION
    EXPENSE_SHOW_REVIEW = WORKFLOW_SHOW_REVIEW
    EXPENSE_REQUEST_SUBMIT = WORKFLOW_REQUEST_SUBMIT
    EXPENSE_CONFIRM_SUBMIT = WORKFLOW_CONFIRM_SUBMIT
    EXPENSE_APPLY_UPDATES = WORKFLOW_APPLY_UPDATES
    SHOW_SESSION_CONTEXT = "show_session_context"
    CLEAR_PENDING_MODIFY = "clear_pending_modify"
    REPLY_POLICY = "reply_policy"
    REPLY_STATUS = "reply_status"
    REPLY_OOS = "reply_oos"
    REPLY_GREETING = "reply_greeting"
    REPLY_CONVERSATIONAL = "reply_conversational"
    REPLY_PLATFORM_CLARIFY = "reply_platform_clarify"
    REPLY_GENERAL_HELP = "reply_general_help"
    REPLY_TODAY_DATE = "reply_today_date"
    REPLY_TRANSLATION = "reply_translation"


INFORMATIONAL_PLAN_OPS = frozenset(
    {
        PlanOp.REPLY_POLICY,
        PlanOp.REPLY_STATUS,
        PlanOp.REPLY_OOS,
        PlanOp.REPLY_GREETING,
        PlanOp.REPLY_CONVERSATIONAL,
        PlanOp.REPLY_PLATFORM_CLARIFY,
        PlanOp.REPLY_GENERAL_HELP,
        PlanOp.REPLY_TODAY_DATE,
        PlanOp.REPLY_TRANSLATION,
    }
)


WORKFLOW_PLAN_OPS = frozenset(
    {
        PlanOp.WORKFLOW_COLLECT,
        PlanOp.WORKFLOW_NEW,
        PlanOp.WORKFLOW_MODIFY,
        PlanOp.WORKFLOW_DELETE,
        PlanOp.WORKFLOW_SWITCH,
        PlanOp.WORKFLOW_CLARIFICATION,
        PlanOp.WORKFLOW_SHOW_REVIEW,
        PlanOp.WORKFLOW_REQUEST_SUBMIT,
        PlanOp.WORKFLOW_CONFIRM_SUBMIT,
        PlanOp.WORKFLOW_APPLY_UPDATES,
        PlanOp.WORKFLOW_CANCEL,
    }
)


@dataclass
class TurnDecision:
    """Decision Core output passed to PlanBuilder (PQ + understanding)."""

    pq: Any = None
    understanding: UnderstandingResult | None = None
    route_source: str = "pending"  # "pending" | "active"

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "route_source": self.route_source,
            "pq": self.pq.to_log_dict() if self.pq else None,
            "understanding": self.understanding.to_dict() if self.understanding else None,
        }


@dataclass
class ExecutionPlan:
    """Ordered ops for one leave turn — built from TurnContext + TurnDecision."""

    ops: list[PlanOp]
    workflow_id: str = "leave"
    reason: str = ""
    plan_schema_version: str = EXECUTION_PLAN_SCHEMA_VERSION

    @property
    def primary_op(self) -> PlanOp:
        return self.ops[0] if self.ops else PlanOp.NONE

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "plan_schema_version": self.plan_schema_version,
            "workflow_id": self.workflow_id,
            "reason": self.reason,
            "ops": [op.value for op in self.ops],
            "primary_op": self.primary_op.value,
        }


@dataclass(frozen=True)
class TurnContext:
    """Immutable snapshot of one chat turn at load time (before any layer mutates memory)."""

    trace_id: str
    session_id: str
    company_id: str
    employee_id: str
    user_message: str
    conversation_history: tuple[str, ...]
    document_text: str | None
    idempotency_key: str
    user_language: str
    reply_language: str
    today_iso: str
    turn_count_at_start: int
    memory_schema_version: int
    active_workflow_id: str | None
    active_workflow_stage: str | None
    draft_id: str | None
    pending_question_field: str | None
    pending_question_prompt: str | None
    pending_question_workflow_id: str | None
    pending_confirmation: str | None
    draft_snapshot: dict[str, Any] | None
    suspended_workflows: tuple[dict[str, Any], ...]
    conversation_facts: dict[str, Any]
    has_active_workflow: bool
    has_pending_question: bool
    has_pending_confirmation: bool
    draft_locked: bool
    wizard_active: bool
    last_assistant_message: str | None = None
    context_schema_version: str = TURN_CONTEXT_SCHEMA_VERSION

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "context_schema_version": self.context_schema_version,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "company_id": self.company_id,
            "employee_id": self.employee_id,
            "user_message": self.user_message,
            "conversation_history_len": len(self.conversation_history),
            "document_text_present": bool(self.document_text),
            "idempotency_key": self.idempotency_key,
            "user_language": self.user_language,
            "reply_language": self.reply_language,
            "today_iso": self.today_iso,
            "turn_count_at_start": self.turn_count_at_start,
            "memory_schema_version": self.memory_schema_version,
            "active_workflow_id": self.active_workflow_id,
            "active_workflow_stage": self.active_workflow_stage,
            "draft_id": self.draft_id,
            "pending_question_field": self.pending_question_field,
            "pending_question_prompt": self.pending_question_prompt,
            "pending_question_workflow_id": self.pending_question_workflow_id,
            "pending_confirmation": self.pending_confirmation,
            "draft_snapshot": self.draft_snapshot,
            "suspended_workflows": list(self.suspended_workflows),
            "conversation_facts": self.conversation_facts,
            "has_active_workflow": self.has_active_workflow,
            "has_pending_question": self.has_pending_question,
            "has_pending_confirmation": self.has_pending_confirmation,
            "draft_locked": self.draft_locked,
            "wizard_active": self.wizard_active,
            "last_assistant_message": self.last_assistant_message,
            "draft_version": (
                (self.draft_snapshot or {}).get("version")
                if self.draft_snapshot
                else None
            ),
        }
