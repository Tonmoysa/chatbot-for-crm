"""Structured session memory persisted in ``ConversationSession.workflow_state``."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from chat.models import ConversationSession

SCHEMA_VERSION = 2


@dataclass
class ActiveWorkflow:
    id: str
    stage: str = "collecting"
    draft_id: str = "default"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ActiveWorkflow | None:
        if not raw or not str(raw.get("id") or "").strip():
            return None
        return cls(
            id=str(raw["id"]).strip(),
            stage=str(raw.get("stage") or "collecting"),
            draft_id=str(raw.get("draft_id") or "default"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "stage": self.stage, "draft_id": self.draft_id}


@dataclass
class SuspendedWorkflow:
    workflow_id: str
    stage: str = "collecting"
    draft_id: str = "default"
    suspended_at_turn: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SuspendedWorkflow | None:
        if not raw or not str(raw.get("workflow_id") or "").strip():
            return None
        return cls(
            workflow_id=str(raw["workflow_id"]).strip(),
            stage=str(raw.get("stage") or "collecting"),
            draft_id=str(raw.get("draft_id") or "default"),
            suspended_at_turn=int(raw.get("suspended_at_turn") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "stage": self.stage,
            "draft_id": self.draft_id,
            "suspended_at_turn": self.suspended_at_turn,
        }


@dataclass
class PendingQuestion:
    field: str
    prompt: str
    workflow_id: str
    asked_at_turn: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PendingQuestion | None:
        if not raw or not str(raw.get("field") or "").strip():
            return None
        return cls(
            field=str(raw["field"]).strip(),
            prompt=str(raw.get("prompt") or ""),
            workflow_id=str(raw.get("workflow_id") or ""),
            asked_at_turn=int(raw.get("asked_at_turn") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "prompt": self.prompt,
            "workflow_id": self.workflow_id,
            "asked_at_turn": self.asked_at_turn,
        }


@dataclass
class WorkflowDraft:
    workflow_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    line_items: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    locked: bool = False
    status: str = "draft"
    submitted_request_id: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> WorkflowDraft | None:
        if not raw:
            return None
        wf_id = str(raw.get("workflow_id") or "").strip()
        if not wf_id:
            return None
        return cls(
            workflow_id=wf_id,
            fields=dict(raw.get("fields") or {}),
            line_items=list(raw.get("line_items") or []),
            version=int(raw.get("version") or 1),
            locked=bool(raw.get("locked")),
            status=str(raw.get("status") or "draft"),
            submitted_request_id=str(raw.get("submitted_request_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "fields": deepcopy(self.fields),
            "line_items": deepcopy(self.line_items),
            "version": self.version,
            "locked": self.locked,
            "status": self.status,
            "submitted_request_id": self.submitted_request_id,
        }


@dataclass
class SessionMemory:
    schema_version: int = SCHEMA_VERSION
    active_workflow: ActiveWorkflow | None = None
    suspended_workflows: list[SuspendedWorkflow] = field(default_factory=list)
    pending_question: PendingQuestion | None = None
    workflow_drafts: dict[str, WorkflowDraft] = field(default_factory=dict)
    conversation_facts: dict[str, Any] = field(default_factory=dict)
    history_summary: str = ""
    last_action: str = ""
    last_entities: dict[str, Any] = field(default_factory=dict)
    pending_confirmation: str | None = None
    events: list[Any] = field(default_factory=list)
    turn_count: int = 0

    @classmethod
    def from_workflow_state(cls, raw: dict[str, Any] | None) -> SessionMemory:
        data = dict(raw or {})
        drafts: dict[str, WorkflowDraft] = {}
        for key, val in (data.get("workflow_drafts") or {}).items():
            draft = WorkflowDraft.from_dict(val if isinstance(val, dict) else None)
            if draft:
                drafts[str(key)] = draft

        suspended: list[SuspendedWorkflow] = []
        for item in data.get("suspended_workflows") or []:
            sw = SuspendedWorkflow.from_dict(item if isinstance(item, dict) else None)
            if sw:
                suspended.append(sw)

        events_raw = data.get("events") or []
        events: list[Any] = []
        if events_raw:
            from chat.services.platform.schemas import WorkflowEvent

            for e in events_raw:
                if isinstance(e, dict):
                    events.append(
                        WorkflowEvent(
                            event_type=str(e.get("event_type") or ""),
                            workflow_id=str(e.get("workflow_id") or ""),
                            payload=dict(e.get("payload") or {}),
                            turn=int(e.get("turn") or 0),
                        )
                    )

        return cls(
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
            active_workflow=ActiveWorkflow.from_dict(data.get("active_workflow")),
            suspended_workflows=suspended,
            pending_question=PendingQuestion.from_dict(data.get("pending_question")),
            workflow_drafts=drafts,
            conversation_facts=dict(data.get("conversation_facts") or {}),
            history_summary=str(data.get("history_summary") or ""),
            last_action=str(data.get("last_action") or ""),
            last_entities=dict(data.get("last_entities") or {}),
            pending_confirmation=data.get("pending_confirmation"),
            events=events,
            turn_count=int(data.get("turn_count") or 0),
        )

    def to_workflow_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "workflow_drafts": {k: v.to_dict() for k, v in self.workflow_drafts.items()},
            "suspended_workflows": [s.to_dict() for s in self.suspended_workflows],
            "conversation_facts": deepcopy(self.conversation_facts),
            "history_summary": self.history_summary,
            "last_action": self.last_action,
            "last_entities": deepcopy(self.last_entities),
            "turn_count": self.turn_count,
            "events": [
                e.to_dict() if hasattr(e, "to_dict") else e for e in self.events[-100:]
            ],
        }
        if self.active_workflow:
            out["active_workflow"] = self.active_workflow.to_dict()
        if self.pending_question:
            out["pending_question"] = self.pending_question.to_dict()
        if self.pending_confirmation:
            out["pending_confirmation"] = self.pending_confirmation
        return out

    def active_draft(self) -> WorkflowDraft | None:
        if not self.active_workflow:
            return None
        return self.workflow_drafts.get(self.active_workflow.draft_id)


def load_session_memory(session: ConversationSession) -> SessionMemory:
    raw = getattr(session, "workflow_state", None) or {}
    if not isinstance(raw, dict):
        raw = {}
    return SessionMemory.from_workflow_state(raw)


def save_session_memory(session: ConversationSession, memory: SessionMemory) -> None:
    session.workflow_state = memory.to_workflow_state()
    session.save(update_fields=["workflow_state", "updated_at"])
