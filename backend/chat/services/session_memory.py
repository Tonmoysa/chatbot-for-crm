"""Structured session memory persisted in ``ConversationSession.workflow_state``.

Phase 7: production reads/writes go through ``chat.services.session_store.SessionStore``.
``load_session_memory`` / ``save_session_memory`` are low-level helpers used by SessionStore.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from chat.models import ConversationSession
from chat.services.platform.schemas import TurnContext
from chat.services.translator import detect_user_language, resolve_reply_language
from django.conf import settings

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
    item_index: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PendingQuestion | None:
        if not raw or not str(raw.get("field") or "").strip():
            return None
        item_index = raw.get("item_index")
        return cls(
            field=str(raw["field"]).strip(),
            prompt=str(raw.get("prompt") or ""),
            workflow_id=str(raw.get("workflow_id") or ""),
            asked_at_turn=int(raw.get("asked_at_turn") or 0),
            item_index=int(item_index) if item_index is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "field": self.field,
            "prompt": self.prompt,
            "workflow_id": self.workflow_id,
            "asked_at_turn": self.asked_at_turn,
        }
        if self.item_index is not None:
            out["item_index"] = self.item_index
        return out


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


def build_turn_context(
    *,
    message: str,
    memory: SessionMemory,
    conversation_history: list[str],
    trace_id: str,
    session_id: str,
    company_id: str,
    employee_id: str,
    document_text: str | None = None,
    idempotency_key: str = "",
) -> TurnContext:
    """Build an immutable turn snapshot immediately after session memory is loaded."""
    aw = memory.active_workflow
    pq = memory.pending_question
    draft = memory.active_draft()
    pending_confirmation = memory.pending_confirmation or None
    user_lang = detect_user_language(message)
    stored = (memory.last_entities or {}).get("reply_language")
    stored_lang = stored if isinstance(stored, str) else None
    reply_lang = resolve_reply_language(message, stored_lang)
    draft_snapshot = deepcopy(draft.to_dict()) if draft else None
    suspended = tuple(deepcopy(sw.to_dict()) for sw in memory.suspended_workflows)
    from chat.services.platform.turn_semantics import last_assistant_message

    return TurnContext(
        trace_id=trace_id,
        session_id=session_id,
        company_id=company_id,
        employee_id=employee_id,
        user_message=(message or "").strip(),
        conversation_history=tuple(conversation_history or ()),
        document_text=document_text,
        idempotency_key=idempotency_key or "",
        user_language=user_lang,
        reply_language=reply_lang,
        today_iso=date.today().isoformat(),
        turn_count_at_start=int(memory.turn_count or 0),
        memory_schema_version=int(memory.schema_version),
        active_workflow_id=aw.id if aw else None,
        active_workflow_stage=aw.stage if aw else None,
        draft_id=aw.draft_id if aw else None,
        pending_question_field=pq.field if pq else None,
        pending_question_prompt=pq.prompt if pq else None,
        pending_question_workflow_id=pq.workflow_id if pq else None,
        pending_confirmation=pending_confirmation,
        draft_snapshot=draft_snapshot,
        suspended_workflows=suspended,
        conversation_facts=deepcopy(memory.conversation_facts or {}),
        has_active_workflow=aw is not None,
        has_pending_question=pq is not None,
        has_pending_confirmation=bool(pending_confirmation),
        draft_locked=bool(draft and draft.locked),
        wizard_active=bool(pq),
        last_assistant_message=last_assistant_message(conversation_history),
    )


def assert_turn_context_parity(ctx: TurnContext, memory: SessionMemory) -> None:
    """Verify TurnContext matches live SessionMemory at turn start (dev/test guard)."""
    aw = memory.active_workflow
    pq = memory.pending_question
    draft = memory.active_draft()
    pending_confirmation = memory.pending_confirmation or None

    assert ctx.turn_count_at_start == int(memory.turn_count or 0)
    assert ctx.memory_schema_version == int(memory.schema_version)
    assert ctx.active_workflow_id == (aw.id if aw else None)
    assert ctx.active_workflow_stage == (aw.stage if aw else None)
    assert ctx.draft_id == (aw.draft_id if aw else None)
    assert ctx.pending_question_field == (pq.field if pq else None)
    assert ctx.pending_question_prompt == (pq.prompt if pq else None)
    assert ctx.pending_question_workflow_id == (pq.workflow_id if pq else None)
    assert ctx.pending_confirmation == pending_confirmation
    assert ctx.has_active_workflow == (aw is not None)
    assert ctx.has_pending_question == (pq is not None)
    assert ctx.has_pending_confirmation == bool(pending_confirmation)
    assert ctx.draft_locked == bool(draft and draft.locked)
    assert ctx.wizard_active == bool(pq)

    if draft is None:
        assert ctx.draft_snapshot is None
    else:
        assert ctx.draft_snapshot == draft.to_dict()

    assert len(ctx.suspended_workflows) == len(memory.suspended_workflows)
    for i, sw in enumerate(memory.suspended_workflows):
        assert ctx.suspended_workflows[i] == sw.to_dict()

    assert ctx.conversation_facts == (memory.conversation_facts or {})

    if ctx.draft_snapshot is not None:
        assert ctx.draft_id is not None
        assert ctx.draft_snapshot.get("workflow_id") == (
            draft.workflow_id if draft else None
        )
        assert ctx.draft_snapshot.get("version") == (
            draft.version if draft else None
        )


def is_leave_platform_scenario(
    memory: SessionMemory,
    *,
    understanding: Any | None = None,
    pq_target_workflow: str | None = None,
) -> bool:
    """True when this turn must stay on the leave platform path (Phase 7)."""
    aw = memory.active_workflow
    if aw and aw.id == "leave":
        return True

    pq = memory.pending_question
    if pq and (pq.workflow_id or "").strip().lower() == "leave":
        return True

    pc = str(memory.pending_confirmation or "")
    if pc == "duplicate_leave":
        return True
    if pc.startswith("switch:") and "leave" in pc:
        return True
    if pc == "submit" and aw and aw.id == "leave":
        return True

    if pq_target_workflow and pq_target_workflow.strip().lower() == "leave":
        return True

    if understanding is not None:
        wf = getattr(understanding, "workflow", "") or ""
        if wf == "leave":
            return True
        is_leave_intent = getattr(understanding, "is_leave_intent", None)
        if callable(is_leave_intent) and is_leave_intent():
            return True

    return False


def is_expense_platform_scenario(
    memory: SessionMemory,
    *,
    understanding: Any | None = None,
    pq_target_workflow: str | None = None,
) -> bool:
    """True when this turn must stay on the expense platform path (Phase 8)."""
    if not getattr(settings, "EXPENSE_NEW_ARCH", False):
        return False

    aw = memory.active_workflow
    if aw and aw.id == "expense":
        return True

    pq = memory.pending_question
    if pq and (pq.workflow_id or "").strip().lower() == "expense":
        return True

    pc = str(memory.pending_confirmation or "")
    if pc == "submit" and aw and aw.id == "expense":
        return True
    if pc.startswith("modify:") and aw and aw.id == "expense":
        return True
    if pc.startswith("switch:") and "expense" in pc:
        return True

    if pq_target_workflow and pq_target_workflow.strip().lower() == "expense":
        return True

    if understanding is not None:
        wf = getattr(understanding, "workflow", "") or ""
        if wf == "expense":
            return True
        is_expense_intent = getattr(understanding, "is_expense_intent", None)
        if callable(is_expense_intent) and is_expense_intent():
            return True

    return False


def is_platform_only_scenario(
    memory: SessionMemory,
    *,
    understanding: Any | None = None,
    pq_target_workflow: str | None = None,
) -> bool:
    """Leave always platform-only; expense when EXPENSE_NEW_ARCH is enabled."""
    if is_leave_platform_scenario(
        memory,
        understanding=understanding,
        pq_target_workflow=pq_target_workflow,
    ):
        return True
    return is_expense_platform_scenario(
        memory,
        understanding=understanding,
        pq_target_workflow=pq_target_workflow,
    )


# --- State reducer (Phase 6) — single writer for workflow session fields ---


def reduce_set_pending_question(memory: SessionMemory, pq: PendingQuestion | None) -> None:
    memory.pending_question = pq


def reduce_clear_pending_question(memory: SessionMemory) -> None:
    memory.pending_question = None


def reduce_set_pending_confirmation(memory: SessionMemory, value: str | None) -> None:
    memory.pending_confirmation = value


def reduce_clear_pending_confirmation(memory: SessionMemory) -> None:
    memory.pending_confirmation = None


def reduce_set_active_workflow(memory: SessionMemory, aw: ActiveWorkflow | None) -> None:
    memory.active_workflow = aw


def reduce_set_active_stage(memory: SessionMemory, stage: str) -> None:
    if memory.active_workflow:
        memory.active_workflow.stage = stage


def reduce_append_suspended_workflow(memory: SessionMemory, sw: SuspendedWorkflow) -> None:
    memory.suspended_workflows.append(sw)


def reduce_remove_suspended_workflow(memory: SessionMemory, workflow_id: str) -> SuspendedWorkflow | None:
    target = workflow_id.strip().lower()
    for i, sw in enumerate(list(memory.suspended_workflows)):
        if sw.workflow_id == target:
            return memory.suspended_workflows.pop(i)
    return None


def reduce_ensure_draft(memory: SessionMemory, draft_id: str, workflow_id: str) -> WorkflowDraft:
    wf_id = workflow_id.strip().lower()
    draft = memory.workflow_drafts.get(draft_id)
    if draft and draft.locked:
        draft = WorkflowDraft(workflow_id=wf_id)
        memory.workflow_drafts[draft_id] = draft
    elif not draft or draft.workflow_id != wf_id:
        draft = WorkflowDraft(workflow_id=wf_id)
        memory.workflow_drafts[draft_id] = draft
    return draft


def reduce_clear_draft_fields(memory: SessionMemory, draft_id: str = "default") -> None:
    draft = memory.workflow_drafts.get(draft_id)
    if draft:
        draft.fields = {}
        draft.version += 1


def reduce_merge_last_entities(memory: SessionMemory, updates: dict[str, Any]) -> None:
    memory.last_entities = {**dict(memory.last_entities or {}), **updates}


def reduce_set_last_entities(memory: SessionMemory, entities: dict[str, Any]) -> None:
    memory.last_entities = dict(entities)


def reduce_pop_last_entity(memory: SessionMemory, key: str) -> Any:
    entities = dict(memory.last_entities or {})
    value = entities.pop(key, None)
    memory.last_entities = entities
    return value


def reduce_record_submitted_leave_range(
    memory: SessionMemory,
    fields: dict[str, Any],
    *,
    request_id: str = "",
) -> None:
    from chat.services.platform.field_extractors.leave import leave_range_from_fields

    rng = leave_range_from_fields(fields)
    if not rng:
        return
    facts = dict(memory.conversation_facts or {})
    rows = list(facts.get("submitted_leave_ranges") or [])
    rows.append({
        "start_date": rng[0],
        "end_date": rng[1],
        "request_id": request_id,
    })
    facts["submitted_leave_ranges"] = rows
    memory.conversation_facts = facts


def reduce_record_submitted_expense(
    memory: SessionMemory,
    fields: dict[str, Any],
    *,
    request_id: str = "",
) -> None:
    facts = dict(memory.conversation_facts or {})
    rows = list(facts.get("submitted_expenses") or [])
    rows.append({
        "request_id": request_id,
        "incurred_date": fields.get("incurred_date"),
        "items": list(fields.get("items") or []),
    })
    facts["submitted_expenses"] = rows
    memory.conversation_facts = facts


def reduce_lock_submitted_draft(memory: SessionMemory, request_id: str) -> None:
    draft = memory.active_draft()
    if not draft:
        return
    draft.locked = True
    draft.status = "submitted"
    draft.submitted_request_id = request_id
    reduce_set_active_stage(memory, "submitted")
    reduce_clear_pending_question(memory)
    reduce_clear_pending_confirmation(memory)


def reduce_suspend_active_workflow(memory: SessionMemory) -> None:
    if not memory.active_workflow:
        return
    aw = memory.active_workflow
    suspended_draft_id = aw.draft_id
    if aw.draft_id == "default":
        draft = memory.active_draft()
        if draft and draft.workflow_id == aw.id:
            suspended_draft_id = aw.id
            memory.workflow_drafts[suspended_draft_id] = deepcopy(draft)
    reduce_append_suspended_workflow(
        memory,
        SuspendedWorkflow(
            workflow_id=aw.id,
            stage=aw.stage,
            draft_id=suspended_draft_id,
            suspended_at_turn=memory.turn_count,
        ),
    )
    reduce_set_active_workflow(memory, None)
    reduce_clear_pending_question(memory)


def reduce_cancel_active_workflow(memory: SessionMemory) -> None:
    aw = memory.active_workflow
    if not aw:
        return
    memory.workflow_drafts.pop(aw.draft_id, None)
    reduce_set_active_workflow(memory, None)
    reduce_clear_pending_question(memory)
    reduce_clear_pending_confirmation(memory)
    memory.last_action = "workflow_cancelled"


def reduce_start_workflow(memory: SessionMemory, workflow_id: str) -> WorkflowDraft:
    wf_id = workflow_id.strip().lower()
    if memory.active_workflow and memory.active_workflow.id != wf_id:
        reduce_suspend_active_workflow(memory)
    draft_id = "default"
    reduce_set_active_workflow(
        memory,
        ActiveWorkflow(id=wf_id, stage="collecting", draft_id=draft_id),
    )
    draft = reduce_ensure_draft(memory, draft_id, wf_id)
    reduce_clear_pending_confirmation(memory)
    return draft


def reduce_resume_suspended_workflow(memory: SessionMemory, sw: SuspendedWorkflow) -> WorkflowDraft | None:
    reduce_set_active_workflow(
        memory,
        ActiveWorkflow(id=sw.workflow_id, stage=sw.stage, draft_id=sw.draft_id),
    )
    return memory.workflow_drafts.get(sw.draft_id)


def reduce_arm_switch_confirm(
    memory: SessionMemory,
    *,
    from_workflow: str,
    to_workflow: str,
    pending_message: str,
) -> None:
    reduce_set_pending_confirmation(memory, f"switch:{from_workflow}:{to_workflow}")
    reduce_merge_last_entities(memory, {"switch_pending_message": pending_message})


def reduce_clear_switch_confirm(memory: SessionMemory) -> str:
    pending = str((memory.last_entities or {}).get("switch_pending_message") or "")
    reduce_clear_pending_confirmation(memory)
    if memory.last_entities and "switch_pending_message" in memory.last_entities:
        reduce_pop_last_entity(memory, "switch_pending_message")
    return pending


def reduce_arm_duplicate_leave(memory: SessionMemory, entities: dict[str, Any]) -> None:
    reduce_set_last_entities(memory, entities)
    reduce_set_pending_confirmation(memory, "duplicate_leave")


def reduce_apply_field_updates(
    memory: SessionMemory,
    draft_id: str,
    updates_raw: list[dict[str, Any]],
    *,
    message: str = "",
    trace_id: str = "",
    review_validated: bool = False,
) -> list[dict[str, Any]]:
    from chat.services.platform.field_engine import FieldEngine, deserialize_field_updates, serialize_field_updates

    draft = memory.workflow_drafts.get(draft_id) or memory.active_draft()
    if not draft:
        return []
    updates = deserialize_field_updates(updates_raw)
    if not updates:
        return []

    if draft.workflow_id == "leave":
        from chat.services.platform.field_extractors.leave import (
            is_garbage_leave_reason_value,
            is_leave_complaint_reason_value,
            is_leave_review_complaint_or_question,
            is_leave_review_mode,
            review_field_updates_from_message,
        )

        if is_leave_review_mode(memory):
            if review_validated:
                if is_leave_review_complaint_or_question(message):
                    return []
                updates = [
                    u
                    for u in updates
                    if u.field != "reason"
                    or (
                        not is_garbage_leave_reason_value(str(u.value or ""))
                        and not is_leave_complaint_reason_value(str(u.value or ""))
                    )
                ]
            else:
                semantic = review_field_updates_from_message(message, memory, trace_id=trace_id)
                allowed = {(u.field, str(u.value)) for u in semantic}
                if semantic:
                    updates = [
                        u
                        for u in updates
                        if (u.field, str(u.value)) in allowed
                    ]
                else:
                    updates = [u for u in updates if u.field != "reason"]

    if draft.workflow_id == "expense":
        from chat.services.platform.field_extractors.expense import (
            filter_expense_updates_for_review,
            is_expense_review_mode,
        )

        if is_expense_review_mode(memory):
            updates = filter_expense_updates_for_review(
                updates,
                message,
                memory=memory,
                trace_id=trace_id,
            )

    before_fields = dict(draft.fields or {})
    if not updates:
        return []
    FieldEngine().apply_updates(draft, updates, message=message or "")

    if updates:
        from chat.services.observability import (
            classify_leave_field_apply_mode,
            log_field_updates_applied,
        )

        log_field_updates_applied(
            trace_id,
            workflow_id=draft.workflow_id,
            draft_id=draft_id,
            updates=serialize_field_updates(updates),
            before_fields=before_fields,
            after_fields=dict(draft.fields or {}),
            apply_mode=classify_leave_field_apply_mode(message, memory=memory),
            message=message or "",
        )
    return serialize_field_updates(updates)


def reduce_remove_draft_field(memory: SessionMemory, draft_id: str, field: str) -> None:
    draft = memory.workflow_drafts.get(draft_id) or memory.active_draft()
    if not draft:
        return
    draft.fields.pop(field, None)
    draft.version += 1


def reduce_set_draft_field(
    memory: SessionMemory,
    draft_id: str,
    field: str,
    value: Any,
) -> None:
    draft = memory.workflow_drafts.get(draft_id) or memory.active_draft()
    if not draft:
        return
    draft.fields[field] = value
    draft.version += 1


def reduce_switch_to_workflow(memory: SessionMemory, target_workflow_id: str) -> WorkflowDraft:
    target = target_workflow_id.strip().lower()
    reduce_suspend_active_workflow(memory)
    sw = reduce_remove_suspended_workflow(memory, target)
    if sw:
        draft = reduce_resume_suspended_workflow(memory, sw)
        if draft:
            reduce_clear_pending_question(memory)
            reduce_clear_pending_confirmation(memory)
            return draft
    return reduce_start_workflow(memory, target)


class StatePatchBuffer:
    """Collect state patches during plan-op execution; flush via ``apply_state_patches``."""

    def __init__(self, memory: SessionMemory, *, trace_id: str = "") -> None:
        self._memory = memory
        self._patches: list[dict[str, Any]] = []
        self._trace_id = trace_id or ""

    @property
    def memory(self) -> SessionMemory:
        return self._memory

    @property
    def patches(self) -> list[dict[str, Any]]:
        return list(self._patches)

    def _resolve_draft_id(self, draft_id: str | None) -> str:
        if draft_id:
            return draft_id
        aw = self._memory.active_workflow
        return aw.draft_id if aw else "default"

    def extend(self, more: list[dict[str, Any]]) -> None:
        self._patches.extend(more)

    def push(self, op: str, **kwargs: Any) -> None:
        patch: dict[str, Any] = {"op": op}
        patch.update(kwargs)
        self._patches.append(patch)

    def _apply_now(self, patch: dict[str, Any]) -> None:
        apply_state_patches(self._memory, [patch])

    def apply_field_updates(
        self,
        updates: list[Any],
        *,
        message: str = "",
        draft_id: str | None = None,
        review_validated: bool = False,
    ) -> list[dict[str, Any]]:
        from chat.services.platform.field_engine import serialize_field_updates

        if not updates:
            return []
        patch = {
            "op": "apply_field_updates",
            "draft_id": self._resolve_draft_id(draft_id),
            "updates": serialize_field_updates(updates),
            "message": message or "",
            "trace_id": self._trace_id,
            "review_validated": review_validated,
        }
        return reduce_apply_field_updates(
            self._memory,
            patch["draft_id"],
            patch["updates"],
            message=patch["message"],
            trace_id=patch["trace_id"],
            review_validated=review_validated,
        )

    def remove_draft_field(self, field: str, *, draft_id: str | None = None) -> None:
        patch = {
            "op": "remove_draft_field",
            "draft_id": self._resolve_draft_id(draft_id),
            "field": field,
        }
        self._apply_now(patch)

    def set_draft_field(self, field: str, value: Any, *, draft_id: str | None = None) -> None:
        patch = {
            "op": "set_draft_field",
            "draft_id": self._resolve_draft_id(draft_id),
            "field": field,
            "value": value,
        }
        self._apply_now(patch)

    def flush(self) -> None:
        if self._patches:
            apply_state_patches(self._memory, self._patches)
            self._patches.clear()

    def ensure_active_draft(self, workflow_id: str) -> WorkflowDraft | None:
        draft = self._memory.active_draft()
        if draft and draft.locked:
            draft_id = self._memory.active_workflow.draft_id if self._memory.active_workflow else "default"
            self.push("ensure_draft", draft_id=draft_id, workflow_id=workflow_id)
            self.flush()
            return self._memory.active_draft()
        if draft:
            return draft
        self.push("start_workflow", value=workflow_id)
        self.flush()
        return self._memory.active_draft()


def apply_state_patches(memory: SessionMemory, patches: list[dict[str, Any]]) -> None:
    """Apply ordered state patches — single reducer entry point (Phase 6/11)."""
    for patch in patches:
        op = str(patch.get("op") or "")
        if op == "set_pending_question":
            pq_raw = patch.get("value")
            reduce_set_pending_question(
                memory,
                PendingQuestion.from_dict(pq_raw) if pq_raw else None,
            )
        elif op == "clear_pending_question":
            reduce_clear_pending_question(memory)
        elif op == "set_pending_confirmation":
            reduce_set_pending_confirmation(memory, patch.get("value"))
        elif op == "clear_pending_confirmation":
            reduce_clear_pending_confirmation(memory)
        elif op == "merge_last_entities":
            reduce_merge_last_entities(memory, dict(patch.get("value") or {}))
        elif op == "set_last_entities":
            reduce_set_last_entities(memory, dict(patch.get("value") or {}))
        elif op == "arm_duplicate_leave":
            reduce_arm_duplicate_leave(memory, dict(patch.get("value") or {}))
        elif op == "clear_draft_fields":
            reduce_clear_draft_fields(memory, str(patch.get("draft_id") or "default"))
        elif op == "set_active_stage":
            reduce_set_active_stage(memory, str(patch.get("value") or "collecting"))
        elif op == "set_active_workflow":
            aw_raw = patch.get("value")
            reduce_set_active_workflow(
                memory,
                ActiveWorkflow.from_dict(aw_raw) if aw_raw else None,
            )
        elif op == "start_workflow":
            reduce_start_workflow(memory, str(patch.get("value") or ""))
        elif op == "ensure_draft":
            reduce_ensure_draft(
                memory,
                str(patch.get("draft_id") or "default"),
                str(patch.get("workflow_id") or "leave"),
            )
        elif op == "suspend_active_workflow":
            reduce_suspend_active_workflow(memory)
        elif op == "cancel_active_workflow":
            reduce_cancel_active_workflow(memory)
        elif op == "switch_to_workflow":
            reduce_switch_to_workflow(memory, str(patch.get("value") or ""))
        elif op == "lock_submitted_draft":
            reduce_lock_submitted_draft(memory, str(patch.get("request_id") or ""))
        elif op == "arm_switch_confirm":
            reduce_arm_switch_confirm(
                memory,
                from_workflow=str(patch.get("from_workflow") or ""),
                to_workflow=str(patch.get("to_workflow") or ""),
                pending_message=str(patch.get("pending_message") or ""),
            )
        elif op == "clear_switch_confirm":
            reduce_clear_switch_confirm(memory)
        elif op == "record_submitted_leave_range":
            reduce_record_submitted_leave_range(
                memory,
                dict(patch.get("fields") or {}),
                request_id=str(patch.get("request_id") or ""),
            )
        elif op == "record_submitted_expense":
            reduce_record_submitted_expense(
                memory,
                dict(patch.get("fields") or {}),
                request_id=str(patch.get("request_id") or ""),
            )
        elif op == "apply_field_updates":
            reduce_apply_field_updates(
                memory,
                str(patch.get("draft_id") or "default"),
                list(patch.get("updates") or []),
                message=str(patch.get("message") or ""),
                trace_id=str(patch.get("trace_id") or ""),
                review_validated=bool(patch.get("review_validated")),
            )
        elif op == "remove_draft_field":
            reduce_remove_draft_field(
                memory,
                str(patch.get("draft_id") or "default"),
                str(patch.get("field") or ""),
            )
        elif op == "set_draft_field":
            reduce_set_draft_field(
                memory,
                str(patch.get("draft_id") or "default"),
                str(patch.get("field") or ""),
                patch.get("value"),
            )


def load_session_memory(session: ConversationSession) -> SessionMemory:
    raw = getattr(session, "workflow_state", None) or {}
    if not isinstance(raw, dict):
        raw = {}
    return SessionMemory.from_workflow_state(raw)


def save_session_memory(session: ConversationSession, memory: SessionMemory) -> None:
    session.workflow_state = memory.to_workflow_state()
    session.save(update_fields=["workflow_state", "updated_at"])
