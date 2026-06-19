"""Workflow lifecycle — start, pause, resume, switch, lock."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chat.services.platform.event_store import EventStore
from chat.services.platform.intent_rules import is_workflow_interrupt_expense
from chat.services.platform.registry import get_workflow_definition
from chat.services.platform.schemas import WorkflowStage, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
)


@dataclass
class WorkflowInterrupt:
    from_workflow: str
    to_workflow: str
    confidence: float
    pending_message: str


class WorkflowManager:
    def __init__(self, events: EventStore | None = None) -> None:
        self.events = events or EventStore()

    def start_workflow(self, memory: SessionMemory, workflow_id: str) -> WorkflowDraft:
        wf_id = workflow_id.strip().lower()
        if memory.active_workflow and memory.active_workflow.id != wf_id:
            self.suspend_active(memory)
        draft_id = "default"
        memory.active_workflow = ActiveWorkflow(id=wf_id, stage=WorkflowStage.COLLECTING.value, draft_id=draft_id)
        draft = memory.workflow_drafts.get(draft_id)
        if draft and draft.locked:
            draft = WorkflowDraft(workflow_id=wf_id)
            memory.workflow_drafts[draft_id] = draft
        elif not draft or draft.workflow_id != wf_id:
            draft = WorkflowDraft(workflow_id=wf_id)
            memory.workflow_drafts[draft_id] = draft
        memory.pending_confirmation = None
        self.events.emit(memory, "workflow_started", wf_id, {"draft_id": draft_id})
        return draft

    def suspend_active(self, memory: SessionMemory) -> None:
        if not memory.active_workflow:
            return
        aw = memory.active_workflow
        memory.suspended_workflows.append(
            SuspendedWorkflow(
                workflow_id=aw.id,
                stage=aw.stage,
                draft_id=aw.draft_id,
                suspended_at_turn=memory.turn_count,
            )
        )
        self.events.emit(memory, "workflow_paused", aw.id, {"draft_id": aw.draft_id})
        memory.active_workflow = None
        memory.pending_question = None

    def switch_to(self, memory: SessionMemory, target_workflow_id: str) -> WorkflowDraft:
        target = target_workflow_id.strip().lower()
        self.suspend_active(memory)
        for i, sw in enumerate(list(memory.suspended_workflows)):
            if sw.workflow_id == target:
                memory.suspended_workflows.pop(i)
                memory.active_workflow = ActiveWorkflow(
                    id=sw.workflow_id, stage=sw.stage, draft_id=sw.draft_id
                )
                draft = memory.workflow_drafts.get(sw.draft_id)
                if draft:
                    self.events.emit(memory, "workflow_resumed", target, {"draft_id": sw.draft_id})
                    return draft
        self.events.emit(memory, "workflow_switched", target, {})
        return self.start_workflow(memory, target)

    def set_stage(self, memory: SessionMemory, stage: str) -> None:
        if memory.active_workflow:
            memory.active_workflow.stage = stage

    def lock_submitted(self, memory: SessionMemory, request_id: str) -> None:
        draft = memory.active_draft()
        if not draft:
            return
        draft.locked = True
        draft.status = "submitted"
        draft.submitted_request_id = request_id
        if memory.active_workflow:
            memory.active_workflow.stage = WorkflowStage.SUBMITTED.value
        memory.pending_question = None
        memory.pending_confirmation = None
        self.events.emit(
            memory,
            "submission_completed",
            draft.workflow_id,
            {"request_id": request_id},
        )

    def is_locked(self, memory: SessionMemory) -> bool:
        draft = memory.active_draft()
        return bool(draft and draft.locked)

    def ensure_definition(self, workflow_id: str):
        defn = get_workflow_definition(workflow_id)
        if not defn:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        return defn

    def detect_interrupt(
        self, message: str, memory: SessionMemory, understanding: UnderstandingResult | None = None
    ) -> WorkflowInterrupt | None:
        """Decide if a new intent should interrupt the active workflow (LLM-first)."""
        if understanding and understanding.interrupt_workflow:
            aw = memory.active_workflow
            if aw and understanding.interrupt_workflow != aw.id:
                return WorkflowInterrupt(
                    from_workflow=aw.id,
                    to_workflow=understanding.interrupt_workflow,
                    confidence=understanding.confidence,
                    pending_message=message,
                )
        aw = memory.active_workflow
        if not aw:
            return None
        if is_workflow_interrupt_expense(message, active_workflow=aw.id):
            from chat.services.platform.intent_rules import expense_signal_strength

            return WorkflowInterrupt(
                from_workflow=aw.id,
                to_workflow="expense",
                confidence=expense_signal_strength(message),
                pending_message=message,
            )
        return None

    def arm_switch_confirm(
        self,
        memory: SessionMemory,
        *,
        from_workflow: str,
        to_workflow: str,
        pending_message: str,
    ) -> None:
        memory.pending_confirmation = f"switch:{from_workflow}:{to_workflow}"
        entities = dict(memory.last_entities or {})
        entities["switch_pending_message"] = pending_message
        memory.last_entities = entities

    @staticmethod
    def parse_switch_reply(message: str, from_workflow: str, to_workflow: str) -> str | None:
        """Return ``continue``, ``switch``, or None."""
        low = (message or "").lower().strip()
        if re.search(rf"\b{re.escape(from_workflow)}\b", low) or re.search(r"\bcontinue\b", low):
            return "continue"
        if re.search(rf"\b{re.escape(to_workflow)}\b", low) or re.search(r"\b(create|new|start)\b", low):
            return "switch"
        if re.search(r"\b(yes|ha+h|confirm|ok|thik\s*ache|হ্যাঁ|ঠিক)\b", low):
            return "switch"
        if re.search(r"\b(no|na|naki|cancel|reject)\b", low):
            return "continue"
        return None

    def clear_switch_confirm(self, memory: SessionMemory) -> str:
        pending = str((memory.last_entities or {}).get("switch_pending_message") or "")
        memory.pending_confirmation = None
        if memory.last_entities:
            memory.last_entities.pop("switch_pending_message", None)
        return pending

    @staticmethod
    def switch_confirm_message(from_workflow: str, to_workflow: str = "expense", *, lang: str = "en") -> str:
        if lang == "bn":
            return (
                f"আপনার **{from_workflow}** request এখনও incomplete।\n"
                f"আমি **{to_workflow}** request শনাক্ত করেছি।\n\n"
                f"**{from_workflow}** চালিয়ে যেতে `{from_workflow}` বলুন, "
                f"নতুন **{to_workflow}** শুরু করতে `{to_workflow}` বা **yes** বলুন।"
            )
        return (
            f"You have an unfinished **{from_workflow}** request.\n"
            f"I detected a **{to_workflow}** request.\n\n"
            f"Reply **{from_workflow}** to continue, or **{to_workflow}** / **yes** to switch."
        )

    @staticmethod
    def switch_retry_message(from_workflow: str, to_workflow: str, *, lang: str = "en") -> str:
        if lang == "bn":
            return (
                f"**{from_workflow}** চালিয়ে যেতে `{from_workflow}` লিখুন, "
                f"অথবা expense তৈরি করতে `{to_workflow}` লিখুন।"
            )
        return f"Reply **{from_workflow}** to continue leave, or **{to_workflow}** to create an expense."
