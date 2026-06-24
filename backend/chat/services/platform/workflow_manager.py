"""Workflow lifecycle — start, pause, resume, switch, lock."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from chat.services.platform.event_store import EventStore
from chat.services.platform.intent_rules import (
    is_workflow_interrupt_expense,
    is_workflow_interrupt_leave,
    leave_signal_strength,
)
from chat.services.platform.registry import get_workflow_definition
from chat.services.platform.schemas import WorkflowStage, UnderstandingResult
from chat.services.session_memory import (
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
    apply_state_patches,
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

    @staticmethod
    def _apply_patches(
        memory: SessionMemory,
        patches: list[dict[str, Any]],
        *,
        state: StatePatchBuffer | None = None,
    ) -> None:
        if state is not None:
            for patch in patches:
                op = str(patch.get("op") or "")
                kwargs = {k: v for k, v in patch.items() if k != "op"}
                state.push(op, **kwargs)
        else:
            apply_state_patches(memory, patches)

    def start_workflow(
        self,
        memory: SessionMemory,
        workflow_id: str,
        *,
        state: StatePatchBuffer | None = None,
    ) -> WorkflowDraft:
        wf_id = workflow_id.strip().lower()
        self._apply_patches(memory, [{"op": "start_workflow", "value": wf_id}], state=state)
        self.events.emit(memory, "workflow_started", wf_id, {"draft_id": "default"})
        return memory.active_draft() or memory.workflow_drafts["default"]

    def suspend_active(self, memory: SessionMemory, *, state: StatePatchBuffer | None = None) -> None:
        if not memory.active_workflow:
            return
        aw = memory.active_workflow
        self._apply_patches(memory, [{"op": "suspend_active_workflow"}], state=state)
        self.events.emit(memory, "workflow_paused", aw.id, {"draft_id": aw.draft_id})

    def switch_to(
        self,
        memory: SessionMemory,
        target_workflow_id: str,
        *,
        state: StatePatchBuffer | None = None,
    ) -> WorkflowDraft:
        target = target_workflow_id.strip().lower()
        resume = next(
            (sw for sw in memory.suspended_workflows if sw.workflow_id == target),
            None,
        )
        self._apply_patches(memory, [{"op": "switch_to_workflow", "value": target}], state=state)
        draft = memory.active_draft()
        if resume:
            self.events.emit(memory, "workflow_resumed", target, {"draft_id": resume.draft_id})
            if draft:
                return draft
        self.events.emit(memory, "workflow_switched", target, {})
        return draft or self.start_workflow(memory, target, state=state)

    def set_stage(self, memory: SessionMemory, stage: str, *, state: StatePatchBuffer | None = None) -> None:
        self._apply_patches(memory, [{"op": "set_active_stage", "value": stage}], state=state)

    def lock_submitted(
        self,
        memory: SessionMemory,
        request_id: str,
        *,
        state: StatePatchBuffer | None = None,
    ) -> None:
        draft = memory.active_draft()
        if not draft:
            return
        self._apply_patches(
            memory,
            [{"op": "lock_submitted_draft", "request_id": request_id}],
            state=state,
        )
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
        aw = memory.active_workflow
        if understanding and aw:
            if understanding.interrupt_workflow and understanding.interrupt_workflow != aw.id:
                return WorkflowInterrupt(
                    from_workflow=aw.id,
                    to_workflow=understanding.interrupt_workflow,
                    confidence=understanding.confidence,
                    pending_message=message,
                )
            target = (understanding.workflow or "").strip().lower()
            if target and target not in ("none", "") and target != aw.id:
                return WorkflowInterrupt(
                    from_workflow=aw.id,
                    to_workflow=target,
                    confidence=understanding.confidence,
                    pending_message=message,
                )
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
        if is_workflow_interrupt_leave(message, active_workflow=aw.id):
            return WorkflowInterrupt(
                from_workflow=aw.id,
                to_workflow="leave",
                confidence=leave_signal_strength(message),
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
        state: StatePatchBuffer | None = None,
    ) -> None:
        self._apply_patches(
            memory,
            [
                {
                    "op": "arm_switch_confirm",
                    "from_workflow": from_workflow,
                    "to_workflow": to_workflow,
                    "pending_message": pending_message,
                }
            ],
            state=state,
        )

    @staticmethod
    def parse_switch_reply(message: str, from_workflow: str, to_workflow: str) -> str | None:
        """Return ``continue``, ``switch``, or None."""
        low = (message or "").lower().strip()
        if (
            re.search(rf"\b{re.escape(from_workflow)}\b", low)
            or re.search(r"\bcontinue\b", low)
            or re.search(rf"\b{re.escape(from_workflow)}\s+e\s+(?:jao|jaw|jai)\b", low)
            or re.search(rf"\b{re.escape(from_workflow)}\s+(?:e|te)\s+(?:jai|jao|fire\s+jao)\b", low)
        ):
            return "continue"
        if (
            re.search(rf"\b{re.escape(to_workflow)}\b", low)
            or re.search(r"\b(create|new|start)\b", low)
            or re.search(rf"\b{re.escape(to_workflow)}\s+e\s+(?:jao|jaw|jai)\b", low)
            or re.search(rf"\b{re.escape(to_workflow)}\s+(?:e|te)\s+(?:jai|jao|fire\s+jao)\b", low)
        ):
            return "switch"
        if re.search(r"\b(yes|ha+h|confirm|ok|thik\s*ache|হ্যাঁ|ঠিক)\b", low):
            return "switch"
        if re.search(r"\b(no|na|naki|cancel|reject)\b", low):
            return "continue"
        return None

    def clear_switch_confirm(self, memory: SessionMemory, *, state: StatePatchBuffer | None = None) -> str:
        pending = str((memory.last_entities or {}).get("switch_pending_message") or "")
        self._apply_patches(memory, [{"op": "clear_switch_confirm"}], state=state)
        return pending

    @staticmethod
    def switch_confirm_message(from_workflow: str, to_workflow: str = "expense", *, lang: str = "en") -> str:
        from chat.services.platform.response_composer import localized

        return localized(
            lang,
            en=(
                f"You still have an unfinished **{from_workflow}** request.\n"
                f"I also noticed you want **{to_workflow}**.\n\n"
                f"Reply **{from_workflow}** to keep working on it, "
                f"or **{to_workflow}** / **yes** to pause **{from_workflow}** and switch."
            ),
            bn=(
                f"আপনার **{from_workflow}** request এখনও incomplete।\n"
                f"আমি **{to_workflow}** request-ও শনাক্ত করেছি।\n\n"
                f"**{from_workflow}** চালিয়ে যেতে **{from_workflow}** বলুন, "
                f"অথবা **{to_workflow}** / **ha** বললে **{from_workflow}** pause করে switch করব।"
            ),
            banglish=(
                f"Apnar **{from_workflow}** request ekhono incomplete.\n"
                f"**{to_workflow}** er kothao bolchen.\n\n"
                f"**{from_workflow}** continue korte **{from_workflow}** bolen, "
                f"na **{to_workflow}** / **ha** bole switch korun — **{from_workflow}** pause hobe."
            ),
        )

    @staticmethod
    def switch_retry_message(from_workflow: str, to_workflow: str, *, lang: str = "en") -> str:
        from chat.services.platform.response_composer import localized

        return localized(
            lang,
            en=(
                f"Reply **{from_workflow}** to continue **{from_workflow}**, "
                f"or **{to_workflow}** / **yes** to switch."
            ),
            bn=(
                f"**{from_workflow}** চালিয়ে যেতে **{from_workflow}** বলুন, "
                f"অথবা switch করতে **{to_workflow}** / **ha** বলুন।"
            ),
            banglish=(
                f"**{from_workflow}** continue korte **{from_workflow}** bolen, "
                f"ba switch korte **{to_workflow}** / **ha** bolen."
            ),
        )
