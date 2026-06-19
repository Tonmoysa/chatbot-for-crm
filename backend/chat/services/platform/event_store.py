"""Append-only workflow events stored in session memory."""

from __future__ import annotations

from chat.services.platform.schemas import WorkflowEvent
from chat.services.session_memory import SessionMemory


class EventStore:
    def emit(
        self,
        memory: SessionMemory,
        event_type: str,
        workflow_id: str = "",
        payload: dict | None = None,
    ) -> None:
        memory.events.append(
            WorkflowEvent(
                event_type=event_type,
                workflow_id=workflow_id,
                payload=dict(payload or {}),
                turn=memory.turn_count,
            )
        )
        memory.last_action = event_type

    def list_events(self, memory: SessionMemory, limit: int = 50) -> list[dict]:
        return [e.to_dict() for e in memory.events[-limit:]]
