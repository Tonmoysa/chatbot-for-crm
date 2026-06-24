"""Phase 7 — unified session store (transcript + workflow state).

Production path: load via ``open()``, persist via ``commit_turn()`` only.
Transcript is read-only context for Understanding; workflow JSON is written
only after the state reducer has applied patches for the turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction

from chat.models import ConversationSession
from chat.services.memory_store import ConversationMemoryStore
from chat.services.session_memory import SessionMemory, load_session_memory, save_session_memory


@dataclass(frozen=True)
class SessionBundle:
    """Loaded session: DB row + workflow memory + transcript lines (context only)."""

    session: ConversationSession
    memory: SessionMemory
    transcript_lines: list[str]


class SessionStore:
    """Single facade for session load and turn commit."""

    def __init__(self, *, max_transcript_turns: int = 30) -> None:
        self._transcript = ConversationMemoryStore(max_turns=max_transcript_turns)

    def open(
        self,
        *,
        company_id: str,
        employee_id: str,
        session_id: str = "",
    ) -> SessionBundle:
        session = self._transcript.get_or_create_session(
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
        )
        memory = load_session_memory(session)
        lines = self._transcript.recent_context_lines(session)
        return SessionBundle(session=session, memory=memory, transcript_lines=lines)

    @transaction.atomic
    def commit_turn(
        self,
        session: ConversationSession,
        memory: SessionMemory,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Persist one completed turn: workflow state then transcript (atomic)."""
        memory.turn_count += 1
        save_session_memory(session, memory)
        self._transcript.append_transcript_turn(
            session,
            user_message=user_message,
            assistant_message=assistant_message,
        )
