"""Conversation scenario test helper."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chat.services.orchestrator import ChatOrchestrator


@pytest.fixture
def chat_runner(db):
    """Orchestrator with LLM disabled (rules-only)."""
    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm, patch("chat.services.conversational.LLMClient") as conv_llm:
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
        conv_llm.return_value.is_configured.return_value = False
        orch = ChatOrchestrator()
        state = {"session_id": ""}

        def send(message: str) -> str:
            result = orch.run_chat(
                message=message,
                session_id=state["session_id"] or None,
                company_id="co-test",
                employee_id="emp-test",
                trace_id="scenario-test",
            )
            state["session_id"] = result.get("_session_id") or state["session_id"]
            return (result.get("response") or {}).get("message") or ""

        yield send
