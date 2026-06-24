"""Conversation scenario test helper."""

from __future__ import annotations

import pytest

from chat.services.orchestrator import ChatOrchestrator
from tests.helpers.yaml_scenario_runner import llm_disabled


@pytest.fixture
def chat_runner(db):
    """Orchestrator with LLM disabled (rules-only)."""
    with llm_disabled():
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
