"""Domain LLM routing — one specialized call per active workflow turn."""

from __future__ import annotations

import json

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.expense_llm_mock import mock_expense_llm, _mock_expense_draft_interpreter


def _active_expense_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={"incurred_date": "2026-06-23", "items": [{"category": "lunch", "amount": 200.0}]},
            )
        },
    )


def test_new_expense_start_uses_single_expense_llm_only():
    """'lunch 100 taka' with no active workflow must not call UNDERSTAND + expense-draft."""
    memory = SessionMemory()
    layer = AIUnderstandingLayer()
    scopes: list[str] = []

    with mock_expense_llm():
        from chat.services.llm_client import LLMClient

        client = LLMClient()

        def _track(*, system_prompt: str, user_prompt: str, **kwargs):
            scopes.append(str(kwargs.get("scope") or "default"))
            assert "Understanding Layer" not in system_prompt
            payload = json.loads(user_prompt)
            return _mock_expense_draft_interpreter(str(payload.get("message") or ""), payload)

        client.chat_json.side_effect = _track
        result = layer.understand(
            "lunch 100 taka",
            memory=memory,
            conversation_history=[],
            trace_id="new-expense-single",
            llm=client,
        )

    assert len(scopes) == 1
    assert scopes[0] == "expense-draft"
    assert result.source == "llm_expense"
    assert result.workflow == "expense"
    assert result.field_updates


def test_active_expense_skips_understand_llm():
    memory = _active_expense_memory()
    layer = AIUnderstandingLayer()
    scopes: list[str] = []

    with mock_expense_llm():
        from chat.services.llm_client import LLMClient

        client = LLMClient()

        def _track(*, system_prompt: str, user_prompt: str, **kwargs):
            scopes.append(str(kwargs.get("scope") or "default"))
            assert "Understanding Layer" not in system_prompt
            payload = json.loads(user_prompt)
            return _mock_expense_draft_interpreter(str(payload.get("message") or ""), payload)

        client.chat_json.side_effect = _track
        result = layer.understand(
            "bus 50 taka",
            memory=memory,
            conversation_history=[],
            trace_id="domain-collect",
            llm=client,
        )

    assert len(scopes) == 1
    assert scopes[0] == "expense-draft"
    assert result.source == "llm_expense"
    assert result.workflow == "expense"
