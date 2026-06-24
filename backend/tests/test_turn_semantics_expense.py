"""Turn semantics helpers for expense conversation context."""

from __future__ import annotations

from chat.services.platform.turn_semantics import (
    expense_conversation_payload,
    recent_user_messages,
)


def test_recent_user_messages_from_history():
    history = [
        "User: dhanmondi to mirpur",
        "Assistant: route?",
        "User: ami tomake route diyechi",
    ]
    assert recent_user_messages(history, limit=2) == [
        "dhanmondi to mirpur",
        "ami tomake route diyechi",
    ]


def test_expense_conversation_payload_shape():
    history = ["User: bus 120 taka", "Assistant: route?"]
    payload = expense_conversation_payload(history)
    assert payload["conversation_history"] == history
    assert payload["recent_user_messages"] == ["bus 120 taka"]
    assert "route" in (payload["last_assistant_message"] or "").lower()
