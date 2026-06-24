"""Phase 7 — SessionStore (transcript + workflow) tests."""

from __future__ import annotations

import pytest

from chat.models import ConversationSession, ConversationTurn
from chat.services.session_memory import SessionMemory, load_session_memory
from chat.services.session_store import SessionStore


@pytest.mark.django_db
def test_session_store_open_returns_bundle():
    store = SessionStore()
    bundle = store.open(company_id="c1", employee_id="e1", session_id="s-open-1")
    assert bundle.session.session_id == "s-open-1"
    assert isinstance(bundle.memory, SessionMemory)
    assert bundle.transcript_lines == []


@pytest.mark.django_db
def test_session_store_commit_turn_persists_workflow_and_transcript():
    store = SessionStore()
    bundle = store.open(company_id="c1", employee_id="e1", session_id="s-commit-1")
    session = bundle.session
    memory = bundle.memory
    memory.last_entities = {"foo": "bar"}

    store.commit_turn(session, memory, user_message="hello", assistant_message="hi there")

    session.refresh_from_db()
    reloaded = load_session_memory(session)
    assert reloaded.turn_count == 1
    assert reloaded.last_entities.get("foo") == "bar"

    turns = list(session.turns.order_by("created_at"))
    assert len(turns) == 2
    assert turns[0].role == ConversationTurn.ROLE_USER
    assert turns[0].content == "hello"
    assert turns[1].role == ConversationTurn.ROLE_ASSISTANT
    assert turns[1].content == "hi there"


@pytest.mark.django_db
def test_session_store_open_includes_prior_transcript():
    store = SessionStore()
    session = ConversationSession.objects.create(
        company_id="c1",
        employee_id="e1",
        session_id="s-history-1",
    )
    ConversationTurn.objects.create(
        session=session,
        role=ConversationTurn.ROLE_USER,
        content="prev question",
    )
    ConversationTurn.objects.create(
        session=session,
        role=ConversationTurn.ROLE_ASSISTANT,
        content="prev answer",
    )

    bundle = store.open(company_id="c1", employee_id="e1", session_id="s-history-1")
    assert bundle.transcript_lines == ["User: prev question", "Assistant: prev answer"]


@pytest.mark.django_db
def test_session_store_commit_increments_turn_count_once():
    store = SessionStore()
    bundle = store.open(company_id="c1", employee_id="e1", session_id="s-count-1")
    session = bundle.session
    memory = bundle.memory

    store.commit_turn(session, memory, user_message="a", assistant_message="b")
    store.commit_turn(session, memory, user_message="c", assistant_message="d")

    session.refresh_from_db()
    assert load_session_memory(session).turn_count == 2
    assert session.turns.count() == 4
