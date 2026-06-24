"""Phase 3 — Understanding layer is SSOT for message interpretation."""

from __future__ import annotations

from unittest.mock import patch

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.schemas import UnderstandingAction
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import SessionMemory


def test_understanding_detects_greeting():
    layer = AIUnderstandingLayer()
    u = layer._understand_rules("hello", memory=SessionMemory(), pending_kind=None)
    assert u.is_greeting is True
    assert u.action == UnderstandingAction.NONE.value


def test_understanding_detects_status_query():
    layer = AIUnderstandingLayer()
    u = layer._understand_rules(
        "What is the status of MOCK-12345?",
        memory=SessionMemory(),
        pending_kind=None,
    )
    assert u.action == UnderstandingAction.STATUS.value


def test_understanding_detects_out_of_scope():
    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    u = layer._understand_rules("When is Eid this year?", memory=memory, pending_kind=None)
    assert u.is_out_of_scope is True


def test_classify_maps_understanding_to_policy():
    engine = PendingQuestionEngine()
    layer = AIUnderstandingLayer()
    u = layer._understand_rules(
        "What is the reimbursement policy?",
        memory=SessionMemory(),
        pending_kind=None,
    )
    decision = engine.classify(
        "What is the reimbursement policy?",
        memory=SessionMemory(),
        conversation_history=[],
        trace_id="ssot-policy",
        understanding=u,
    )
    assert decision.kind == MessageIntentKind.ASK_POLICY


def test_orchestrator_skips_understanding_for_today_date(db):
    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm, patch(
        "chat.services.conversational.LLMClient"
    ) as conv_llm:
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
        conv_llm.return_value.is_configured.return_value = False

        from chat.services.orchestrator import ChatOrchestrator

        calls: list[str] = []

        def _track_understand(*args, **kwargs):
            calls.append("understand")
            raise AssertionError("understand() must not run for today-date shortcut")

        orch = ChatOrchestrator()
        with patch.object(orch.understanding_layer, "understand", side_effect=_track_understand):
            out = orch.run_chat(
                message="What is today's date?",
                session_id=None,
                company_id="co-test",
                employee_id="emp-test",
                trace_id="ssot-today",
            )
        assert calls == []
        assert "calendar_date" in str(out.get("entities") or {})
