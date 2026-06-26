"""Phase 2 — answers_pending_field, pending_kind, contextual clarification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.platform.turn_semantics import (
    enrich_answers_pending_field,
    infer_pending_kind,
    is_workflow_meta_complaint,
)
from chat.services.platform.response_composer import ResponseComposer
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
    build_turn_context,
)


def _memory_pending_reason() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        pending_question=PendingQuestion(
            field="reason",
            prompt="What is the reason for your leave?",
            workflow_id="leave",
        ),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "start_date": "2026-08-05",
                    "end_date": "2026-08-09",
                },
            ),
        },
    )


def test_infer_pending_kind():
    memory = _memory_pending_reason()
    assert infer_pending_kind(memory) == "answer_pending"
    memory.pending_question = None
    assert infer_pending_kind(memory) is None


def test_enrich_summary_not_slot_answer():
    memory = _memory_pending_reason()
    result = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.7,
        field_updates=[],
    )
    out = enrich_answers_pending_field("leave er summery ta daw", memory, result)
    assert out.answers_pending_field is False
    assert out.action == UnderstandingAction.REVIEW.value
    assert out.field_updates == []


def test_enrich_leave_summary_during_expense_submit_review():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit", draft_id="expense"),
        pending_confirmation="submit",
        workflow_drafts={
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-26",
                    "items": [{"category": "lunch", "amount": 200.0}],
                },
            ),
            "leave": WorkflowDraft(
                workflow_id="leave",
                fields={"reason": "Leave tomorrow", "leave_type": "sick"},
            ),
        },
        suspended_workflows=[
            {"workflow_id": "leave", "stage": "collecting", "draft_id": "leave", "suspended_at_turn": 12},
        ],
    )
    result = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.55,
        entities={"expense_intent": "conversation"},
        field_updates=[],
    )
    out = enrich_answers_pending_field("leave er summery ta daw", memory, result)
    assert out.action == UnderstandingAction.REVIEW.value
    assert out.workflow == "leave"
    assert (out.entities or {}).get("show_workflow_target") == "leave"


def test_enrich_osusto_is_slot_answer():
    memory = _memory_pending_reason()
    result = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.8,
    )
    out = enrich_answers_pending_field("osusto", memory, result)
    assert out.answers_pending_field is True


def test_meta_complaint_detection():
    assert is_workflow_meta_complaint("reason keno modify korcho?")
    assert is_workflow_meta_complaint("tumi amar context bujhtecho nah")


def test_contextual_meta_response_shows_draft():
    memory = _memory_pending_reason()
    composer = ResponseComposer()
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
        entities={"meta_complaint": True},
    )
    msg = composer.clarification(u, lang="en", memory=memory)
    assert "current leave draft" in msg.lower() or "leave draft" in msg.lower()
    assert "chuti nite chan" not in msg.lower()
    assert "Mone hocche" not in msg


def test_build_turn_context_includes_last_assistant_message():
    memory = SessionMemory()
    ctx = build_turn_context(
        message="tahole summary daw",
        memory=memory,
        conversation_history=[
            "User: hi",
            "Assistant: You already have an active leave request.",
        ],
        trace_id="t2",
        session_id="s2",
        company_id="c2",
        employee_id="e2",
    )
    assert ctx.last_assistant_message == "You already have an active leave request."


@pytest.fixture
def engine() -> PendingQuestionEngine:
    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm:
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
        yield PendingQuestionEngine()


def test_pqe_meta_complaint_not_answer_pending(engine):
    memory = _memory_pending_reason()
    layer = AIUnderstandingLayer()
    understanding = layer.understand(
        "tumi bolcho active leave ache, tahole summary daw",
        memory=memory,
        conversation_history=[
            "Assistant: You already have an active leave request.",
        ],
        trace_id="phase2-meta",
        pending_kind="answer_pending",
    )
    decision = engine.classify(
        "tumi bolcho active leave ache, tahole summary daw",
        memory=memory,
        conversation_history=[],
        trace_id="phase2-meta",
        understanding=understanding,
    )
    assert decision.kind in (
        MessageIntentKind.SHOW_REVIEW,
        MessageIntentKind.CLARIFICATION_NEEDED,
    )
    assert decision.kind != MessageIntentKind.ANSWER_PENDING


def test_gatekeeper_overrides_llm_collect_on_summary():
    layer = AIUnderstandingLayer()
    memory = _memory_pending_reason()
    client = MagicMock()
    client.is_configured.return_value = True
    llm_wrong = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.85,
        answers_pending_field=True,
        field_updates=[],
        reasoning="User gave reason.",
        source="llm",
    )
    with patch.object(layer, "_understand_llm", return_value=llm_wrong):
        result = layer.understand(
            "leave er summery ta daw",
            memory=memory,
            conversation_history=[],
            trace_id="phase2-gatekeeper",
            llm=client,
            pending_kind="answer_pending",
        )
    assert result.source == "rules_gatekeeper"
    assert result.answers_pending_field is False
    assert result.action == UnderstandingAction.REVIEW.value


@pytest.mark.django_db
def test_orchestrator_passes_pending_kind_and_contextual_clarify():
    from chat.services.orchestrator import ChatOrchestrator

    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm, patch("chat.services.conversational.LLMClient") as conv_llm:
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
        conv_llm.return_value.is_configured.return_value = False

        orch = ChatOrchestrator()
        out1 = orch.run_chat(
            message="annual leave 5 august theke 9 august",
            session_id=None,
            company_id="co-p2",
            employee_id="emp-p2",
            trace_id="orch-p2-seed",
        )
        sid = out1.get("_session_id") or ""

        out = orch.run_chat(
            message="reason keno modify korcho?",
            session_id=sid,
            company_id="co-p2",
            employee_id="emp-p2",
            trace_id="orch-p2-meta",
        )

    msg = (out.get("response") or {}).get("message") or ""
    assert msg
    assert "Mone hocche apni chuti nite chan" not in msg
    assert "chuti nite chan" not in msg.lower() or "summary" in msg.lower()
