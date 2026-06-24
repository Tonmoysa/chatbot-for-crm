"""Phase 1 — workflow navigation beats pending slot; parallel block shows draft."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chat.services.platform.intent_rules import (
    is_same_workflow_navigation,
    is_switch_request,
    is_workflow_show_request,
)
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.response_composer import ResponseComposer
from chat.services.platform.schemas import PlanOp, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.test_plan_builder import _ctx


def _leave_memory_with_pending_reason() -> SessionMemory:
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
                    "day_scope": "full_day",
                },
            ),
        },
    )


@pytest.fixture
def engine() -> PendingQuestionEngine:
    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm:
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
        yield PendingQuestionEngine()


def test_workflow_show_request_covers_transcript_phrases():
    assert is_workflow_show_request("leave er summery ta daw", workflow_id="leave")
    assert is_workflow_show_request("okay leave e back koro", workflow_id="leave")
    assert is_workflow_show_request("where is my leave?", workflow_id="leave")


def test_same_workflow_navigation_not_switch():
    assert is_same_workflow_navigation("leave e jao", active_workflow_id="leave")
    assert not is_switch_request("leave e jao", active_workflow_id="leave")


def test_summary_during_pending_reason_is_show_review(engine):
    from chat.services.platform.ai_understanding import AIUnderstandingLayer

    memory = _leave_memory_with_pending_reason()
    layer = AIUnderstandingLayer()
    understanding = layer.understand(
        "leave er summery ta daw",
        memory=memory,
        conversation_history=[],
        trace_id="phase1-summary",
    )
    decision = engine.classify(
        "leave er summery ta daw",
        memory=memory,
        conversation_history=[],
        trace_id="phase1-summary",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.SHOW_REVIEW
    assert understanding.action == UnderstandingAction.REVIEW.value


def test_plan_builder_show_review_during_active_leave():
    from chat.services.pending_question_engine import PendingQuestionDecision

    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.REVIEW.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(
            user_message="leave er summery ta daw",
            active_workflow_id="leave",
            has_active_workflow=True,
            pending_question_field="reason",
            has_pending_question=True,
        ),
        TurnDecision(
            pq=PendingQuestionDecision(
                kind=MessageIntentKind.SHOW_REVIEW,
                confidence=0.92,
                reasoning="show draft",
                source="rules",
                blocks_new_workflow=True,
                target_workflow="leave",
            ),
            understanding=u,
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_SHOW_REVIEW


def test_leave_e_jao_shows_review_not_empty():
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = _leave_memory_with_pending_reason()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.SHOW_REVIEW,
        confidence=0.92,
        reasoning="same-workflow navigation",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    msg, envelope = handle_with_rules_understanding(
        pipeline,
        "leave e jao",
        memory=memory,
        pq_decision=pq,
        trace_id="phase1-leave-e-jao",
        route_source="active",
    )
    assert msg
    assert "Leave" in msg or "leave" in msg.lower()
    assert envelope.get("outcome") in ("NEEDS_INPUT", "INFORMATIONAL")


def test_bare_modify_not_stored_as_reason():
    from chat.services.platform.field_extractors.leave import (
        is_garbage_leave_reason,
        parse_leave_field,
    )

    assert is_garbage_leave_reason("modify")
    assert parse_leave_field("modify", "reason") is None


def test_parallel_block_includes_draft_summary():
    memory = _leave_memory_with_pending_reason()
    composer = ResponseComposer()
    msg = composer.active_leave_parallel_block(memory, lang="en")
    assert "active" in msg.lower()
    assert "2026-08-05" in msg or "05 August" in msg
    assert "summary" in msg.lower()


@pytest.mark.django_db
def test_orchestrator_summary_during_pending_reason_not_reason_loop():
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
            company_id="co-p1",
            employee_id="emp-p1",
            trace_id="orch-phase1-seed",
        )
        sid = out1.get("_session_id") or ""

        out = orch.run_chat(
            message="leave er summery ta daw",
            session_id=sid,
            company_id="co-p1",
            employee_id="emp-p1",
            trace_id="orch-phase1-summary",
        )

    msg = (out.get("response") or {}).get("message") or ""
    assert msg
    assert "clear leave reason" not in msg.lower()
    assert "summery" not in msg.lower() or "summary" in msg.lower() or "august" in msg.lower()
