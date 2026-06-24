"""Phase 3 — Banglish modify patterns, hybrid LLM gap-fill, gatekeeper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.field_extractors.leave import (
    leave_modify_updates_as_dict,
    merge_leave_field_dicts,
)
from chat.services.platform.intent_rules import is_resume_workflow_request
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.schemas import FieldUpdate, PlanOp, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.test_plan_builder import _ctx


def test_kar_ta_reason_modify():
    from tests.helpers.leave_llm_mock import mock_leave_llm, review_memory

    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("kar ta family wedding koro", memory=memory)
    assert mod.get("reason")
    assert "wedding" in mod["reason"].lower()


def test_reason_change_koro_modify():
    from tests.helpers.leave_llm_mock import mock_leave_llm, review_memory

    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("reason change koro family program", memory=memory)
    assert mod.get("reason")
    assert "family" in mod["reason"].lower()


def test_bare_date_koro_modify():
    from tests.helpers.leave_llm_mock import mock_leave_llm, review_memory

    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("3 july koro", memory=memory)
    assert mod.get("start_date") or mod.get("end_date")


def test_shesh_din_end_date_modify():
    from tests.helpers.leave_llm_mock import mock_leave_llm, review_memory

    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("shesh din 5 july koro", memory=memory)
    assert mod.get("end_date") == "2026-07-05" or mod.get("start_date")


def test_resume_review_phrases():
    assert is_resume_workflow_request("review dekhao", workflow_id="leave")
    assert is_resume_workflow_request("abar dekhao", workflow_id="leave")
    assert is_resume_workflow_request("leave e back koro", workflow_id="leave")


def test_merge_gap_fills_leave_type_from_llm():
    merged = merge_leave_field_dicts(
        {},
        {
            "leave_type": "sick",
            "start_date": "2026-08-18",
            "end_date": "2026-08-21",
            "reason": "long narrative should not win",
        },
        "18 August theke 21 August sick leave",
    )
    assert merged.get("leave_type") == "sick"
    assert merged.get("start_date") == "2026-08-18"


def test_merge_llm_leave_type_wins_over_rules():
    merged = merge_leave_field_dicts(
        {"leave_type": "annual", "start_date": "2026-08-22"},
        {"leave_type": "sick"},
        "22 august annual",
    )
    assert merged["leave_type"] == "sick"


def test_gatekeeper_overrides_llm_on_modify_command():
    layer = AIUnderstandingLayer()
    client = MagicMock()
    client.is_configured.return_value = True
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "reason": "family program",
                },
            )
        },
        pending_confirmation="submit",
    )
    llm_weak = UnderstandingResult(
        goal="Chat",
        workflow="none",
        action=UnderstandingAction.NONE.value,
        confidence=0.7,
        is_greeting=True,
        reasoning="Conversational.",
        source="llm",
    )
    with patch.object(layer, "_understand_llm", return_value=llm_weak):
        result = layer.understand(
            "kar ta family wedding koro",
            memory=memory,
            conversation_history=[],
            trace_id="phase3-modify-gatekeeper",
            llm=client,
        )
    assert result.source == "rules_gatekeeper"
    assert result.action == UnderstandingAction.MODIFY.value
    assert any(u.field == "reason" for u in (result.field_updates or []))


def test_review_dekhao_during_submit_shows_review():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "reason": "family program",
                },
            )
        },
        pending_confirmation="submit",
    )
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.REVIEW.value,
        confidence=0.88,
    )
    plan = PlanBuilder.build(
        _ctx(
            user_message="review dekhao",
            pending_confirmation="submit",
            has_pending_confirmation=True,
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_SHOW_REVIEW


def test_bare_date_modify_during_review_keeps_reason():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "reason": "family program",
                },
            )
        },
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="modify",
        source="rules",
        blocks_new_workflow=True,
    )
    from tests.helpers.leave_llm_mock import mock_leave_llm

    with mock_leave_llm():
        handle_with_rules_understanding(
            pipeline,
            "3 july koro",
            memory=memory,
            pq_decision=pq,
            trace_id="phase3-bare-date",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("start_date") == "2026-07-03"
    assert draft.fields.get("reason") == "family program"
