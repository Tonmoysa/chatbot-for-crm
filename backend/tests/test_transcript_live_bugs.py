"""Regression tests from live transcript (greeting, leave dates, expense switch)."""

from __future__ import annotations

import pytest

from chat.services.platform.intent_rules import is_greeting_or_chitchat, is_workflow_interrupt_expense
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)


LEAVE_NARRATIVE_BN = (
    "Actually last kichudin dhore family related kichu urgent issue cholche. "
    "Amar mama onekdin dhore osustho chilen, ekhon take Dhaka te niye treatment korate hobe "
    "ebong family member hisebe amar uposthit thaka khub dorkar. "
    "Ei karone ami Monday (14th July) theke Wednesday (16th July) porjonto "
    "office e attend korte parbo na."
)


def test_greeting_detection():
    assert is_greeting_or_chitchat("hi")
    assert is_greeting_or_chitchat("hello")
    assert is_greeting_or_chitchat("Hello!")
    assert not is_greeting_or_chitchat("hello I need leave")


def test_bare_expense_not_interrupt():
    assert not is_workflow_interrupt_expense("expense", active_workflow="leave")
    assert WorkflowManager.parse_switch_reply("expense", "leave", "expense") == "switch"


def test_switch_confirm_resolves_expense_reply():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting")
    memory.workflow_drafts["default"] = WorkflowDraft(
        workflow_id="leave",
        fields={"leave_type": "annual", "reason": "family"},
    )
    memory.pending_question = PendingQuestion(
        field="start_date",
        prompt="Start date?",
        workflow_id="leave",
        asked_at_turn=1,
    )
    memory.pending_confirmation = "switch:leave:expense"
    memory.last_entities = {"switch_pending_message": "100 taka bus expense"}

    pipeline = WorkflowPipeline()
    msg, decision = pipeline.handle(
        "expense",
        memory=memory,
        pq_decision=__import__(
            "chat.services.pending_question_engine", fromlist=["PendingQuestionDecision"]
        ).PendingQuestionDecision(
            kind=MessageIntentKind.SWITCH_WORKFLOW,
            confidence=0.9,
            reasoning="test",
            source="test",
            blocks_new_workflow=True,
            target_workflow="expense",
        ),
        conversation_history=[],
        trace_id="test-switch-resolve",
    )
    assert memory.active_workflow and memory.active_workflow.id == "expense"
    assert memory.pending_confirmation is None
    assert "expense" in msg.lower() or "bus" in msg.lower() or "100" in msg


def test_leave_narrative_merges_dates_on_understand():
    from unittest.mock import patch

    from chat.services.platform.ai_understanding import AIUnderstandingLayer

    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = False
        u = layer.understand(
            LEAVE_NARRATIVE_BN,
            memory=memory,
            conversation_history=[],
            trace_id="test-leave-dates",
        )
    fields = {upd.field: upd.value for upd in u.field_updates}
    assert "start_date" in fields
    assert "end_date" in fields
    assert u.action != "submit"


def test_pq_greeting_not_out_of_scope():
    engine = PendingQuestionEngine()
    decision = engine.classify(
        "hello",
        memory=SessionMemory(),
        conversation_history=[],
        trace_id="test-hello-pq",
    )
    assert decision.kind != MessageIntentKind.OUT_OF_SCOPE


def test_family_illness_not_sick_leave():
    """Mama osustho = family caregiver leave, NOT employee sick leave."""
    from chat.services.platform.ai_understanding import AIUnderstandingLayer

    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    u = layer._sanitize_leave_result(
        LEAVE_NARRATIVE_BN,
        layer._parse_result(
            {
                "goal": "Start leave",
                "workflow": "leave",
                "action": "start",
                "confidence": 0.9,
                "field_updates": [
                    {"field": "leave_type", "value": "sick"},
                    {"field": "start_date", "value": "2026-07-11"},
                    {"field": "end_date", "value": "2026-07-13"},
                    {"field": "day_scope", "value": "full_day"},
                    {"field": "reason", "value": "Mama osustho"},
                ],
                "reasoning": "Sick leave for family illness",
            },
            source="llm",
        ),
        memory=memory,
    )
    fields = {upd.field: upd.value for upd in u.field_updates}
    assert fields.get("leave_type") is None
    assert fields.get("start_date") == "2026-07-14"
    assert fields.get("end_date") == "2026-07-16"
    assert "reason" in fields


def test_self_sick_may_infer_sick_leave():
    from chat.services.platform.field_extractors.leave import infer_leave_type_from_text

    assert infer_leave_type_from_text("ami onek osustho, kal theke sick leave lagbe") == "sick"
    assert infer_leave_type_from_text("pet betha, doctor dekhabo") == "sick"


def test_medical_document_nai_not_saved():
    from chat.services.platform.field_extractors.leave import parse_medical_document_field

    assert parse_medical_document_field("medical document nai") is None
    assert parse_medical_document_field("false") is None


def test_medical_document_decline_resets_sick_leave():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "sick",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "day_scope": "full_day",
                },
            )
        },
        pending_question=PendingQuestion(
            field="medical_document",
            prompt="Upload medical document",
            workflow_id="leave",
            asked_at_turn=1,
        ),
    )
    pipeline = WorkflowPipeline()
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    msg, decision = pipeline.handle(
        "medical document nai",
        memory=memory,
        pq_decision=pq,
        conversation_history=[],
        trace_id="test-med-decline",
    )
    draft = memory.active_draft()
    assert draft.fields.get("leave_type") is None
    assert "medical_document" not in draft.fields or draft.fields.get("medical_document") in (None, "")
    assert _any(msg, r"annual|sick|lwop|leave type|ধরন")
    assert decision.get("outcome") == "NEEDS_INPUT"


def test_review_arms_submit_confirmation():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "reason": "family emergency",
                },
            )
        },
    )
    pipeline = WorkflowPipeline()
    msg, decision = pipeline._continue_collection(
        memory,
        pipeline.manager.ensure_definition("leave"),
        lang="en",
        prefix="Saved day scope.",
    )
    assert memory.pending_confirmation == "submit"
    assert "yes" in msg.lower()
    assert decision.get("awaiting_confirmation") is True


def test_yes_after_review_submits_leave(monkeypatch):
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="review"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "reason": "family emergency",
                },
            )
        },
        pending_confirmation="submit",
    )
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.95,
        reasoning="submit confirm",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    pipeline = WorkflowPipeline()

    class _CRM:
        def create_request(self, **kwargs):
            return {"request_id": "LV-TEST-1", "record": {}}

    monkeypatch.setattr(
        "chat.services.platform.submission_engine.get_crm_adapter",
        lambda: _CRM(),
    )
    msg, decision = pipeline.handle(
        "yes",
        memory=memory,
        pq_decision=pq,
        conversation_history=[],
        trace_id="test-yes-submit",
        company_id="c1",
        employee_id="e1",
        session_id="s1",
    )
    assert decision.get("outcome") == "SUBMITTED"
    assert memory.active_draft().locked
    assert "LV-TEST-1" in msg


def test_submitted_same_dates_blocked(monkeypatch):
    memory = SessionMemory(
        conversation_facts={
            "submitted_leave_ranges": [
                {"start_date": "2026-07-14", "end_date": "2026-07-16", "request_id": "LV-1"},
            ]
        },
    )
    from chat.services.platform.schemas import UnderstandingResult, FieldUpdate
    from chat.services.platform.pipeline import WorkflowPipeline

    pipeline = WorkflowPipeline()
    u = UnderstandingResult(
        goal="Start leave",
        workflow="leave",
        action="start",
        confidence=0.9,
        field_updates=[
            FieldUpdate(field="start_date", value="2026-07-14"),
            FieldUpdate(field="end_date", value="2026-07-16"),
        ],
        source="rules",
    )
    blocked = pipeline._block_submitted_leave_overlap(memory, u, "en")
    assert blocked is not None
    assert "already submitted" in blocked[0].lower() or "ইতিমধ্যে" in blocked[0]


def _any(text: str, pattern: str) -> bool:
    import re
    return bool(re.search(pattern, text, re.I))
