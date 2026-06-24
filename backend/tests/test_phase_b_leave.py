"""Phase B — skip/yes loop fix + multi-day auto full_day."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.leave import (
    apply_leave_derived_fields,
    apply_multi_day_scope_to_fields,
    leave_span_days,
    merge_leave_field_dicts,
)
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.registry import get_workflow_definition
from chat.services.platform.schemas import UnderstandingAction
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    WorkflowDraft,
)
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def test_multi_day_span_auto_full_day():
    fields = apply_multi_day_scope_to_fields(
        {"start_date": "2026-09-14", "end_date": "2026-09-17", "leave_type": "annual"},
        "",
    )
    assert fields["day_scope"] == "full_day"
    assert leave_span_days(fields) == 4


def test_single_day_does_not_auto_full_day_without_dates_span():
    fields = apply_multi_day_scope_to_fields(
        {"start_date": "2026-09-14", "end_date": "2026-09-14"},
        "",
    )
    assert "day_scope" not in fields


def test_merge_leave_field_dicts_sets_multi_day_scope():
    merged = merge_leave_field_dicts(
        {},
        {"start_date": "2026-09-14", "end_date": "2026-09-17", "leave_type": "annual"},
        "14 September theke 17 September annual leave",
    )
    assert merged.get("day_scope") == "full_day"


def test_multi_day_leave_skips_day_scope_question():
    engine = FieldEngine()
    defn = get_workflow_definition("leave")
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "day_scope": "full_day",
                },
            )
        },
    )
    draft = memory.active_draft()
    missing = engine.missing_fields(draft, defn)
    assert "day_scope" not in missing
    pq = engine.next_question(memory, draft, defn, lang="banglish")
    assert pq is None or pq.field != "day_scope"


def _leave_collecting_near_review() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                },
            )
        },
        pending_question=PendingQuestion(
            field="reason",
            prompt="Karon janate chan?",
            workflow_id="leave",
            asked_at_turn=2,
        ),
    )


def test_reason_skip_goes_to_review_not_shuru_korbo():
    memory = _leave_collecting_near_review()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="answer reason",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    with llm_disabled():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "skip",
            memory=memory,
            pq_decision=pq,
            trace_id="phase-b-skip",
            route_source="active",
        )
    low = msg.lower()
    assert "shuru korbo" not in low
    assert "chuti nite chan" not in low
    assert "submit" in low or "review" in low or "porjalochona" in low
    assert memory.pending_confirmation == "submit"


def test_yes_after_leave_clarify_with_draft_starts_not_loops():
    memory = SessionMemory(
        last_entities={"leave_start_clarify": True},
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                },
            )
        },
    )
    apply_leave_derived_fields(memory.workflow_drafts["default"])
    with llm_disabled():
        from chat.services.platform.ai_understanding import AIUnderstandingLayer

        layer = AIUnderstandingLayer()
        u = layer.understand("yes", memory=memory, conversation_history=[], trace_id="phase-b-yes")
    assert u.workflow == "leave"
    assert u.action == UnderstandingAction.START.value

    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="confirm start",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    with llm_disabled():
        msg, _ = handle_with_rules_understanding(
            pipeline,
            "yes",
            memory=memory,
            pq_decision=pq,
            trace_id="phase-b-yes-2",
            route_source="pending",
        )
    assert "shuru korbo" not in msg.lower()
