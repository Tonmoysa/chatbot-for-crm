"""Leave modify mode — Banglish commands during review/collect (LLM-driven)."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.leave import (
    is_garbage_leave_reason,
    is_leave_modify_message,
    leave_modify_updates_as_dict,
)
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.schemas import PlanOp, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft, build_turn_context
from tests.helpers.leave_llm_mock import mock_leave_llm, review_memory
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def test_parse_reason_modify_banglish():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("reason ta familly program koro", memory=memory)
    assert mod.get("reason")
    assert "familly" in mod["reason"].lower() or "family" in mod["reason"].lower()
    assert "koro" not in mod["reason"].lower()


def test_parse_reason_modify_poriborton():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("reason ta poriborton kore family program koro", memory=memory)
    assert mod.get("reason")
    assert "family" in mod["reason"].lower()


def test_parse_date_range_modify():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("date ta 23 august theke 25th august koro", memory=memory)
    assert mod.get("start_date") == "2026-08-23"
    assert mod.get("end_date") == "2026-08-25"
    assert "reason" not in mod


def test_parse_leave_type_modify():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("leave type ta lwop koro", memory=memory)
    assert mod.get("leave_type") == "lwop"
    assert "reason" not in mod


def test_is_leave_modify_message():
    memory = review_memory()
    with mock_leave_llm():
        assert is_leave_modify_message("reason ta family program koro", memory=memory)
        assert is_leave_modify_message("date ta 23 august koro", memory=memory)
        assert not is_leave_modify_message("amar 22 august leave lagbe", memory=memory)


def test_garbage_reason_detection():
    assert is_garbage_leave_reason("date ta 23 august theke 25th august koro")
    assert is_garbage_leave_reason("leave")
    assert is_garbage_leave_reason("end date 7 july hobe")
    assert is_garbage_leave_reason("leave e back koro..ami kichu besoy modify korbo")
    assert not is_garbage_leave_reason("Family program")


def test_parse_end_date_only_modify():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("end date 7 july hobe", memory=memory)
    assert mod.get("end_date") == "2026-07-07"
    assert "start_date" not in mod
    assert "reason" not in mod


def test_parse_last_date_modify():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("last date 3 july koro", memory=memory)
    assert mod.get("end_date") == "2026-07-03"
    assert "start_date" not in mod


def test_parse_last_date_mody_kore():
    memory = review_memory()
    with mock_leave_llm():
        mod = leave_modify_updates_as_dict("last date ta mody kore 3 july koro", memory=memory)
    assert mod.get("end_date") == "2026-07-03"


def test_end_date_modify_keeps_reason_and_start():
    memory = review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        handle_with_rules_understanding(
            pipeline,
            "end date 7 july hobe",
            memory=memory,
            pq_decision=pq,
            trace_id="test-end-date-only",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("start_date") == "2026-06-29"
    assert draft.fields.get("end_date") == "2026-07-07"
    assert draft.fields.get("reason") == "Father unwell; Hospital/treatment visit"


def test_back_koro_message_does_not_overwrite_reason():
    memory = review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.55,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        handle_with_rules_understanding(
            pipeline,
            "leave e back koro..ami kichu besoy modify korbo",
            memory=memory,
            pq_decision=pq,
            trace_id="test-back-no-reason",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("reason") == "Father unwell; Hospital/treatment visit"


def test_garbage_modify_during_review_returns_error_not_reason_overwrite():
    memory = review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        msg, _ = handle_with_rules_understanding(
            pipeline,
            "last date 3 july koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-last-date-review",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("reason") == "Father unwell; Hospital/treatment visit"
    assert draft.fields.get("end_date") == "2026-07-03"
    assert "reason" in msg.lower() or "review" in msg.lower() or "yes" in msg.lower()


def test_plan_builder_modify_during_submit():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.MODIFY.value,
        confidence=0.9,
        field_updates=[],
    )
    ctx = build_turn_context(
        message="date ta 23 august koro",
        memory=SessionMemory(
            active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit", draft_id="default"),
            workflow_drafts={
                "default": WorkflowDraft(
                    workflow_id="leave",
                    fields={
                        "leave_type": "annual",
                        "day_scope": "full_day",
                        "start_date": "2026-08-22",
                        "end_date": "2026-08-22",
                        "reason": "leave",
                    },
                )
            },
            pending_confirmation="submit",
        ),
        conversation_history=[],
        trace_id="t1",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    plan = PlanBuilder.build(
        ctx,
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert PlanOp.WORKFLOW_MODIFY in plan.ops


def test_leave_modify_during_review_updates_reason_only():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "end_date": "2026-08-22",
                    "reason": "leave",
                },
            )
        },
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "reason ta family program koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-leave-modify-reason",
            route_source="active",
        )
    draft = memory.active_draft()
    assert "family" in (draft.fields.get("reason") or "").lower()
    assert "koro" not in (draft.fields.get("reason") or "").lower()
    assert decision.get("awaiting_confirmation") is True
    assert "review" in msg.lower() or "yes" in msg.lower()


def test_leave_modify_date_keeps_reason():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "end_date": "2026-08-22",
                    "reason": "Family program",
                },
            )
        },
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        handle_with_rules_understanding(
            pipeline,
            "date ta 23 august koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-leave-modify-date",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("start_date") == "2026-08-23"
    assert draft.fields.get("reason") == "Family program"


def test_leave_review_modify_start_date_surur_din():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "urgent family issue",
                },
            )
        },
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        handle_with_rules_understanding(
            pipeline,
            "surur din ta 13th september",
            memory=memory,
            pq_decision=pq,
            trace_id="test-surur-din",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("start_date") == "2026-09-13"
    assert draft.fields.get("end_date") == "2026-09-17"


def test_leave_review_modify_start_date_suru_tarik():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "lwop",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "dadi onekdin dhore osustho hobe",
                },
            )
        },
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    with mock_leave_llm():
        handle_with_rules_understanding(
            pipeline,
            "leave suru hobe 13 tarik theke",
            memory=memory,
            pq_decision=pq,
            trace_id="test-suru-tarik",
            route_source="active",
        )
    draft = memory.active_draft()
    assert draft.fields.get("start_date") == "2026-09-13"


def test_validator_blocks_garbage_reason_on_submit():
    defn = __import__(
        "chat.services.platform.registry", fromlist=["get_workflow_definition"]
    ).get_workflow_definition("leave")
    draft = WorkflowDraft(
        workflow_id="leave",
        fields={
            "leave_type": "annual",
            "day_scope": "full_day",
            "start_date": "2026-08-23",
            "end_date": "2026-08-25",
            "reason": "date ta 23 august theke 25th august koro",
        },
    )
    errors = ValidationEngine().validate(draft, defn, lang="en")
    assert any("reason" in e.lower() for e in errors)


def test_yes_after_leave_clarify_starts_workflow():
    memory = SessionMemory(last_entities={"leave_start_clarify": True})
    with llm_disabled():
        layer = AIUnderstandingLayer()
        u = layer.understand("yes", memory=memory, conversation_history=[], trace_id="leave-clarify-yes")
    assert u.workflow == "leave"
    assert u.action == UnderstandingAction.START.value

    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="confirm start after clarify",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    handle_with_rules_understanding(
        pipeline,
        "yes",
        memory=memory,
        pq_decision=pq,
        trace_id="leave-clarify-yes-2",
        route_source="pending",
    )
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "leave"
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields.get("reason") != "yes"
