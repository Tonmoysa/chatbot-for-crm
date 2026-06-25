"""Expense review — bare line items like 'bike 120 taka' must append, not re-show submit."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import (
    expense_turn_to_field_updates,
    filter_expense_updates_for_review,
)
from chat.services.platform.pipeline import PlanBuilder, WorkflowPipeline
from chat.services.platform.schemas import FieldUpdate, PlanOp, TurnContext, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def _lunch_review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [{"category": "lunch", "amount": 100.0}],
                },
            )
        },
        pending_confirmation="submit",
    )


def test_bike_line_parses_at_review_with_llm_off():
    memory = _lunch_review_memory()
    with llm_disabled():
        turn, updates = expense_turn_to_field_updates("bike 120 taka", memory)
    assert turn.get("intent") == "add"
    assert updates
    assert updates[0].action == "append"
    assert updates[0].value.get("category") == "bike"
    assert float(updates[0].value.get("amount") or 0) == 120.0


def test_filter_review_passes_append_updates():
    memory = _lunch_review_memory()
    updates = [
        FieldUpdate(
            field="items",
            value={"category": "bike", "amount": 120.0},
            action="append",
        )
    ]
    kept = filter_expense_updates_for_review(updates, "bike 120 taka", memory=memory)
    assert len(kept) == 1
    assert kept[0].action == "append"


def test_plan_builder_bike_line_routes_collect_during_submit():
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.55,
        entities={"expense_intent": "conversation", "expense_llm_degraded": True},
    )
    plan = PlanBuilder.build(
        TurnContext(
            trace_id="t1",
            session_id="s1",
            company_id="c1",
            employee_id="e1",
            user_message="bike 120 taka",
            conversation_history=(),
            document_text=None,
            idempotency_key="",
            user_language="bn",
            reply_language="banglish",
            today_iso="2026-06-24",
            turn_count_at_start=2,
            memory_schema_version=1,
            active_workflow_id="expense",
            active_workflow_stage="confirm_submit",
            draft_id="default",
            pending_question_field=None,
            pending_question_prompt=None,
            pending_question_workflow_id=None,
            pending_confirmation="submit",
            draft_snapshot={
                "workflow_id": "expense",
                "fields": {
                    "incurred_date": "2026-06-24",
                    "items": [{"category": "lunch", "amount": 100.0}],
                },
            },
            suspended_workflows=(),
            conversation_facts={},
            has_active_workflow=True,
            has_pending_question=False,
            has_pending_confirmation=True,
            draft_locked=False,
            wizard_active=False,
        ),
        TurnDecision(pq=None, understanding=u, route_source="pending"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_COLLECT


def test_bike_line_appended_at_review_not_resubmit_prompt():
    memory = _lunch_review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.5,
        reasoning="add line at review",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with llm_disabled():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "bike 120 taka",
            memory=memory,
            pq_decision=pq,
            trace_id="review-bike-add",
            route_source="pending",
        )
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 2
    assert any(i.get("category") == "bike" and float(i.get("amount") or 0) == 120.0 for i in items)
    assert "Submit korar age" not in (msg or "")
    assert decision.get("outcome") != "NEEDS_CLARIFICATION"
