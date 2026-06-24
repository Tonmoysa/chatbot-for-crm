"""Expense review submit — bare 'submit koro' must confirm, not recap summary."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import interpret_expense_draft_turn
from chat.services.platform.intent_rules import parse_submit_workflow
from chat.services.platform.pipeline import PlanBuilder
from chat.services.platform.schemas import PlanOp, TurnContext, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def _review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 140.0},
                        {"category": "metro", "amount": 90.0},
                        {"category": "bus", "amount": 40.0},
                    ],
                },
            )
        },
        pending_confirmation="submit",
    )


def test_parse_submit_bare_phrase_at_expense_review():
    assert parse_submit_workflow("submit koro", active_workflow_id="expense") == "expense"


def test_interpret_submit_koro_at_review_is_confirm():
    with llm_disabled():
        turn = interpret_expense_draft_turn("submit koro", memory=_review_memory())
    assert turn.get("intent") == "confirm"


def test_plan_builder_submit_koro_during_expense_review():
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
            user_message="submit koro",
            conversation_history=(),
            document_text=None,
            idempotency_key="",
            user_language="bn",
            reply_language="banglish",
            today_iso="2026-06-24",
            turn_count_at_start=53,
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
                    "items": [{"category": "bus", "amount": 120.0}],
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
    assert plan.primary_op == PlanOp.RESOLVE_SUBMIT_CONFIRMATION
    assert plan.workflow_id == "expense"


def test_submit_koro_at_review_not_expense_summary(monkeypatch):
    from chat.services.platform import pipeline as pipeline_mod

    memory = _review_memory()
    submitted = {"called": False}

    def _fake_confirm_submit(*args, **kwargs):
        submitted["called"] = True
        return "Expense submitted.", {"outcome": "SUBMITTED", "rules_applied": ["EXPENSE_SUBMIT"]}

    monkeypatch.setattr(pipeline_mod.WorkflowPipeline, "_confirm_submit", _fake_confirm_submit)

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.5,
        reasoning="pending submit",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline_mod.WorkflowPipeline(),
        "submit koro",
        memory=memory,
        pq_decision=pq,
        trace_id="review-submit-koro",
        route_source="pending",
    )
    assert submitted["called"] is True
    assert decision.get("outcome") == "SUBMITTED"
    assert "বর্তমান খরচ" not in (msg or "")
