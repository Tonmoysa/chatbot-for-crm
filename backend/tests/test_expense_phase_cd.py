"""Phase C+D — expense wizard fallback + observability regression lock."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.field_extractors.expense import (
    build_wizard_fallback_turn,
    expense_turn_to_field_updates,
    split_expense_clauses,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.pending_question_engine import PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    WorkflowDraft,
    build_turn_context,
)
from tests.helpers.yaml_scenario_runner import llm_disabled

COMPOUND_EXPENSE = (
    "Aj office jawar somoy bus e 120 taka lagse. Dupure lunch korlam 280 taka. "
    "Bikale ekta snack kheyechi 70 taka. Ferar somoy metro te 90 taka lagse "
    "Mirpur theke Agargaon porjonto. Ar ekta 150 taka expense hoise but category mone nei ekhon."
)


def test_split_expense_clauses():
    parts = split_expense_clauses(COMPOUND_EXPENSE)
    assert len(parts) >= 4
    assert any("120" in p for p in parts)
    assert any("lunch" in p.lower() for p in parts)


def test_wizard_fallback_builds_multiple_items_without_llm():
    turn = build_wizard_fallback_turn(COMPOUND_EXPENSE)
    assert turn.get("wizard_fallback") is True
    patches = turn.get("item_patches") or []
    assert len(patches) >= 4
    amounts = {float(p.get("amount") or 0) for p in patches}
    assert 120.0 in amounts
    assert 280.0 in amounts
    assert 150.0 in amounts


def test_interpret_expense_uses_wizard_when_llm_disabled():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    with llm_disabled():
        turn, updates = expense_turn_to_field_updates(
            COMPOUND_EXPENSE,
            memory,
            trace_id="phase-c-wizard-off",
        )
    assert turn.get("wizard_fallback") is True
    assert len(updates) >= 4


def test_wizard_fallback_pipeline_collects_without_mock_llm():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    layer = AIUnderstandingLayer()
    engine = PendingQuestionEngine()
    pipeline = WorkflowPipeline()
    with llm_disabled():
        u = layer.understand(
            COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-c-pipeline-wizard",
        )
        pq = engine.classify(
            COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-c-pipeline-wizard",
            understanding=u,
        )
        turn_context = build_turn_context(
            message=COMPOUND_EXPENSE,
            memory=memory,
            conversation_history=[],
            trace_id="phase-c-pipeline-wizard",
            session_id="test-session",
            company_id="test-company",
            employee_id="test-employee",
            idempotency_key="",
        )
        msg, meta = pipeline.execute_workflow_turn(
            COMPOUND_EXPENSE,
            memory=memory,
            understanding=u,
            pq_decision=pq,
            conversation_history=[],
            trace_id="phase-c-pipeline-wizard",
            turn_context=turn_context,
            route_source="active",
        )
    draft = memory.active_draft()
    items = (draft.fields if draft else {}).get("items") or []
    assert len(items) >= 4
    assert "EXPENSE_WIZARD_FALLBACK" in (meta.get("rules_applied") or [])
    assert "limit" in msg.lower() or "120" in msg or "jog" in msg.lower()


def test_observability_logs_expense_draft_turn(caplog):
    import logging

    from chat.services.observability import log_expense_draft_turn

    caplog.set_level(logging.INFO, logger="hr_chatbot")
    log_expense_draft_turn(
        "trace-obs",
        message="bus 120 taka",
        turn={"intent": "add", "item_patches": [{"action": "append", "amount": 120}]},
        llm_used=False,
        wizard_fallback=True,
    )
    assert any("expense_draft_turn" in r.message for r in caplog.records)


def test_llm_fail_falls_back_to_wizard_not_empty_summary():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    with patch("chat.services.llm_client.LLMClient") as mock_cls:
        client = mock_cls.return_value
        client.is_configured.return_value = True
        client.chat_json.return_value = None
        turn, updates = expense_turn_to_field_updates(
            COMPOUND_EXPENSE,
            memory,
            trace_id="phase-c-llm-fail-wizard",
        )
    assert turn.get("wizard_fallback") is True
    assert len(updates) >= 4
