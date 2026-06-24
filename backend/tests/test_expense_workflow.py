"""Expense workflow tests — LLM-driven extraction."""

from __future__ import annotations

from chat.services.platform.field_extractors.expense import (
    expense_fields_from_message,
    expense_item_gaps,
    normalize_expense_category,
)
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.registry import get_workflow_definition
from chat.services.session_memory import ActiveWorkflow, PendingQuestion, SessionMemory, WorkflowDraft
from tests.helpers.expense_llm_mock import mock_expense_llm


def test_normalize_expense_categories():
    assert normalize_expense_category("Lunch") == "lunch"
    assert normalize_expense_category("Metro Rail") == "metro_rail"
    assert normalize_expense_category("uber") is None


def test_expense_item_gaps():
    items = [
        {"amount": 100},
        {"category": "bus", "amount": 100},
        {"category": "lunch", "amount": 250},
    ]
    gaps = expense_item_gaps(items)
    assert (0, "item_category") in gaps
    assert (1, "item_route") in gaps
    assert (2, "item_category") not in gaps and (2, "item_route") not in gaps


def test_extract_lunch_complete_via_llm():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    with mock_expense_llm():
        fields = expense_fields_from_message("lunch 250 taka", memory)
    assert fields["items"][0]["category"] == "lunch"
    assert fields["items"][0]["amount"] == 250.0


def test_expense_next_question_asks_category():
    engine = FieldEngine()
    defn = get_workflow_definition("expense")
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-23",
                    "items": [{"amount": 100}],
                },
            )
        },
    )
    draft = memory.active_draft()
    pq = engine.next_question(memory, draft, defn)
    assert pq is not None
    assert pq.field == "item_category"
    assert pq.item_index == 0


def test_expense_next_question_asks_route():
    engine = FieldEngine()
    defn = get_workflow_definition("expense")
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-23",
                    "items": [{"category": "bus", "amount": 100}],
                },
            )
        },
    )
    draft = memory.active_draft()
    pq = engine.next_question(memory, draft, defn)
    assert pq is not None
    assert pq.field == "item_route"
    assert pq.item_index == 0
