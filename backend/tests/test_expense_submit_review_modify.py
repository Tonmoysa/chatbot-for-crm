"""Expense submit with compound items in one message."""

from __future__ import annotations

from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.field_extractors.expense import (
    expense_turn_to_field_updates,
    extract_expense_compound_items_rules,
    interpret_expense_draft_turn,
    message_has_new_expense_items,
)
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import ActiveWorkflow, PendingQuestion, SessionMemory, WorkflowDraft
from tests.helpers.yaml_scenario_runner import llm_disabled

COMPOUND_SUBMIT_MSG = (
    "amar ajke expense hoyeche 100 taka bus e ..train e 30 taka mirpur to uttora "
    "then tarpor lunch 120 taka...tumi amar hoye eta ektu submit kore daw"
)


def _pending_items_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={"incurred_date": "2026-06-27", "items": []},
            ),
        },
        pending_question=PendingQuestion(
            field="items",
            prompt="Tell me about the expense — category and amount.",
            workflow_id="expense",
            asked_at_turn=4,
        ),
    )


def test_rules_extract_compound_items_with_submit_tail():
    patches = extract_expense_compound_items_rules(COMPOUND_SUBMIT_MSG)
    assert len(patches) >= 3
    amounts = {float(p["amount"]) for p in patches}
    assert 100.0 in amounts
    assert 30.0 in amounts
    assert 120.0 in amounts
    categories = {p.get("category") for p in patches}
    assert "bus" in categories
    assert "train" in categories
    assert "lunch" in categories


def test_compound_submit_message_has_new_expense_items():
    assert message_has_new_expense_items(COMPOUND_SUBMIT_MSG)


def test_compound_submit_message_extracts_items_not_bare_confirm():
    memory = _pending_items_memory()
    with llm_disabled():
        turn = interpret_expense_draft_turn(
            COMPOUND_SUBMIT_MSG,
            memory,
            trace_id="compound-submit-items",
        )
    assert turn.get("intent") == "add"
    assert len(turn.get("item_patches") or []) >= 3
    assert turn.get("submit_after_edit") is True


def test_compound_submit_applies_field_updates():
    memory = _pending_items_memory()
    with llm_disabled():
        turn, updates = expense_turn_to_field_updates(
            COMPOUND_SUBMIT_MSG,
            memory,
            trace_id="compound-submit-updates",
        )
    assert len(turn.get("item_patches") or []) >= 3
    assert len(updates) >= 3


def test_bare_submit_without_items_stays_confirm():
    memory = _pending_items_memory()
    with llm_disabled():
        turn = interpret_expense_draft_turn(
            "submit kore daw",
            memory,
            trace_id="bare-submit",
        )
    assert turn.get("intent") == "confirm"
    assert not turn.get("item_patches")


def test_pending_engine_routes_compound_submit_to_mutation():
    memory = _pending_items_memory()
    layer = AIUnderstandingLayer()
    engine = PendingQuestionEngine()
    with llm_disabled():
        understanding = layer.understand(
            COMPOUND_SUBMIT_MSG,
            memory=memory,
            conversation_history=[],
            trace_id="compound-submit-pqe",
        )
        decision = engine.classify(
            COMPOUND_SUBMIT_MSG,
            memory=memory,
            conversation_history=[],
            trace_id="compound-submit-pqe",
            understanding=understanding,
        )
    assert len(understanding.field_updates or []) >= 3
    assert decision.kind in (
        MessageIntentKind.ANSWER_PENDING,
        MessageIntentKind.MODIFY_DATA,
    )
    if decision.kind == MessageIntentKind.ANSWER_PENDING:
        assert decision.extracted_entities.get("interrupts_pending") is False
