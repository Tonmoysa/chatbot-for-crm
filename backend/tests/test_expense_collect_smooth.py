"""Expense draft editor — collect flow (P0/P1/P2)."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import sync_expense_draft
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
)
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def _expense_collect_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 280.0},
                        {"category": "snack", "amount": 70.0},
                        {
                            "category": "metro",
                            "amount": 90.0,
                            "from_location": "Mirpur",
                            "to_location": "Agargaon",
                        },
                        {"amount": 150.0},
                    ],
                },
            )
        },
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Where did you travel from and to for the first expense?",
            workflow_id="expense",
            asked_at_turn=1,
            item_index=0,
        ),
    )


def test_collect_mode_skips_incomplete_item_validation():
    memory = _expense_collect_memory()
    draft = memory.active_draft()
    sync_expense_draft(draft)
    engine = ValidationEngine()
    from chat.services.platform.registry import get_workflow_definition

    defn = get_workflow_definition("expense")
    strict = engine.validate(draft, defn, lang="en", collect_mode=False)
    soft = engine.validate(draft, defn, lang="en", collect_mode=True)
    assert strict
    assert not soft


def test_route_answer_updates_bus_not_new_bike_item():
    memory = _expense_collect_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="route answer",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        handle_with_rules_understanding(
            pipeline,
            "mirpur to motejheel and category hobe bike",
            memory=memory,
            pq_decision=pq,
            trace_id="test-route-answer",
            route_source="active",
        )
    draft = memory.active_draft()
    sync_expense_draft(draft)
    items = draft.fields.get("items") or []
    assert len(items) == 5
    bus = items[0]
    assert bus.get("from_location") == "Mirpur"
    assert bus.get("to_location") == "Motijheel"
    assert bus.get("category") == "bus"


def test_pending_category_bus_not_lunch():
    memory = _expense_collect_memory()
    items = list(memory.active_draft().fields.get("items") or [])
    memory.pending_question = PendingQuestion(
        field="item_category",
        prompt="category?",
        workflow_id="expense",
        asked_at_turn=2,
        item_index=4,
    )
    with mock_expense_llm():
        from chat.services.platform.field_extractors.expense import expense_turn_to_field_updates, sync_expense_draft

        turn, updates = expense_turn_to_field_updates("bus", memory, trace_id="test-cat-bus")
    assert turn.get("intent") == "answer_pending"
    assert updates
    pipeline = WorkflowPipeline()
    state = StatePatchBuffer(memory)
    from chat.services.platform.registry import get_workflow_definition

    pipeline._finish_expense_update_turn(
        memory,
        get_workflow_definition("expense"),
        updates=updates,
        message="bus",
        lang="en",
        state=state,
    )
    state.flush()
    sync_expense_draft(memory.active_draft())
    item5 = memory.active_draft().fields["items"][4]
    assert item5.get("category") == "bus"


def test_review_add_duplicate_bus_allowed():
    memory = _expense_collect_memory()
    sync_expense_draft(memory.active_draft())
    for item in memory.active_draft().fields["items"]:
        if item.get("category") == "bus" and item.get("amount") == 120:
            item["from_location"] = "rampura"
            item["to_location"] = "badda"
    memory.pending_confirmation = "submit"
    memory.pending_question = None
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="add at review",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "bus 120 taka add koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-review-add-bus",
            route_source="active",
        )
    items = memory.active_draft().fields.get("items") or []
    bus_120 = [i for i in items if i.get("category") == "bus" and float(i.get("amount") or 0) == 120]
    assert len(bus_120) >= 2
    assert "summary" not in msg.lower() or "added" in msg.lower() or "bus" in msg.lower()
    assert decision.get("outcome") != "NEEDS_CLARIFICATION"


def test_expense_reference_amount_update():
    memory = _expense_collect_memory()
    items = list(memory.active_draft().fields.get("items") or [])
    items.append({"category": "bike", "from_location": "Mirpur", "to_location": "Motijheel"})
    memory.active_draft().fields["items"] = items
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="amount fix",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "expense 6 130 taka",
            memory=memory,
            pq_decision=pq,
            trace_id="test-expense-6-amount",
            route_source="active",
        )
    draft = memory.active_draft()
    items = draft.fields.get("items") or []
    assert len(items) == 6
    assert float(items[5].get("amount") or 0) == 130.0
    assert "greater than zero" not in msg.lower()
    assert decision.get("outcome") != "BLOCKED"


def test_numbered_bus_modify_updates_item_one_not_append():
    """Transcript: '1 number bus 130 taka koro' while route pending on item 1."""
    memory = _expense_collect_memory()
    memory.pending_question = PendingQuestion(
        field="item_category",
        prompt="Expense 5 — ? — 150.0 taka: what category?",
        workflow_id="expense",
        asked_at_turn=2,
        item_index=4,
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="modify bus amount",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with llm_disabled():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "1 number bus 130 taka koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-numbered-bus-modify",
            route_source="active",
        )
    draft = memory.active_draft()
    items = draft.fields.get("items") or []
    assert len(items) == 5
    assert float(items[0].get("amount") or 0) == 130.0
    assert "1.0" not in msg
    assert decision.get("outcome") != "BLOCKED"


def test_numbered_bus_hobe_modify_without_koro():
    memory = _expense_collect_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="modify",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with llm_disabled():
        handle_with_rules_understanding(
            pipeline,
            "1 number bus ta 130 taka hobe",
            memory=memory,
            pq_decision=pq,
            trace_id="test-bus-hobe-modify",
            route_source="active",
        )
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 5
    assert float(items[0].get("amount") or 0) == 130.0
