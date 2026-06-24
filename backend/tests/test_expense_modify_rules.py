"""Rules-first expense item modify/delete — Banglish numbered references."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.modify import parse_delete_request, parse_modify_request
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.helpers.yaml_scenario_runner import llm_disabled


def _sample_items():
    return [
        {"category": "bus", "amount": 120.0, "description": "bus e 120 taka lagse"},
        {"category": "lunch", "amount": 280.0},
        {"category": "snack", "amount": 70.0},
        {"category": "metro", "amount": 90.0},
        {"amount": 150.0},
    ]


def _memory_with_items(items=None):
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={"incurred_date": "2026-06-24", "items": items or _sample_items()},
            )
        },
    )


def test_parse_numbered_bus_modify_130():
    parsed = parse_modify_request("1 number bus 130 taka koro", _sample_items())
    assert parsed is not None
    assert parsed["item_index"] == 0
    assert parsed["amount"] == 130.0


def test_parse_expense_six_130():
    parsed = parse_modify_request("expense 6 130 taka", _sample_items() + [{}])
    assert parsed is not None
    assert parsed["item_index"] == 5
    assert parsed["amount"] == 130.0


def test_does_not_confuse_item_number_with_amount():
    parsed = parse_modify_request("1 number bus ta 130 taka hobe", _sample_items())
    assert parsed is not None
    assert parsed["amount"] == 130.0
    assert parsed["amount"] != 1.0


def test_ambiguous_bus_modify_requests_clarify():
    items = [
        {"category": "bus", "amount": 120.0},
        {"category": "lunch", "amount": 280.0},
        {"category": "bus", "amount": 150.0},
    ]
    parsed = parse_modify_request("bus 130 taka koro", items)
    assert parsed is not None
    assert parsed.get("needs_clarify") is True
    assert parsed.get("candidate_indices") == [0, 2]


def test_parse_delete_by_number():
    parsed = parse_delete_request("5 no bad dao", _sample_items())
    assert parsed is not None
    assert parsed["item_index"] == 4


def test_parse_delete_last_bus():
    items = [
        {"category": "bus", "amount": 120.0, "description": "Mirpur Agargaon"},
        {"category": "lunch", "amount": 280.0},
        {"category": "snack", "amount": 70.0},
        {"category": "metro", "amount": 90.0},
        {"category": "bus", "amount": 150.0, "description": "Dhanmondi Dhaka"},
        {"category": "bus", "amount": 1.0, "description": "Dhaa Mirpur"},
    ]
    parsed = parse_delete_request("last bus ta delete koro", items)
    assert parsed is not None
    assert parsed["item_index"] == 5
    assert parsed.get("needs_clarify") is not True


def test_parse_delete_sesh_bus_banglish():
    items = [
        {"category": "bus", "amount": 120.0},
        {"category": "bus", "amount": 150.0},
    ]
    parsed = parse_delete_request("sesh bus ta muche felo", items)
    assert parsed is not None
    assert parsed["item_index"] == 1


def test_last_bus_delete_on_expense_review_not_leave_switch():
    items = [
        {"category": "bus", "amount": 120.0},
        {"category": "lunch", "amount": 280.0},
        {"category": "snack", "amount": 70.0},
        {"category": "metro", "amount": 90.0},
        {"category": "bus", "amount": 150.0},
        {"category": "bus", "amount": 1.0, "description": "Dhaa Mirpur"},
    ]
    from chat.services.session_memory import SuspendedWorkflow

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={"incurred_date": "2026-06-24", "items": items},
            ),
            "leave-draft": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "lwop",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "Grandfather unwell",
                },
            ),
        },
        suspended_workflows=[
            SuspendedWorkflow(
                workflow_id="leave",
                stage="confirm_submit",
                draft_id="leave-draft",
                suspended_at_turn=2,
            )
        ],
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.DELETE_DATA,
        confidence=0.9,
        reasoning="delete last bus",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with llm_disabled():
        msg, meta = handle_with_rules_understanding(
            pipeline,
            "last bus ta delete koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-last-bus-delete",
            route_source="active",
        )
    assert memory.active_workflow.id == "expense"
    remaining = memory.active_draft().fields.get("items") or []
    assert len(remaining) == 5
    assert all(float(i.get("amount") or 0) != 1.0 for i in remaining)
    low = msg.lower()
    assert "resuming your leave" not in low
    assert "leave request" not in low or "expense" in low


def test_pipeline_delete_item_five():
    memory = _memory_with_items()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.DELETE_DATA,
        confidence=0.9,
        reasoning="delete",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with llm_disabled():
        handle_with_rules_understanding(
            pipeline,
            "5 no bad dao",
            memory=memory,
            pq_decision=pq,
            trace_id="test-delete-5",
            route_source="active",
        )
    items = memory.active_draft().fields.get("items") or []
    assert len(items) == 4
    assert all(float(i.get("amount") or 0) != 150.0 for i in items)


def test_expense_cancel_rules():
    memory = _memory_with_items()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CANCEL_WORKFLOW,
        confidence=0.9,
        reasoning="cancel",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with llm_disabled():
        _, meta = handle_with_rules_understanding(
            pipeline,
            "expense cancel koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-expense-cancel",
            route_source="active",
        )
    assert meta.get("outcome") == "CANCELLED"
    assert memory.active_workflow is None
