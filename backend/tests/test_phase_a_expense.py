"""Phase A — expense list routing, amount-only items, route validation."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import (
    expense_fields_from_message,
    is_valid_expense_route,
)
from chat.services.platform.intent_rules import is_expense_list_request, should_resume_expense_for_list
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
)
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def test_is_valid_expense_route_rejects_hallucinated_office_jawar():
    assert not is_valid_expense_route("office", "jawar")
    assert is_valid_expense_route("Mirpur", "Agargaon")


def test_extract_amount_only_item_category_mone_nei():
    msg = "Ar ekta 150 taka expense hoise but category mone nei ekhon."
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
    )
    with mock_expense_llm():
        from chat.services.platform.field_extractors.expense import expense_turn_to_field_updates

        turn, updates = expense_turn_to_field_updates(msg, memory, trace_id="test-150")
    assert turn.get("item_patches") or updates
    append = next((u for u in updates if u.field == "items" and u.action == "append"), None)
    assert append is not None
    assert float(append.value.get("amount") or 0) == 150.0


def test_is_expense_list_request_matches_transcript_phrases():
    assert is_expense_list_request("aj saradin ki ki expense korchi tar list ta dekhao toh")
    assert is_expense_list_request("ami expense er list jante ceyechi")
    assert not is_expense_list_request("leave er summery ta dekhao")


def test_compound_expense_includes_150_and_no_bus_route_hallucination():
    msg = (
        "Aj office jawar somoy bus e 120 taka lagse. Dupure lunch korlam 280 taka. "
        "Snack 70 taka. Metro te 90 taka Mirpur theke Agargaon. "
        "Ar ekta 150 taka expense hoise but category mone nei ekhon."
    )
    with mock_expense_llm():
        memory = SessionMemory(
            active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
            workflow_drafts={"default": WorkflowDraft(workflow_id="expense", fields={"items": []})},
        )
        fields = expense_fields_from_message(msg, memory)
    items = fields.get("items") or []
    amounts = sorted(float(i.get("amount") or 0) for i in items)
    assert 150.0 in amounts
    bus = next(i for i in items if i.get("category") == "bus")
    assert bus.get("amount") == 120.0
    assert not bus.get("from_location")
    metro = next(i for i in items if i.get("category") == "metro")
    assert metro.get("from_location") == "Mirpur"
    assert metro.get("to_location") == "Agargaon"


def _leave_active_expense_suspended_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "osusto",
                },
            ),
            "expense": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-23",
                    "items": [
                        {"category": "bus", "amount": 120.0},
                        {"category": "lunch", "amount": 280.0},
                    ],
                },
            ),
        },
        suspended_workflows=[
            SuspendedWorkflow(
                workflow_id="expense",
                stage="confirm_submit",
                draft_id="expense",
                suspended_at_turn=3,
            )
        ],
        pending_confirmation="submit",
    )


def test_expense_list_during_leave_shows_expense_not_leave():
    memory = _leave_active_expense_suspended_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.9,
        reasoning="expense list",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="expense",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        "aj saradin ki ki expense korchi tar list ta dekhao toh",
        memory=memory,
        pq_decision=pq,
        trace_id="test-expense-list-leave",
        route_source="active",
    )
    low = msg.lower()
    assert "chuti" not in low or "expense" in low
    assert "lunch" in low or "bus" in low or "expense summary" in low
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"
    assert decision.get("outcome") in ("INFORMATIONAL", "NEEDS_INPUT", None)


def test_should_resume_expense_for_list():
    memory = _leave_active_expense_suspended_memory()
    assert should_resume_expense_for_list(
        message="ami expense er list jante ceyechi",
        active_workflow_id="leave",
        suspended_workflows=memory.suspended_workflows,
    )
