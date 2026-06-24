"""Phase 2/3/4 — history backfill, repair/undo, complaint UX."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.field_extractors.expense import (
    infer_expense_slot_from_history,
    interpret_expense_draft_turn,
    is_expense_anti_summary_request,
    is_expense_collect_complaint,
    sync_expense_draft,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.turn_semantics import is_workflow_meta_complaint
from chat.services.platform.response_composer import ResponseComposer
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
)
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.helpers.pipeline_handle import handle_with_rules_understanding
from tests.test_expense_conversation_phase1 import _route_pending_memory


def test_history_backfill_route_when_llm_returns_empty():
    memory = _route_pending_memory()
    sync_expense_draft(memory.active_draft())
    history = [
        "User: dhanmondi to mirpur",
        "Assistant: Still needed - Route",
    ]
    with mock_expense_llm():
        slot_turn = infer_expense_slot_from_history(
            memory,
            history,
            trace_id="test-backfill-route",
        )
    assert slot_turn
    assert slot_turn["item_patches"][0].get("from_location") == "Dhanmondi"


def test_undo_last_append_on_keno_abar():
    memory = _route_pending_memory()
    items = list(memory.active_draft().fields.get("items") or [])
    items.append({"amount": 150.0})
    memory.active_draft().fields["items"] = items
    memory.last_entities = {
        "expense_last_ops": {"appended": True, "item_count_before": 5, "notes": ["added item 6"]},
    }
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="repair",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        handle_with_rules_understanding(
            pipeline,
            "keno abar 150 taka add koro",
            memory=memory,
            pq_decision=pq,
            trace_id="test-undo-append",
            route_source="active",
        )
    sync_expense_draft(memory.active_draft())
    assert len(memory.active_draft().fields.get("items") or []) == 5


def test_anti_summary_intent():
    with mock_expense_llm():
        turn = interpret_expense_draft_turn(
            "ami summery chai ni",
            _route_pending_memory(),
            trace_id="test-anti-summary",
        )
    assert turn.get("intent") == "anti_summary"
    assert is_expense_anti_summary_request("ami summery chai ni")


def test_expense_complaint_uses_frustration_reply_not_robotic_clarify():
    memory = _route_pending_memory()
    from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult

    u = UnderstandingResult(
        goal="complaint",
        workflow="expense",
        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
        confidence=0.8,
        entities={"meta_complaint": True, "anti_summary": True},
        source="rules",
    )
    msg = ResponseComposer().clarification(u, memory=memory, lang="en")
    assert "summary" not in msg.lower() or "won't show" in msg.lower() or "unless you ask" in msg.lower()
    assert "or are you answering" not in msg.lower()
    assert is_expense_anti_summary_request("ami toh present expense chai ni")
    assert is_workflow_meta_complaint("eta ki diccho tumi") or is_expense_collect_complaint(
        "ami summery chai ni"
    )


def test_frustration_reply_via_pipeline():
    memory = _route_pending_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.85,
        reasoning="complaint",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "ami summery chai ni",
            memory=memory,
            pq_decision=pq,
            trace_id="test-pipeline-anti-summary",
            route_source="active",
        )
    assert decision.get("outcome") in ("NEEDS_INPUT", "NEEDS_CLARIFICATION", "INFORMATIONAL")
    assert "or are you answering" not in msg.lower()
