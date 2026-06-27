"""Regression: expense route answer must not resume suspended leave."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.pipeline import PlanBuilder, PlanOp, WorkflowPipeline
from chat.services.platform.schemas import FieldUpdate, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    SuspendedWorkflow,
    WorkflowDraft,
    build_turn_context,
)
from tests.helpers.expense_llm_mock import mock_expense_llm


def _expense_route_pending_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="collecting"),
        suspended_workflows=[
            SuspendedWorkflow(workflow_id="leave", stage="confirm_submit", draft_id="leave-draft"),
        ],
        pending_question=PendingQuestion(
            field="item_route",
            prompt="Where did you travel from and to for the first expense?",
            workflow_id="expense",
            item_index=0,
        ),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-23",
                    "items": [
                        {
                            "category": "bus",
                            "amount": 120,
                            "missing_fields": ["route"],
                            "status": "incomplete",
                        },
                    ],
                },
            ),
            "leave-draft": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "Grandfather unwell; family emergency in village",
                },
            ),
        },
    )


def test_plan_builder_collects_route_not_switch_to_leave():
    memory = _expense_route_pending_memory()
    understanding = UnderstandingResult(
        goal="Answer route",
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.9,
        answers_pending_field=True,
        field_updates=[
            FieldUpdate(
                field="items",
                value={"from_location": "Mirpur", "to_location": "Motijheel"},
                item_index=0,
                action="update",
            ),
        ],
        source="llm",
    )
    decision = PendingQuestionEngine().classify(
        "mirpur to motejheel",
        memory=memory,
        conversation_history=[],
        trace_id="test-plan-route",
        understanding=understanding,
    )
    assert decision.kind == MessageIntentKind.ANSWER_PENDING

    ctx = build_turn_context(
        message="mirpur to motejheel",
        memory=memory,
        conversation_history=[],
        trace_id="test-plan-route",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    plan = PlanBuilder._build_expense_plan(
        ctx,
        TurnDecision(pq=decision, understanding=understanding, route_source="pending"),
    )
    assert plan is not None
    assert PlanOp.WORKFLOW_SWITCH not in plan.ops
    assert PlanOp.WORKFLOW_COLLECT in plan.ops


def test_pipeline_applies_route_and_stays_on_expense():
    memory = _expense_route_pending_memory()
    pipeline = WorkflowPipeline()
    message = "mirpur to motejheel"
    with mock_expense_llm():
        misclassified = UnderstandingResult(
            goal="Answer route",
            workflow="leave",
            action=UnderstandingAction.COLLECT.value,
            confidence=0.9,
            interrupt_workflow="leave",
            source="llm",
        )
        understanding = FieldEngine().ground_expense_understanding(
            message,
            misclassified,
            memory=memory,
            trace_id="test-pipeline-route",
        )
        decision = PendingQuestionEngine().classify(
            message,
            memory=memory,
            conversation_history=[
                "Assistant: Where did you travel from and to for the first expense?",
            ],
            trace_id="test-pipeline-route",
            understanding=understanding,
        )
        ctx = build_turn_context(
            message=message,
            memory=memory,
            conversation_history=[],
            trace_id="test-pipeline-route",
            session_id="s1",
            company_id="c1",
            employee_id="e1",
        )
        msg, meta = pipeline.execute_workflow_turn(
            message,
            memory=memory,
            understanding=understanding,
            pq_decision=decision,
            conversation_history=[],
            trace_id="test-pipeline-route",
            turn_context=ctx,
            route_source="pending",
        )
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"
    low = (msg or "").lower()
    assert "resuming your" not in low
    items = memory.active_draft().fields.get("items") or []
    assert items[0].get("from_location") == "Mirpur"
    assert items[0].get("to_location") == "Motijheel"
    assert meta.get("outcome") != "CANCELLED"


def test_dhanmondi_to_mirpur_route_coerced_without_llm():
    """Pending route answer 'dhanmondi to mirpur' must apply without LLM."""
    from chat.services.platform.field_extractors.expense import (
        coerce_pending_expense_turn,
        expense_turn_to_field_updates,
        sync_expense_draft,
    )
    from chat.services.platform.field_engine import FieldEngine
    from tests.helpers.yaml_scenario_runner import llm_disabled

    memory = _expense_route_pending_memory()
    memory.pending_question.prompt = "Expense 1 — Bus — 120.0 taka: where did you travel from and to?"
    turn = coerce_pending_expense_turn("dhanmondi to mirpur", memory)
    assert turn is not None
    assert turn.get("intent") == "answer_pending"
    patch = turn["item_patches"][0]
    assert patch.get("from_location") == "Dhanmondi"
    assert patch.get("to_location") == "Mirpur"

    with llm_disabled():
        _, updates = expense_turn_to_field_updates("dhanmondi to mirpur", memory)
    assert updates
    assert updates[0].field == "items"
    assert updates[0].value.get("from_location") == "Dhanmondi"
    assert updates[0].value.get("to_location") == "Mirpur"

    draft = memory.active_draft()
    engine = FieldEngine()
    engine.apply_updates(draft, updates)
    sync_expense_draft(draft)
    bus = draft.fields["items"][0]
    assert bus.get("from_location") == "Dhanmondi"
    assert bus.get("to_location") == "Mirpur"
    assert "route" not in (bus.get("missing_fields") or [])


def test_mirpur_to_office_route_applies_without_llm():
    """Pending route answer 'mirpur to office' must apply when LLM is rate-limited."""
    from chat.services.platform.field_extractors.expense import (
        coerce_pending_expense_turn,
        expense_turn_to_field_updates,
        sync_expense_draft,
    )
    from chat.services.platform.field_engine import FieldEngine
    from tests.helpers.yaml_scenario_runner import llm_disabled

    memory = _expense_route_pending_memory()
    memory.pending_question.prompt = (
        "Expense 4 — Bus — 30.0 taka: where did you travel from and to?"
    )
    memory.pending_question.item_index = 3
    memory.workflow_drafts["default"].fields["items"] = [
        {
            "category": "bus",
            "amount": 30.0,
            "status": "complete",
            "from_location": "Kamlapur",
            "to_location": "Dhanmondi",
        },
        {"category": "lunch", "amount": 120.0, "status": "complete"},
        {
            "category": "bike",
            "amount": 130.0,
            "status": "complete",
            "from_location": "Dhanmondi",
            "to_location": "Mirpur",
        },
        {"category": "bus", "amount": 30.0, "missing_fields": ["route"], "status": "incomplete"},
    ]

    turn = coerce_pending_expense_turn("mirpur to office", memory)
    assert turn is not None
    assert turn.get("intent") == "answer_pending"
    patch = turn["item_patches"][0]
    assert patch.get("item_index") == 3
    assert patch.get("from_location") == "Mirpur"
    assert patch.get("to_location") == "Office"

    with llm_disabled():
        _, updates = expense_turn_to_field_updates("mirpur to office", memory)
    assert updates
    draft = memory.active_draft()
    engine = FieldEngine()
    engine.apply_updates(draft, updates)
    sync_expense_draft(draft)
    bus = draft.fields["items"][3]
    assert bus.get("from_location") == "Mirpur"
    assert bus.get("to_location") == "Office"
    assert "route" not in (bus.get("missing_fields") or [])


def _mirpur_to_office_pending_memory() -> SessionMemory:
    memory = _expense_route_pending_memory()
    memory.pending_question.prompt = (
        "Expense 4 — Bus — 30.0 taka: where did you travel from and to?"
    )
    memory.pending_question.item_index = 3
    memory.workflow_drafts["default"].fields["items"] = [
        {
            "category": "bus",
            "amount": 30.0,
            "status": "complete",
            "from_location": "Kamlapur",
            "to_location": "Dhanmondi",
        },
        {"category": "lunch", "amount": 120.0, "status": "complete"},
        {
            "category": "bike",
            "amount": 130.0,
            "status": "complete",
            "from_location": "Dhanmondi",
            "to_location": "Mirpur",
        },
        {"category": "bus", "amount": 30.0, "missing_fields": ["route"], "status": "incomplete"},
    ]
    return memory


def test_mirpur_to_office_not_treated_as_draft_mutation():
    """Route slot answers must not be blocked by modify/LLM heuristics."""
    from chat.services.platform.field_extractors.expense import (
        interpret_expense_draft_turn,
        is_expense_draft_mutation_message,
        is_expense_pending_field_value_answer,
    )

    memory = _mirpur_to_office_pending_memory()
    msg = "mirpur to office"
    assert not is_expense_draft_mutation_message(msg, memory)
    assert is_expense_pending_field_value_answer(msg, memory)

    turn = interpret_expense_draft_turn(msg, memory, trace_id="mirpur-office-rules")
    assert turn.get("intent") == "answer_pending"
    assert turn.get("llm_used") is False
    patch = turn["item_patches"][0]
    assert patch.get("from_location") == "Mirpur"
    assert patch.get("to_location") == "Office"


def test_bike_route_answer_not_marked_out_of_scope_when_scope_llm_misfires():
    """Regression: 'badda to gulshan' answering pending bike route must not OOS."""
    from unittest.mock import patch

    from chat.services.platform.ai_understanding import AIUnderstandingLayer
    from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult

    memory = _expense_route_pending_memory()
    memory.pending_question = PendingQuestion(
        field="item_route",
        prompt="Expense 3 — Bike — 120.0 taka: where did you travel from and to?",
        workflow_id="expense",
        item_index=2,
    )
    memory.workflow_drafts["default"].fields["items"] = [
        {"category": "lunch", "amount": 100.0, "status": "complete"},
        {
            "category": "bus",
            "amount": 40.0,
            "status": "complete",
            "from_location": "Mirpur",
            "to_location": "Badda",
        },
        {"category": "bike", "amount": 120.0, "missing_fields": ["route"], "status": "incomplete"},
    ]

    scope_oos = UnderstandingResult(
        goal="Out of scope",
        workflow="none",
        action=UnderstandingAction.NONE.value,
        confidence=0.9,
        is_out_of_scope=True,
        reasoning="Place names without HR context.",
        source="llm_hr_scope",
    )

    with patch(
        "chat.services.platform.hr_assistant_scope.resolve_hr_assistant_scope",
        return_value=scope_oos,
    ):
        understanding = AIUnderstandingLayer().understand(
            "badda to gulshan",
            memory=memory,
            conversation_history=[
                "Assistant: Expense 3 — Bike — 120.0 taka: where did you travel from and to?",
            ],
            trace_id="test-bike-route-oos",
        )

    assert not understanding.is_out_of_scope
    assert understanding.workflow == "expense"
    updates = understanding.field_updates or []
    assert updates
    assert updates[0].value.get("from_location") == "Badda"
    assert updates[0].value.get("to_location") == "Gulshan"
    decision = PendingQuestionEngine().classify(
        "badda to gulshan",
        memory=memory,
        conversation_history=[],
        trace_id="test-bike-route-oos",
        understanding=understanding,
    )
    assert decision.kind in (
        MessageIntentKind.ANSWER_PENDING,
        MessageIntentKind.MODIFY_DATA,
    )


def test_leave_back_during_expense_pending_switches_not_slot_answer():
    """'leave e back koro' must resume leave — not answer expense category slot."""
    memory = _expense_route_pending_memory()
    memory.pending_question = PendingQuestion(
        field="item_category",
        prompt="Expense 5 — ? — 150.0 taka: what category was it?",
        workflow_id="expense",
        asked_at_turn=2,
        item_index=4,
    )
    memory.workflow_drafts["default"].fields["items"] = [
        {"category": "bus", "amount": 120.0, "id": "a", "status": "complete"},
        {"category": "lunch", "amount": 280.0, "id": "b", "status": "complete"},
        {"category": "snack", "amount": 70.0, "id": "c", "status": "complete"},
        {"category": "metro", "amount": 90.0, "id": "d", "status": "complete"},
        {"amount": 150.0, "id": "e", "missing_fields": ["category"], "status": "incomplete"},
    ]
    misclassified = UnderstandingResult(
        goal="leave",
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.9,
        entities={"expense_intent": "answer_pending"},
        field_updates=[
            FieldUpdate(
                field="items",
                value={"category": "bus", "amount": 150.0},
                item_index=4,
                action="update",
            ),
        ],
        interrupt_workflow="expense",
        answers_pending_field=True,
        reasoning="User wants to review the leave draft.",
        source="llm",
    )
    decision = PendingQuestionEngine().classify(
        "leave e back koro",
        memory=memory,
        conversation_history=[],
        trace_id="test-leave-back-pending",
        understanding=misclassified,
    )
    assert decision.kind == MessageIntentKind.SWITCH_WORKFLOW
    assert decision.target_workflow == "leave"

    ctx = build_turn_context(
        message="leave e back koro",
        memory=memory,
        conversation_history=[],
        trace_id="test-leave-back-pending",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    plan = PlanBuilder._build_expense_plan(
        ctx,
        TurnDecision(pq=decision, understanding=misclassified, route_source="pending"),
    )
    assert plan is not None
    assert PlanOp.WORKFLOW_SWITCH in plan.ops
    assert PlanOp.WORKFLOW_COLLECT not in plan.ops
