"""Phase 5 — PlanBuilder + leave execution plan tests."""

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.pipeline import PlanBuilder
from chat.services.platform.schemas import FieldUpdate, PlanOp, TurnContext, TurnDecision, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, PendingQuestion, SessionMemory, SuspendedWorkflow, WorkflowDraft


def _ctx(**overrides) -> TurnContext:
    base = dict(
        trace_id="t1",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
        user_message="ha",
        conversation_history=(),
        document_text=None,
        idempotency_key="",
        user_language="en",
        reply_language="en",
        today_iso="2026-06-21",
        turn_count_at_start=1,
        memory_schema_version=1,
        active_workflow_id="leave",
        active_workflow_stage="collecting",
        draft_id="default",
        pending_question_field=None,
        pending_question_prompt=None,
        pending_question_workflow_id=None,
        pending_confirmation="submit",
        draft_snapshot={"workflow_id": "leave", "fields": {"leave_type": "annual"}},
        suspended_workflows=(),
        conversation_facts={},
        has_active_workflow=True,
        has_pending_question=False,
        has_pending_confirmation=True,
        draft_locked=False,
        wizard_active=False,
    )
    base.update(overrides)
    return TurnContext(**base)


def test_plan_builder_submit_confirmation():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.CONFIRM.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(user_message="ha"),
        TurnDecision(
            pq=None,
            understanding=u,
            route_source="active",
        ),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.RESOLVE_SUBMIT_CONFIRMATION


def test_plan_builder_leave_submit_request():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.SUBMIT.value,
        confidence=0.88,
    )
    plan = PlanBuilder.build(
        _ctx(
            user_message="leave submit koro",
            pending_confirmation=None,
            has_pending_confirmation=False,
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.LEAVE_REQUEST_SUBMIT


def test_plan_builder_active_leave_apply_includes_duplicate_guard():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.85,
        field_updates=[FieldUpdate(field="start_date", value="2026-08-20")],
    )
    plan = PlanBuilder.build(
        _ctx(
            user_message="agami 20 august leave chai",
            pending_confirmation=None,
            has_pending_confirmation=False,
            draft_snapshot={
                "workflow_id": "leave",
                "fields": {"start_date": "2026-08-15"},
            },
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.ops == [PlanOp.MAYBE_DUPLICATE_LEAVE, PlanOp.LEAVE_APPLY_UPDATES]


def test_plan_builder_leave_active_expense_interrupt():
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.85,
        field_updates=[
            FieldUpdate(
                field="items",
                value={"category": "travel", "amount": 100.0, "description": "bus"},
                action="append",
            )
        ],
    )
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="leave",
            pending_confirmation="submit",
            has_pending_confirmation=True,
            user_message="amar ajke expense hoyeche 100 taka for bus",
            draft_snapshot={"workflow_id": "leave", "fields": {"leave_type": "annual"}},
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_SWITCH
    assert plan.workflow_id == "expense"


def test_plan_builder_answer_pending_collect_chain():
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="slot",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.85,
        field_updates=[],
    )
    plan = PlanBuilder.build(
        _ctx(
            pending_confirmation=None,
            has_pending_confirmation=False,
            pending_question_field="end_date",
            pending_question_workflow_id="leave",
            has_pending_question=True,
            wizard_active=True,
        ),
        TurnDecision(pq=pq, understanding=u, route_source="pending"),
    )
    assert plan is not None
    assert plan.ops == [
        PlanOp.MAYBE_DUPLICATE_LEAVE,
        PlanOp.MAYBE_WORKFLOW_SWITCH,
        PlanOp.LEAVE_COLLECT,
    ]


def test_plan_builder_expense_collect_during_submit_pending():
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.85,
        field_updates=[
            FieldUpdate(
                field="items",
                value={"category": "meals", "amount": 150.0, "description": "lunch"},
                action="append",
            )
        ],
    )
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="expense",
            pending_confirmation="submit",
            has_pending_confirmation=True,
            draft_snapshot={"workflow_id": "expense", "fields": {"items": [{"amount": 100}]}},
            user_message="lunch korechi 150 taka",
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.EXPENSE_APPLY_UPDATES


def test_plan_builder_expense_submit_confirmation():
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.CONFIRM.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="expense",
            pending_confirmation="submit",
            has_pending_confirmation=True,
            draft_snapshot={"workflow_id": "expense", "fields": {"items": [{"amount": 100}]}},
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.RESOLVE_SUBMIT_CONFIRMATION
    assert plan.workflow_id == "expense"


def test_plan_builder_resume_suspended_leave_switches_directly():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.SWITCH.value,
        interrupt_workflow="leave",
        confidence=0.88,
    )
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="expense",
            active_workflow_stage="collecting",
            pending_confirmation=None,
            has_pending_confirmation=False,
            user_message="continue leave",
            draft_snapshot={"workflow_id": "expense", "fields": {"items": [{"amount": 100}]}},
            suspended_workflows=(
                SuspendedWorkflow(
                    workflow_id="leave",
                    stage="collecting",
                    draft_id="leave-draft",
                    suspended_at_turn=1,
                ).to_dict(),
            ),
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_SWITCH


def test_plan_builder_last_bus_delete_does_not_resume_suspended_leave():
    from chat.services.platform.schemas import TargetRef

    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.DELETE.value,
        confidence=0.9,
        targets=[TargetRef(field="items", item_index=2)],
    )
    items = [
        {"category": "bus", "amount": 120.0},
        {"category": "lunch", "amount": 280.0},
        {"category": "bus", "amount": 1.0},
    ]
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="expense",
            active_workflow_stage="confirm_submit",
            pending_confirmation="submit",
            has_pending_confirmation=True,
            user_message="last bus ta delete koro",
            draft_snapshot={"workflow_id": "expense", "fields": {"items": items}},
            suspended_workflows=(
                SuspendedWorkflow(
                    workflow_id="leave",
                    stage="confirm_submit",
                    draft_id="leave-draft",
                    suspended_at_turn=1,
                ).to_dict(),
            ),
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_DELETE
    assert plan.workflow_id == "expense"


def test_plan_builder_misclassified_leave_on_expense_delete_stays_expense():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.88,
        entities={"expense_intent": "delete"},
    )
    items = [
        {"category": "bus", "amount": 120.0},
        {"category": "bus", "amount": 1.0},
    ]
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="expense",
            active_workflow_stage="confirm_submit",
            pending_confirmation="submit",
            has_pending_confirmation=True,
            user_message="last bus ta delete koro",
            draft_snapshot={"workflow_id": "expense", "fields": {"items": items}},
            suspended_workflows=(
                SuspendedWorkflow(
                    workflow_id="leave",
                    stage="confirm_submit",
                    draft_id="leave-draft",
                    suspended_at_turn=1,
                ).to_dict(),
            ),
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op != PlanOp.WORKFLOW_SWITCH
    assert plan.workflow_id == "expense"


    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="start expense",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="expense",
    )
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.START.value,
        confidence=0.88,
    )
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id=None,
            has_active_workflow=False,
            pending_confirmation=None,
            has_pending_confirmation=False,
        ),
        TurnDecision(pq=pq, understanding=u, route_source="pending"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.EXPENSE_NEW


def test_plan_builder_expense_collect_chain():
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="slot",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.COLLECT.value,
        confidence=0.85,
    )
    plan = PlanBuilder.build(
        _ctx(
            active_workflow_id="expense",
            pending_question_field="from_location",
            pending_question_workflow_id="expense",
            has_pending_question=True,
            wizard_active=True,
            pending_confirmation=None,
            has_pending_confirmation=False,
        ),
        TurnDecision(pq=pq, understanding=u, route_source="pending"),
    )
    assert plan is not None
    assert plan.ops == [PlanOp.MAYBE_WORKFLOW_SWITCH, PlanOp.EXPENSE_COLLECT]


def test_plan_builder_unrelated_returns_informational_fallback():
    u = UnderstandingResult(workflow="none", action=UnderstandingAction.NONE.value, confidence=0.4)
    plan = PlanBuilder.build(
        _ctx(active_workflow_id=None, has_active_workflow=False, pending_confirmation=None),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.workflow_id == "informational"
    assert plan.primary_op == PlanOp.REPLY_GENERAL_HELP


def test_plan_builder_locked_leave_summary():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.REVIEW.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(
            draft_locked=True,
            active_workflow_stage="submitted",
            pending_confirmation=None,
            has_pending_confirmation=False,
            draft_snapshot={
                "workflow_id": "leave",
                "locked": True,
                "submitted_request_id": "LV-1",
                "fields": {"leave_type": "annual", "start_date": "2026-08-05"},
            },
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.ops == [PlanOp.WORKFLOW_SHOW_REVIEW]


def test_plan_builder_locked_leave_allows_new_request():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.START.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(
            draft_locked=True,
            user_message="amar notun leave lagbe next week",
            active_workflow_stage="submitted",
            pending_confirmation=None,
            has_pending_confirmation=False,
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.ops == [PlanOp.WORKFLOW_NEW]


def test_plan_builder_locked_leave_blocks_modify():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.MODIFY.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(
            draft_locked=True,
            active_workflow_stage="submitted",
            pending_confirmation=None,
            has_pending_confirmation=False,
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.ops == [PlanOp.LOCKED_RESPONSE]


def test_plan_builder_locked_leave_expense_interrupt():
    u = UnderstandingResult(
        workflow="expense",
        action=UnderstandingAction.START.value,
        confidence=0.9,
        field_updates=[
            FieldUpdate(
                field="items",
                value={"category": "bus", "amount": 120.0},
                action="append",
            )
        ],
    )
    plan = PlanBuilder.build(
        _ctx(
            draft_locked=True,
            active_workflow_id="leave",
            active_workflow_stage="submitted",
            pending_confirmation=None,
            has_pending_confirmation=False,
            user_message="ajke bus e 120 taka lunch 280 taka",
            draft_snapshot={
                "workflow_id": "leave",
                "locked": True,
                "submitted_request_id": "LV-1",
                "fields": {"leave_type": "lwop", "start_date": "2026-09-13"},
            },
            conversation_facts={
                "submitted_leave_ranges": [
                    {"start_date": "2026-09-13", "end_date": "2026-09-18", "request_id": "LV-1"},
                ]
            },
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.workflow_id == "expense"
    assert plan.primary_op == PlanOp.WORKFLOW_NEW


def test_plan_builder_locked_leave_review_with_new_dates_starts_leave():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.REVIEW.value,
        confidence=0.9,
    )
    plan = PlanBuilder.build(
        _ctx(
            draft_locked=True,
            active_workflow_id="leave",
            active_workflow_stage="submitted",
            user_message=(
                "14 September 2026 theke 17 September 2026 annual leave "
                "review dekhao"
            ),
            draft_snapshot={
                "workflow_id": "leave",
                "locked": True,
                "fields": {"leave_type": "lwop", "start_date": "2026-09-13"},
            },
            conversation_facts={
                "submitted_leave_ranges": [
                    {"start_date": "2026-09-13", "end_date": "2026-09-18", "request_id": "LV-1"},
                ]
            },
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.WORKFLOW_NEW


def test_plan_builder_locked_leave_overlap_blocks():
    u = UnderstandingResult(
        workflow="leave",
        action=UnderstandingAction.START.value,
        confidence=0.9,
        field_updates=[
            FieldUpdate(field="start_date", value="2026-09-13"),
            FieldUpdate(field="end_date", value="2026-09-18"),
        ],
    )
    plan = PlanBuilder.build(
        _ctx(
            draft_locked=True,
            active_workflow_id="leave",
            user_message="13 september theke 18 september leave chai",
            draft_snapshot={"workflow_id": "leave", "locked": True, "fields": {}},
            conversation_facts={
                "submitted_leave_ranges": [
                    {"start_date": "2026-09-13", "end_date": "2026-09-18", "request_id": "LV-1"},
                ]
            },
        ),
        TurnDecision(pq=None, understanding=u, route_source="active"),
    )
    assert plan is not None
    assert plan.primary_op == PlanOp.SUBMITTED_LEAVE_OVERLAP
