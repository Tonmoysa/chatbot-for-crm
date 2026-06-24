"""Regression tests from live transcript (greeting, leave dates, expense switch)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chat.services.platform.intent_rules import is_greeting_or_chitchat, is_workflow_interrupt_expense
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionEngine
from chat.services.session_memory import (
    ActiveWorkflow,
    PendingQuestion,
    SessionMemory,
    StatePatchBuffer,
    SuspendedWorkflow,
    WorkflowDraft,
)
from tests.helpers.pipeline_handle import handle_with_rules_understanding


LEAVE_NARRATIVE_BN = (
    "Actually last kichudin dhore family related kichu urgent issue cholche. "
    "Amar mama onekdin dhore osustho chilen, ekhon take Dhaka te niye treatment korate hobe "
    "ebong family member hisebe amar uposthit thaka khub dorkar. "
    "Ei karone ami Monday (14th July) theke Wednesday (16th July) porjonto "
    "office e attend korte parbo na."
)


def test_greeting_detection():
    assert is_greeting_or_chitchat("hi")
    assert is_greeting_or_chitchat("hello")
    assert is_greeting_or_chitchat("Hello!")
    assert not is_greeting_or_chitchat("hello I need leave")


def test_bare_expense_interrupts_leave():
    assert is_workflow_interrupt_expense("expense", active_workflow="leave")
    assert WorkflowManager.parse_switch_reply("expense", "leave", "expense") == "switch"


def test_leave_submit_compound_expense_switches_to_expense():
    """Leave submit review + compound expense message must switch to expense workflow."""
    from chat.services.platform.ai_understanding import AIUnderstandingLayer
    from chat.services.session_memory import build_turn_context
    from tests.helpers.expense_llm_mock import mock_expense_llm
    from tests.helpers.yaml_scenario_runner import llm_disabled

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-09-14",
                    "end_date": "2026-09-17",
                    "reason": "Grandfather unwell; family emergency in village",
                },
            )
        },
        pending_confirmation="submit",
    )
    msg = (
        "Aj office jawar somoy bus e 120 taka lagse. Dupure lunch korlam 280 taka. "
        "Bikale ekta snack kheyechi 70 taka. Ferar somoy metro te 90 taka lagse "
        "Mirpur theke Agargaon porjonto. Ar ekta 150 taka expense hoise but category mone nei ekhon."
    )
    engine = PendingQuestionEngine()
    layer = AIUnderstandingLayer()
    with mock_expense_llm(), llm_disabled():
        u = layer.understand(
            msg,
            memory=memory,
            conversation_history=[],
            trace_id="test-leave-submit-expense-switch",
        )
        pq = engine.classify(
            msg,
            memory=memory,
            conversation_history=[],
            trace_id="test-leave-submit-expense-switch",
            understanding=u,
        )
        assert pq.kind == MessageIntentKind.SWITCH_WORKFLOW
        assert pq.target_workflow == "expense"
        assert u.workflow == "expense" or u.interrupt_workflow == "expense"

        pipeline = WorkflowPipeline()
        turn_context = build_turn_context(
            message=msg,
            memory=memory,
            conversation_history=[],
            trace_id="test-leave-submit-expense-switch",
            session_id="test-session",
            company_id="test-company",
            employee_id="test-employee",
            idempotency_key="",
        )
        resp, decision = pipeline.execute_workflow_turn(
            msg,
            memory=memory,
            understanding=u,
            pq_decision=pq,
            conversation_history=[],
            trace_id="test-leave-submit-expense-switch",
            turn_context=turn_context,
            route_source="active",
        )
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"
    draft = memory.active_draft()
    assert draft is not None
    assert draft.workflow_id == "expense"
    items = draft.fields.get("items") or []
    amounts = {float(i.get("amount") or 0) for i in items}
    assert 120.0 in amounts
    assert "chuti saransho" not in resp.lower()
    assert decision.get("outcome") != "BLOCKED"


def test_switch_confirm_understands_bn_navigation_replies():
    assert WorkflowManager.parse_switch_reply("leave e jao", "expense", "leave") == "switch"
    assert WorkflowManager.parse_switch_reply("expense e jao", "leave", "expense") == "switch"
    assert WorkflowManager.parse_switch_reply("continue leave", "leave", "expense") == "continue"


def test_switch_confirm_resolves_expense_reply():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting")
    memory.workflow_drafts["default"] = WorkflowDraft(
        workflow_id="leave",
        fields={"leave_type": "annual", "reason": "family"},
    )
    memory.pending_question = PendingQuestion(
        field="start_date",
        prompt="Start date?",
        workflow_id="leave",
        asked_at_turn=1,
    )
    memory.pending_confirmation = "switch:leave:expense"
    memory.last_entities = {"switch_pending_message": "100 taka bus expense"}

    pipeline = WorkflowPipeline()
    from chat.services.pending_question_engine import PendingQuestionDecision

    msg, decision = handle_with_rules_understanding(
        pipeline,
        "expense",
        memory=memory,
        pq_decision=PendingQuestionDecision(
            kind=MessageIntentKind.SWITCH_WORKFLOW,
            confidence=0.9,
            reasoning="test",
            source="test",
            blocks_new_workflow=True,
            target_workflow="expense",
        ),
        trace_id="test-switch-resolve",
    )
    assert memory.active_workflow and memory.active_workflow.id == "expense"
    assert memory.pending_confirmation is None
    assert "expense" in msg.lower() or "bus" in msg.lower() or "100" in msg


def test_switch_confirm_resolves_leave_reply_without_expense_corruption():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="expense", stage="collecting")
    memory.workflow_drafts["default"] = WorkflowDraft(
        workflow_id="expense",
        fields={"items": [{"category": "meals", "amount": 120.0}]},
    )
    memory.suspended_workflows.append(
        SuspendedWorkflow(
            workflow_id="leave",
            stage="collecting",
            draft_id="leave-draft",
            suspended_at_turn=1,
        )
    )
    memory.workflow_drafts["leave-draft"] = WorkflowDraft(
        workflow_id="leave",
        fields={"leave_type": "annual", "day_scope": "full_day"},
    )
    memory.pending_confirmation = "switch:expense:leave"
    memory.last_entities = {"switch_pending_message": "agami 20 august annual leave chai"}

    pipeline = WorkflowPipeline()
    from chat.services.pending_question_engine import PendingQuestionDecision

    def _mock_extract(msg, mem=None, **kwargs):
        low = (msg or "").lower()
        if "20 august" in low:
            return {"leave_type": "annual", "start_date": "2026-08-20", "day_scope": "full_day"}
        return {}

    with patch(
        "chat.services.platform.field_extractors.leave.extract_leave_fields_via_llm",
        side_effect=_mock_extract,
    ):
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "leave e jao",
            memory=memory,
            pq_decision=PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=0.9,
                reasoning="resume leave",
                source="test",
                blocks_new_workflow=True,
                target_workflow="leave",
            ),
            trace_id="test-switch-resolve-leave",
        )
    assert memory.active_workflow and memory.active_workflow.id == "leave"
    draft = memory.active_draft()
    assert draft is not None
    assert draft.workflow_id == "leave"
    assert draft.fields.get("start_date") == "2026-08-20"
    assert "items" not in draft.fields
    assert decision.get("outcome") == "NEEDS_INPUT"
    assert "leave" in msg.lower() or "august" in msg.lower()


def test_leave_narrative_merges_dates_on_understand():
    from unittest.mock import patch

    from chat.services.platform.ai_understanding import AIUnderstandingLayer

    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    with patch("chat.services.platform.ai_understanding.LLMClient") as mock_cls:
        mock_cls.return_value.is_configured.return_value = False
        with patch(
            "chat.services.platform.field_extractors.leave.extract_leave_fields_via_llm",
            return_value={"start_date": "2026-07-14", "end_date": "2026-07-16"},
        ):
            u = layer.understand(
                LEAVE_NARRATIVE_BN,
                memory=memory,
                conversation_history=[],
                trace_id="test-leave-dates",
            )
    fields = {upd.field: upd.value for upd in u.field_updates}
    assert "start_date" in fields
    assert "end_date" in fields
    assert u.action != "submit"


def test_pq_greeting_not_out_of_scope():
    from chat.services.platform.ai_understanding import AIUnderstandingLayer

    engine = PendingQuestionEngine()
    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    u = layer.understand("hello", memory=memory, conversation_history=[], trace_id="test-hello-pq")
    decision = engine.classify(
        "hello",
        memory=memory,
        conversation_history=[],
        trace_id="test-hello-pq",
        understanding=u,
    )
    assert decision.kind != MessageIntentKind.OUT_OF_SCOPE


def test_family_illness_not_sick_leave():
    """Family illness narrative — LLM omits sick leave_type; system must not add it via rules."""
    from chat.services.platform.ai_understanding import AIUnderstandingLayer

    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    u = layer._sanitize_leave_result(
        LEAVE_NARRATIVE_BN,
        layer._parse_result(
            {
                "goal": "Start leave",
                "workflow": "leave",
                "action": "start",
                "confidence": 0.9,
                "field_updates": [
                    {"field": "start_date", "value": "2026-07-11"},
                    {"field": "end_date", "value": "2026-07-13"},
                    {"field": "day_scope", "value": "full_day"},
                    {"field": "reason", "value": "Mama needs treatment in Dhaka"},
                ],
                "reasoning": "Family care leave — ask annual or lwop",
            },
            source="llm",
        ),
        memory=memory,
    )
    fields = {upd.field: upd.value for upd in u.field_updates}
    assert fields.get("leave_type") is None
    assert fields.get("start_date")
    assert fields.get("end_date")
    assert "reason" in fields


def test_semantic_sick_inference_deferred_to_llm():
    from chat.services.platform.field_extractors.leave import infer_leave_type_from_text

    assert infer_leave_type_from_text("ami onek osustho, kal theke sick leave lagbe") is None
    assert infer_leave_type_from_text("sick") is None


def test_explicit_sick_leave_type_via_llm_collect():
    from chat.services.session_memory import ActiveWorkflow, PendingQuestion, SessionMemory
    from tests.helpers.leave_llm_mock import mock_leave_llm

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        pending_question=PendingQuestion(
            field="leave_type",
            prompt="Type?",
            workflow_id="leave",
            asked_at_turn=1,
        ),
    )
    with mock_leave_llm():
        from chat.services.platform.field_extractors.leave import interpret_leave_collect_message

        out = interpret_leave_collect_message("sick", memory)
    assert out.get("leave_type") == "sick"


def test_leave_collection_asks_reason_once_then_allows_skip():
    from chat.services.platform.field_engine import FieldEngine
    from chat.services.platform.field_extractors.leave import is_reason_skip_message
    from chat.services.platform.registry import get_workflow_definition

    engine = FieldEngine()
    defn = get_workflow_definition("leave")
    draft = WorkflowDraft(
        workflow_id="leave",
        fields={"leave_type": "annual", "day_scope": "full_day", "start_date": "2026-08-10"},
    )
    assert engine.missing_fields(draft, defn) == ["reason"]
    assert is_reason_skip_message("skip")
    draft.fields["reason_skipped"] = True
    assert engine.missing_fields(draft, defn) == []


def test_medical_document_nai_not_saved():
    from chat.services.platform.field_extractors.leave import parse_medical_document_field

    assert parse_medical_document_field("medical document nai") is None
    assert parse_medical_document_field("false") is None


def test_medical_document_decline_keeps_sick_leave():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "sick",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "day_scope": "full_day",
                },
            )
        },
        pending_question=PendingQuestion(
            field="medical_document",
            prompt="Upload medical document",
            workflow_id="leave",
            asked_at_turn=1,
        ),
    )
    pipeline = WorkflowPipeline()
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="test",
        source="rules",
        blocks_new_workflow=True,
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        "medical document nai",
        memory=memory,
        pq_decision=pq,
        trace_id="test-med-decline",
    )
    draft = memory.active_draft()
    assert draft.fields.get("leave_type") == "sick"
    assert draft.fields.get("medical_document_skipped") is True
    assert _any(msg, r"later|continuing|sick|চালিয়ে")
    assert decision.get("outcome") in ("NEEDS_INPUT", "INFORMATIONAL")


def test_review_arms_submit_confirmation():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "reason": "family emergency",
                },
            )
        },
    )
    pipeline = WorkflowPipeline()
    state = StatePatchBuffer(memory)
    msg, decision = pipeline._continue_collection(
        memory,
        pipeline.manager.ensure_definition("leave"),
        lang="en",
        prefix="Saved day scope.",
        state=state,
    )
    state.flush()
    assert memory.pending_confirmation == "submit"
    assert "yes" in msg.lower()
    assert decision.get("awaiting_confirmation") is True


def test_yes_after_review_submits_leave(monkeypatch):
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="review"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "reason": "family emergency",
                },
            )
        },
        pending_confirmation="submit",
    )
    from chat.services.pending_question_engine import PendingQuestionDecision

    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.95,
        reasoning="submit confirm",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )
    pipeline = WorkflowPipeline()

    class _CRM:
        def create_request(self, **kwargs):
            return {"request_id": "LV-TEST-1", "record": {}}

    monkeypatch.setattr(
        "chat.services.platform.submission_engine.get_crm_adapter",
        lambda: _CRM(),
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        "yes",
        memory=memory,
        pq_decision=pq,
        trace_id="test-yes-submit",
        company_id="c1",
        employee_id="e1",
        session_id="s1",
    )
    assert decision.get("outcome") == "SUBMITTED"
    assert memory.active_draft().locked
    assert "LV-TEST-1" in msg


def test_submitted_same_dates_blocked(monkeypatch):
    memory = SessionMemory(
        conversation_facts={
            "submitted_leave_ranges": [
                {"start_date": "2026-07-14", "end_date": "2026-07-16", "request_id": "LV-1"},
            ]
        },
    )
    from chat.services.platform.schemas import UnderstandingResult, FieldUpdate

    pipeline = WorkflowPipeline()
    u = UnderstandingResult(
        goal="Start leave",
        workflow="leave",
        action="start",
        confidence=0.9,
        field_updates=[
            FieldUpdate(field="start_date", value="2026-07-14"),
            FieldUpdate(field="end_date", value="2026-07-16"),
        ],
        source="rules",
    )
    blocked = pipeline._block_submitted_leave_overlap(memory, u, "en")
    assert blocked is not None
    assert "already submitted" in blocked[0].lower() or "ইতিমধ্যে" in blocked[0]


def _locked_submitted_leave_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="submitted", draft_id="default"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                locked=True,
                status="submitted",
                submitted_request_id="MOCK-DA56C8AEC4",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-05",
                    "end_date": "2026-08-09",
                    "reason": "Younger sister's wedding",
                },
            )
        },
        conversation_facts={
            "submitted_leave_ranges": [
                {
                    "start_date": "2026-08-05",
                    "end_date": "2026-08-09",
                    "request_id": "MOCK-DA56C8AEC4",
                }
            ]
        },
    )


def test_post_submit_leave_summary():
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = _locked_submitted_leave_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.CLARIFICATION_NEEDED,
        confidence=0.9,
        reasoning="summary",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        ".leave er summery ta dekhao",
        memory=memory,
        pq_decision=pq,
        trace_id="test-post-submit-summary",
    )
    assert decision.get("outcome") == "INFORMATIONAL"
    assert "MOCK-DA56C8AEC4" in msg
    assert "05" in msg and "august" in msg.lower()
    assert "annual" in msg.lower()


def test_post_submit_new_non_overlapping_leave(monkeypatch):
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = _locked_submitted_leave_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="new leave",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        "Hello, amar ekta leave apply korte hobe. Next Wednesday theke Friday porjonto ami office e aste parbo na. Annual Leave hisebe consider korle bhalo hoy.",
        memory=memory,
        pq_decision=pq,
        trace_id="test-post-submit-new-leave",
        route_source="active",
    )
    draft = memory.active_draft()
    assert draft is not None
    assert not draft.locked
    assert decision.get("outcome") != "BLOCKED"
    assert "already submitted" not in msg.lower()
    assert draft.fields.get("start_date") or "leave" in msg.lower()


def test_post_submit_expense_after_locked_leave(monkeypatch):
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = _locked_submitted_leave_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="expense after submit",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="expense",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        "Aj office jawar somoy bus e 120 taka lagse. Dupure lunch korlam 280 taka.",
        memory=memory,
        pq_decision=pq,
        trace_id="test-post-submit-expense",
        route_source="active",
    )
    assert "already submitted" not in msg.lower()
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "expense"
    draft = memory.active_draft()
    assert draft is not None
    assert draft.workflow_id == "expense"
    assert not draft.locked


def test_post_submit_new_leave_with_review_phrase(monkeypatch):
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = _locked_submitted_leave_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="new leave",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        (
            "Hello, amar ekta leave apply korte hobe. "
            "14 September 2026 theke 17 September 2026 porjonto office attend korte parbo na. "
            "Eta Annual Leave hisebe apply korte chai. Ekhon review dekhao."
        ),
        memory=memory,
        pq_decision=pq,
        trace_id="test-post-submit-review-new-dates",
        route_source="active",
    )
    draft = memory.active_draft()
    assert draft is not None
    assert not draft.locked
    assert "already submitted" not in msg.lower() or draft.fields.get("start_date")
    assert decision.get("outcome") != "BLOCKED"


def test_post_submit_same_dates_overlap_message(monkeypatch):
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = _locked_submitted_leave_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.NEW_WORKFLOW,
        confidence=0.9,
        reasoning="duplicate dates",
        source="rules",
        blocks_new_workflow=False,
        target_workflow="leave",
    )
    msg, decision = handle_with_rules_understanding(
        pipeline,
        "5 august theke 9 august annual leave chai",
        memory=memory,
        pq_decision=pq,
        trace_id="test-post-submit-overlap",
        route_source="active",
    )
    low = msg.lower()
    assert "already submitted" in low or "ইতিমধ্যে" in msg or "submitted leave" in low


def _any(text: str, pattern: str) -> bool:
    import re
    return bool(re.search(pattern, text, re.I))


def test_reducer_reason_immunity_during_submit_review():
    from chat.services.session_memory import reduce_apply_field_updates

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "reason": "family program",
                },
            )
        },
        pending_confirmation="submit",
    )
    reduce_apply_field_updates(
        memory,
        "default",
        [{"field": "reason", "value": "leave e back koro garbage", "action": "set"}],
        message="leave e back koro..ami kichu besoy modify korbo",
    )
    assert memory.active_draft().fields.get("reason") == "family program"


def test_end_date_modify_at_submit_preserves_reason():
    from chat.services.pending_question_engine import PendingQuestionDecision

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-08-22",
                    "end_date": "2026-08-22",
                    "reason": "family program",
                },
            )
        },
        pending_confirmation="submit",
    )
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="end date modify",
        source="rules",
        blocks_new_workflow=True,
    )
    handle_with_rules_understanding(
        pipeline,
        "end date 7 july hobe",
        memory=memory,
        pq_decision=pq,
        trace_id="transcript-end-date-safe",
        route_source="active",
    )
    draft = memory.active_draft()
    assert draft.fields.get("reason") == "family program"
    assert draft.fields.get("end_date") == "2026-07-07"
