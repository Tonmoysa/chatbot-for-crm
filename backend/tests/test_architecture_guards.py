"""Architecture guard tests — single understand, plan-first executor."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import UnderstandingResult
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    WorkflowDraft,
    apply_state_patches,
    build_turn_context,
)


def test_pipeline_has_no_understand_calls():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "understand":
                calls.append(node.lineno)
    assert calls == [], f"pipeline must not call understand(); found at lines {calls}"


def test_pipeline_has_no_direct_reduce_calls():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id.startswith("reduce_"):
                calls.append(node.lineno)
    assert calls == [], f"pipeline must use StatePatchBuffer/apply_state_patches; found reduce_* at {calls}"


def test_pipeline_has_no_direct_field_apply_updates():
    """Phase 6 — draft mutations go through StatePatchBuffer.apply_field_updates."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    assert "fields.apply_updates(" not in source
    assert "draft.fields[" not in source
    assert "draft.fields.pop(" not in source


def test_execute_workflow_turn_requires_understanding():
    pipeline = WorkflowPipeline()
    memory = SessionMemory()
    from chat.services.session_memory import build_turn_context

    ctx = build_turn_context(
        message="hi",
        memory=memory,
        conversation_history=[],
        trace_id="t1",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    assert pipeline.execute_workflow_turn(
        "hi",
        memory=memory,
        understanding=None,  # type: ignore[arg-type]
        pq_decision=None,
        conversation_history=[],
        trace_id="t1",
        turn_context=ctx,
    ) is None


def test_apply_state_patches_roundtrip():
    memory = SessionMemory()
    apply_state_patches(
        memory,
        [
            {"op": "set_pending_confirmation", "value": "submit"},
            {"op": "merge_last_entities", "value": {"foo": "bar"}},
        ],
    )
    assert memory.pending_confirmation == "submit"
    assert memory.last_entities.get("foo") == "bar"
    apply_state_patches(memory, [{"op": "clear_pending_confirmation"}])
    assert memory.pending_confirmation is None


def test_orchestrator_has_no_inline_help_copy():
    """Phase 9 — orchestrator fallbacks must use ResponseComposer / informational_responses."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    assert "How can I help? You can ask about" not in source
    assert "Translation is briefly unavailable" not in source
    assert "_RULES_FOOTER_EN" not in source


def _service_sources_excluding_tests() -> list[tuple[str, Path]]:
    root = Path(__file__).resolve().parents[1]
    services = root / "chat" / "services"
    allowed = {
        services / "orchestrator.py",
        services / "session_memory.py",
    }
    hits: list[tuple[str, Path]] = []
    for path in services.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "build_turn_context(" in text:
            hits.append((str(path.relative_to(root)), path))
    return hits


def test_only_orchestrator_builds_turn_context_in_services():
    """Phase 1 — TurnContext is built once per turn in ChatOrchestrator only."""
    hits = _service_sources_excluding_tests()
    assert hits == [], f"build_turn_context forbidden outside orchestrator/session_memory: {hits}"


def test_pipeline_has_no_turn_context_fallback():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    assert "build_turn_context(" not in source
    assert "turn_context or " not in source


def test_orchestrator_has_no_workflow_routing():
    """Phase 2 — orchestrator must not route workflows or retry execution."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    forbidden = [
        "execute_workflow_turn(",
        "_run_workflow_turn",
        "_synthetic_leave_decision",
        "_synthetic_expense_decision",
        "_workflow_decision_candidates",
        "decide_turn(",
        "is_hr_today_date_query",
        "conversational_reply(",
    ]
    hits = [token for token in forbidden if token in source]
    assert hits == [], f"orchestrator must not contain routing logic: {hits}"


def test_orchestrator_delegates_to_decision_core():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    assert "decide_and_execute_turn(" in source


def test_decision_core_has_authoritative_entry():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    assert "def decide_and_execute_turn(" in source
    assert "authoritative_decision" in source


def test_execute_workflow_turn_called_once_per_chat_turn(db):
    """Phase 2 — one workflow execution path per user message."""
    from unittest.mock import patch

    from chat.services.orchestrator import ChatOrchestrator

    calls: list[str] = []

    def _track_execute(*args, **kwargs):
        calls.append("execute")
        return (
            "Tracked.",
            {
                "outcome": "COLLECTING",
                "execution_plan": {
                    "workflow_id": "leave",
                    "primary_op": "workflow_new",
                    "ops": ["workflow_new"],
                },
            },
        )

    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm, patch(
        "chat.services.conversational.LLMClient"
    ) as conv_llm, patch.object(
        WorkflowPipeline,
        "execute_workflow_turn",
        side_effect=_track_execute,
    ):
        pq_llm.return_value.is_configured.return_value = False
        ai_llm.return_value.is_configured.return_value = False
        conv_llm.return_value.is_configured.return_value = False
        orch = ChatOrchestrator()
        orch.run_chat(
            message="I need sick leave tomorrow",
            session_id=None,
            company_id="co-test",
            employee_id="emp-test",
            trace_id="arch-guard-1",
        )
    assert len(calls) == 1


def test_orchestrator_has_single_understand_call():
    """Phase 3 — production orchestrator calls understand() exactly once per turn."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    count = source.count(".understand(")
    assert count == 1, f"orchestrator must call understand() once; found {count}"


def test_classify_does_not_call_classify_rules():
    """Phase 3 — Decision Core classify maps from Understanding, not legacy _classify_rules."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    classify_block = source.split("def classify(", 1)[1].split("\n    def ", 1)[0]
    assert "_classify_rules(" not in classify_block


def test_ai_understanding_owns_greeting_signal():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "ai_understanding.py").read_text(encoding="utf-8")
    assert "is_greeting_or_chitchat" in source
    pqe = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    classify_block = pqe.split("def classify(", 1)[1].split("\n    def ", 1)[0]
    assert "is_greeting_or_chitchat" not in classify_block


def test_legacy_classify_rules_removed():
    """Phase 3 — legacy _classify_rules path removed from Decision Core."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    assert "def _classify_rules(" not in source
    assert "def _classify_llm(" not in source


def test_decision_core_route_turn_removed():
    """Phase 4.4 — legacy route_turn deleted; plan path is authoritative."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    assert "def route_turn(" not in source
    assert "_execute_planned_turn(" in source
    assert "execute_workflow_turn(" in source


def test_decision_core_no_inline_conversational_routing():
    """Phase 4.4 — greeting/conversational fallbacks live in PlanBuilder ops only."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    decide_block = source.split("def decide_and_execute_turn(", 1)[1].split("\n    def ", 1)[0]
    assert "conversational_reply(" not in decide_block
    assert "is_platform_only_scenario(" not in decide_block
    assert "build_platform_response(" not in decide_block


def test_plan_builder_policy_wins_during_active_leave():
    """Phase 4.4 — policy/status/OOS use informational plan even during leave wizard."""
    from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
    from chat.services.platform.pipeline import PlanBuilder
    from chat.services.platform.schemas import PlanOp, TurnDecision, UnderstandingAction, UnderstandingResult

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting", draft_id="default"),
    )
    ctx = build_turn_context(
        message="what is sick leave policy?",
        memory=memory,
        conversation_history=[],
        trace_id="guard-policy-leave",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    plan = PlanBuilder.build(
        ctx,
        TurnDecision(
            pq=PendingQuestionDecision(
                kind=MessageIntentKind.ASK_POLICY,
                confidence=0.9,
                reasoning="policy",
                source="rules",
                blocks_new_workflow=True,
            ),
            understanding=UnderstandingResult(
                workflow="none",
                action=UnderstandingAction.CLARIFICATION_NEEDED.value,
                confidence=0.9,
            ),
            route_source="pending",
        ),
    )
    assert plan is not None
    assert plan.workflow_id == "informational"
    assert plan.primary_op == PlanOp.REPLY_POLICY


def test_classify_does_not_invoke_understanding_layer():
    """Phase 3 — classify maps provided Understanding; does not re-run understand()."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    classify_block = source.split("def classify(", 1)[1].split("\n    def ", 1)[0]
    assert "AIUnderstandingLayer" not in classify_block
    assert ".understand(" not in classify_block


def test_phase5_unified_workflow_plan_ops():
    """Phase 5 — leave/expense PlanOps alias unified WORKFLOW_* handlers."""
    from chat.services.platform.schemas import PlanOp

    assert PlanOp.LEAVE_COLLECT is PlanOp.WORKFLOW_COLLECT
    assert PlanOp.EXPENSE_APPLY_UPDATES is PlanOp.WORKFLOW_APPLY_UPDATES
    assert PlanOp.LEAVE_COLLECT.value == "workflow_collect"


def test_phase5_single_executor_dispatch():
    """Phase 5 — workflow ops dispatch through _run_workflow_plan_op, not duplicated branches."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    assert "def _run_workflow_plan_op(" in source
    assert "def _build_workflow_plan(" in source
    assert "def _maybe_execute_workflow_plan(" not in source
    run_block = source.split("def _run_plan_op(", 1)[1].split("\n    def ", 1)[0]
    assert "elif op == PlanOp.LEAVE_COLLECT:" not in run_block
    assert "elif op == PlanOp.EXPENSE_COLLECT:" not in run_block
    assert "WORKFLOW_PLAN_OPS" in run_block


def test_phase5_execute_workflow_turn_is_ssot():
    """Phase 5 — production path calls _maybe_execute_turn_plan from execute_workflow_turn."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    exec_block = source.split("def execute_workflow_turn(", 1)[1].split("\n    def ", 1)[0]
    assert "_maybe_execute_turn_plan(" in exec_block
    assert "_maybe_execute_workflow_plan(" not in exec_block
    pqe = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    assert "execute_workflow_turn(" in pqe


def test_phase4_no_early_exit_bypass():
    """Phase 4 — today/translation use PlanBuilder ops, not try_early_exits finalize."""
    root = Path(__file__).resolve().parents[1]
    orch = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    pqe = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    pipeline = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    assert "try_early_exits(" not in orch
    assert "def try_early_exits(" not in pqe
    assert "def _route_early_exits(" not in pqe
    assert "REPLY_TODAY_DATE" in pipeline
    assert "REPLY_TRANSLATION" in pipeline
    assert "informational_fallback_plan" in pipeline
    assert "decision_core_fallback" not in pqe.split("def decide_and_execute_turn(", 1)[1].split("\n    def ", 1)[0]


def test_phase4_informational_priority_ssot():
    """Phase 4 — message-level policy/status/today rules live in one helper."""
    root = Path(__file__).resolve().parents[1]
    pqe = (root / "chat" / "services" / "pending_question_engine.py").read_text(encoding="utf-8")
    assert "def informational_priority_decision(" in pqe
    classify_block = pqe.split("def classify(", 1)[1].split("\n    def ", 1)[0]
    assert "informational_priority_decision(" in classify_block


def test_orchestrator_uses_session_store_not_dual_writes():
    """Phase 7 — orchestrator loads/commits via SessionStore only."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    assert "SessionStore" in source
    assert "session_store.open(" in source
    assert "session_store.commit_turn(" in source
    assert "ConversationMemoryStore" not in source
    assert "load_session_memory(" not in source
    assert "save_session_memory(" not in source
    assert "memory.append(" not in source


def test_memory_store_no_turn_count_side_effect_on_append():
    """Phase 7 — turn_count bumps only in SessionStore.commit_turn."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "memory_store.py").read_text(encoding="utf-8")
    assert "_bump_turn_count" not in source
    assert "append_transcript_turn(" in source


def test_pipeline_uses_response_composer_facade_only():
    """Phase 8 — pipeline informational replies go through ResponseComposer."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "platform" / "pipeline.py").read_text(encoding="utf-8")
    assert "self.composer.policy_turn(" in source
    assert "self.composer.status_turn(" in source
    assert "self.composer.conversational(" in source
    assert "self.composer.out_of_scope(" in source
    assert "self.composer.today_date(" in source
    assert "conversational_reply" not in source
    assert "build_user_message" not in source
    assert "resolve_request_status_turn" not in source
    assert "build_out_of_scope_message" not in source
    assert "format_today_date_reply" not in source
    assert "informational_responses" not in source


def test_phase9_yaml_scenario_runner_wired():
    """Phase 9 — conversation regressions live in tests/scenarios/*.yaml."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "tests" / "helpers" / "yaml_scenario_runner.py").is_file()
    assert (root / "tests" / "test_yaml_scenarios.py").is_file()
    yaml_files = list((root / "tests" / "scenarios").glob("*.yaml"))
    assert yaml_files, "expected tests/scenarios/*.yaml"
    from tests.helpers.yaml_scenario_runner import load_paraphrase_cases, load_yaml_scenarios

    scenarios = load_yaml_scenarios()
    assert len(scenarios) >= 10
    ids = {s["id"] for s in scenarios}
    assert "expense_during_leave_asks_switch" in ids
    paraphrases = load_paraphrase_cases()
    assert len(paraphrases) >= 30
    assert (root / "tests" / "test_phase3_paraphrase_scenarios.py").is_file()
    assert (root / "tests" / "test_phase3_llm_regression.py").is_file()


def test_phase3_llm_regression_gated():
    """Phase 3 — LLM tests require explicit opt-in."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "tests" / "helpers" / "yaml_scenario_runner.py").read_text(encoding="utf-8")
    assert "def llm_tests_enabled(" in source
    assert "def load_paraphrase_cases(" in source
    assert "def run_paraphrase_case(" in source


def test_phase10_architecture_doc_linked():
    """Phase 10 — architecture reference doc present."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "ARCHITECTURE.md").is_file()


def test_phase4_reason_cannot_change_without_modify_op():
    """Phase 4 — reducer blocks unvalidated reason updates during review."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "session_memory.py").read_text(encoding="utf-8")
    block = source.split("def reduce_apply_field_updates(", 1)[1].split("\ndef ", 1)[0]
    assert "review_validated" in block
    assert "is_leave_complaint_reason_value" in block

    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "reason": "keep me",
                    "start_date": "2026-07-01",
                },
            )
        },
        pending_confirmation="submit",
    )
    from chat.services.session_memory import reduce_apply_field_updates

    reduce_apply_field_updates(
        memory,
        "default",
        [{"field": "reason", "value": "overwrite attempt", "action": "set"}],
        message="random unclear text",
    )
    assert memory.active_draft().fields.get("reason") == "keep me"

    reduce_apply_field_updates(
        memory,
        "default",
        [{"field": "reason", "value": "family wedding", "action": "set"}],
        message="reason ta family wedding",
        review_validated=True,
    )
    assert memory.active_draft().fields.get("reason") == "family wedding"


def test_phase_f_leave_prompts_centralized_in_composer():
    """Phase F / 5 — leave slot/validation copy SSOT lives in response_composer (EN/BN/Banglish)."""
    root = Path(__file__).resolve().parents[1]
    composer_source = (root / "chat" / "services" / "platform" / "response_composer.py").read_text(encoding="utf-8")
    field_engine_source = (root / "chat" / "services" / "platform" / "field_engine.py").read_text(encoding="utf-8")
    assert "LEAVE_FIELD_PROMPTS" in composer_source
    assert "banglish" in composer_source
    assert "normalize_reply_lang" in composer_source
    assert "leave_field_prompt(" in composer_source
    assert "leave_field_prompt(" in field_engine_source
    assert "What type of leave would you like?" not in field_engine_source
