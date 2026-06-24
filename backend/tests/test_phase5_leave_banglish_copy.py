"""Phase 5 — Banglish copy, sticky reply language, localized leave prompts."""

from __future__ import annotations

from chat.services.orchestrator import ChatOrchestrator
from chat.services.platform.response_composer import (
    ResponseComposer,
    leave_field_prompt,
    leave_validation_message,
    localized,
    normalize_reply_lang,
)
from chat.services.session_memory import (
    ActiveWorkflow,
    SessionMemory,
    WorkflowDraft,
    build_turn_context,
)
from chat.services.translator import resolve_reply_language
from tests.helpers.yaml_scenario_runner import llm_disabled


def test_normalize_reply_lang_buckets():
    assert normalize_reply_lang("banglish") == "banglish"
    assert normalize_reply_lang("bn") == "bn"
    assert normalize_reply_lang("en") == "en"


def test_leave_field_prompt_banglish():
    prompt = leave_field_prompt("day_scope", lang="banglish")
    assert "Puro din" in prompt or "ordho" in prompt
    assert leave_validation_message("end_date_gte_start", lang="banglish") == (
        "Shesh tarikh shurur tarikher pore hote hobe."
    )


def test_sticky_banglish_on_weak_replies():
    assert resolve_reply_language("agami 22 august leave chai") == "banglish"
    assert resolve_reply_language("full day", "banglish") == "banglish"
    assert resolve_reply_language("annual", "banglish") == "banglish"


def test_build_turn_context_uses_stored_reply_language():
    from chat.services.session_memory import ActiveWorkflow

    memory = SessionMemory(
        last_entities={"reply_language": "banglish"},
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={"leave_type": "annual", "start_date": "2026-08-22"},
            )
        },
    )
    ctx = build_turn_context(
        message="full day",
        memory=memory,
        conversation_history=[],
        trace_id="phase5-sticky",
        session_id="s1",
        company_id="c1",
        employee_id="e1",
    )
    assert ctx.reply_language == "banglish"


def test_leave_review_banglish_copy():
    from chat.services.platform.registry import get_workflow_definition

    composer = ResponseComposer()
    draft = WorkflowDraft(
        workflow_id="leave",
        fields={
            "leave_type": "annual",
            "day_scope": "full_day",
            "start_date": "2026-08-22",
            "reason": "family program",
        },
    )
    review = composer.leave_review(draft, get_workflow_definition("leave"), lang="banglish")
    assert "porjalochona" in review.lower() or "Chuti abedon" in review
    assert "ha" in review.lower()
    assert localized("banglish", en="yes", bn="ha", banglish="ha") == "ha"


def test_banglish_leave_collect_prompt_via_field_engine():
    from chat.services.platform.field_engine import FieldEngine
    from chat.services.platform.registry import get_workflow_definition

    engine = FieldEngine()
    defn = get_workflow_definition("leave")
    memory = SessionMemory()
    draft = WorkflowDraft(
        workflow_id="leave",
        fields={"leave_type": "annual", "start_date": "2026-08-22", "day_scope": "full_day"},
    )
    pq = engine.next_question(memory, draft, defn, lang="banglish")
    assert pq is not None
    assert pq.field == "reason"
    assert "karon" in pq.prompt.lower() or "skip" in pq.prompt.lower()


def test_orchestrator_persists_reply_language(db):
    with llm_disabled():
        orch = ChatOrchestrator()
        envelope = orch.run_chat(
            message="agami 22 august leave chai",
            session_id=None,
            company_id="co-phase5",
            employee_id="emp-phase5",
            trace_id="phase5-persist-lang-1",
        )
        session_id = envelope.get("_session_id") or ""
        bundle = orch.session_store.open(
            company_id="co-phase5",
            employee_id="emp-phase5",
            session_id=session_id,
        )
        assert bundle.memory.last_entities.get("reply_language") == "banglish"
