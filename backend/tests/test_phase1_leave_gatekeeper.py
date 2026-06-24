"""Phase 1 — leave gatekeeper: rules override LLM, contextual submit, conversational guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chat.services.conversational import _sanitize_fake_workflow_claims, conversational_reply
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.intent_rules import parse_submit_workflow
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from chat.services.session_store import SessionStore


LONG_LEAVE_NARRATIVE = (
    "Hi, amar ekta leave apply korte hobe. Amar ma onek osustho, tai take hospital e niye "
    "jete hobe ebong tar dekhashona korte hobe. Ei karone ami 15 July 2026 theke 18 July 2026 "
    "porjonto office e aste parbo na. Mot 4 diner leave lagbe. Eta Annual Leave hisebe apply "
    "korte chai. Amar team lead ke agei inform kore diyechi. Ekhon amar leave request ta "
    "prepare kore dao."
)

_BAD_LLM_LEAVE = UnderstandingResult(
    goal="Greeting",
    workflow="none",
    action=UnderstandingAction.NONE.value,
    confidence=0.92,
    is_greeting=True,
    reasoning="User greeted and described situation conversationally.",
    source="llm",
)


def test_parse_bare_submit_with_active_leave():
    assert parse_submit_workflow("okay ekhon submit koro", active_workflow_id="leave") == "leave"
    assert parse_submit_workflow("submit koro", active_workflow_id="leave") == "leave"
    assert parse_submit_workflow("okay ekhon submit koro", active_workflow_id=None) is None


def test_gatekeeper_overrides_llm_on_long_leave_narrative():
    layer = AIUnderstandingLayer()
    client = MagicMock()
    client.is_configured.return_value = True

    with patch.object(layer, "_understand_llm", return_value=_BAD_LLM_LEAVE):
        result = layer.understand(
            LONG_LEAVE_NARRATIVE,
            memory=SessionMemory(),
            conversation_history=[],
            trace_id="gatekeeper-leave",
            llm=client,
        )

    assert result.source == "rules_gatekeeper"
    assert result.workflow == "leave"
    assert result.action == UnderstandingAction.START.value
    assert any(u.field == "start_date" for u in (result.field_updates or []))


def test_gatekeeper_contextual_submit_with_active_leave():
    layer = AIUnderstandingLayer()
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting", draft_id="default")
    memory.workflow_drafts = {
        "default": WorkflowDraft(
            workflow_id="leave",
            fields={"leave_type": "annual", "start_date": "2026-07-15"},
        )
    }

    bad_llm = UnderstandingResult(
        workflow="none",
        action=UnderstandingAction.CLARIFICATION_NEEDED.value,
        confidence=0.5,
        reasoning="Unclear.",
        source="llm",
    )
    client = MagicMock()
    client.is_configured.return_value = True

    with patch.object(layer, "_understand_llm", return_value=bad_llm):
        result = layer.understand(
            "okay ekhon submit koro",
            memory=memory,
            conversation_history=[],
            trace_id="gatekeeper-submit",
            llm=client,
        )

    assert result.source == "rules_gatekeeper"
    assert result.workflow == "leave"
    assert result.action == UnderstandingAction.SUBMIT.value


def test_conversational_sanitizer_strips_fake_success():
    fake = "আপনার লেভ রিকোয়েস্ট প্রস্তুত করা হয়েছে। আমি এটা আপনার টিম লিডের সাথে শেয়ার করব।"
    safe = _sanitize_fake_workflow_claims(fake, user_lang="bn")
    assert "প্রস্তুত" not in safe
    assert "টিম লিড" not in safe
    assert "workflow" in safe.lower() or "leave" in safe.lower()


def test_conversational_sanitizer_allows_normal_reply():
    normal = "ভালো আছি, ধন্যবাদ! আপনি কেমন আছেন?"
    assert _sanitize_fake_workflow_claims(normal, user_lang="bn") == normal


def test_conversational_reply_sanitizes_configured_llm_output():
    client = MagicMock()
    client.is_configured.return_value = True
    client.chat_text.return_value = "Your leave request is prepared and shared with your team lead."

    out = conversational_reply(
        message="amar leave lagbe",
        context_lines=[],
        trace_id="conv-guard",
        llm=client,
    )
    assert out is not None
    assert "prepared" not in out.lower()
    assert "team lead" not in out.lower()


@pytest.mark.django_db
def test_orchestrator_long_narrative_starts_leave_when_llm_misroutes():
    from chat.services.orchestrator import ChatOrchestrator

    client = MagicMock()
    client.is_configured.return_value = True

    extract_fields = {
        "leave_type": "annual",
        "start_date": "2026-07-15",
        "end_date": "2026-07-18",
        "day_scope": "full_day",
    }

    with patch("chat.services.platform.ai_understanding.LLMClient", return_value=client), patch(
        "chat.services.conversational.LLMClient"
    ) as conv_llm, patch(
        "chat.services.platform.field_extractors.leave.extract_leave_fields_via_llm",
        return_value=extract_fields,
    ):
        conv_llm.return_value.is_configured.return_value = False
        layer = AIUnderstandingLayer()

        def _bad_llm(*args, **kwargs):
            return _BAD_LLM_LEAVE

        with patch.object(AIUnderstandingLayer, "_understand_llm", _bad_llm):
            orch = ChatOrchestrator()
            out = orch.run_chat(
                message=LONG_LEAVE_NARRATIVE,
                session_id=None,
                company_id="co-test",
                employee_id="emp-test",
                trace_id="orch-gatekeeper",
            )

    msg = (out.get("response") or {}).get("message") or ""
    assert "প্রস্তুত করা হয়েছে" not in msg
    assert "team lead" not in msg.lower()

    bundle = SessionStore().open(
        company_id="co-test",
        employee_id="emp-test",
        session_id=out.get("_session_id") or "",
    )
    memory = bundle.memory
    assert memory.active_workflow is not None
    assert memory.active_workflow.id == "leave"
    draft = memory.active_draft()
    assert draft is not None
    assert draft.fields.get("start_date")
    assert draft.fields.get("leave_type") == "annual"
