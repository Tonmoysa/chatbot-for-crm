"""Phase 11 — legacy module deprecation and debug endpoint gating."""

from __future__ import annotations

import warnings

import pytest
from django.test import override_settings
from django.urls import reverse

from chat.services.informational_responses import evaluate_request_status_decision, resolve_request_status_turn
from chat.services.orchestrator import ChatOrchestrator
from chat.services.reference_extractors import extract_reference_entities
from chat.services.session_memory import ActiveWorkflow, SessionMemory
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision


def test_reference_extractors_finds_request_id():
    entities = extract_reference_entities("status for ref HR-2025-00123")
    assert entities["request_id"] == "HR-2025-00123"


def test_status_decision_not_found():
    decision = evaluate_request_status_decision(
        entities={"request_id": "HR-9999"},
        crm_context={"status": "NOT_FOUND"},
    )
    assert decision["outcome"] == "NEEDS_CLARIFICATION"


def test_resolve_request_status_turn_without_id():
    msg, status, decision, entities, request_id = resolve_request_status_turn(
        "what is my request status?",
        company_id="c1",
        employee_id="e1",
        session_id="s1",
    )
    assert request_id == ""
    assert status == "needs_input"
    assert "reference" in msg.lower()


def test_legacy_path_always_disabled():
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting")
    u = UnderstandingResult(workflow="leave", action=UnderstandingAction.COLLECT.value)
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.ANSWER_PENDING,
        confidence=0.9,
        reasoning="test",
        source="test",
        blocks_new_workflow=True,
    )
    assert ChatOrchestrator._legacy_path_allowed(memory, u, pq) is False


@override_settings(ENABLE_LEGACY_DEBUG_ENDPOINTS=False)
def test_intent_debug_endpoint_returns_410(api_client):
    url = reverse("chat:intent")
    resp = api_client.post(
        url,
        {"message": "leave please", "company_id": "c1", "employee_id": "e1", "session_id": "s1"},
        format="json",
    )
    assert resp.status_code == 410
    assert resp.data["status"] == "deprecated"


@override_settings(ENABLE_LEGACY_DEBUG_ENDPOINTS=False)
def test_extract_debug_endpoint_returns_410(api_client):
    url = reverse("chat:extract")
    resp = api_client.post(
        url,
        {"message": "ref HR-1", "intent": "REQUEST_STATUS", "company_id": "c1", "employee_id": "e1", "session_id": "s1"},
        format="json",
    )
    assert resp.status_code == 410


@override_settings(ENABLE_LEGACY_DEBUG_ENDPOINTS=False)
def test_decision_debug_endpoint_returns_410(api_client):
    url = reverse("chat:decision")
    resp = api_client.post(
        url,
        {"intent": "REQUEST_STATUS", "entities": {}, "company_id": "c1", "employee_id": "e1", "session_id": "s1"},
        format="json",
    )
    assert resp.status_code == 410


def test_deprecated_intent_detector_warns():
    from chat.services.intent_detector import IntentDetector

    with pytest.warns(DeprecationWarning):
        IntentDetector().detect("WFH tomorrow")


def test_response_formatter_reexports_build_user_message():
    from chat.services.response_formatter import build_user_message as legacy_build

    msg, st = legacy_build(
        intent="UNKNOWN",
        entities={},
        decision={"outcome": "NEEDS_CLARIFICATION", "reason": "test"},
        crm_payload={},
    )
    assert st == "needs_input"
    assert msg == "test"
