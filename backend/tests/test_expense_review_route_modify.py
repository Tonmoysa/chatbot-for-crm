"""Regression: expense review-stage route modify must not corrupt amount/item index."""

from __future__ import annotations

from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.field_extractors.expense import (
    coerce_expense_route_modify_turn,
    filter_expense_updates_for_review,
    is_expense_review_mode,
)
from chat.services.platform.field_extractors.modify import parse_route_modify_request
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.platform.schemas import FieldUpdate, UnderstandingAction, UnderstandingResult
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from chat.services.translator import resolve_reply_language
from tests.helpers.expense_llm_mock import mock_expense_llm
from tests.helpers.pipeline_handle import handle_with_rules_understanding


def _five_item_review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-24",
                    "items": [
                        {"category": "bus", "amount": 120.0, "from_location": "Mirpur", "to_location": "Agargaon"},
                        {"category": "lunch", "amount": 140.0},
                        {"category": "metro", "amount": 90.0, "from_location": "Mirpur", "to_location": "Agargaon"},
                        {
                            "category": "bus",
                            "amount": 60.0,
                            "from_location": "Dhanmondi",
                            "to_location": "Dhaka",
                            "description": "Ar ekta 150 taka expense hoise but category mone nei ekhon",
                        },
                        {"category": "bus", "amount": 40.0, "from_location": "Dhaa", "to_location": "Mirpur"},
                    ],
                },
            )
        },
        pending_confirmation="submit",
        last_entities={"reply_language": "banglish"},
    )


def test_parse_route_modify_resolves_fifth_item_with_numer_typo():
    memory = _five_item_review_memory()
    items = list(memory.active_draft().fields["items"])
    parsed = parse_route_modify_request(
        "5 numer expense er route hobe dhaka to mirpur",
        items,
    )
    assert parsed is not None
    assert parsed["item_index"] == 4
    assert parsed["from_location"] == "Dhaka"
    assert parsed["to_location"] == "Mirpur"


def test_coerce_route_modify_keeps_amount_on_fourth_item():
    memory = _five_item_review_memory()
    turn = coerce_expense_route_modify_turn(
        "5 numer expense er route hobe dhaka to mirpur",
        memory,
    )
    assert turn is not None
    assert turn["intent"] == "modify_review"
    patch = turn["item_patches"][0]
    assert patch["item_index"] == 4
    assert patch["from_location"] == "Dhaka"
    assert patch["to_location"] == "Mirpur"
    assert "amount" not in patch


def test_sanitize_strips_hallucinated_amount_from_llm_patch():
    memory = _five_item_review_memory()
    bad = [
        FieldUpdate(
            field="items",
            value={
                "amount": 150.0,
                "description": "Ar ekta 150 taka expense hoise but category mone nei ekhon",
                "category": "bus",
                "from_location": "Dhaka",
                "to_location": "Mirpur",
            },
            item_index=3,
            action="update",
        )
    ]
    cleaned = filter_expense_updates_for_review(
        bad,
        "5 numer expense er route hobe dhaka to mirpur",
        memory=memory,
    )
    assert len(cleaned) == 1
    assert cleaned[0].item_index == 4
    assert cleaned[0].value["from_location"] == "Dhaka"
    assert cleaned[0].value["to_location"] == "Mirpur"
    assert "amount" not in cleaned[0].value


def test_review_route_modify_end_to_end():
    memory = _five_item_review_memory()
    pipeline = WorkflowPipeline()
    pq = PendingQuestionDecision(
        kind=MessageIntentKind.MODIFY_DATA,
        confidence=0.9,
        reasoning="route modify at review",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
    )
    with mock_expense_llm():
        msg, decision = handle_with_rules_understanding(
            pipeline,
            "5 numer expense er route hobe dhaka to mirpur",
            memory=memory,
            pq_decision=pq,
            trace_id="review-route-modify",
            route_source="active",
        )
    items = memory.active_draft().fields["items"]
    assert items[3]["amount"] == 60.0
    assert items[4]["from_location"] == "Dhaka"
    assert items[4]["to_location"] == "Mirpur"
    assert items[4]["amount"] == 40.0
    assert "150" not in msg or "60" in msg
    assert decision.get("outcome") == "NEEDS_INPUT"


def test_domain_expense_understanding_during_review():
    """Active expense uses one domain LLM call — no UNDERSTAND + gatekeeper override."""
    memory = _five_item_review_memory()
    layer = AIUnderstandingLayer()
    scopes: list[str] = []

    with mock_expense_llm() as mock_cls:
        from chat.services.llm_client import LLMClient

        client = LLMClient()
        original_chat = client.chat_json.side_effect

        def _track_chat_json(*, system_prompt: str, user_prompt: str, **kwargs):
            scopes.append(str(kwargs.get("scope") or "default"))
            return original_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                trace_id=kwargs.get("trace_id") or "",
                **{k: v for k, v in kwargs.items() if k not in ("system_prompt", "user_prompt", "trace_id")},
            )

        client.chat_json.side_effect = _track_chat_json
        mock_cls.return_value.chat_json.side_effect = _track_chat_json
        result = layer.understand(
            "5 numer expense er route hobe dhaka to mirpur",
            memory=memory,
            conversation_history=[],
            trace_id="domain-route",
            llm=client,
        )

    assert len(scopes) <= 1
    if scopes:
        assert scopes[0] == "expense-draft"
    assert result.workflow == "expense"
    assert result.entities.get("expense_intent") in ("modify_review", "update")
    assert "Understanding Layer" not in " ".join(scopes)
    assert is_expense_review_mode(memory)


def test_reply_language_stays_sticky_after_banglish_session():
    """Session language should not flip to English on mixed workflow edits."""
    lang = resolve_reply_language(
        "5 numer expense er route hobe dhaka to mirpur",
        stored="banglish",
    )
    assert lang == "banglish"
    assert resolve_reply_language("yes", stored="banglish") == "banglish"
    assert resolve_reply_language("in english", stored="banglish") == "en"
