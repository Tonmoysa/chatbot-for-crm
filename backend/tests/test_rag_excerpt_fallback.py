"""RAG excerpt fallback keeps named-policy retrieval filter."""

from __future__ import annotations

from types import SimpleNamespace

from chat.services.rag_excerpt_fallback import excerpt_result_from_hits


def _hit(*, doc_title: str, section: str, body: str, score: float = 0.5):
    return SimpleNamespace(
        score=score,
        payload={
            "document_title": doc_title,
            "section_title": section,
            "chunk_text": body,
        },
    )


def test_excerpt_prefers_named_leave_policy_over_company_policy():
    hits = [
        _hit(
            doc_title="Company Policy",
            section="Internet Use",
            body="Pirated software installation is strictly prohibited.",
            score=0.58,
        ),
        _hit(
            doc_title="Leave Policy",
            section="Annual Leave",
            body="Employees receive 18 days of annual leave per year.",
            score=0.57,
        ),
        _hit(
            doc_title="Leave Policy",
            section="Sick Leave",
            body="Sick leave requires a medical note after two consecutive days.",
            score=0.56,
        ),
    ]
    result = excerpt_result_from_hits(
        hits,
        "trace-leave-excerpt",
        company_id="co1",
        retrieval_query="Policy title: Leave Policy. leave policy ta bolo",
    )
    assert result is not None
    assert result["mode"] == "rag_excerpt"
    text = result["text"]
    assert "Leave Policy" in text
    assert "annual leave" in text.lower() or "sick leave" in text.lower()
    assert "Pirated software" not in text


def test_compose_policy_turn_skips_translation_for_rag_excerpt():
    from unittest.mock import patch

    from chat.services.informational_responses import compose_policy_turn

    excerpt = (
        "Here is what I found in your uploaded HR policies:\n\n"
        "**Leave Policy**\nEmployees receive 18 days of annual leave."
    )
    with patch("knowledge_base.services.rag_pipeline.try_hr_policy_rag") as rag_mock:
        rag_mock.return_value = {"text": excerpt, "mode": "rag_excerpt"}
        with patch("chat.services.translator.align_policy_answer_language") as align_mock:
            msg, _status, _decision = compose_policy_turn(
                "leave policy ta bolo",
                document_text=None,
                pq_decision_log=None,
                conversation_history=[],
                company_id="co1",
                trace_id="compose-excerpt",
            )
    align_mock.assert_not_called()
    assert "annual leave" in msg
    assert "Apnar uploaded HR policy" in msg
