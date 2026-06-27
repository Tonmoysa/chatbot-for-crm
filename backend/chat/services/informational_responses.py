"""Informational / status response builders (Phase 8/11 SSOT)."""

from __future__ import annotations

from typing import Any

from chat.constants import (
    INTENT_HR_POLICY,
    INTENT_REQUEST_STATUS,
    INTENT_UNKNOWN,
)
from chat.services.crm.factory import get_crm_adapter
from chat.services.reference_extractors import extract_reference_entities

RULES_FOOTER_EN = (
    "_(Answers come from your uploaded policies; ask using the policy title or topic.)_"
)
RULES_FOOTER_BN = (
    "_(উত্তর আপনার আপলোড করা পলিসি থেকে আসে; পলিসির নাম বা বিষয় লিখে জিজ্ঞাসা করুন।)_"
)


def policy_rules_footer(*, lang: str) -> str:
    return RULES_FOOTER_BN if lang in ("bn", "banglish") else RULES_FOOTER_EN


def evaluate_request_status_decision(
    *,
    entities: dict[str, Any],
    crm_context: dict[str, Any],
) -> dict[str, Any]:
    st = str((crm_context or {}).get("status") or "")
    if st == "NOT_FOUND":
        return {
            "outcome": "NEEDS_CLARIFICATION",
            "reason": "Request not found.",
            "rules_applied": ["REQUEST_STATUS_NOT_FOUND"],
        }
    return {
        "outcome": "INFORMATIONAL",
        "reason": "Request status lookup.",
        "rules_applied": ["REQUEST_STATUS_LOOKUP"],
    }


def build_user_message(
    *,
    intent: str,
    entities: dict[str, Any],
    decision: dict[str, Any],
    crm_payload: dict[str, Any],
) -> tuple[str, str]:
    """Returns (message, status) for response envelope."""
    outcome = (decision or {}).get("outcome", "")
    reason = (decision or {}).get("reason", "")

    if intent == INTENT_REQUEST_STATUS:
        st = crm_payload.get("status")
        if st and st != "NOT_FOUND":
            rid = (
                str(crm_payload.get("request_id") or "")
                or str(entities.get("request_id") or "")
            ).strip()
            if rid:
                return (f"Request **`{rid}`** status: **{st}**.", "success")
            return (f"Current request status: **{st}**.", "success")
        if st == "NOT_FOUND":
            rid = (
                str(crm_payload.get("request_id") or "")
                or str(entities.get("request_id") or "")
            ).strip()
            rid_part = f" (`{rid}`)" if rid else ""
            return (
                f"I couldn't find any request{rid_part} in the system. "
                "Please double-check the reference ID and try again.",
                "needs_input",
            )
        return ("Please provide a request reference to look up status.", "needs_input")

    if outcome == "NEEDS_CLARIFICATION":
        if intent == INTENT_UNKNOWN:
            return (
                reason
                or "Could you clarify your question? I can help with company HR policies, "
                "attendance, WFH, and request status.",
                "needs_input",
            )
        return (reason or "Could you share a bit more detail?", "needs_input")

    if outcome == "INFORMATIONAL":
        if intent == INTENT_HR_POLICY:
            rules_answer = (crm_payload.get("rules_answer") or "").strip()
            if rules_answer:
                return (rules_answer, "success")
            topic = entities.get("policy_topic") or "general HR policy"
            return (
                f"Regarding {topic}: refer to the official employee handbook. "
                "This assistant provides guidance only; decisions follow company policy.",
                "success",
            )
        return (reason or "Here is the information you requested.", "success")

    if outcome == "ERROR":
        return (reason or "An error occurred.", "error")

    if outcome in ("PENDING_APPROVAL", "PENDING_REVIEW"):
        rid = crm_payload.get("request_id", "")
        msg = reason or "Your request is submitted for review."
        if rid:
            msg += f" Reference: {rid}."
        return (msg, "pending")

    if outcome == "REJECTED":
        return (reason or "The request could not be approved.", "rejected")

    return ("Request processed.", "success")


def resolve_request_status_turn(
    message: str,
    *,
    company_id: str,
    employee_id: str,
    session_id: str,
    pq_decision_log: dict[str, Any] | None = None,
    rules_tag: str = "DECISION_CORE",
) -> tuple[str, str, dict[str, Any], dict[str, Any], str]:
    """Build status lookup response from user message."""
    entities = extract_reference_entities(message)
    crm = get_crm_adapter()
    crm_context: dict[str, Any] = {}
    request_id = ""
    rid = str(entities.get("request_id") or "").strip()
    if rid:
        crm_context = crm.get_request_status(
            rid,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
        )
        request_id = rid

    decision = evaluate_request_status_decision(entities=entities, crm_context=crm_context)
    if pq_decision_log:
        decision["pending_question_decision"] = pq_decision_log
    decision["rules_applied"] = list(decision.get("rules_applied") or []) + [rules_tag]

    msg, resp_status = build_user_message(
        intent=INTENT_REQUEST_STATUS,
        entities=entities,
        decision=decision,
        crm_payload=crm_context,
    )
    return msg, resp_status, decision, entities, request_id


def compose_policy_turn(
    message: str,
    *,
    document_text: str | None,
    pq_decision_log: dict[str, Any] | None,
    conversation_history: list[str],
    company_id: str,
    trace_id: str,
    pq_reasoning: str = "",
) -> tuple[str, str, dict[str, Any]]:
    """Policy RAG + envelope message (Phase 8 — moved from pipeline)."""
    from chat.services.policy_intent_helpers import is_policy_kb_query, is_rules_query
    from chat.services.translator import align_policy_answer_language, resolve_reply_language
    from knowledge_base.services.rag_pipeline import hr_policy_not_found_message, try_hr_policy_rag

    policy_entities: dict[str, Any] = {}
    if document_text:
        policy_entities["document_text"] = document_text
        policy_entities["document_read"] = True
    decision: dict[str, Any] = {
        "outcome": "INFORMATIONAL",
        "reason": pq_reasoning or "Policy query.",
        "rules_applied": ["ASK_POLICY", "PLAN_REPLY_POLICY"],
        "pending_question_decision": pq_decision_log,
    }
    crm_payload: dict[str, Any] = {}
    rag = try_hr_policy_rag(message, trace_id, company_id=company_id)
    answer_text = (rag or {}).get("text") or (rag or {}).get("answer")
    if rag and answer_text:
        reply_lang = resolve_reply_language(message)
        if (rag.get("mode") or "") == "rag_excerpt":
            # Excerpt fallback is already grounded in retrieved chunks — skip a
            # second LLM pass that can hallucinate or fail under rate limits.
            answer = str(answer_text).rstrip()
            if reply_lang in ("bn", "banglish"):
                answer = answer.replace(
                    "Here is what I found in your uploaded HR policies:",
                    "Apnar uploaded HR policy theke je ta pelam:",
                )
        else:
            answer = align_policy_answer_language(
                str(answer_text),
                user_message=message,
                target_lang=reply_lang,
                trace_id=trace_id,
            )
        crm_payload["rules_answer"] = answer.rstrip() + "\n\n" + policy_rules_footer(lang=reply_lang)
        decision = {
            **decision,
            "reason": "Policy answer from knowledge base.",
            "rules_applied": list(set(list(decision.get("rules_applied") or []) + ["HR_POLICY_RAG"])),
        }
    elif is_policy_kb_query(message) or is_rules_query(message):
        crm_payload["rules_answer"] = hr_policy_not_found_message()
        decision = {
            **decision,
            "reason": crm_payload["rules_answer"],
            "rules_applied": list(
                set(list(decision.get("rules_applied") or []) + ["HR_POLICY_RAG_NOT_FOUND"])
            ),
        }
    msg, resp_status = build_user_message(
        intent=INTENT_HR_POLICY,
        entities=policy_entities,
        decision=decision,
        crm_payload=crm_payload,
    )
    return msg, resp_status, decision
