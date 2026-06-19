from typing import Any

from chat.constants import (
    INTENT_HR_POLICY,
    INTENT_REQUEST_STATUS,
    INTENT_UNKNOWN,
)


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
