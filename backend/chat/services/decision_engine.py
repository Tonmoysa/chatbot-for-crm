from typing import Any

from chat.constants import (
    INTENT_APPROVAL_ESCALATION,
    INTENT_ATTENDANCE_CORRECTION,
    INTENT_HR_POLICY,
    INTENT_REQUEST_STATUS,
    INTENT_UNKNOWN,
    INTENT_WFH_REQUEST,
)


class DecisionEngine:
    """Rule-based source of truth for approval outcomes."""

    def evaluate(
        self,
        *,
        intent: str,
        entities: dict[str, Any],
        crm_context: dict[str, Any],
    ) -> dict[str, Any]:
        doc_text = str(entities.get("document_text") or "")
        if entities.get("document_read"):
            if not doc_text.strip():
                return {
                    "outcome": "NEEDS_CLARIFICATION",
                    "reason": (
                        "I couldn't extract readable text from that document. "
                        "Please upload a text-based PDF or contact HR."
                    ),
                    "rules_applied": ["DOCUMENT_TEXT_EMPTY_OR_UNREADABLE"],
                }
            snippet = doc_text.strip()[:2500]
            truncated = "..." if len(doc_text.strip()) > 2500 else ""
            return {
                "outcome": "INFORMATIONAL",
                "reason": f"Document text preview:\n\n{snippet}{truncated}",
                "rules_applied": ["DOCUMENT_TEXT_PREVIEW"],
            }

        if intent == INTENT_HR_POLICY:
            return {
                "outcome": "INFORMATIONAL",
                "reason": "Policy lookup.",
                "rules_applied": ["HR_POLICY_LOOKUP"],
            }

        if intent == INTENT_REQUEST_STATUS:
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

        if intent == INTENT_WFH_REQUEST:
            return {
                "outcome": "PENDING_APPROVAL",
                "reason": "WFH requests require manager approval.",
                "rules_applied": ["WFH_PENDING_MANAGER"],
                "route_to": "MANAGER",
            }

        if intent == INTENT_ATTENDANCE_CORRECTION:
            return {
                "outcome": "PENDING_APPROVAL",
                "reason": "Attendance corrections are reviewed by HR.",
                "rules_applied": ["ATTENDANCE_PENDING_HR"],
                "route_to": "HR",
            }

        if intent == INTENT_APPROVAL_ESCALATION:
            return {
                "outcome": "PENDING_APPROVAL",
                "reason": "Your escalation has been noted for HR follow-up.",
                "rules_applied": ["ESCALATION_PENDING_HR"],
                "route_to": "HR",
            }

        if intent == INTENT_UNKNOWN:
            return {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": "",
                "rules_applied": ["UNKNOWN_INTENT"],
            }

        return {
            "outcome": "INFORMATIONAL",
            "reason": "Request noted.",
            "rules_applied": ["DEFAULT_INFORMATIONAL"],
        }
