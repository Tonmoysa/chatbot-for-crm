import re
from typing import Any

from chat.constants import (
    INTENT_APPROVAL_ESCALATION,
    INTENT_ATTENDANCE_CORRECTION,
    INTENT_HR_POLICY,
    INTENT_REQUEST_STATUS,
    INTENT_UNKNOWN,
    INTENT_WFH_REQUEST,
)
from chat.services.policy_intent_helpers import is_rules_query


def _strong_hr_policy(message: str) -> bool:
    low = (message or "").lower()
    raw = message or ""
    if re.search(
        r"\b(rules?\s+(?:and|&)\s+regulations?|handbook|employee\s+handbook|"
        r"company\s+(?:rules?|regulations?|policy|policies)|"
        r"hr\s+(?:rule|rules|policy|policies)|guideline|guidelines)\b",
        low,
    ):
        return True
    if re.search(r"\b(rule|rules|regulation|regulations|policy|policies)\b", low):
        return True
    if re.search(r"(সব\s*নিয়ম|সকল\s*নিয়ম|নিয়ম|বিধি|নীতি|হ্যান্ডবুক|রুলস|পলিসি)", raw):
        return True
    if re.search(
        r"\b(shob|sob|sokol)?\s*(niyom|niyam|bidhi|niti|rules?|policy|policies|handbook)\b",
        low,
    ):
        return True
    return False


def _looks_like_request_status(message: str) -> bool:
    low = (message or "").lower()
    if re.search(r"\b(ref|reference|request\s*id|rid|status|track)\b", low):
        return True
    if re.search(r"(রেফারেন্স|স্ট্যাটাস|ট্র্যাক)", message or "", re.I):
        return True
    if re.search(r"\b[A-Z]{2,}-\d{4,}\b", message or ""):
        return True
    return False


def _looks_like_wfh(message: str) -> bool:
    low = (message or "").lower()
    return bool(
        re.search(r"\b(wfh|work\s+from\s+home|remote\s+work)\b", low)
        or re.search(r"(বাড়ি\s*থেকে|হোম\s*অফিস|রিমোট)", message or "", re.I)
    )


def _looks_like_attendance(message: str) -> bool:
    low = (message or "").lower()
    return bool(
        re.search(r"\b(attendance|clock\s*in|clock\s*out|punch|timesheet)\b", low)
        or re.search(r"(উপস্থিতি|পাঞ্চ|ক্লক)", message or "", re.I)
    )


def _looks_like_escalation(message: str) -> bool:
    low = (message or "").lower()
    return bool(
        re.search(r"\b(escalat|speak\s+to\s+hr|talk\s+to\s+manager|complaint)\b", low)
        or re.search(r"(এসকালেট|অভিযোগ|ম্যানেজার\s*কাছ)", message or "", re.I)
    )


class IntentDetector:
    def detect(self, message: str, trace_id: str = "") -> dict[str, Any]:
        raw = (message or "").strip()
        if not raw:
            return {"intent": INTENT_UNKNOWN, "confidence": 0.0, "source": "rules"}

        if _looks_like_request_status(raw):
            return {"intent": INTENT_REQUEST_STATUS, "confidence": 0.9, "source": "rules"}
        if _looks_like_wfh(raw):
            return {"intent": INTENT_WFH_REQUEST, "confidence": 0.85, "source": "rules"}
        if _looks_like_attendance(raw):
            return {"intent": INTENT_ATTENDANCE_CORRECTION, "confidence": 0.85, "source": "rules"}
        if _looks_like_escalation(raw):
            return {"intent": INTENT_APPROVAL_ESCALATION, "confidence": 0.8, "source": "rules"}
        if _strong_hr_policy(raw) or is_rules_query(raw):
            return {"intent": INTENT_HR_POLICY, "confidence": 0.85, "source": "rules"}

        return {"intent": INTENT_UNKNOWN, "confidence": 0.3, "source": "rules"}
