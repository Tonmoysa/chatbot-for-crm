"""Deterministic leave field extraction — Universal Field Engine."""

from __future__ import annotations

import re
from typing import Any

from chat.services.platform.field_extractors.date import parse_leave_dates, parse_relative_date

# Third-party illness: family member / relative sick — NOT employee sick leave.
_THIRD_PARTY_SUBJECT_RE = re.compile(
    r"\b(?:amar|my|his|her|their|tar|tader|mama|mam[ao]|kaka|kaku|baba|father|ma|mother|"
    r"bon|apa|bhai|brother|sister|uncle|aunt|family\s+member|relative|relatives|"
    r"wife|husband|son|daughter|shoshur|shashuri|nana|nani)\b",
    re.I | re.UNICODE,
)
_SICKNESS_WORD_RE = re.compile(
    r"\b(?:osustho|osustha|sick|ill|unwell|fever|jor|jvar|pet\s+betha|bma|hospital|"
    r"treatment|operation|surgery|bed\s*rest)\b|"
    r"(অসুস্থ|জ্বর|চিকিৎস|হাসপাতাল)",
    re.I | re.UNICODE,
)
_SELF_SICK_RE = re.compile(
    r"\b(?:ami|amr|amra|I|I'm|I\s+am|myself)\b.{0,40}\b(?:osustho|osustha|sick|ill|unwell|"
    r"fever|jor|pet\s+betha|hospitalized)\b|"
    r"\b(?:osustho|sick|ill|fever|jor|pet\s+betha)\b.{0,20}\b(?:ami|I)\b|"
    r"\bpet\s+betha\b|\bI\s+am\s+sick\b|\bI'm\s+sick\b",
    re.I | re.UNICODE,
)
_EXPLICIT_LEAVE_TYPE_RE = re.compile(
    r"\b(?:sick|annual|lwop|unpaid|leave without pay)\s+leave\b|"
    r"^\s*(?:sick|annual|lwop|unpaid)\s*\.?\s*$",
    re.I | re.UNICODE,
)
_MEDICAL_DECLINE_RE = re.compile(
    r"\b(?:nai|nei|no|n/a|na\b|don't\s+have|do\s+not\s+have|dewa\s+nai|"
    r"upload\s+kora\s+nai|document\s*nai|medical\s+document\s*nai)\b|"
    r"(নাই|নেই|দেওয়া\s+নাই)",
    re.I | re.UNICODE,
)


def text_has_third_party_sick_signal(text: str) -> bool:
    """Someone else (family/relative) is unwell — not grounds for employee sick leave."""
    raw = (text or "").strip()
    if not raw or not _SICKNESS_WORD_RE.search(raw):
        return False
    if _THIRD_PARTY_SUBJECT_RE.search(raw):
        return True
    if re.search(
        r"\b(?:take|niye|neiye|nye|accompany)\b.{0,30}\b(?:treatment|hospital|Dhaka)\b",
        raw,
        re.I,
    ):
        return True
    if re.search(r"\bfamily\s+member\b", raw, re.I) and _SICKNESS_WORD_RE.search(raw):
        return True
    return False


def text_has_self_sick_signal(text: str) -> bool:
    """Employee themselves is unwell — may justify sick leave."""
    raw = (text or "").strip()
    if not raw:
        return False
    if text_has_third_party_sick_signal(raw):
        return False
    if _SELF_SICK_RE.search(raw):
        return True
    if _EXPLICIT_LEAVE_TYPE_RE.search(raw) and re.search(r"\bsick\b", raw, re.I):
        return True
    low = raw.lower()
    if re.match(r"^\s*sick\s*\.?\s*$", low) or re.match(r"^\s*sick\s+leave\s*\.?\s*$", low):
        return True
    return False


def infer_leave_type_from_text(text: str) -> str | None:
    """Infer leave_type only when explicit or clearly self-sickness."""
    low = (text or "").lower().strip()
    for lt in ("annual", "sick", "lwop", "unpaid"):
        if re.match(rf"^\s*{lt}\s*\.?$", low) or re.search(rf"\b{lt}\s+leave\b", low):
            if lt == "unpaid":
                lt = "lwop"
            if lt == "sick" and text_has_third_party_sick_signal(text):
                return None
            return lt
    if re.search(r"\b(?:leave without pay|lwop)\b", low):
        return "lwop"
    if text_has_self_sick_signal(text):
        return "sick"
    return None


def sanitize_leave_type_value(message: str, leave_type: str | None) -> str | None:
    """Drop auto-inferred sick leave when illness refers to a family member."""
    if not leave_type:
        return None
    lt = str(leave_type).strip().lower()
    if lt != "sick":
        return lt
    if text_has_third_party_sick_signal(message):
        return None
    if not text_has_self_sick_signal(message) and not re.search(
        r"\bsick\s+leave\b", (message or ""), re.I
    ):
        return None
    return "sick"


def is_medical_document_unavailable(message: str) -> bool:
    low = (message or "").lower().strip()
    if not low:
        return False
    if _MEDICAL_DECLINE_RE.search(low):
        return True
    if low in ("false", "no", "na", "nai", "nei"):
        return True
    return False


def parse_medical_document_field(message: str) -> Any:
    if is_medical_document_unavailable(message):
        return None
    val = (message or "").strip()
    return val[:500] if val else None


def extract_leave_fields(message: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    low = (message or "").lower()

    dates = parse_leave_dates(message)
    fields.update(dates)

    if re.search(
        r"\b(mother|father|mama|family|emergency|sick|ill|hospital|treatment|uposthit)\b",
        low,
    ) and len(message.split()) > 8:
        fields["reason"] = message.strip()[:500]

    if re.search(r"\breason\b", low) or "personal" in low:
        fields["reason"] = message.strip()[:500]

    inferred = infer_leave_type_from_text(message)
    if inferred:
        fields["leave_type"] = inferred

    if re.search(r"\bhalf\s*day\b", low):
        fields["day_scope"] = "half_day"
    elif re.search(r"\bfull\s*day\b", low) or re.search(r"\bporjonto\b", low):
        fields["day_scope"] = "full_day"
    elif dates.get("start_date") and dates.get("end_date") and dates["start_date"] != dates["end_date"]:
        fields["day_scope"] = "full_day"

    return fields


def parse_leave_field(message: str, field: str) -> Any:
    """Parse a single leave field from message — no intent classification."""
    low = (message or "").lower().strip()
    if field == "leave_type":
        if re.search(r"\b(?:leave without pay|lwop)\b", low):
            return "lwop"
        for lt in ("annual", "sick", "lwop", "unpaid"):
            if re.match(rf"^\s*{lt}\s*\.?$", low) or re.search(rf"\b{lt}\b", low):
                val = "lwop" if lt == "unpaid" else lt
                return sanitize_leave_type_value(message, val)
        return None
    if field in ("start_date", "end_date"):
        dates = parse_leave_dates(message)
        if field == "start_date":
            return dates.get("start_date") or parse_relative_date(message)
        return dates.get("end_date")
    if field == "day_scope":
        if re.search(r"\bhalf", low):
            return "half_day"
        if re.search(r"\bfull", low):
            return "full_day"
        return None
    if field == "reason":
        return message.strip()[:500]
    if field == "half_day_period":
        if "morning" in low or "সকাল" in message:
            return "morning"
        if "afternoon" in low or "বিকেল" in message:
            return "afternoon"
        return None
    if field == "medical_document":
        return parse_medical_document_field(message)
    return None


def merge_deterministic_leave_dates(fields: dict[str, Any], message: str) -> dict[str, Any]:
    """Prefer regex-parsed dates from the user message over LLM hallucinations."""
    parsed = parse_leave_dates(message)
    if not parsed:
        return fields
    out = dict(fields)
    if parsed.get("start_date"):
        out["start_date"] = parsed["start_date"]
    if parsed.get("end_date"):
        out["end_date"] = parsed["end_date"]
    return out


def leave_range_from_fields(fields: dict[str, Any]) -> tuple[str, str] | None:
    start = fields.get("start_date")
    if not start:
        return None
    end = fields.get("end_date") or start
    return str(start)[:10], str(end)[:10]


def leave_ranges_overlap(
    start_a: str, end_a: str, start_b: str, end_b: str
) -> bool:
    from datetime import date

    try:
        sa = date.fromisoformat(start_a[:10])
        ea = date.fromisoformat(end_a[:10])
        sb = date.fromisoformat(start_b[:10])
        eb = date.fromisoformat(end_b[:10])
    except ValueError:
        return False
    return sa <= eb and sb <= ea


def find_submitted_leave_overlap(
    memory, start_date: str, end_date: str | None = None
) -> dict[str, Any] | None:
    end_date = end_date or start_date
    for entry in (memory.conversation_facts or {}).get("submitted_leave_ranges") or []:
        if not isinstance(entry, dict):
            continue
        s = entry.get("start_date")
        e = entry.get("end_date") or s
        if s and leave_ranges_overlap(str(start_date)[:10], str(end_date)[:10], str(s)[:10], str(e)[:10]):
            return entry
    return None


def record_submitted_leave_range(memory, fields: dict[str, Any], *, request_id: str = "") -> None:
    rng = leave_range_from_fields(fields)
    if not rng:
        return
    facts = dict(memory.conversation_facts or {})
    rows = list(facts.get("submitted_leave_ranges") or [])
    rows.append({
        "start_date": rng[0],
        "end_date": rng[1],
        "request_id": request_id,
    })
    facts["submitted_leave_ranges"] = rows
    memory.conversation_facts = facts
