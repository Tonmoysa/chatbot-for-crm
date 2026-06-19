"""Intent classification rules — AI Understanding Layer (rules fallback when LLM off).

Classification only. No field parsing — use field_extractors / FieldEngine for that.
"""

from __future__ import annotations

import re

from chat.services.platform.field_extractors.amount import parse_amount
from chat.services.platform.field_extractors.date import parse_leave_dates, parse_relative_date
from chat.services.platform.field_extractors.expense import extract_expense_items

_TRAVEL_WORDS = re.compile(
    r"\b(bus|train|bike|uber|taxi|cng|metro|transport|travel|office jete|commute)\b|"
    r"(বাস|ট্রেন|যাতায়|যাত্রা|অফিস\s*য)",
    re.I | re.UNICODE,
)

_MEAL_WORDS = re.compile(
    r"\b(lunch|dinner|breakfast|meal|snack|nasta|nasto|nosto|lanch|tiffin|khabar|"
    r"khawa|khete|korechi|khabar)\b|"
    r"(লাঞ্চ|দুপুর|নasta|নাস্তা|খাবার|খেয়েছি|খেয়েছি|হালকা\s*নasta|হালকা\s*নাস্তা|বিকাল)",
    re.I | re.UNICODE,
)

_BUS_WORD = re.compile(r"\b(bus|বাস)\b", re.I | re.UNICODE)

_PROGRAMMING_RE = re.compile(
    r"^\s*(?:what\s+is|what's)\s+(python|javascript|js|golang|go|java|c\+\+|rust|ruby|php|typescript)\s*\??\s*$|"
    r"^\s*(python|javascript|js|golang|go|java|c\+\+|rust|ruby|php|typescript)\s+ki\s*\??\s*$|"
    r"^\s*(python|javascript|js|golang|go)\s+(?:ki|kemon|kivabe)\s*\??\s*$",
    re.I | re.UNICODE,
)

_SUBMIT_RE = re.compile(
    r"(?:^|\b)(?:(?:amar|my)\s+)?(leave|expense|claim)\s+submit\s+(?:koro|kor|dao|de|den|do)\b|"
    r"\b(?:submit|joma)\s+(?:koro|kor|dao|de|den|do)\b.{0,20}\b(leave|expense|claim)\b|"
    r"\b(leave|expense)\s+submit\b",
    re.I | re.UNICODE,
)

_SUMMARY_RE = re.compile(
    r"\b(summary|summery|dekhao|show|review|total)\b|"
    r"(summary\s*dekhao|summery\s*dekhao|total\s*koto|koto\s*hoise|"
    r"amar\s+(?:leave|expense)\s+summary|expense\s+summary|leave\s+summary|"
    r"(?:leave|expense).{0,20}(?:summary|summery|dekhao|review)\s*(?:ta\s*)?(?:daw|dao|de|den)?)",
    re.I | re.UNICODE,
)

_RESUME_LEAVE_RE = re.compile(
    r"(?:"
    r"\b(?:back|return|go)\s+(?:to|2)\s+leave\b|"
    r"\bcontinue\s+(?:my\s+)?leave\b|"
    r"\bleave\s+(?:summary|summery)\b|"
    r"(?:leave|chuti|chhuti|ছুটি).{0,20}(?:summary|summery|dekhao|review|back|continue|jaw|ja[oow])|"
    r"(?:leave|chuti)\s+e\s+(?:jaw|ja[oow]|back|continue)"
    r")",
    re.I | re.UNICODE,
)

_DELETE_VAGUE_RE = re.compile(
    r"^\s*(?:delete|remove|muche|muchey|muche\s*felo|bad\s*dao)\s*(?:koro|kor|dao|de|den|felo)?\s*\.?$",
    re.I | re.UNICODE,
)

_BARE_CONFIRM_RE = re.compile(
    r"^(?:ha+|ha+h|yes|yep|yeah|confirm|confirmed|ok|okay|thik\s*ache|ঠিক\s*আছে|হ্যাঁ)\.?$",
    re.I | re.UNICODE,
)

_GREETING_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|hola|sup|yo|salam|thanks?|thank\s*you|bye|"
    r"kemon\s*ach[oe]n?|ki\s*khobor|"
    r"হ্যালো|হাই|ধন্যবাদ|কেমন\s*আছ"
    r")\s*[!.?,…]*\s*$",
    re.I | re.UNICODE,
)


def is_compound_expense_message(message: str) -> bool:
    return len(extract_expense_items(message)) >= 2


def is_expense_message(message: str) -> bool:
    low = (message or "").lower()
    if is_compound_expense_message(message):
        return True
    if parse_amount(message):
        if (
            _MEAL_WORDS.search(low)
            or _TRAVEL_WORDS.search(low)
            or _BUS_WORD.search(message or "")
            or re.search(r"\b(taka|tk|khoroch|expense|claim|খরচ|টাকা|lagse|hoise|hoyeche)\b", low)
        ):
            return True
    if re.search(r"\b(khoroch|expense|claim|খরচ)\b", low):
        return True
    return False


def expense_signal_strength(message: str) -> float:
    low = (message or "").lower()
    if is_compound_expense_message(message):
        return 0.95
    if re.search(r"\b(expense|khoroch|hoyeche|hoise|claim|খরচ)\b", low):
        return 0.9
    if is_expense_message(message) and parse_amount(message):
        return 0.85
    return 0.0


def is_workflow_interrupt_expense(message: str, *, active_workflow: str | None) -> bool:
    if active_workflow != "leave":
        return False
    if is_bare_confirmation(message) or is_greeting_or_chitchat(message):
        return False
    if is_summary_request(message):
        return False
    low = (message or "").strip().lower()
    # Bare workflow name is a switch reply, not a new expense claim.
    if low in ("expense", "leave", "claim", "wfh"):
        return False
    return expense_signal_strength(message) >= 0.85


def leave_signal_strength(message: str) -> float:
    if is_leave_message(message):
        dates = parse_leave_dates(message)
        if dates.get("start_date"):
            return 0.92
        return 0.85
    return 0.0


def is_workflow_interrupt_leave(message: str, *, active_workflow: str | None) -> bool:
    if active_workflow != "expense":
        return False
    if is_bare_confirmation(message) or is_greeting_or_chitchat(message):
        return False
    if is_summary_request(message) and "leave" not in (message or "").lower():
        return False
    low = (message or "").strip().lower()
    if low in ("expense", "leave", "claim", "wfh"):
        return False
    return leave_signal_strength(message) >= 0.85


def is_resume_workflow_request(message: str, *, workflow_id: str) -> bool:
    low = (message or "").lower()
    if workflow_id == "leave":
        return bool(_RESUME_LEAVE_RE.search(message or ""))
    if workflow_id == "expense":
        return bool(re.search(
            r"\b(?:back|return|go)\s+(?:to|2)\s+expense\b|"
            r"\bcontinue\s+(?:my\s+)?expense\b|"
            r"\bexpense\s+(?:summary|summery)\b",
            low,
        ))
    return False


def is_leave_message(message: str) -> bool:
    low = (message or "").lower()
    if re.search(r"\b(policy|policies|niyom|niti)\b", low) and re.search(r"\b(leave|chuti|sick|annual)\b", low):
        return False
    dates = parse_leave_dates(message)
    if dates.get("start_date") and re.search(
        r"\b(mother|father|mama|family|emergency|sick|ill|hospital|treatment|uposthit)\b",
        low,
    ):
        return True
    if dates.get("start_date") and re.search(
        r"\b(cannot|can't|unable|attend|office|parbo\s+na|porjonto)\b",
        low,
    ):
        return True
    if re.search(
        r"\b(mother|father|family|sick|ill|hospital).{0,40}(?:days?\s+off|leave|chuti|from\s+work)\b|"
        r"\b(need|want).{0,20}(?:days?\s+off|leave|time\s+away)\b",
        low,
    ):
        return True
    if re.search(r"\b(leave|chuti|chhuti|ছুটি)\s*(?:chai|chah|lagbe|lagi|nibo|nit)\b", low):
        return True
    if re.search(r"\b(leave|chuti|chhuti|ছুটি)\b", low) and parse_relative_date(message):
        return True
    if re.search(r"\breason\b", low) and "personal" in low:
        return True
    return False


def is_programming_question(message: str) -> bool:
    return bool(_PROGRAMMING_RE.search((message or "").strip()))


def parse_submit_workflow(message: str) -> str | None:
    m = _SUBMIT_RE.search(message or "")
    if not m:
        if re.search(r"\bexpense\s+submit\b", (message or "").lower()):
            return "expense"
        if re.search(r"\bleave\s+submit\b", (message or "").lower()):
            return "leave"
        return None
    for g in m.groups():
        if g:
            return g.lower()
    return None


def is_summary_request(message: str) -> bool:
    return bool(_SUMMARY_RE.search(message or ""))


def is_total_request(message: str) -> bool:
    low = (message or "").lower()
    return bool(re.search(r"total\s*koto|koto\s*hoise|মোট\s*কত", low))


def is_vague_delete(message: str) -> bool:
    return bool(_DELETE_VAGUE_RE.match((message or "").strip()))


def is_bare_confirmation(message: str) -> bool:
    return bool(_BARE_CONFIRM_RE.match((message or "").strip()))


def is_greeting_or_chitchat(message: str) -> bool:
    """Short greetings/thanks — not out-of-scope, not a workflow action."""
    return bool(_GREETING_RE.match((message or "").strip()))
