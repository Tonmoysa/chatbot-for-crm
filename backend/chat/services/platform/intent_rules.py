"""Intent classification rules — AI Understanding Layer (rules fallback when LLM off).

Classification only. No field parsing — use field_extractors / FieldEngine for that.
"""

from __future__ import annotations

import re
from typing import Any

from chat.services.platform.field_extractors.amount import parse_amount
from chat.services.platform.field_extractors.date import parse_leave_dates, parse_relative_date

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
    r"^\s*(?:what\s+is|what's|what\s+are)\s+"
    r"(python|javascript|js|golang|go|java|c\+\+|rust|ruby|php|typescript|"
    r"json|xml|html|css|sql|api|kubernetes|docker|react|vue|angular|node\.?js|"
    r"mongodb|redis|graphql|yaml|toml|markdown|git|github|linux|windows|macos|"
    r"android|ios|flutter|django|flask|fastapi|numpy|pandas|tensorflow|pytorch)"
    r"\s*\??\s*$|"
    r"^\s*(python|javascript|js|golang|go|java|c\+\+|rust|ruby|php|typescript|json|sql|api)"
    r"\s+ki\s*\??\s*$|"
    r"^\s*(python|javascript|js|golang|go|json|sql)\s+(?:ki|kemon|kivabe)\s*\??\s*$",
    re.I | re.UNICODE,
)

_SUBMIT_RE = re.compile(
    r"(?:^|\b)(?:(?:amar|my)\s+)?(leave|expense|claim)\s+submit\s+(?:koro|kor|dao|de|den|do)\b|"
    r"\b(?:submit|joma)\s+(?:koro|kor|dao|de|den|do)\b.{0,20}\b(leave|expense|claim)\b|"
    r"\b(leave|expense)\s+submit\b",
    re.I | re.UNICODE,
)

_BARE_SUBMIT_RE = re.compile(
    r"(?:^|\b)(?:ok(?:ay)?|ekhon|এখন|thik|ha+h?)?\s*(?:,)?\s*(?:submit|joma)\s+(?:koro|kor|dao|de|den|do)\b",
    re.I | re.UNICODE,
)

_SUMMARY_RE = re.compile(
    r"\b(summary|summery|dekhao|show|review|total|report|reporting|status\s*report)\b|"
    r"(summary\s*dekhao|summery\s*dekhao|summery\s*ta|total\s*koto|koto\s*hoise|"
    r"amar\s+(?:leave|expense)\s+(?:summary|summery|report|status)|"
    r"expense\s+summary|leave\s+summary|leave\s+er\s+(?:summary|summery|report|status)|"
    r"(?:leave|expense).{0,25}(?:summary|summery|report|dekhao|review|status)\s*(?:ta\s*)?(?:daw|dao|de|den)?)",
    re.I | re.UNICODE,
)

_RESUME_LEAVE_RE = re.compile(
    r"(?:"
    r"\b(?:back|return|go)\s+(?:to|2)\s+leave\b|"
    r"\bcontinue\s+(?:my\s+)?leave\b|"
    r"\bleave\s+(?:summary|summery)\b|"
    r"\b(?:review|summery|summary)\s+dekhao\b|"
    r"\babar\s+dekhao\b|"
    r"\bback\s+koro\b|"
    r"(?:leave|chuti|chhuti|ছুটি).{0,20}(?:summary|summery|dekhao|review|back|continue|jaw|ja[oow])|"
    r"(?:leave|chuti|chhuti|ছুটি)\s+(?:e|te)\s+(?:jaw|ja[oow]|jai|back|continue|fire\s+ja[oow])"
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

_BARE_REJECTION_RE = re.compile(
    r"^(?:no|na|nah|nope|n|lagbe\s*nah|lagbe\s*na|lage\s*nah|lage\s*na|"
    r"dorkar\s*nai|chahina|hobe\s*na|habe\s*na|korbo\s*na)\.?$",
    re.I | re.UNICODE,
)

_BARE_CANCEL_RE = re.compile(
    r"^\s*(?:cancel|batil|বাতিল|bandho|abort|discard|stop)\s*"
    r"(?:koro|kor|dao|de|den|felo)?\s*\.?$",
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
    low = (message or "").lower()
    if re.search(r"\bthen\b", low) and parse_amount(message):
        return True
    amounts = re.findall(r"\d+(?:\.\d+)?\s*(?:taka|tk|টাকা)", low)
    return len(amounts) >= 2


def is_expense_add_request(message: str) -> bool:
    """Explicit add-more phrasing — not narrative extraction."""
    low = (message or "").strip().lower()
    return any(
        p in low
        for p in (
            "add koro",
            "add kore",
            "add kor",
            "jog koro",
            "jog kore",
            "notun expense",
            "ar ekta",
            "aro ekta",
            "add korte",
        )
    )


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
    if low in ("expense", "claim"):
        return True
    if low in ("leave", "wfh"):
        return False
    return expense_signal_strength(message) >= 0.85


def leave_signal_strength(message: str) -> float:
    if is_leave_message(message):
        dates = parse_leave_dates(message)
        if dates.get("start_date"):
            return 0.92
        return 0.85
    return 0.0


def is_leave_navigation_from_expense(message: str) -> bool:
    """User wants to resume/show suspended leave while expense is active."""
    low = (message or "").lower()
    return (
        is_resume_workflow_request(message, workflow_id="leave")
        or is_workflow_show_request(message, workflow_id="leave")
        or (is_switch_request(message, active_workflow_id="expense") and "leave" in low)
    )


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
    if is_leave_navigation_from_expense(message):
        return True
    return leave_signal_strength(message) >= 0.85


def is_resume_workflow_request(message: str, *, workflow_id: str) -> bool:
    low = (message or "").lower()
    if workflow_id == "leave":
        if is_expense_navigation_message(message):
            return False
        return bool(_RESUME_LEAVE_RE.search(message or ""))
    if workflow_id == "expense":
        if is_expense_navigation_message(message) and is_pure_expense_navigation(message):
            return True
        return bool(re.search(
            r"\b(?:back|return|go)\s+(?:to|2)\s+expense\b|"
            r"\bcontinue\s+(?:my\s+)?expense\b|"
            r"\bexpense\s+(?:summary|summery)\b|"
            r"(?:expense|claim|খরচ)\s+(?:e|te)\s+(?:jaw|ja[oow]|jai|back|continue|fire\s+ja[oow])",
            low,
        ))
    return False


def is_workflow_application_message(message: str) -> bool:
    """Leave/expense application narrative — must not route to policy KB."""
    return (
        is_leave_message(message)
        or is_strong_new_workflow_message(message)
        or is_expense_message(message)
    )


def is_leave_message(message: str) -> bool:
    """Weak pre-filter for leave start — final routing uses LLM + draft context."""
    low = (message or "").lower()
    if re.search(r"\b(policy|policies|niyom|niti)\b", low) and re.search(
        r"\b(leave|chuti|sick|annual)\b", low
    ):
        return False
    if re.search(
        r"\b(leave|chuti|chhuti|ছুটি)\s*(?:chai|chah|lagbe|lagi|nibo|nit|apply|request)\b",
        low,
    ):
        return True
    if re.search(r"\bleave\s+apply\b", low):
        return True
    if re.search(
        r"\b(?:need|want).{0,30}(?:days?\s+off|leave|time\s+away)\b",
        low,
    ):
        return True
    dates = parse_leave_dates(message)
    if dates.get("start_date") and re.search(
        r"\b(?:parbo\s+na|attend\s+korte\s+parbo\s+na|office\s+e\s+aste\s+parbo\s+na)\b",
        low,
    ):
        return True
    return False


_TECH_TERM_RE = re.compile(
    r"\b("
    r"json|xml|html|css|javascript|js|typescript|python|sql|api|docker|kubernetes|"
    r"react|vue|angular|node\.?js|mongodb|redis|graphql|yaml|markdown|git|linux|"
    r"android|ios|flutter|django|flask|fastapi|java|golang|rust|ruby|php"
    r")\b",
    re.I | re.UNICODE,
)

_GENERAL_TECH_QUESTION_RE = re.compile(
    r"^\s*(?:what\s+is|what's|what\s+are|explain|define|tell\s+me\s+about)\b",
    re.I | re.UNICODE,
)


def is_programming_question(message: str) -> bool:
    raw = (message or "").strip()
    if not raw:
        return False
    if _PROGRAMMING_RE.search(raw):
        return True
    low = raw.lower()
    if _GENERAL_TECH_QUESTION_RE.search(low) and _TECH_TERM_RE.search(low):
        return True
    if re.search(r"^\s*(?:json|python|javascript|js|sql|api)\s+ki\s*\??\s*$", low):
        return True
    return False


def is_clearly_off_hr_question(message: str) -> bool:
    """Off-HR trivia/programming — must win over active workflow routing."""
    from chat.services.policy_intent_helpers import (
        is_general_knowledge_out_of_scope,
        is_hr_assistant_in_scope,
        is_off_topic_for_hr_assistant,
    )

    raw = (message or "").strip()
    if not raw:
        return False
    if is_programming_question(raw):
        return True
    if is_general_knowledge_out_of_scope(raw):
        return True
    return bool(
        is_off_topic_for_hr_assistant(raw, wizard_active=False)
        and not is_hr_assistant_in_scope(raw)
    )


def is_workflow_turn_message(message: str, *, memory=None) -> bool:
    """Message plausibly continues leave/expense work — not general off-HR chat."""
    raw = (message or "").strip()
    if not raw:
        return False

    from chat.services.policy_intent_helpers import is_general_knowledge_out_of_scope

    if is_programming_question(raw) or is_general_knowledge_out_of_scope(raw):
        return False

    active_id = ""
    if memory is not None and getattr(memory, "active_workflow", None):
        active_id = str(memory.active_workflow.id or "").strip().lower()

    from chat.services.platform.turn_semantics import is_process_question, is_workflow_meta_complaint

    if is_bare_confirmation(raw) or is_bare_rejection(raw):
        return True
    if _BARE_CANCEL_RE.match(raw):
        return True
    if is_delete_request(raw) or is_modify_request(raw):
        return True
    if is_cancel_workflow_message(raw, workflow_id=active_id or None):
        return True
    if active_id and is_workflow_show_request(raw, workflow_id=active_id):
        return True
    if active_id and is_resume_workflow_request(raw, workflow_id=active_id):
        return True
    if is_switch_request(raw, active_workflow_id=active_id or None):
        return True
    if parse_submit_workflow(raw, active_workflow_id=active_id or None):
        return True
    if is_expense_add_request(raw) or is_expense_list_request(raw) or is_expense_draft_query(raw):
        return True
    if is_strong_new_workflow_message(raw) or is_leave_message(raw):
        return True
    if is_expense_message(raw) or is_compound_expense_message(raw):
        return True
    if active_id == "leave" and is_workflow_interrupt_expense(raw, active_workflow="leave"):
        return True
    if active_id == "expense" and is_leave_navigation_from_expense(raw):
        return True
    if is_process_question(raw) or is_workflow_meta_complaint(raw):
        return True
    if is_status_query(raw):
        return True
    if re.match(r"^\s*(?:expense\s+)?\d+\s*\.?\s*$", raw, re.I):
        return True

    if active_id == "expense" and memory is not None:
        from chat.services.platform.field_extractors.modify import (
            looks_like_expense_item_delete,
            parse_delete_request,
            parse_modify_request,
        )

        if looks_like_expense_item_delete(raw):
            return True
        draft = memory.active_draft() if hasattr(memory, "active_draft") else None
        items = list((draft.fields.get("items") or []) if draft else [])
        if items and (parse_delete_request(raw, items) or parse_modify_request(raw, items)):
            return True
    return False


def is_off_hr_topic_message(message: str, *, memory=None) -> bool:
    """General / programming questions unrelated to the active HR workflow."""
    from chat.services.policy_intent_helpers import (
        is_general_knowledge_out_of_scope,
        is_hr_assistant_in_scope,
        is_off_topic_for_hr_assistant,
    )

    raw = (message or "").strip()
    if not raw:
        return False
    if is_workflow_turn_message(raw, memory=memory):
        return False
    if is_clearly_off_hr_question(raw):
        return True
    if is_off_topic_for_hr_assistant(raw, wizard_active=False) and not is_hr_assistant_in_scope(raw):
        return True
    return False


def parse_submit_workflow(message: str, *, active_workflow_id: str | None = None) -> str | None:
    m = _SUBMIT_RE.search(message or "")
    if not m:
        if re.search(r"\bexpense\s+submit\b", (message or "").lower()):
            return "expense"
        if re.search(r"\bleave\s+submit\b", (message or "").lower()):
            return "leave"
    else:
        for g in m.groups():
            if g:
                return g.lower()
    active = (active_workflow_id or "").strip().lower()
    if active in ("leave", "expense") and _BARE_SUBMIT_RE.search(message or ""):
        return active
    if active == "claim" and _BARE_SUBMIT_RE.search(message or ""):
        return "expense"
    return None


def is_summary_request(message: str) -> bool:
    return bool(_SUMMARY_RE.search(message or ""))


_EXPENSE_LIST_RE = re.compile(
    r"(?:"
    r"\b(?:expense|expenses|khoroch|kharcha|claim)\b.{0,45}\b(?:list|ki\s+ki|koto\s+ta)\b|"
    r"\b(?:list|ki\s+ki)\b.{0,45}\b(?:expense|expenses|khoroch|kharcha|claim)\b|"
    r"\bexpense\s+er\s+list\b|"
    r"\bami\s+expense\s+er\s+list\b|"
    r"\b(?:aj|ajke|ajkar|today|saradin|sara\s*din)\b.{0,35}\b(?:expense|khoroch|kharcha)\b|"
    r"\b(?:expense|khoroch|kharcha)\b.{0,35}\b(?:aj|ajke|ajkar|today|saradin|sara\s*din)\b"
    r")",
    re.I | re.UNICODE,
)


def is_expense_list_request(message: str) -> bool:
    """User wants today's / pending expense item list — not leave summary."""
    raw = (message or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if not re.search(r"\b(?:expense|expenses|khoroch|kharcha|claim)\b", low):
        return False
    if re.search(r"\bleave\b", low) and not re.search(r"\bexpense", low):
        return False
    if _EXPENSE_LIST_RE.search(raw):
        return True
    if re.search(r"\b(?:list|ki\s+ki|jante\s+ceyechi|jante\s+chai)\b", low) and re.search(
        r"\b(?:expense|khoroch|kharcha)\b", low
    ):
        return True
    if is_summary_request(raw) and re.search(r"\b(?:expense|khoroch|kharcha)\b", low):
        if not re.search(r"\bleave\b", low) or re.search(r"\bexpense\b", low):
            return True
    return False


def suspended_workflow_ids(suspended_workflows: list | tuple | None) -> set[str]:
    ids: set[str] = set()
    for sw in suspended_workflows or ():
        if isinstance(sw, dict):
            wf = str(sw.get("workflow_id") or "").strip().lower()
        else:
            wf = str(getattr(sw, "workflow_id", "") or "").strip().lower()
        if wf:
            ids.add(wf)
    return ids


def should_resume_expense_for_list(
    *,
    message: str,
    active_workflow_id: str | None,
    suspended_workflows: list | tuple | None,
) -> bool:
    """Leave (or other) active but user asked to see suspended expense draft."""
    if not is_expense_list_request(message):
        return False
    if "expense" not in suspended_workflow_ids(suspended_workflows):
        return False
    active = (active_workflow_id or "").strip().lower()
    return active != "expense"


def is_expense_navigation_message(message: str) -> bool:
    """User wants to view or resume a suspended/active expense draft."""
    low = (message or "").strip().lower()
    if not low:
        return False
    mentions_expense = any(
        tok in low for tok in ("expense", "claim", "khoroch", "kharcha", "খরচ")
    )
    if not mentions_expense:
        return False
    if re.search(r"\bleave\b", low) and not re.search(r"\bexpense\b", low):
        return False
    nav_tokens = (
        "summary",
        "summery",
        "summry",
        "continue",
        "back",
        "resume",
        "list",
        "dekhao",
        "daw",
        "dao",
        "jao",
        "jai",
        "fire",
        "return",
        "abr",
        "ekhon",
        "koi",
        "kothay",
        "kothai",
        "where",
    )
    if any(re.search(rf"\b{re.escape(tok)}\b", low) for tok in nav_tokens):
        return True
    if "back koro" in low or "e back" in low or "te jaw" in low or "te jao" in low:
        return True
    return is_expense_list_request(message)


def is_pure_expense_navigation(message: str) -> bool:
    """Expense nav phrasing without amounts or compound expense data in the same message."""
    if not is_expense_navigation_message(message):
        return False
    if parse_amount(message):
        return False
    if is_compound_expense_message(message):
        return False
    return True


def expense_navigation_kind(message: str) -> str:
    """Whether user wants expense summary view or to continue collecting."""
    low = (message or "").strip().lower()
    if any(
        tok in low
        for tok in ("continue", "back", "fire", "resume", "jao", "jai", "jaw", "back koro")
    ):
        return "continue"
    return "summary"


def is_expense_draft_query(message: str) -> bool:
    """User asks where/summary/status of their expense draft — not a new line item."""
    raw = (message or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if not re.search(r"\b(?:expense|expenses|khoroch|kharcha|claim|খরচ)\b", low):
        return False
    if is_compound_expense_message(raw):
        return False
    if parse_amount(raw) and not is_pure_expense_navigation(raw):
        return False
    if is_expense_navigation_message(raw) or is_expense_list_request(raw):
        return True
    if _WORKFLOW_WHERE_RE.search(raw) and re.search(
        r"\b(?:expense|claim|khoroch|kharcha)\b", low
    ):
        return True
    if re.search(
        r"\b(?:amar|my)\s+(?:active\s+)?(?:expense|claim|khoroch|kharcha)\b", low
    ) and re.search(r"\b(?:koi|kothay|kothai|where|ache|hollo|ki|status)\b", low):
        return True
    return False


def should_resume_suspended_expense(
    *,
    message: str,
    active_workflow_id: str | None,
    suspended_workflows: list | tuple | None,
) -> bool:
    """Another workflow is active but user navigates to a suspended expense draft."""
    if "expense" not in suspended_workflow_ids(suspended_workflows):
        return False
    active = (active_workflow_id or "").strip().lower()
    if active == "expense":
        return False
    return is_expense_draft_query(message)


def find_submitted_leave_overlap_from_message(
    message: str,
    submitted_ranges: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return submitted range when message repeats the same start/end dates."""
    from chat.services.platform.field_extractors.date import parse_leave_dates
    from chat.services.platform.field_extractors.leave import leave_date_ranges_match

    dates = parse_leave_dates(message)
    start = dates.get("start_date")
    if not start:
        return None
    end = dates.get("end_date") or start
    for entry in submitted_ranges or []:
        if not isinstance(entry, dict):
            continue
        s = entry.get("start_date")
        e = entry.get("end_date") or s
        if s and leave_date_ranges_match(str(start)[:10], str(end)[:10], str(s)[:10], str(e)[:10]):
            return entry
    return None


def message_has_new_leave_date_range(
    message: str,
    *,
    submitted_ranges: list[dict[str, Any]] | None = None,
) -> bool:
    """True when message contains leave dates that are not an exact submitted duplicate."""
    dates_start = parse_leave_dates(message).get("start_date")
    if not dates_start:
        return False
    return find_submitted_leave_overlap_from_message(message, submitted_ranges) is None


def should_route_expense_after_submitted_leave(
    *,
    draft_locked: bool,
    active_workflow_id: str | None,
    message: str,
    understanding: Any | None = None,
    pq_target_workflow: str | None = None,
) -> bool:
    """Post-submit leave on screen — expense messages must start expense, not lock."""
    if not draft_locked or (active_workflow_id or "").strip().lower() != "leave":
        return False
    if is_expense_message(message) and not is_leave_message(message):
        return True
    if understanding is not None:
        wf = str(getattr(understanding, "workflow", "") or "").strip().lower()
        interrupt = str(getattr(understanding, "interrupt_workflow", "") or "").strip().lower()
        if wf == "expense" or interrupt == "expense":
            return True
        if getattr(understanding, "is_expense_intent", None) and understanding.is_expense_intent():
            return True
    if (pq_target_workflow or "").strip().lower() == "expense":
        return True
    if is_workflow_interrupt_expense(message, active_workflow="leave"):
        return True
    return False


_WORKFLOW_WHERE_RE = re.compile(
    r"(?:"
    r"\bwhere(?:\s+is)?\s+(?:my\s+)?(?:active\s+)?(?:leave|chuti|expense|claim)\b|"
    r"\b(?:kothay|kothai|where|koi)\s+(?:amar|my)\s+(?:active\s+)?(?:leave|chuti|expense|claim)\b|"
    r"\b(?:leave|chuti|expense|claim)\s+(?:ta\s+)?(?:kothay|kothai|where|koi|ache|hollo|ki\s*hollo)\b|"
    r"\b(?:amar|my)\s+(?:active\s+)?(?:leave|chuti|expense|claim)\b.{0,30}"
    r"\b(?:kothay|kothai|where|koi|dekhao|daw|summary|summery|ki)\b"
    r")",
    re.I | re.UNICODE,
)

_DRAFT_FIELDS_STATUS_RE = re.compile(
    r"(?:"
    r"\bki\s+ki\s+(?:fill|bhora|likha|deya|diyechi)\b|"
    r"\bkoto\s+(?:ta\s+)?(?:field|leave|din)\b|"
    r"\bwhat\s+(?:did\s+i\s+)?(?:fill|enter)\b|"
    r"\bwhich\s+fields?\b"
    r")",
    re.I | re.UNICODE,
)


def is_workflow_show_request(message: str, *, workflow_id: str | None = None) -> bool:
    """User wants to see the active/suspended draft — not answer a pending slot."""
    raw = (message or "").strip()
    if not raw:
        return False
    if workflow_id == "expense":
        from chat.services.platform.field_extractors.expense import is_expense_anti_summary_request

        if is_expense_anti_summary_request(raw):
            return False
        if is_compound_expense_message(raw):
            return False
        if is_expense_message(raw) and parse_amount(raw) and not is_pure_expense_navigation(raw):
            return False
    if workflow_id == "leave" and (
        is_expense_navigation_message(raw) or is_expense_draft_query(raw)
    ):
        return False
    if is_summary_request(raw) or is_total_request(raw):
        if workflow_id == "leave" and is_expense_navigation_message(raw):
            return False
        return True
    if workflow_id and is_resume_workflow_request(raw, workflow_id=workflow_id):
        return True
    if _WORKFLOW_WHERE_RE.search(raw):
        return True
    if workflow_id and _DRAFT_FIELDS_STATUS_RE.search(raw):
        return True
    return False


def is_same_workflow_navigation(message: str, *, active_workflow_id: str) -> bool:
    """Navigate within the active workflow (show/resume) — not cross-workflow switch."""
    active = (active_workflow_id or "").strip().lower()
    if not active:
        return False
    if active == "leave" and is_expense_draft_query(message):
        return False
    if is_workflow_show_request(message, workflow_id=active):
        return True
    low = (message or "").strip().lower()
    if active == "leave" and re.search(
        r"(?:leave|chuti|chhuti|ছুটি)\s+(?:e|te)\s+"
        r"(?:jaw|ja[oow]|jai|back|continue|fire\s+ja[oow])",
        low,
    ):
        return True
    if active == "expense" and re.search(
        r"(?:expense|claim|খরচ)\s+(?:e|te)\s+"
        r"(?:jaw|ja[oow]|jai|back|continue|fire\s+ja[oow])",
        low,
    ):
        return True
    return False


def is_total_request(message: str) -> bool:
    low = (message or "").lower()
    return bool(re.search(r"total\s*koto|koto\s*hoise|মোট\s*কত", low))


def is_vague_delete(message: str) -> bool:
    return bool(_DELETE_VAGUE_RE.match((message or "").strip()))


def is_bare_confirmation(message: str) -> bool:
    return bool(_BARE_CONFIRM_RE.match((message or "").strip()))


def is_bare_rejection(message: str) -> bool:
    """Short no / lagbe nah — declines submit confirm or cancels intent."""
    return bool(_BARE_REJECTION_RE.match((message or "").strip()))


def is_greeting_or_chitchat(message: str) -> bool:
    """Short greetings/thanks — not out-of-scope, not a workflow action."""
    return bool(_GREETING_RE.match((message or "").strip()))


_STATUS_QUERY_RE = re.compile(
    r"\b(ref|reference|request\s*id|rid|status|track)\b|"
    r"(রেফারেন্স|স্ট্যাটাস|ট্র্যাক)|"
    r"\b[A-Z]{2,}-\d{4,}\b",
    re.I | re.UNICODE,
)

_SWITCH_REQUEST_RE = re.compile(
    r"(?:"
    r"\b(switch|move\s+to|back\s+to|resume|continue)\b.{0,25}\b(leave|expense|claim|wfh)\b|"
    r"\b(leave|expense|claim)\b.{0,25}\b(instead|first|age|আগে)|"
    r"(?:ekhon|এখন).{0,20}(?:expense|leave|খরচ|ছুটি)|"
    r"(?:leave|expense|claim|wfh|খরচ|ছুটি)\s+(?:e|te)\s+(?:ja[oow]|jaw|jai|back|continue|fire\s+ja[oow])"
    r")",
    re.I | re.UNICODE,
)

_STRONG_NEW_WORKFLOW_RE = re.compile(
    r"(?:"
    r"\b(apply|request|submit|start|open|new)\b.{0,30}\b(leave|expense|claim|wfh)\b|"
    r"\b(leave|expense|claim)\b.{0,30}\b(apply|request|submit|start|new|chah[iy]|lagbe|lage)\b|"
    r"(ছুটি|খরচ).{0,30}(চাই|লাগবে|apply|request|submit|নিতে)|"
    r"(?:i\s+)?(?:want|need)\s+(?:to\s+)?(?:apply|take|request)\s+(?:for\s+)?(?:a\s+)?leave\b|"
    r"(?:log|add|submit)\s+(?:an?\s+)?expense\b"
    r")",
    re.I | re.UNICODE,
)


def is_status_query(message: str) -> bool:
    return bool(_STATUS_QUERY_RE.search(message or ""))


def is_switch_request(message: str, *, active_workflow_id: str | None = None) -> bool:
    if active_workflow_id and is_same_workflow_navigation(
        message, active_workflow_id=active_workflow_id
    ):
        return False
    if not _SWITCH_REQUEST_RE.search(message or ""):
        return False
    # "ekhon amar leave request prepare kore dao" is a new leave start, not a switch.
    if is_strong_new_workflow_message(message):
        return False
    return True


def is_strong_new_workflow_message(message: str) -> bool:
    if parse_submit_workflow(message):
        return False
    return bool(_STRONG_NEW_WORKFLOW_RE.search(message or ""))


def infer_switch_target(message: str) -> str | None:
    low = (message or "").lower()
    if re.search(r"\b(expense|claim|খরচ)\b", low) or "খরচ" in message:
        return "expense"
    if re.search(r"\b(leave|wfh|ছুটি|chuti)\b", low) or "ছুটি" in message:
        return "leave"
    return None


def infer_new_workflow_target(message: str) -> str | None:
    low = (message or "").lower()
    if re.search(r"\b(expense|claim|খরচ|reimburse)\b", low) or "খরচ" in message:
        return "expense"
    if re.search(r"\b(wfh|work\s+from\s+home)\b", low):
        return "wfh"
    if re.search(r"\b(leave|chuti|chhuti|ছুটি)\b", low) or "ছুটি" in message:
        return "leave"
    return None


_CANCEL_WORKFLOW_RE = re.compile(
    r"(?:"
    r"\b(?:cancel|abort|discard|stop|bandho|batil|বাতিল)\b.{0,30}\b(?:leave|chuti|chhuti|request|ছুটি|expense|claim|khoroch|খরচ)\b|"
    r"\b(?:leave|chuti|chhuti|ছুটি|expense|claim|khoroch|খরচ).{0,25}\b(?:cancel|bandho|batil|বাতিল)\b|"
    r"^\s*cancel\s+(?:my\s+)?(?:leave|expense|request)\s*\.?\s*$"
    r")",
    re.I | re.UNICODE,
)


def is_cancel_workflow_message(message: str, *, workflow_id: str | None = None) -> bool:
    """User wants to abandon the active workflow draft."""
    raw = (message or "").strip()
    if not raw:
        return False
    if workflow_id and workflow_id not in ("leave", "expense"):
        return False
    if workflow_id and _BARE_CANCEL_RE.match(raw):
        return True
    return bool(_CANCEL_WORKFLOW_RE.search(raw))


_MODIFY_HINT_RE = re.compile(
    r"(?:"
    r"\b(change|update|modify|edit|correct|fix|instead|rather|use)\b|"
    r"\b(instead\s+of|not\s+\d+|make\s+it)\b|"
    r"\bamount\s*ta\b|"
    r"(?:lunch|bus|nasta|prothom|first).{0,25}(?:amount|taka|tk).{0,20}(?:kore|kor|koro|dao|hobe|habe)|"
    r"\d+\s*(?:no|number|nombor|numer).{0,30}(?:taka|tk).{0,15}(?:koro|kor|hobe|habe|kore)|"
    r"\d+\s*(?:no|number|nombor|numer).{0,40}\broute\b|"
    r"\broute\b.{0,25}(?:hobe|habe|koro|kor|dao|change)|"
    r"prothom\s*ta\b|"
    r"(?:reason|date|tarikh|leave\s*type|type|day\s*scope)\s*ta\b|"
    r"(?:বদল|পরিবর্তন|ঠিক|instead|change\s*koro)"
    r")",
    re.I | re.UNICODE,
)

_DELETE_HINT_RE = re.compile(
    r"(?:"
    r"\b(delete|remove|drop|cancel\s+that|undo)\b|"
    r"(?:মুছ|ডিলিট|remove\s*koro|bad\s*d[aeiou]?[oy]?|muche|মুছে)"
    r")",
    re.I | re.UNICODE,
)


def is_modify_request(message: str) -> bool:
    return bool(_MODIFY_HINT_RE.search(message or ""))


def is_delete_request(message: str) -> bool:
    return bool(_DELETE_HINT_RE.search(message or ""))
