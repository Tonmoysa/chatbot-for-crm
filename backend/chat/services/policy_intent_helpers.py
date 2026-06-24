"""Lightweight policy / rules topic detection for routing (no static handbook)."""

from __future__ import annotations

import random
import re

_RULES_QUERY_PATTERNS = (
    r"\b(rule|rules|regulation|regulations|policy|policies|handbook|guideline|guidelines)\b",
    r"\b(allowed|prohibited|must|mustn't|forbidden|mandatory|required|may\s+not)\b",
)

_BENGALI_RULES_HINT = (
    r"(নিয়ম|বিধি|নীতি|হ্যান্ডবুক|রুলস|পলিসি)",
    r"\b(niyom|niyam|bidhi|niti|rules?|policy|policies|handbook)\b",
)


def is_rules_query(message: str) -> bool:
    """True if the message is about rules / regulations / handbook topics."""
    if not message:
        return False
    if is_policy_handbook_complaint(message):
        return False
    try:
        from chat.services.platform.intent_rules import is_workflow_application_message

        if is_workflow_application_message(message):
            return False
    except Exception:
        pass
    low = message.lower()
    for pat in _RULES_QUERY_PATTERNS:
        if re.search(pat, low):
            return True
    for pat in _BENGALI_RULES_HINT:
        if re.search(pat, message) or re.search(pat, low):
            return True
    return False


_BAD_ANSWER_COMPLAINT_RE = re.compile(
    r"(relation\s*nai|related\s*na|relevant\s*na|not\s*related|no\s*relation|"
    r"wrong\s*answer|hallucinat|manasse\s*nai|"
    r"প্রাসঙ্গিক\s*না|সম্পর্ক\s*নেই|মিল\s*নেই|ভুল\s*উত্তর|এই\s*উত্তর|"
    r"ei\s*ans|amar\s*question.{0,40}(sathe|satha).{0,20}(nai|ney|na))",
    re.I | re.UNICODE,
)

_POLICY_HANDBOOK_COMPLAINT_RE = re.compile(
    r"(?:"
    r"policy\s*te\s*nai|policies?\s*te\s*nai|rules?\s*te\s*nai|"
    r"পলিসি\s*তে\s*না[ইে]|নীতি\s*তে\s*না[ইে]|"
    r"ei\s*dhoroner|এই\s*ধরনের|ei\s*besoy|এই\s*বিষয়|ei\s*bishoy|"
    r"kivabe\s*pele|কিভাবে\s*পেল|pele\s*keno|"
    r"tahole\s*tumi|তাহলে\s*তুমি|eta\s*kivabe|এটা\s*কিভাবে"
    r")",
    re.I | re.UNICODE,
)

_CALENDAR_QUESTION_RE = re.compile(
    r"(?:"
    r"\b(?:when\s+is|what\s+(?:day|date)\s+is|which\s+day\s+is)\b|"
    r"\b(?:kobe|kon\s*din|ki\s*din|ki\s*dibosh|ki\s*disbosh|ki\s*dibos)\b|"
    r"(?:কবে|কী\s*দিন|কি\s*দিন|কী\s*দিবস|কি\s*দিবস|কোন\s*দিবস|কোন\s*দিন)"
    r")",
    re.I | re.UNICODE,
)

_FESTIVAL_OR_OCCASION_RE = re.compile(
    r"(?:"
    r"\beid\b|ঈদ|bijoy|বিজয়|victory\s+day|independence|স্বাধীন|"
    r"26\s*march|২৬\s*মার্চ|durga|puja|pujo|পূজা|"
    r"\bdibosh\b|\bdebosh\b|\bdisbosh\b|দিবস|"
    r"divali|diwali|christmas|xmas|boro\s*din|বড়\s*দিন|pohela|নববর্ষ"
    r")",
    re.I | re.UNICODE,
)

_MONTH_NAME_RE = re.compile(
    r"\b(?:"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|"
    r"জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর"
    r")\b",
    re.I | re.UNICODE,
)

_ORDINAL_DATE_RE = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
    r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.I,
)

_WHEN_WORD_RE = re.compile(r"(?:\bkobe\b|কবে|\bwhen\b)", re.I)

_QUESTION_SHAPE_RE = re.compile(
    r"(?:"
    r"\?\s*$|"
    r"^\s*(?:what|when|where|why|how|who|which|can|could|is|are|do|does)\b|"
    r"\b(?:ki|kobe|keno|kothay|kemon|kon)\b|"
    r"(?:কী|কি|কেন|কোথায়|কখন|কিভাবে|কোন)"
    r")",
    re.I | re.UNICODE,
)

_HR_IN_SCOPE_RE = re.compile(
    r"\b("
    r"salary|beton|payroll|overtime|payslip|"
    r"attendance|clock|punch|timesheet|"
    r"policy|policies|rule|rules|regulation|regulations|handbook|guideline|"
    r"wfh|remote|work\s+from\s+home|"
    r"status|track|request|ticket|reference|ref|"
    r"manager|supervisor|hr|approval|escalat|"
    r"dress\s*code|ppe|safety|benefits|"
    r"document|receipt|invoice|bill"
    r")\b",
    re.I,
)

_HR_IN_SCOPE_BN_RE = re.compile(
    r"(বেতন|নিয়ম|বিধি|পলিসি|হ্যান্ডবুক|উপস্থিতি|"
    r"অনুমোদন|ম্যানেজার|সুবিধা)",
    re.UNICODE,
)

_PURE_CHITCHAT_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|hola|sup|yo|salam|thanks?|thank\s*you|bye|ok|okay|"
    r"kemon\s*ach[oe]n?|ki\s*khobor|"
    r"হ্যালো|হাই|ধন্যবাদ|কেমন\s*আছ"
    r")\s*[!.?,…]*\s*$",
    re.I | re.UNICODE,
)

_COMPANY_POLICY_SCOPE_RE = re.compile(
    r"(?:"
    r"\b(?:policy|policies|rules?|regulation|handbook|niyom|niti|bidhi)\b|"
    r"(?:নিয়ম|বিধি|নীতি|পলিসি|হ্যান্ডবুক)|"
    r"(?:policy|niyom|niti|company|কোম্পানি).{0,40}(?:allowed|grant|apply|company|কোম্পানি)"
    r")",
    re.I | re.UNICODE,
)


def is_irrelevant_answer_complaint(message: str) -> bool:
    if not message:
        return False
    if _BAD_ANSWER_COMPLAINT_RE.search(message):
        return True
    return bool(_POLICY_HANDBOOK_COMPLAINT_RE.search(message))


def is_policy_handbook_complaint(message: str) -> bool:
    if not message:
        return False
    return bool(_POLICY_HANDBOOK_COMPLAINT_RE.search(message))


def is_company_policy_about_occasion(message: str) -> bool:
    if not message:
        return False
    if is_rules_query(message):
        return True
    raw = message or ""
    if not _FESTIVAL_OR_OCCASION_RE.search(raw):
        return False
    return bool(_COMPANY_POLICY_SCOPE_RE.search(raw))


def is_hr_assistant_in_scope(message: str) -> bool:
    if not message:
        return False
    raw = (message or "").strip()
    low = raw.lower()
    if is_rules_query(raw):
        return True
    if _HR_IN_SCOPE_RE.search(low) or _HR_IN_SCOPE_BN_RE.search(raw):
        return True
    try:
        from knowledge_base.services.sanitization import extract_policy_title_phrases

        if extract_policy_title_phrases(raw):
            return True
    except Exception:
        pass
    return False


def is_policy_kb_query(message: str) -> bool:
    if not message or is_policy_handbook_complaint(message):
        return False
    if is_rules_query(message):
        return True
    try:
        from knowledge_base.services.sanitization import extract_policy_title_phrases

        if extract_policy_title_phrases(message):
            return True
    except Exception:
        pass
    low = (message or "").lower()
    if re.search(
        r"\b(?:company|office|employer|hr|কোম্পানি|অফিস)\b.{0,40}"
        r"\b(?:policy|policies|rules?|niyom|niti)\b",
        low,
    ):
        return True
    if re.search(
        r"\b(?:policy|policies|rules?|niyom|niti)\b.{0,40}"
        r"\b(?:company|office|employer|hr|কোম্পানি)\b",
        low,
    ):
        return True
    return False


_TODAY_DATE_QUERY_RE = re.compile(
    r"(?:"
    r"ajker\s+date|ajke\s+tarikh|aaj\s+ki\s+din|today'?s?\s+date|what\s+date\s+is\s+it|"
    r"আজকের\s+তারিখ|আজ\s+কী\s+তারিখ|আজ\s+কি\s+তারিখ|আজ\s+কত\s+তারিখ|"
    r"ajke\s+kon\s+din|ajke\s+ki\s+din"
    r")",
    re.I | re.UNICODE,
)


def is_hr_today_date_query(message: str) -> bool:
    raw = (message or "").strip()
    if not raw:
        return False
    return bool(_TODAY_DATE_QUERY_RE.search(raw))


def format_today_date_reply(*, today_iso: str, lang: str = "en") -> str:
    if lang == "bn":
        return f"আজকের তারিখ: **{today_iso}**।"
    if lang == "banglish":
        return f"Ajker date: **{today_iso}**."
    return f"Today's date is **{today_iso}**."


def is_general_knowledge_out_of_scope(message: str) -> bool:
    if not message or is_company_policy_about_occasion(message):
        return False
    if is_hr_today_date_query(message):
        return False
    raw = (message or "").strip()
    low = raw.lower()
    festival = bool(_FESTIVAL_OR_OCCASION_RE.search(raw))
    calendar_q = bool(_CALENDAR_QUESTION_RE.search(raw) or _CALENDAR_QUESTION_RE.search(low))
    has_month = bool(_MONTH_NAME_RE.search(raw) or _MONTH_NAME_RE.search(low))
    has_ordinal = bool(_ORDINAL_DATE_RE.search(low))
    named_date = bool(
        has_month
        or has_ordinal
        or re.search(
            r"\b\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|march)\b",
            low,
        )
        or re.search(r"(?:২৬\s*মার্চ|26\s*march)", raw, re.I)
    )
    if calendar_q and (festival or named_date):
        return True
    if has_ordinal and re.search(r"(?:eta\s*)?ki\s*din|what\s+day", low):
        return True
    if festival and _WHEN_WORD_RE.search(raw):
        return True
    words = re.findall(r"\S+", raw)
    if len(words) <= 6 and festival and _WHEN_WORD_RE.search(raw):
        return True
    return False


def is_off_topic_for_hr_assistant(
    message: str,
    *,
    wizard_active: bool = False,
) -> bool:
    if not message or is_hr_assistant_in_scope(message):
        return False
    if is_general_knowledge_out_of_scope(message):
        return True
    raw = (message or "").strip()
    if _PURE_CHITCHAT_RE.match(raw):
        return False
    if not _QUESTION_SHAPE_RE.search(raw):
        return False
    words = re.findall(r"\S+", raw)
    return len(words) >= 2


_OUT_OF_SCOPE_BN: tuple[str, ...] = (
    (
        "এ ধরনের সাধারণ প্রশ্ন (তারিখ, আবহাওয়া, trivia) **কোম্পানি HR**-এর বাইরে — "
        "আমি attendance, WFH ও **আপলোড করা পলিসি** নিয়ে কাজ করি।\n"
        "পলিসি চাইলে বিষয় লিখুন (যেমন: Attendance Policy)।"
    ),
    (
        "বুঝতে পারছি — তবে এটা আমার স্কোপের বাইরে; আমি **অফিস HR সহকারী** "
        "(attendance, WFH, company policy)।\n"
        "HR বিষয় হলে পলিসির নাম বা টপিক স্পষ্ট করে জিজ্ঞাসা করুন।"
    ),
)

_OUT_OF_SCOPE_EN: tuple[str, ...] = (
    (
        "That's general knowledge — outside **company HR** (attendance, WFH, uploaded policies).\n"
        "For HR rules, ask with a **policy name or topic** (e.g. Attendance Policy)."
    ),
    (
        "I can't help with that one — I'm your **workplace HR assistant**, not a general chatbot.\n"
        "Try attendance, WFH, or a named **company policy**."
    ),
)


def _last_assistant_text(context_lines: list[str] | None) -> str:
    for line in reversed(context_lines or []):
        if line.startswith("Assistant:"):
            return line[len("Assistant:") :].strip()
    return ""


def _pick_out_of_scope_variant(
    pool: tuple[str, ...],
    context_lines: list[str] | None,
) -> str:
    last = _last_assistant_text(context_lines)
    if last:
        candidates = [v for v in pool if v not in last and last not in v]
        if candidates:
            return random.choice(candidates)
    return random.choice(pool)


def build_out_of_scope_message(
    message: str,
    *,
    lang: str | None = None,
    context_lines: list[str] | None = None,
    trace_id: str | None = None,
) -> str:
    from chat.services.translator import detect_user_language

    user_lang = lang or detect_user_language(message)
    pool = _OUT_OF_SCOPE_BN if user_lang == "bn" else _OUT_OF_SCOPE_EN
    return _pick_out_of_scope_variant(pool, context_lines)


def is_policy_interrupt_message(message: str) -> bool:
    raw = message or ""
    low = raw.lower()
    if re.search(
        r"\b(payslip|pay\s*slip|salary\s*slip|payroll|payslips)\b",
        low,
    ):
        return True
    return bool(
        re.search(r"(পেস্লিপ|বেতন\s*স্লিপ|বেতনের\s*স্লিপ)", raw, re.I)
    )
