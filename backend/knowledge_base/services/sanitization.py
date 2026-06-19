"""Sanitize extracted document text and retrieved snippets before LLM context."""

from __future__ import annotations

import re
from typing import Final

# Patterns that often appear in prompt-injection attempts in documents / OCR.
_INJECTION_HINTS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?", re.I),
    re.compile(r"disregard\s+(?:the\s+)?system\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"\[\s*INST\s*\]", re.I),
)


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\x00", " ")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def sanitize_for_indexing(text: str, *, max_chars: int | None = None) -> str:
    """Normalize + strip control chars; safe for storage and embedding."""
    t = normalize_whitespace(text)
    t = "".join(ch for ch in t if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    if max_chars is not None and len(t) > max_chars:
        t = t[:max_chars].rstrip()
    return t


def sanitize_retrieval_context(text: str, *, max_chars: int = 12_000) -> str:
    """
    Harden retrieved chunks before they are placed in an LLM prompt.
    Does not remove factual content; neutralizes obvious injection scaffolding.
    """
    t = sanitize_for_indexing(text, max_chars=max_chars)
    for pat in _INJECTION_HINTS:
        t = pat.sub("[redacted]", t)
    t = t.replace("```", "'''")
    return t


def preprocess_query(text: str) -> str:
    """Lightweight multilingual-safe query normalization for retrieval."""
    return normalize_whitespace(text)[:4000]


# Ordered most-specific first — only the first matching hint is used (avoids mixed-policy noise).
_EMBEDDING_TOPIC_HINTS: Final[tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]] = (
    (
        re.compile(
            r"\b(?:cyber|cyber\s*security|cybersecurity|infosec|information\s+security|"
            r"it\s+security|network\s+security|data\s+protection)\b|"
            r"সাইবার|ইনফরমেশন\s*(?:সুরক্ষা|সিকিউরিটি)",
            re.I | re.UNICODE,
        ),
        ("information security cybersecurity policy",),
    ),
    (
        re.compile(
            r"\bacceptable\s+use\b|\bpersonal\b.*illegal.*assets\b|"
            r"\bcompany\s+(?:devices|equipment|computers|laptops)\b",
            re.I | re.UNICODE,
        ),
        ("acceptable use company devices policy",),
    ),
    (
        re.compile(
            r"\b(?:daily\s+)?allowance\b|\bta\s*/\s*da\b|\btada\b|"
            r"travel\s+allowance|dearness\s+allowance|"
            r"দৈনিক\s*ভাতা|ভাতা\s*কত|টিএ|ডিএ|"
            r"\b(?:per\s*day|protidin).{0,30}\b(?:koto|limit|cap|allowance)\b",
            re.I | re.UNICODE,
        ),
        (
            "daily travel allowance TA DA per day expense reimbursement limit entitlement",
            "field staff conveyance meal transport policy",
        ),
    ),
    (
        re.compile(
            r"\b(?:sick|casual|bereavement|maternity|paternity|marriage|study)\s+leave\b|\blwop\b|"
            r"উ\/এল\b|উ\s*\/?\s*এল\b|বিশেষ ছুটি|ক্যাশুয়াল\b",
            re.I | re.UNICODE,
        ),
        ("sick casual special leave entitlement", "LWOP unpaid leave policy"),
    ),
    (
        re.compile(
            r"leave\s*policy|policy\s*(?:on|about|for)?\s*leave\b|"
            r"leave\s*(?:rules?|regulations?)\b|"
            r"ছুটি\s*(?:নীতি|পলিসি|শর্ত|নিয়ম)|"
            r"(?:jante|জানতে|somporke|about).*(?:leave|ছুটি)|"
            r"(?:leave|ছুটি).*(?:jante|জানতে|somporke|bolo|বল)",
            re.I | re.UNICODE,
        ),
        ("employee leave policy rules application approval",),
    ),
    (
        re.compile(
            r"(carry\s*-?\s*forward|carried\s+forward|carrying\s+forward|\bcarry\b\s*-?\s*over\b|carryover|rollover|opening\s*balance\b|unused\s+leave|"
            r"annual\s+(?:credit|leaves?|leave|vacation|days)\b|\bpto\b|\bavl\b|\bleave\s+accrual\b|"
            r"vacation\b.*(?:accru|balance)\b|"
            r"ছুটি.*(?:ফরওয়ার্ড|ফরয়াওয়ার্ড|ক্যারি|বয়ে\s*যাবে|স্থানান্তর|জমা)|বছর\b.*ছুটি|বছরান্ত\b.*ছুটি|"
            r"কত\b.*ছুটি|ছুটি.*(?:কত|ফর|S|মেয়াদ|মেয়াদ)|kotodin\b|koydin\b|kondo?in\b|\bbaki\b.*ছুটি|ছুটি.*\bbaki\b)",
            re.I | re.UNICODE,
        ),
        (
            "annual leave entitlement",
            "vacation PTO accrued leave balance",
            "leave carry forward carryover rollover forfeiture expiry",
        ),
    ),
    (
        re.compile(r"\b(?:remote\s+work|telework\b|wfh\b|ওএফএইচ)\b", re.I | re.UNICODE),
        ("work from home remote work policy",),
    ),
    (
        re.compile(
            r"\bউপস্থিতি\b|\bhajira\b|\battendance\b|"
            r"\bflex\s*time\b|\bbio\b.*time\b|\bflexible\b.*working\b|\bfingerprint\b|\bbiometric\b",
            re.I | re.UNICODE,
        ),
        ("attendance working hours policy", "presence punctuality biometric time"),
    ),
    (
        re.compile(
            r"\b(?:policy|rules?\b.*regulations?|handbook|guideline\b.*HR)\b|"
            r"হ্যান্ডবুক|নিয়ম\b",
            re.I | re.UNICODE,
        ),
        ("human resources employee handbook excerpt",),
    ),
)

_EXPLICIT_POLICY_TITLE_RES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(
            r"\binformation\s+security\s*(?:policy|policies|rules?)?\b",
            re.I,
        ),
        "Information Security Policy",
    ),
    (
        re.compile(
            r"\bdata\s+privacy\b(?:\s*(?:&|and)\s*confidentiality)?\s*(?:policy)?\b",
            re.I,
        ),
        "Data Privacy and Confidentiality Policy",
    ),
    (
        re.compile(
            r"\b(?:cybersecurity|cyber\s+security)\s*(?:policy|rules?)?\b",
            re.I,
        ),
        "Cybersecurity Rules",
    ),
    (
        re.compile(
            r"\bcasual\s+leave\s*(?:policy|rules?)?\b",
            re.I,
        ),
        "Casual Leave Policy",
    ),
    (
        re.compile(
            r"\bsick\s+leave\s*(?:policy|rules?)?\b",
            re.I,
        ),
        "Sick Leave Policy",
    ),
    (
        re.compile(
            r"\bannual\s+leave\s*(?:policy|rules?)?\b",
            re.I,
        ),
        "Annual Leave Policy",
    ),
    (
        re.compile(
            r"\battendance\s*(?:rules?|policy|policies)?\b",
            re.I,
        ),
        "Attendance Rules",
    ),
    (
        re.compile(
            r"\bacceptable\s+use\s*(?:policy)?\b",
            re.I,
        ),
        "Acceptable Use Policy",
    ),
    (
        re.compile(
            r"\bsoftware\s+development\s*(?:policy)?\b",
            re.I,
        ),
        "Software Development Policy",
    ),
    (
        re.compile(
            r"\bemail\s*(?:&|and)\s*communication\s*(?:policy)?\b",
            re.I,
        ),
        "Email and Communication Policy",
    ),
    (
        re.compile(r"\bleave\s+without\s+pay\b|\blwop\b", re.I),
        "Leave Without Pay",
    ),
    (
        re.compile(r"\bleave\s+poli\b", re.I),
        "Leave Policy",
    ),
    (
        re.compile(
            r"\bleave\s*(?:policy|policies|rules?|regulations?)\b",
            re.I,
        ),
        "Leave Policy",
    ),
    (
        re.compile(r"ছুটি\s*(?:নীতি|পলিসি|নিয়ম|শর্ত)", re.UNICODE),
        "Leave Policy",
    ),
)


_POLICY_DOCUMENT_ASK_RE = re.compile(
    r"poli(?:cy|cies)?|poli\b|"
    r"rules?\s*(?:ta|er)?\s*(?:bolo|bole|daw|de|den|tell|explain)|"
    r"handbook|নীতি|পলিসি|"
    r"bolo\s*(?:amake|me)?|amake\s+.*\s+bolo|"
    r"\bt\s*bolo|somporke\s*(?:jante|chai)|"
    r"what\s+is\s+(?:the\s+)?[\w\s]+\s+policy|"
    r"tell\s+me\s+(?:about\s+)?(?:the\s+)?",
    re.I | re.UNICODE,
)


def extract_policy_title_phrases(message: str) -> list[str]:
    """Named policy titles when the user asks to see/read a policy document."""
    raw = preprocess_query(message)
    if not raw:
        return []
    if not _POLICY_DOCUMENT_ASK_RE.search(raw):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for pat, title in _EXPLICIT_POLICY_TITLE_RES:
        if pat.search(raw):
            key = title.lower()
            if key not in seen:
                seen.add(key)
                out.append(title)
    return out


def _unique_join_phrases(phrases: tuple[str, ...], *, hint_cap: int) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        p = " ".join(phrase.split())
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(p)
        if sum(len(x) + 1 for x in out) >= hint_cap:
            break
    return ". ".join(out)


def hr_retrieval_hint_line(query_normalized: str, *, phrase_cap_chars: int = 650) -> str:
    if extract_policy_title_phrases(query_normalized):
        return ""
    for pat, tup in _EMBEDDING_TOPIC_HINTS:
        if pat.search(query_normalized):
            return _unique_join_phrases(tup, hint_cap=phrase_cap_chars)
    return ""


def build_hr_policy_retrieval_query(message: str) -> str:
    """
    Focus vector search on the policy the user named — never replace the question
    with a generic list of all leave types (that pulls wrong policy chunks).
    """
    raw = preprocess_query(message)
    if not raw:
        return ""
    titles = extract_policy_title_phrases(raw)
    if titles:
        return f"Policy title: {titles[0]}. {raw}"[:4000]
    return raw


def build_retrieval_embedding_text(query: str, *, max_chars: int = 3800) -> str:
    """
    Text passed to embedding for vector search — may include multilingual-safe HR
    paraphrases to improve recall versus English-heavy policy corpuses.

    Caller must ensure the conversational RAG pathway still forwards the user's
    original question to grounded_user_prompt unchanged.
    """
    base = normalize_whitespace(query)[:max_chars]
    if not base:
        return ""
    hint = hr_retrieval_hint_line(base[:2000])
    if not hint:
        return base
    suffix = (
        "[HR handbook retrieval context]\n"
        + hint[: min(len(hint), 900)]
    ).strip()
    sep = "\n\n---\n\n"
    room = max(0, max_chars - len(sep) - len(suffix))
    head = base[:room] if room < len(base) else base
    return (head + sep + suffix).strip()[:max_chars]
