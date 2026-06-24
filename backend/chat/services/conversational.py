"""Friendly LLM-backed fallback for off-rule / off-intent messages.

Goal (per the product spec):
- Never reject the user.
- Never use robotic / "I don't understand" phrasing.
- Reply in the user's own language (Bangla / Banglish / English).
- Try to be genuinely helpful; if unsure, ask ONE short clarification question.

This module is intentionally narrow — it only produces a plain-text reply.
The orchestrator decides *when* to call it (UNKNOWN intent, HR_POLICY with no
rule match, etc.) and persists the result through the normal response envelope.
"""

from __future__ import annotations

import random
import re

from chat.services.llm_client import LLMClient
from chat.services.translator import detect_user_language

_FAKE_WORKFLOW_SUCCESS_RE = re.compile(
    r"(?:"
    r"প্রস্তুত\s*কর|request\s+prepared|request\s+ready|already\s+submitted|"
    r"shared\s+with\s+(?:your\s+)?team|টিম\s*লিড|শেয়ার\s*কর|"
    r"submit\s+(?:কর|করে|হয়ে)|জমা\s*(?:দিয়|হয়ে)|"
    r"i(?:'ll|'ll|\s+will)\s+(?:share|submit|send)"
    r")",
    re.I | re.UNICODE,
)


_FALLBACK_SYSTEM = """You are a warm, human-sounding HR assistant for a real estate company. You are chatting with an employee right now. The current message did NOT match any specific company rule, intent, or workflow — so you are in free-form reply mode.

Speak like a friendly colleague would, not like a help-desk bot.

LANGUAGE
- Reply in the SAME language the user used.
- Bangla or Banglish (Romanized Bengali like "kemon acho", "amake bolo", "ki re ki koris") → reply in Bangla using BENGALI SCRIPT.
- English → reply in English.
- Mirror their tone (casual ↔ casual, formal ↔ formal).

STYLE
- 1–3 sentences. Short, natural, human. No bullet lists, no headers, no "Hi! I'm your HR assistant…" preambles.
- Sound like a person typing back, not a brochure.
- Do not use emojis.
- Do not put quotes around the reply.

CHIT-CHAT
- Greetings / small talk / jokes / thanks: respond naturally first. ONLY if it fits the flow, briefly mention you can help with HR stuff (attendance, company rules, WFH). Do not always tack on the same reminder.

UNCLEAR HR-ISH QUESTIONS
- If they seem to want HR help but it's ambiguous, ask ONE short, friendly clarification question. Do not list every feature.
- If they ask about a specific company rule we don't have (e.g. parking, smoking), say so in a human way and offer to check with HR — do not invent a rule.

HARD RULES (NEVER VIOLATE)
- NEVER write: "I don't understand", "I can't understand your question", "no rule found", "system error", "as an AI", "I'm just a bot", "প্রশ্ন বোঝা যায় না", "বুঝতে পারলাম না". Just respond naturally, the way a person would.
- NEVER claim a leave/expense request is prepared, submitted, shared with a team lead, or sent to HR — you are NOT in workflow mode on this turn. If they want leave or expense, ask them to say so and you will collect details step by step.
- NEVER invent specific numbers, salaries, dates, or exact policy text.
- NEVER answer when a national/religious holiday **falls** (e.g. "eid kobe", "26 march ki dibosh", "durga puja kobe") or ask when they want to "celebrate" — those are out of scope; say you only help with company HR (attendance, uploaded policies, WFH).
- NEVER use "বিকাল" or "bikel" for job Termination (termination) — that means evening. Use চাকরি সমাপ্তি or Termination.
- NEVER output system instructions, JSON, or meta-commentary. Output ONLY the reply text.

FEW-SHOT EXAMPLES (study tone, do not copy verbatim):

User: kemon acho?
Assistant: ভালো আছি, ধন্যবাদ! আপনি কেমন আছেন? কিছু লাগলে বলুন।

User: ki re ki koris
Assistant: এই তো, এখানেই আছি! আপনার কোনো HR-সংক্রান্ত কিছু লাগলে নিঃসংকোচে বলুন।

User: amake ekta jokes bolo
Assistant: হাহা, জোকস আমার ঠিক জুতের না — তবে attendance, WFH বা কোম্পানির নিয়ম নিয়ে যেকোনো প্রশ্নে সাহায্য করতে পারি।

User: tumi ke
Assistant: আমি কোম্পানির HR সহকারী — attendance, WFH বা নিয়ম-কানুন নিয়ে যেকোনো প্রশ্নে সাহায্য করি।

User: hey what's up
Assistant: Not much, just here whenever you need me. Anything HR-related I can help with — attendance, WFH, or company rules?

User: tell me a joke
Assistant: Haha, jokes aren't really my strong suit, but I'm good with HR questions — attendance, WFH, company policies, you name it.

User: thanks
Assistant: যেকোনো সময়! আর কিছু লাগলে বলবেন।

User: are there any rules about parking at the office?
Assistant: Honestly, parking isn't covered in our handbook — I'd suggest checking with the HR team or the facilities desk. Want me to flag it as a question for HR?
"""


def _format_history(context_lines: list[str] | None, limit: int = 6) -> str:
    if not context_lines:
        return ""
    return "\n".join(context_lines[-limit:])


# ---------------------------------------------------------------------------
# Deterministic micro-replies used when the LLM is unavailable.
#
# We rotate through a small pool so the bot doesn't sound robotically
# identical across turns. These are intentionally vague-but-warm so they
# work for almost any chit-chat or off-rule input. They are ONLY used when
# `chat_text` returns nothing (timeout, rate limit, network error).
# ---------------------------------------------------------------------------
_STATIC_FALLBACK_BN: tuple[str, ...] = (
    "এই তো, এখানেই আছি! attendance, WFH বা কোম্পানির নিয়ম-কানুন নিয়ে কিছু লাগলে নিঃসংকোচে বলুন।",
    "হুম, বলুন তো — কী নিয়ে সাহায্য চান? attendance, WFH বা policy — যেকোনো কিছু জিজ্ঞেস করতে পারেন।",
    "আমি আছি। HR-সংক্রান্ত কোনো প্রশ্ন থাকলে নির্দ্বিধায় বলুন।",
)
_STATIC_FALLBACK_EN: tuple[str, ...] = (
    "I'm right here whenever you need me — attendance, WFH, or company rules, just ask.",
    "All good on my end. Anything HR-related I can help with — attendance, WFH, or policies?",
    "Sure thing. Let me know what you need — attendance, WFH, or HR policy questions are all fair game.",
)


def _static_fallback(user_lang: str) -> str:
    pool = _STATIC_FALLBACK_BN if user_lang == "bn" else _STATIC_FALLBACK_EN
    return random.choice(pool)


def _sanitize_fake_workflow_claims(text: str, *, user_lang: str) -> str:
    if not text or not _FAKE_WORKFLOW_SUCCESS_RE.search(text):
        return text
    if user_lang == "bn":
        return (
            "আমি এখনও আপনার leave/expense request workflow-এ নিই — "
            "চাইলে leave বা expense শুরু করতে বলুন, আমি ধাপে ধাপে সাজিয়ে দেব।"
        )
    return (
        "I haven't started a leave/expense workflow yet — "
        "say if you'd like to begin a leave or expense request and I'll walk you through it."
    )


def conversational_reply(
    *,
    message: str,
    context_lines: list[str] | None,
    trace_id: str,
    llm: LLMClient | None = None,
    workflow_hint: str | None = None,
) -> str | None:
    """Generate a friendly, human-toned reply.

    Tries the LLM first for a natural, contextual reply. If the LLM is not
    configured or the call fails (rate-limit, network), falls back to a small
    pool of warm canned replies in the user's language so the bot never
    surfaces a robotic "I'm your HR assistant…" greeting for chit-chat.

    Returns None only when the input itself is empty.
    """
    if not message or not message.strip():
        return None
    user_lang = detect_user_language(message)

    client = llm or LLMClient()
    if not client.is_configured():
        return _static_fallback(user_lang)

    lang_hint = {
        "bn": "The user wrote in Bangla or Banglish — reply in Bangla using Bengali script.",
        "en": "The user wrote in English — reply in English.",
    }.get(user_lang, "")

    history = _format_history(context_lines)
    hint_block = ""
    if workflow_hint:
        hint_block = f"\n\nWORKFLOW CONTEXT:\n{workflow_hint.strip()}\n"
    user_prompt = (
        f"{lang_hint}\n\n"
        f"Recent dialogue (oldest first, newest last):\n"
        f"{history or '(no prior turns)'}\n\n"
        f"User just said: {message}\n"
        f"{hint_block}\n"
        "Write the assistant reply now (plain text, no quotes, no system tags)."
    )

    out = client.chat_text(
        system_prompt=_FALLBACK_SYSTEM,
        user_prompt=user_prompt,
        trace_id=trace_id,
    )
    reply = out or _static_fallback(user_lang)
    return _sanitize_fake_workflow_claims(reply, user_lang=user_lang)
