"""Language detection + on-demand translation for the chatbot.

Why this module exists:
- The rules handbook content is stored in English. When a user writes in
  Bangla / Banglish, we want the answer in Bangla without duplicating every
  rule string.
- When a user explicitly asks "banglai bolo" / "translate this" we need to
  translate the *previous* assistant message rather than dropping into the
  generic "I didn't understand" greeting.

Both flows lean on the existing LLMClient. Detection is deterministic so the
orchestrator can route correctly even before any LLM call is made.
"""

from __future__ import annotations

import logging
import re
import time

from chat.services.llm_client import LLMClient

logger = logging.getLogger("hr_chatbot")

# Bengali Unicode block.
_BENGALI_CHAR_RE = re.compile(r"[\u0980-\u09FF]")

# Common Romanized-Bangla (Banglish) function words. Used only as a *signal* —
# they are short and would create false positives if matched as standalone
# English, so we require at least one of them to appear as a clue.
_BANGLISH_WORDS: frozenset[str] = frozenset(
    {
        "ami", "amar", "amake", "amra", "amader", "tumi", "tomar", "tomake",
        "apni", "apnar", "apnake",
        "ki", "kothay", "kobe", "keno", "kemne", "kemon", "kotha",
        "lagbe", "lage", "dorkar", "chai", "chaina",
        "bolo", "bolun", "bolen", "boltesi",
        "korbo", "korba", "korben", "kortesi", "korte", "korlo", "kore",
        "gula", "guli", "gulo",
        "hoyeche", "hoyese", "hocche", "ache", "achi", "asho", "ashbe",
        "kalke", "ajke", "porshu", "aaj", "shokal", "rat",
        "shob", "sob", "shokol", "kichu", "kichui",
        "thik", "shotti",
        "chuti", "chhuti",
        "kharcha", "khoroch", "taka",
        "banglai", "bangla", "banglate", "bangalir",
        "anubad", "anuvad",
        "porte", "pori", "poro",
        "niyom", "niyam", "bidhi", "niti",
        "khub", "onek", "ektu",
    }
)


def has_bengali_chars(message: str) -> bool:
    return bool(_BENGALI_CHAR_RE.search(message or ""))


def has_banglish_words(message: str) -> bool:
    low = (message or "").lower()
    tokens = set(re.findall(r"[a-z]+", low))
    return bool(tokens & _BANGLISH_WORDS)


def detect_explicit_reply_language(message: str) -> str | None:
    """
    User explicitly asks for a reply language (overrides Banglish word heuristics).
    ``banglai`` / ``bangla te`` → Bengali script, not Romanized Banglish.
    """
    if not message:
        return None
    low = message.lower()
    raw = message
    if re.search(r"\b(banglish|romanized|latin\s+letters)\b", low):
        return "banglish"
    if re.search(r"(বাংলায়|বাংলা\s*তে|বাংলা\s*করে|বাংলায়\s*বল|বাংলায়\s*বুঝ)", raw):
        return "bn"
    if re.search(
        r"(^|\b)(banglai|banglay|banglate)(\b|$)|"
        r"\bin\s+bangla\b|\bin\s+bengali\b|"
        r"bangla\s*te\b|bangla\s*kore\b|bangla\s*version\b|"
        r"explain.{0,30}\b(bangla|bengali|banglai|banglay)\b|"
        r"\b(bangla|bengali|banglai|banglay)\b.{0,30}\b(explain|bolo|bol|bujhao|bujhay|describe)\b|"
        r"\b(explain|bujhao|bujhay|describe|anubad)\b.{0,30}\b(banglai|banglay|banglate|bangla)\b",
        low,
    ):
        return "bn"
    if re.search(
        r"\bin\s+english\b|\benglish\s*e\b|\benglish\s*version\b|ইংরেজিতে",
        low,
    ) or re.search(r"ইংরেজিতে", raw):
        return "en"
    return None


def detect_reply_language(message: str) -> str:
    """
    Language the assistant should use in its reply: ``en``, ``bn`` (Bengali script),
    or ``banglish`` (Romanized Bengali, Latin letters only).
    """
    if not message:
        return "en"
    explicit = detect_explicit_reply_language(message)
    if explicit:
        return explicit
    if has_bengali_chars(message):
        return "bn"
    if has_banglish_words(message):
        return "banglish"
    return "en"


# Short confirmations / wizard tokens — keep the session reply language.
_WEAK_REPLY_RE = re.compile(
    r"^(?:"
    r"yes|no|yep|yeah|y|n|ok|okay|submit|summary|summery|clear|confirm|cancel|stop|done|"
    r"thanks|thank\s*you|"
    r"হ্যাঁ|না|ঠিক|শেষ|জমা|সারাংশ|"
    r"ha|hae|han|na"
    r")(?:[.!?]*)$",
    re.I | re.UNICODE,
)


def is_weak_language_signal(message: str) -> bool:
    """True when the message does not carry enough language signal on its own."""
    text = (message or "").strip()
    if not text:
        return True
    if _WEAK_REPLY_RE.match(text):
        return True
    # e.g. "yes please" / "no thanks" — still weak if no Bangla/Banglish cues.
    if len(text) <= 16 and not has_bengali_chars(text) and not has_banglish_words(text):
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if words and len(words) <= 3 and all(
            w in {"yes", "no", "yep", "yeah", "ok", "okay", "submit", "summary", "clear", "confirm", "cancel", "done", "thanks", "thank", "please"}
            for w in words
        ):
            return True
    return False


def resolve_reply_language(message: str, stored: str | None = None) -> str:
    """
    Pick reply language for this turn.

    Explicit requests (``banglai bolo``, ``in english``) win. Ambiguous tokens
    such as ``yes`` / ``submit`` / ``summary`` keep ``stored`` when set.
    """
    explicit = detect_explicit_reply_language(message)
    if explicit:
        return explicit
    if stored and is_weak_language_signal(message):
        return stored
    return detect_reply_language(message)


def detect_content_language(text: str) -> str:
    """Rough language of existing assistant/policy text (for alignment)."""
    if not (text or "").strip():
        return "en"
    if has_bengali_chars(text):
        return "bn"
    if has_banglish_words(text):
        return "banglish"
    return "en"


def detect_user_language(message: str) -> str:
    """Return 'bn' for Bangla/Banglish queries, otherwise 'en' (legacy binary API)."""
    lang = detect_reply_language(message)
    return "bn" if lang in ("bn", "banglish") else "en"


def is_translation_request(message: str) -> str | None:
    """
    Return the target language code if the user is asking to translate
    the previous answer. Returns None when the message is not a
    translation request.

    Supported phrasings:
      - English: "translate", "translate this into bangla", "translate to english"
      - Banglish: "banglai bolo", "bangla te bolo", "english e bolo"
      - Bengali: "বাংলায় বলো", "বাংলায় অনুবাদ করো", "ইংরেজিতে বলুন"
    """
    if not message:
        return None
    low = message.lower()
    raw = message

    # Bengali script phrasings — check first so we never miscategorize them.
    if re.search(
        r"(বাংলায়|বাংলা\s*তে|বাংলা\s*ভাষায়|বাংলা\s*করে|বাংলা\s*অনুবাদ|"
        r"অনুবাদ\s*কর|অনুবাদ\s*করুন|বাংলা\s*ভার্সন)",
        raw,
    ):
        return "bn"
    if re.search(r"(ইংরেজিতে|ইংরেজি\s*ভাষায়|ইংরেজি\s*করে|ইংরেজি\s*অনুবাদ)", raw):
        return "en"

    # Banglish phrasings.
    if re.search(
        r"\b(banglai|banglay|banglate|bangla\s*te|bangla\s*kore|bangla\s*version)"
        r"\b.*\b(bolo|bol|bolun|bolen|likho|likh|lekho|bujhao|bujhayo|bujhao)\b",
        low,
    ) or re.search(
        r"\b(bangla(i|y|te)?|bangla\s+te|bangla\s+kore)\b\s+(bolo|bol|bolun|likho|"
        r"lekho|bujhao)\b",
        low,
    ) or re.search(r"\bbanglai\s+bolo\b", low) or re.search(r"\bbangla\s+korun\b", low):
        return "bn"
    if re.search(
        r"\b(english\s+(e|te|version|kore)|english\s+e\s+bolo|english\s+kore\s+bolo)\b",
        low,
    ):
        return "en"

    # English phrasings.
    if re.search(
        r"\b(translate|translation|convert\s+(?:this|it)?\s*(?:into|to))\b",
        low,
    ):
        if re.search(r"\b(bangla|bengali|bangali)\b", low):
            return "bn"
        if re.search(r"\benglish\b", low):
            return "en"
        # Most uses in this product translate EN → BN; default to Bangla.
        return "bn"

    # "explain in bangla", "termination policy explain koro banglai"
    if re.search(r"\bexplain\b", low) and re.search(
        r"\b(bangla|bengali|banglai|banglay|banglate)\b", low
    ):
        return "bn"
    if re.search(r"\b(explain|bujhao|bujhay|describe|anubad)\b", low) and re.search(
        r"\b(banglai|banglay|banglate|bangla\s*te)\b", low
    ):
        return "bn"
    if re.search(r"\b(banglai|banglay|banglate|bangla\s*te)\b", low) and re.search(
        r"\b(explain|bolo|bol|koro|kor[eo]|bujhao|bujhay|anubad|translate)\b",
        low,
    ):
        return "bn"

    return None


_POLICY_FOOTER_TAIL_RE = re.compile(
    r"\n*_\([^)]*(?:uploaded polic|আপলোড|tomar uploaded|policy name)[^)]*\)_\s*$",
    re.I | re.DOTALL,
)


def strip_policy_footer(text: str) -> str:
    """Remove policy citation footer before translating or re-aligning language."""
    return _POLICY_FOOTER_TAIL_RE.sub("", (text or "")).rstrip()


_TRANSLATION_SYSTEM = (
    "You are a precise HR-domain translator. Translate the user's message into {lang_name}. "
    "Strict rules:\n"
    "- Preserve markdown exactly: ### headings, **bold**, - bullet lists, --- separators, blank lines.\n"
    "- Preserve numbers, section numbers, dates, times, and abbreviations exactly "
    "(e.g. PPE, ERP, CRM, KPI, HR, PIP, BDT, 9:00 AM, 3–6 months).\n"
    "- Preserve English proper nouns and product names that should not be localized.\n"
    "- HR term safety: \"Termination\" / termination policy is job exit — NEVER translate as "
    "\"বিকাল\" or \"bikel\" (that means evening). Use \"Termination\", \"চাকরি সমাপ্তি\", or "
    "\"termination\" as appropriate.\n"
    "- For Bangla output, do NOT invent unusual Bangla compounds for established HR/business terms. "
    "Either keep them in English or use the common Banglish transliteration. "
    "Examples (Bangla output): \"Casual Leave\" → \"ক্যাজুয়াল ছুটি\" (or keep \"Casual Leave\"); "
    "\"Sick Leave\" → \"সিক লিভ\" / \"অসুস্থতার ছুটি\"; "
    "\"Annual Leave\" → \"বার্ষিক ছুটি\"; "
    "\"Maternity / Paternity Leave\" → \"ম্যাটার্নিটি / প্যাটার্নিটি ছুটি\"; "
    "\"Emergency Leave\" → \"জরুরি ছুটি\"; "
    "\"Leave Without Pay\" or \"LWOP\" → \"বেতন ছাড়া ছুটি (LWOP)\"; "
    "\"Manager\" → \"ম্যানেজার\". Never produce literal/awkward word-for-word coinings.\n"
    "- Do not add commentary, introductions, or explanations. Output only the translation.\n"
    "- Keep the tone professional but simple and human-friendly.\n"
    "- Do not invent any rules or numbers that are not in the source text."
)

_BANGLISH_REWRITE_SYSTEM = (
    "You rewrite HR policy text in natural Banglish for Bangladesh office chat.\n"
    "Strict rules:\n"
    "- Use ONLY Latin letters (a-z). Do NOT use Bengali Unicode script.\n"
    "- Preserve markdown: **bold**, - bullets, blank lines.\n"
    "- Keep numbers, dates, and standard HR English terms when clearer: Termination, "
    "Gross misconduct, Security breach, Casual Leave, manager, HR.\n"
    "- NEVER use \"bikel\" for Termination (bikel means evening). Use \"termination\" or "
    "\"job termination\".\n"
    "- Rewrite ONLY what is in the source — do NOT invent policy rules, day counts, "
    "entitlements, or rhetorical questions (keno, ki ki) that are not in the source.\n"
    "- Do not add conversational commentary or duplicate advice.\n"
    "- Output only the rewritten text — no commentary."
)


def _lang_name(code: str) -> str:
    return {
        "bn": "Bangla (Bengali script — Unicode)",
        "en": "English",
        "banglish": "Banglish (Romanized Bengali, Latin letters only)",
    }.get(code, code)


def _translation_system_for(target_lang: str) -> str:
    if target_lang == "banglish":
        return _BANGLISH_REWRITE_SYSTEM
    return _TRANSLATION_SYSTEM.format(lang_name=_lang_name(target_lang))


def align_policy_answer_language(
    answer: str,
    *,
    user_message: str,
    trace_id: str,
    llm: LLMClient | None = None,
) -> str:
    """
    If the grounded policy answer language does not match how the user wrote,
    translate or rewrite so English → English, Bangla script → Bangla, Banglish → Banglish.
    """
    if not (answer or "").strip():
        return answer
    target = detect_reply_language(user_message)
    current = detect_content_language(answer)
    if target == current:
        return answer
    translated, ok = translate_text(
        answer,
        target_lang=target,
        trace_id=trace_id,
        llm=llm,
    )
    return translated if ok else answer


def align_workflow_answer_language(
    answer: str,
    *,
    user_message: str,
    stored_lang: str | None = None,
    trace_id: str,
    llm: LLMClient | None = None,
) -> str:
    """
    Align assistant copy to the user's language.

    Uses sticky ``stored_lang`` so short replies like ``yes`` stay in the
    language the user started the workflow in.
    """
    if not (answer or "").strip():
        return answer
    target = resolve_reply_language(user_message, stored_lang)
    current = detect_content_language(answer)
    if target == current:
        return answer
    translated, ok = translate_text(
        answer,
        target_lang=target,
        trace_id=trace_id,
        llm=llm,
    )
    return translated if ok else answer


# Soft cap per chunk. Bangla output uses ~2–3× more tokens than the same
# English source, so we keep chunks modest (~2500 chars in) to ensure the
# Bangla translation always fits within the LLM's output token budget
# without truncation. Smaller chunks mean a few more LLM calls but reliable
# full translations every time.
_TRANSLATION_CHUNK_CHARS = 2500

# Token budget for each chunk's translated output. Bangla expansion needs
# more tokens; we give plenty of headroom so the LLM never has to cut off.
_TRANSLATION_MAX_TOKENS = 8192

# Polite pause between chunk requests so we never burst the provider.
# Many free / shared LLM endpoints enforce per-second AND per-minute caps;
# a ~2.5s pace keeps us comfortably under both.
_INTER_CHUNK_DELAY_SEC = 2.5

# Retry backoff (seconds) after a failed chunk call. The provider's rate
# limit window is typically per-minute, so later waits are deliberately
# generous to ride out 429 / quota errors.
_RETRY_BACKOFFS = (3.0, 8.0, 15.0)


def _split_for_translation(text: str, max_chars: int = _TRANSLATION_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into translation-friendly chunks.

    Splitting strategy (in order of preference):
      1. ``### `` section headers (each handbook section is self-contained).
      2. Paragraph breaks (blank lines).
      3. Single newlines.
    A chunk is never broken mid-sentence; if a single section already
    exceeds ``max_chars`` we recurse into paragraphs.
    """
    if len(text) <= max_chars:
        return [text]

    # Step 1: split on section markers but KEEP them at the start of each piece.
    parts = re.split(r"(?m)(?=^### )", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_chars:
            # Flush whatever is buffered first.
            if buf:
                chunks.append(buf)
                buf = ""
            # Section too big on its own — break it down by paragraph.
            paras = part.split("\n\n")
            inner = ""
            for para in paras:
                candidate = inner + ("\n\n" + para if inner else para)
                if len(candidate) <= max_chars:
                    inner = candidate
                else:
                    if inner:
                        chunks.append(inner)
                    if len(para) <= max_chars:
                        inner = para
                    else:
                        # Worst case — break the paragraph by single newlines.
                        lines = para.split("\n")
                        line_buf = ""
                        for line in lines:
                            cand = line_buf + ("\n" + line if line_buf else line)
                            if len(cand) <= max_chars:
                                line_buf = cand
                            else:
                                if line_buf:
                                    chunks.append(line_buf)
                                line_buf = line
                        if line_buf:
                            inner = line_buf
            if inner:
                chunks.append(inner)
            continue

        candidate = buf + part if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = part
    if buf:
        chunks.append(buf)
    return chunks


def translate_text(
    text: str,
    *,
    target_lang: str,
    trace_id: str,
    llm: LLMClient | None = None,
) -> tuple[str, bool]:
    """Translate ``text`` into ``target_lang``. Returns (translated_text, ok).

    Long inputs (e.g. the full rules handbook) are split into self-contained
    chunks at section / paragraph boundaries and translated independently so
    the LLM's output token limit can never truncate the answer. The chunked
    translations are then joined back into a single Markdown string.

    On any LLM error or when the client is not configured the original text
    is returned with ok=False so callers can degrade gracefully without
    surfacing an error to the user.
    """
    if not text or not text.strip():
        return ("", False)
    client = llm or LLMClient()
    if not client.is_configured():
        return (text, False)
    system = _translation_system_for(target_lang)

    chunks = _split_for_translation(text)
    logger.info(
        "translation_chunks trace_id=%s chunks=%d total_chars=%d",
        trace_id,
        len(chunks),
        len(text),
    )

    def _translate_one(chunk: str, idx_label: str) -> str | None:
        """Translate a single chunk with retries. Handles transient
        rate-limit / 5xx errors. Returns None only when all retries fail."""
        attempts = len(_RETRY_BACKOFFS) + 1
        for attempt in range(attempts):
            part = client.chat_text(
                system_prompt=system,
                user_prompt=chunk,
                trace_id=f"{trace_id}-{idx_label}-a{attempt + 1}",
                max_tokens=_TRANSLATION_MAX_TOKENS,
            )
            if part:
                return part.strip()
            if attempt < len(_RETRY_BACKOFFS):
                time.sleep(_RETRY_BACKOFFS[attempt])
        logger.warning(
            "translation_chunk_failed trace_id=%s idx=%s",
            trace_id,
            idx_label,
        )
        return None

    if len(chunks) == 1:
        out = _translate_one(chunks[0], "single")
        if not out:
            return (text, False)
        return (out, True)
    # else: fall through to multi-chunk path below.

    fallback_note = (
        "\n\n_(এই অংশটি এই মুহূর্তে অনুবাদ করা যাচ্ছে না — কিছুক্ষণ পরে আবার চেষ্টা করুন)_"
        if target_lang == "bn"
        else "\n\n_(this section could not be translated right now — try again shortly)_"
    )

    translated_parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if idx > 1:
            time.sleep(_INTER_CHUNK_DELAY_SEC)
        part = _translate_one(chunk, f"c{idx}of{len(chunks)}")
        if part:
            translated_parts.append(part)
        else:
            # Keep this chunk in the source language so the user still gets
            # most of the answer. Mark it visibly so they know which section
            # to retry if they want a fully translated version.
            translated_parts.append(chunk + fallback_note)

    joined = "\n\n".join(translated_parts)
    # Partial success still counts as a success — surfacing the global
    # "translation unavailable" banner just because one chunk failed would
    # discard 80%+ of useful Bangla output.
    return (joined, True)
