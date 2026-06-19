"""Strict grounded-RAG prompts (informational only; no operational authority)."""

from __future__ import annotations

from knowledge_base.services.sanitization import extract_policy_title_phrases

GROUNDED_SYSTEM = """You are an HR policy assistant for employees. You write clear, well-organized answers.

NON-NEGOTIABLE RULES
- Use ONLY the evidence excerpts in the user message under EVIDENCE. Do not use outside knowledge.
- If the evidence does not clearly answer the question—including when excerpts only vaguely relate while omitting the specific fact requested (figures, quotas, durations, approvals, thresholds), set insufficient_evidence to true and answer with EXACTLY:
  "I could not find this policy in the handbook."
- Never invent policy details, numbers, deadlines, or approval rules not present in EVIDENCE.
- Never tell the user to submit requests, approve anything, or take operational actions; informational only.
- Do not output JSON outside the required schema.

LANGUAGE (follow REPLY_LANGUAGE in the user message exactly)
- English REPLY_LANGUAGE → entire answer in clear English; no Bengali Unicode.
- Bangla REPLY_LANGUAGE → entire answer in Bengali script; simple everyday words.
- Banglish REPLY_LANGUAGE → Romanized Bengali in Latin letters only; no Bengali Unicode.
- HR term safety: Termination / termination policy = job exit — NEVER "বিকাল" or "bikel" (evening).
  Use Termination (English), চাকরি সমাপ্তি (Bangla), or termination (Banglish).
- Do not translate evidence into a different language than REPLY_LANGUAGE.

ANSWER LAYOUT (inside the "answer" string — use markdown)
- Start with one short summary line (bold title ok, e.g. **ছুটি নীতি**).
- Group content with **section headings** when EVIDENCE has multiple topics.
- Use markdown bullet lists: each item on ONE line starting with "- " (never one word per line).
- Keep paragraphs short (2–4 sentences). Do not dump the entire handbook—only what answers the question.
- Do not repeat the same rule in multiple sections.

OUTPUT FORMAT
Return a single JSON object ONLY:
{"answer":"<string>","insufficient_evidence":<true|false>}
"""


def reply_language_instruction(lang: str) -> str:
    if lang == "en":
        return (
            "REPLY_LANGUAGE: English. Write the full answer in clear, simple English only. "
            "Do not use Bengali Unicode. Keep HR terms in standard English "
            "(Termination, Gross misconduct, Security breach)."
        )
    if lang == "banglish":
        return (
            "REPLY_LANGUAGE: Banglish. Use Latin letters only (Romanized Bengali + common English HR words). "
            "Do not use Bengali Unicode. Example style: \"termination policy te gross misconduct er rules\". "
            "Never use \"bikel\" for Termination."
        )
    return (
        "REPLY_LANGUAGE: Bangla (Bengali script). Write in clear, simple Bengali. "
        "Termination → চাকরি সমাপ্তি (NEVER বিকাল — that means evening). "
        "Prefer familiar HR phrasing: ক্যাজুয়াল ছুটি, জরুরি ছুটি, ম্যানেজার."
    )


def grounded_user_prompt(
    *,
    user_query: str,
    evidence_blocks: list[str],
    reply_language: str = "en",
) -> str:
    joined = "\n\n---\n\n".join(evidence_blocks)
    lang_line = reply_language_instruction(reply_language)
    focus = ""
    titles = extract_policy_title_phrases(user_query)
    if titles:
        focus = (
            f"\nFOCUS: The user asked about \"{titles[0]}\". "
            "Answer ONLY from evidence excerpts whose title matches that policy. "
            "Ignore excerpts from other policies even if they look related.\n"
        )
    return (
        f"{lang_line}{focus}\n\n"
        f"USER_QUESTION:\n{user_query}\n\n"
        f"EVIDENCE (excerpts from the official knowledge base; cite mentally from these only):\n{joined}\n"
    )
