"""Orchestrator-facing RAG entry: retrieve → ground → citations."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from django.conf import settings

# from chat.services.rules_handbook import wants_full_handbook  # disabled: RAG-only policy answers
from chat.services.llm_client import LLMClient
from chat.services.translator import detect_reply_language
from chat.services.observability import log_step
from knowledge_base.services.citation_builder import build_sources
from knowledge_base.services.prompts import GROUNDED_SYSTEM, grounded_user_prompt
from knowledge_base.services.retriever import retrieve_for_query
from chat.services.message_polish import polish_policy_answer
from knowledge_base.services.sanitization import (
    build_hr_policy_retrieval_query,
    extract_policy_title_phrases,
    sanitize_retrieval_context,
)

logger = logging.getLogger("hr_chatbot")

_NOT_FOUND = (
    "I could not find this policy (or a clear answer) in your uploaded policies. "
    "Try asking with the policy name or topic (for example, \"expense policy\" or "
    "\"daily expense limit\"), or contact HR for confirmation."
)


def hr_policy_not_found_message() -> str:
    """User-visible text when RAG is enabled but no grounded answer is produced."""
    custom = getattr(settings, "KB_RAG_NOT_FOUND_MESSAGE", None)
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    return _NOT_FOUND


def _payload_from_hit(hit: Any) -> dict[str, Any]:
    pl = getattr(hit, "payload", None) or {}
    if isinstance(pl, dict):
        return pl
    md = getattr(pl, "model_dump", None)
    if callable(md):
        dumped = md()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _title_matches_named_policy(query: str, payload: dict[str, Any]) -> bool:
    from knowledge_base.services.retriever import _title_match_boost

    return _title_match_boost(query, payload) > 0.0


def _prefer_named_policy_hits(
    hits: list[Any], query: str, trace_id: str
) -> list[Any]:
    titles = extract_policy_title_phrases(query)
    if not titles:
        return hits
    matched = [h for h in hits if _title_matches_named_policy(query, _payload_from_hit(h))]
    if matched:
        log_step(
            trace_id,
            "rag_title_filter",
            {"named_policy": titles[0], "kept": len(matched), "pool": len(hits)},
        )
        return matched
    return hits


def try_hr_policy_rag(
    message: str,
    trace_id: str,
    *,
    company_id: str,
    department: str | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any] | None:
    """
    Returns a dict with keys: hit (bool), text (str), sources (list), mode ('rag').
    Returns None when RAG is disabled or hard infrastructure failure before retrieval.
    """
    if not getattr(settings, "KB_RAG_ENABLED", True):
        return None

    # Static handbook "show all" bypass removed — broad queries also go through retrieve + LLM.
    # if wants_full_handbook(message or ""):
    #     log_step(trace_id, "rag_skip_full_handbook", {})
    #     return None

    msg = (message or "").strip()
    if not msg:
        return None

    retrieval_query = build_hr_policy_retrieval_query(msg) or msg
    if retrieval_query.strip() != msg:
        log_step(
            trace_id,
            "rag_query_rewritten",
            {"original_chars": len(msg), "retrieval_chars": len(retrieval_query)},
        )

    t0 = time.perf_counter()
    hits, _emb_ms = retrieve_for_query(
        retrieval_query,
        trace_id,
        company_id=company_id,
        department=department,
        top_k=int(getattr(settings, "RAG_TOP_K", 8)),
        score_threshold=float(getattr(settings, "RAG_SCORE_THRESHOLD", 0.45)),
    )
    if not hits:
        log_step(
            trace_id,
            "rag_no_hits",
            {"ms": int((time.perf_counter() - t0) * 1000), "company_id": company_id},
        )
        return None

    hits = _prefer_named_policy_hits(hits, retrieval_query, trace_id)

    blocks: list[str] = []
    max_ctx = int(getattr(settings, "RAG_MAX_CONTEXT_CHARS", 10_000))
    running = 0
    for h in hits:
        payload = _payload_from_hit(h)
        if not payload:
            continue
        title = str(payload.get("section_title") or payload.get("document_title") or "Policy")
        body = sanitize_retrieval_context(str(payload.get("chunk_text") or ""), max_chars=4000)
        if not body:
            continue
        piece = f"[{title}]\n{body}"
        if running + len(piece) > max_ctx:
            break
        blocks.append(piece)
        running += len(piece)

    if not blocks:
        log_step(trace_id, "rag_empty_context", {})
        return None

    client = llm or LLMClient()
    if not client.is_configured():
        from chat.services.rag_excerpt_fallback import excerpt_result_from_hits

        excerpt = excerpt_result_from_hits(
            hits,
            trace_id,
            company_id=company_id,
            retrieval_query=retrieval_query,
        )
        if excerpt:
            return excerpt
        return None

    reply_lang = detect_reply_language(msg)
    user_prompt = grounded_user_prompt(
        user_query=msg,
        evidence_blocks=blocks,
        reply_language=reply_lang,
    )
    log_step(trace_id, "rag_reply_language", {"lang": reply_lang})
    t1 = time.perf_counter()
    parsed = client.chat_json(
        system_prompt=GROUNDED_SYSTEM,
        user_prompt=user_prompt,
        trace_id=trace_id,
    )
    gen_ms = int((time.perf_counter() - t1) * 1000)
    log_step(
        trace_id,
        "rag_generation_done",
        {"ms": gen_ms, "ok": bool(parsed)},
    )

    if not isinstance(parsed, dict):
        from chat.services.rag_excerpt_fallback import excerpt_result_from_hits

        excerpt = excerpt_result_from_hits(
            hits,
            trace_id,
            company_id=company_id,
            retrieval_query=retrieval_query,
        )
        if excerpt:
            return excerpt
        return None

    insufficient = bool(parsed.get("insufficient_evidence"))
    answer = str(parsed.get("answer") or "").strip()
    if re.search(r"could not find this policy", answer, re.I):
        insufficient = True
        answer = ""
    if insufficient or not answer:
        log_step(
            trace_id,
            "rag_insufficient_evidence",
            {
                "insufficient": insufficient,
                "hits": len(hits),
                "had_answer": bool(answer),
            },
        )
        from chat.services.rag_excerpt_fallback import excerpt_result_from_hits

        excerpt = excerpt_result_from_hits(
            hits,
            trace_id,
            company_id=company_id,
            retrieval_query=retrieval_query,
        )
        if excerpt:
            return excerpt
        return None

    answer = polish_policy_answer(answer)

    sources = build_sources(hits)
    log_step(trace_id, "rag_success", {"sources": len(sources), "company_id": company_id})
    return {
        "hit": True,
        "text": answer,
        "sources": sources,
        "mode": "rag",
    }
