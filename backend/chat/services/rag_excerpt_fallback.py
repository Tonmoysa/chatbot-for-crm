"""RAG excerpt fallback when grounded LLM is rate-limited or unavailable."""

from __future__ import annotations

from typing import Any

from django.conf import settings


def format_rag_excerpt(blocks: list[str], *, max_chars: int = 1200) -> str:
    """Readable policy excerpt from retrieved chunks — no generation LLM."""
    parts: list[str] = []
    running = 0
    for block in blocks:
        piece = block.strip()
        if not piece:
            continue
        if running + len(piece) + 2 > max_chars:
            remaining = max_chars - running - 2
            if remaining > 80:
                parts.append(piece[:remaining].rstrip() + "…")
            break
        parts.append(piece)
        running += len(piece) + 2
    if not parts:
        return ""
    return (
        "Here is what I found in your uploaded HR policies:\n\n"
        + "\n\n".join(parts)
    )


def payload_from_hit(hit: Any) -> dict[str, Any]:
    pl = getattr(hit, "payload", None) or {}
    if isinstance(pl, dict):
        return pl
    md = getattr(pl, "model_dump", None)
    if callable(md):
        dumped = md()
        if isinstance(dumped, dict):
            return dumped
    return {}


def build_excerpt_blocks(
    hits: list[Any],
    *,
    limit: int = 3,
    body_max_chars: int = 900,
) -> list[str]:
    from knowledge_base.services.sanitization import sanitize_retrieval_context

    blocks: list[str] = []
    for h in hits[:limit]:
        payload = payload_from_hit(h)
        if not payload:
            continue
        title = str(payload.get("section_title") or payload.get("document_title") or "Policy")
        body = sanitize_retrieval_context(str(payload.get("chunk_text") or ""), max_chars=body_max_chars)
        if body:
            blocks.append(f"**{title}**\n{body}")
    return blocks


def excerpt_result_from_hits(
    hits: list[Any],
    trace_id: str,
    *,
    company_id: str,
    retrieval_query: str,
) -> dict[str, Any] | None:
    """Build a no-LLM policy answer from retrieval hits (named-policy filter applied)."""
    if not hits:
        return None

    import knowledge_base.services.rag_pipeline as rag_module
    from chat.services.observability import log_step
    from knowledge_base.services.citation_builder import build_sources

    filtered = rag_module._prefer_named_policy_hits(hits, retrieval_query, trace_id)
    blocks = build_excerpt_blocks(filtered)
    excerpt = format_rag_excerpt(blocks)
    if not excerpt:
        return None

    log_step(trace_id, "rag_excerpt_fallback", {"blocks": len(blocks), "company_id": company_id})
    return {
        "hit": True,
        "text": excerpt,
        "sources": build_sources(filtered),
        "mode": "rag_excerpt",
    }


def apply_rag_excerpt_patch() -> None:
    """Wrap try_hr_policy_rag to return excerpts when generation LLM fails."""
    import knowledge_base.services.rag_pipeline as rag_module

    original = rag_module.try_hr_policy_rag

    def patched_try_hr_policy_rag(message, trace_id, *, company_id, department=None, llm=None):
        result = original(
            message,
            trace_id,
            company_id=company_id,
            department=department,
            llm=llm,
        )
        if result and (result.get("text") or result.get("answer")):
            return result

        if not getattr(settings, "KB_RAG_ENABLED", True):
            return result

        from knowledge_base.services.retriever import retrieve_for_query
        from knowledge_base.services.sanitization import build_hr_policy_retrieval_query

        msg = (message or "").strip()
        if not msg:
            return result

        retrieval_query = build_hr_policy_retrieval_query(msg) or msg
        hits, _ = retrieve_for_query(
            retrieval_query,
            trace_id,
            company_id=company_id,
            department=department,
            top_k=int(getattr(settings, "RAG_TOP_K", 8)),
            score_threshold=float(getattr(settings, "RAG_SCORE_THRESHOLD", 0.45)),
        )
        return excerpt_result_from_hits(
            hits,
            trace_id,
            company_id=company_id,
            retrieval_query=retrieval_query,
        )

    rag_module.try_hr_policy_rag = patched_try_hr_policy_rag
