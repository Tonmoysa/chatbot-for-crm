"""Semantic retrieval against Qdrant with embedding."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from django.conf import settings

from chat.services.llm_client import LLMClient
from chat.services.observability import log_step
from knowledge_base.services.qdrant_service import search_vectors
from knowledge_base.services.sanitization import (
    build_retrieval_embedding_text,
    extract_policy_title_phrases,
    preprocess_query,
)

logger = logging.getLogger("hr_chatbot")


def _tenant_filter(*, company_id: str, department: str | None):
    company = (company_id or "").strip()
    if not company:
        raise ValueError("company_id is required for tenant-scoped retrieval.")
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(
                key="company_id",
                match=MatchValue(value=company),
            )
        ]
        if department and str(department).strip():
            must.append(
                FieldCondition(
                    key="department",
                    match=MatchValue(value=str(department).strip()),
                )
            )
        return Filter(
            must=must
        )
    except Exception:
        raise


def _hit_score(hit: Any) -> float:
    return float(getattr(hit, "score", 0.0) or 0.0)


def _payload_dict(hit: Any) -> dict[str, Any]:
    pl = getattr(hit, "payload", None) or {}
    if isinstance(pl, dict):
        return pl
    md = getattr(pl, "model_dump", None)
    if callable(md):
        dumped = md()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _payload_title_surface(payload: dict[str, Any]) -> str:
    doc = str(payload.get("document_title") or payload.get("source_document") or "")
    sec = str(payload.get("section_title") or "")
    return f"{doc} {sec}".lower()


def _title_match_boost(query: str, payload: dict[str, Any]) -> float:
    titles = extract_policy_title_phrases(query)
    if not titles:
        return 0.0
    surface = _payload_title_surface(payload)
    if not surface.strip():
        return 0.0
    boost = 0.0
    for title in titles:
        tl = title.lower()
        if tl in surface:
            boost = max(boost, 0.28)
            continue
        tokens = [t for t in re.split(r"\W+", tl) if len(t) >= 3]
        if not tokens:
            continue
        matched = sum(1 for t in tokens if t in surface)
        if matched == len(tokens):
            boost = max(boost, 0.22)
        elif matched >= max(1, len(tokens) - 1):
            boost = max(boost, 0.14 * matched / len(tokens))
    return boost


def _rerank_by_policy_title(hits: list[Any], query: str) -> list[Any]:
    if not hits or not extract_policy_title_phrases(query):
        return hits
    return sorted(
        hits,
        key=lambda h: _hit_score(h) + _title_match_boost(query, _payload_dict(h)),
        reverse=True,
    )


def _min_results_floor(top_k: int, pool_size: int) -> int:
    """Minimum hits to return on graceful degradation (never zero if pool non-empty)."""
    if pool_size <= 0:
        return 0
    return min(pool_size, max(1, min(3, top_k)))


def _apply_soft_ranking(
    hits: list[Any],
    *,
    top_k: int,
    min_sim: float,
    trace_id: str,
    used_relaxed: bool,
) -> list[Any]:
    """
    Prefer chunks at or above ``min_sim``; if that would drop everything, fall back to
    the best-ranked raw Qdrant hits so grounded generation can still run.
    """
    if not hits:
        return []

    ranked = sorted(hits, key=_hit_score, reverse=True)
    max_score = _hit_score(ranked[0])
    scores_preview = [round(_hit_score(h), 4) for h in ranked[:8]]

    preferred = [h for h in ranked if _hit_score(h) >= min_sim]
    if preferred:
        out = preferred[:top_k]
        log_step(
            trace_id,
            "rag_retrieval_ranked",
            {
                "max_score": round(max_score, 4),
                "min_similarity": min_sim,
                "mode": "preferred",
                "pool": len(ranked),
                "returned": len(out),
                "relaxed_pass": used_relaxed,
                "scores": scores_preview,
            },
        )
        return out

    # Graceful degradation: Qdrant returned candidates but all are below min_sim.
    take = max(_min_results_floor(top_k, len(ranked)), min(top_k, len(ranked)))
    out = ranked[:take]
    log_step(
        trace_id,
        "rag_retrieval_fallback_raw_topk",
        {
            "max_score": round(max_score, 4),
            "min_similarity": min_sim,
            "mode": "fallback_raw_topk",
            "pool": len(ranked),
            "returned": len(out),
            "relaxed_pass": used_relaxed,
            "scores": scores_preview,
            "reason": "all_below_min_similarity",
        },
    )
    logger.info(
        "rag_retrieval_fallback_raw_topk trace_id=%s max_score=%.4f min_sim=%.4f returned=%s",
        trace_id,
        max_score,
        min_sim,
        len(out),
    )
    return out


def _coerce_embedding_vector(vec: Any, *, trace_id: str) -> list[float] | None:
    """Match ingest path: plain ``list[float]`` for Qdrant ``query_points``."""
    if vec is None:
        return None
    if isinstance(vec, (list, tuple)):
        try:
            return [float(x) for x in vec]
        except (TypeError, ValueError):
            return None
    try:
        import numpy as np

        if isinstance(vec, np.ndarray):
            flat = np.asarray(vec, dtype=np.float64).reshape(-1)
            return [float(x) for x in flat.tolist()]
    except Exception:
        pass
    if hasattr(vec, "tolist"):
        try:
            raw = vec.tolist()
            if isinstance(raw, (int, float)):
                return [float(raw)]
            if isinstance(raw, list):
                return [float(x) for x in raw]
        except (TypeError, ValueError):
            return None
    logger.warning(
        "rag_embed_vector_coerce_unsupported trace_id=%s type=%s",
        trace_id,
        type(vec).__name__,
    )
    return None


def retrieve_for_query(
    query: str,
    trace_id: str,
    *,
    company_id: str,
    department: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> tuple[list[Any], int]:
    """
    Returns (scored_points, embedding_latency_ms for the query embedding call).

    Uses the same ``LLMClient.embed_texts`` path as ingestion. If the strict
    Qdrant ``score_threshold`` returns no points, retries without a server-side
    threshold against a wider candidate pool, then soft-ranks by similarity.
    Chunks below ``RAG_MIN_SIMILARITY`` are deprioritized; if that would remove
    every hit, returns the top raw Qdrant results (at least one, up to three).
    """
    if not getattr(settings, "KB_RAG_ENABLED", True):
        return [], 0

    q = preprocess_query(query)
    if not q:
        return [], 0

    embedding_prompt = build_retrieval_embedding_text(q)
    if embedding_prompt.strip() != q.strip():
        log_step(
            trace_id,
            "rag_embedding_query_augmented",
            {"prompt_chars": len(embedding_prompt), "original_chars": len(q)},
        )

    llm = LLMClient()
    if not llm.is_embedding_configured():
        return [], 0

    t0 = time.perf_counter()
    vectors = llm.embed_texts([embedding_prompt], trace_id)
    emb_ms = int((time.perf_counter() - t0) * 1000)
    if not vectors:
        log_step(trace_id, "rag_embed_query_failed", {"ms": emb_ms})
        return [], emb_ms
    raw_vec = vectors[0]
    if raw_vec is None:
        log_step(trace_id, "rag_embed_query_failed", {"ms": emb_ms, "reason": "null_vector"})
        return [], emb_ms
    if isinstance(raw_vec, (list, tuple)) and len(raw_vec) == 0:
        log_step(trace_id, "rag_embed_query_failed", {"ms": emb_ms, "reason": "empty_vector"})
        return [], emb_ms

    qv = _coerce_embedding_vector(raw_vec, trace_id=trace_id)
    if not qv:
        log_step(trace_id, "rag_embed_vector_coerce_failed", {"ms": emb_ms})
        return [], emb_ms

    expected = int(getattr(settings, "QDRANT_VECTOR_SIZE", 0) or 0)
    if expected and len(qv) != expected:
        log_step(
            trace_id,
            "rag_query_vector_dim_mismatch",
            {"expected": expected, "got": len(qv), "ms": emb_ms},
        )
        logger.warning(
            "rag_query_vector_dim_mismatch trace_id=%s expected=%s got=%s",
            trace_id,
            expected,
            len(qv),
        )
        return [], emb_ms

    log_step(
        trace_id,
        "rag_query_vector_ready",
        {
            "dim": len(qv),
            "sample_head": [round(x, 5) for x in qv[:3]],
            "embedding_ms": emb_ms,
        },
    )

    k = int(top_k or getattr(settings, "RAG_TOP_K", 8))
    thr = score_threshold
    if thr is None:
        thr = float(getattr(settings, "RAG_SCORE_THRESHOLD", 0.45))
    min_sim = float(getattr(settings, "RAG_MIN_SIMILARITY", 0.3))
    cand_mult = max(2, int(getattr(settings, "RAG_RELAXED_CANDIDATE_MULTIPLIER", 3)))
    relaxed_limit = max(k * cand_mult, k + 16)

    flt = _tenant_filter(company_id=company_id, department=department)
    t1 = time.perf_counter()
    try:
        hits = search_vectors(
            qv,
            limit=k,
            score_threshold=thr,
            payload_filter=flt,
            trace_id=trace_id,
        )
        used_relaxed = False
        if not hits:
            log_step(
                trace_id,
                "rag_retrieval_relaxed_retry",
                {
                    "strict_threshold": thr,
                    "reason": "no_server_hits_with_threshold",
                },
            )
            hits = search_vectors(
                qv,
                limit=relaxed_limit,
                score_threshold=None,
                payload_filter=flt,
                trace_id=trace_id,
            )
            used_relaxed = True

        raw_pool = list(hits)
        hits = _apply_soft_ranking(
            raw_pool,
            top_k=k,
            min_sim=min_sim,
            trace_id=trace_id,
            used_relaxed=used_relaxed,
        )
        hits = _rerank_by_policy_title(hits, q)
    except Exception as exc:
        log_step(
            trace_id,
            "rag_qdrant_search_failed",
            {
                "error": type(exc).__name__,
                "detail": (str(exc) or "")[:500],
            },
        )
        logger.warning(
            "rag_qdrant_search_failed trace_id=%s err=%s detail=%s",
            trace_id,
            type(exc).__name__,
            str(exc)[:300],
        )
        return [], emb_ms

    ret_ms = int((time.perf_counter() - t1) * 1000)
    max_score = round(_hit_score(hits[0]), 4) if hits else None
    log_step(
        trace_id,
        "rag_retrieval_done",
        {
            "embedding_ms": emb_ms,
            "retrieval_ms": ret_ms,
            "hits": len(hits),
            "max_score": max_score,
            "scores": [round(_hit_score(h), 4) for h in hits[:8]],
            "collection": getattr(settings, "QDRANT_COLLECTION", ""),
            "company_id": company_id,
        },
    )

    return hits, emb_ms
