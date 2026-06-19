"""Unified synchronous document ingestion service.

Admin uploads and API uploads both enter through process_document(document):
extract -> sanitize -> chunk -> embed -> Qdrant -> ORM chunks.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from django.conf import settings
from django.utils import timezone
from qdrant_client.models import PointStruct

from chat.services.document_reader import DocumentExtractResult, extract_text_from_upload
from chat.services.llm_client import LLMClient
from chat.services.observability import log_step
from chat.services.translator import detect_user_language
from knowledge_base.models import DocumentStatus, DocumentType, KnowledgeChunk, KnowledgeDocument
from knowledge_base.services.chunker import chunk_policy_text, count_tokens
from knowledge_base.services.qdrant_service import delete_by_document_id, upsert_points
from knowledge_base.services.sanitization import sanitize_for_indexing

logger = logging.getLogger("hr_chatbot")


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _embedding_version() -> str:
    explicit = getattr(settings, "EMBEDDING_VERSION", "")
    if explicit:
        return str(explicit)
    backend = getattr(settings, "EMBEDDING_BACKEND", "unknown")
    model = (
        getattr(settings, "OPENAI_EMBED_MODEL", "")
        if backend == "openai"
        else getattr(settings, "LOCAL_EMBED_MODEL", "")
    )
    size = getattr(settings, "QDRANT_VECTOR_SIZE", "")
    return f"{backend}:{model}:{size}"


def _indexed_chunk_surface(doc_title: str, section_title: str, chunk_body: str) -> str:
    """Prefix document + section context for embedding and citation alignment."""
    d = (doc_title or "").strip()
    s = (section_title or "").strip()
    b = (chunk_body or "").strip()
    bits = [f"Policy title: {d}"] if d else []
    if s:
        bits.append(f"Section: {s}")
    if bits:
        return ("; ".join(bits) + "\n\n" + b).strip()
    return b


def read_policy_file(
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    max_chars: int | None = None,
) -> DocumentExtractResult:
    """TXT/Markdown inline decode; PDF/images via ``document_reader``."""
    cap = max_chars or int(getattr(settings, "KB_MAX_EXTRACT_CHARS", 200_000))
    name = (filename or "").lower()
    if name.endswith((".md", ".markdown", ".txt")):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        text = sanitize_for_indexing(text, max_chars=cap)
        return DocumentExtractResult(text=text, warnings=[], source="text_file")
    return extract_text_from_upload(
        filename=filename,
        content_type=content_type,
        data=data,
        max_chars=cap,
    )


def _document_bytes(document: KnowledgeDocument) -> tuple[bytes, str | None, str | None]:
    metadata = document.metadata or {}
    if document.file:
        document.file.open("rb")
        try:
            return (
                document.file.read(),
                Path(document.file.name).name,
                str(metadata.get("content_type") or "") or None,
            )
        finally:
            document.file.close()
    if document.source_path:
        path = Path(document.source_path)
        return path.read_bytes(), path.name, "application/octet-stream"
    raise ValueError("document_has_no_file")


def _mark_failed(
    document: KnowledgeDocument,
    *,
    error: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document.status = DocumentStatus.FAILED
    document.total_chunks = 0
    if metadata is not None:
        document.metadata = metadata
    document.save(update_fields=["status", "total_chunks", "metadata"])
    return {
        "document_id": str(document.pk),
        "chunks_created": 0,
        "status": "failed",
        "error": error,
    }


def process_document(
    document: KnowledgeDocument,
    *,
    trace_id: str = "",
    reindex: bool = False,
) -> dict[str, Any]:
    """
    Synchronously index one KnowledgeDocument.

    This is the only ingestion pipeline used by Admin, API uploads, and the
    legacy management command wrapper.
    """
    t0 = time.perf_counter()
    company_id = (document.company_id or "").strip()
    if not company_id:
        raise ValueError("company_id_required")
    if not (document.uploaded_by_employee_id or "").strip():
        raise ValueError("uploaded_by_employee_id_required")

    data, filename, content_type = _document_bytes(document)
    max_upload = int(getattr(settings, "KB_MAX_UPLOAD_BYTES", 26_214_400))
    if len(data) > max_upload:
        raise ValueError("upload_too_large")

    chk = _checksum(data)
    metadata = dict(document.metadata or {})
    metadata.update({"filename": filename or "", "content_type": content_type or ""})

    existing = (
        KnowledgeDocument.objects.filter(
            company_id=company_id,
            checksum=chk,
            status=DocumentStatus.INDEXED,
        )
        .exclude(pk=document.pk)
        .first()
    )
    if existing and not reindex:
        document.checksum = chk
        document.status = DocumentStatus.INDEXED
        document.total_chunks = existing.total_chunks
        document.metadata = {
            **metadata,
            "deduped_from_document_id": existing.pk,
            "embedding_version": _embedding_version(),
        }
        document.save(update_fields=["checksum", "status", "total_chunks", "metadata"])
        log_step(
            trace_id,
            "kb_ingest_deduped",
            {"document_id": document.pk, "existing_document_id": existing.pk, "company_id": company_id},
        )
        return {
            "document_id": str(existing.pk),
            "chunks_created": existing.total_chunks,
            "status": "deduped",
            "checksum": chk,
        }

    document.checksum = chk
    document.status = DocumentStatus.PROCESSING
    document.total_chunks = 0
    document.metadata = metadata
    document.save(update_fields=["checksum", "status", "total_chunks", "metadata"])

    extracted = read_policy_file(data=data, filename=filename, content_type=content_type)
    text = sanitize_for_indexing(extracted.text or "")
    metadata.update({"warnings": extracted.warnings, "source": extracted.source})
    if not text.strip():
        return _mark_failed(document, error="empty_extract", metadata=metadata)

    chunks = chunk_policy_text(
        text,
        target_tokens=int(getattr(settings, "KB_CHUNK_TARGET_TOKENS", 500)),
        overlap_tokens=int(getattr(settings, "KB_CHUNK_OVERLAP_TOKENS", 100)),
    )
    if not chunks:
        return _mark_failed(document, error="no_chunks", metadata=metadata)

    surface_texts = [
        _indexed_chunk_surface(document.title, ch.section_title, ch.text) for ch in chunks
    ]

    embed_batches: list[list[float]] = []
    llm = LLMClient()
    batch = int(getattr(settings, "EMBED_BATCH_SIZE", 64))
    for i in range(0, len(surface_texts), batch):
        part = surface_texts[i : i + batch]
        t_emb = time.perf_counter()
        vecs = llm.embed_texts(part, trace_id) if llm.is_embedding_configured() else None
        emb_ms = int((time.perf_counter() - t_emb) * 1000)
        ok = bool(vecs) and len(vecs or []) == len(part)
        log_step(
            trace_id,
            "kb_ingest_embed_batch",
            {"offset": i, "size": len(part), "ms": emb_ms, "ok": ok, "company_id": company_id},
        )
        if not ok:
            log_step(trace_id, "embedding_failure", {"document_id": document.pk, "company_id": company_id})
            return _mark_failed(document, error="embedding_failed", metadata=metadata)
        embed_batches.extend(vecs or [])

    KnowledgeChunk.objects.filter(document=document).delete()
    delete_by_document_id(document.pk, company_id=company_id, trace_id=trace_id)

    upload_ts = timezone.now().isoformat()
    points: list[PointStruct] = []
    lang = detect_user_language(text[:500])
    dept = str(metadata.get("department") or "")
    policy_type = str(metadata.get("policy_type") or document.document_type)
    embedding_version = _embedding_version()

    for i, ch in enumerate(chunks):
        pid = str(uuid4())
        surface = surface_texts[i]
        payload = {
            "company_id": company_id,
            "uploaded_by_employee_id": document.uploaded_by_employee_id,
            "chunk_text": surface[:8000],
            "document_title": document.title,
            "source_document": document.title,
            "section_title": ch.section_title or "",
            "chunk_index": i,
            "policy_type": policy_type,
            "department": dept,
            "language": lang,
            "upload_timestamp": upload_ts,
            "created_at": upload_ts,
            "document_db_id": document.pk,
            "document_checksum": chk,
            "embedding_version": embedding_version,
        }
        points.append(PointStruct(id=pid, vector=embed_batches[i], payload=payload))

    KnowledgeChunk.objects.bulk_create(
        [
            KnowledgeChunk(
                company_id=company_id,
                document=document,
                chunk_index=i,
                chunk_text=surface_texts[i],
                token_count=count_tokens(surface_texts[i]),
                qdrant_point_id=str(points[i].id),
                language=lang,
                metadata={
                    "section_title": ch.section_title,
                    "policy_type": policy_type,
                    "embedding_version": embedding_version,
                },
            )
            for i, ch in enumerate(chunks)
        ]
    )

    try:
        upsert_points(points, trace_id=trace_id)
    except Exception:
        logger.exception("kb_ingest_qdrant_failed trace_id=%s doc_id=%s", trace_id, document.pk)
        KnowledgeChunk.objects.filter(document=document).delete()
        return _mark_failed(document, error="qdrant_upsert_failed", metadata=metadata)

    document.total_chunks = len(chunks)
    document.status = DocumentStatus.INDEXED
    document.metadata = {**metadata, "embedding_version": embedding_version}
    document.save(update_fields=["total_chunks", "status", "metadata"])

    total_ms = int((time.perf_counter() - t0) * 1000)
    log_step(
        trace_id,
        "kb_ingest_done",
        {"document_id": document.pk, "chunks": len(chunks), "ms": total_ms, "company_id": company_id},
    )
    return {
        "document_id": str(document.pk),
        "chunks_created": len(chunks),
        "status": "indexed",
        "checksum": chk,
        "ingestion_ms": total_ms,
    }
