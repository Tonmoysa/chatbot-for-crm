"""Backward-compatible wrappers around the unified ingestion service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from django.core.files.base import ContentFile

from knowledge_base.models import DocumentType, KnowledgeDocument
from knowledge_base.services.ingestion_service import (
    _checksum,
    _indexed_chunk_surface,
    process_document,
    read_policy_file,
)


def ingest_bytes(
    *,
    data: bytes,
    title: str,
    filename: str | None,
    content_type: str | None,
    document_type: str = DocumentType.POLICY,
    source_path: str = "",
    uploaded_by_id: int | None = None,
    trace_id: str,
    metadata: dict[str, Any] | None = None,
    reindex: bool = False,
    company_id: str = "",
    uploaded_by_employee_id: str = "",
) -> dict[str, Any]:
    """
    Legacy entry point retained for tests/commands.

    New API and Admin code create KnowledgeDocument directly and call
    ingestion_service.process_document(document).
    """
    meta = dict(metadata or {})
    if content_type:
        meta["content_type"] = content_type
    if uploaded_by_id is not None:
        meta["legacy_uploaded_by_id"] = uploaded_by_id
    doc = KnowledgeDocument.objects.create(
        company_id=company_id,
        title=title,
        source_path=source_path,
        checksum=_checksum(data),
        document_type=document_type,
        uploaded_by_employee_id=uploaded_by_employee_id,
        metadata=meta,
    )
    if data:
        doc.file.save(filename or "policy.txt", ContentFile(data), save=True)
    return process_document(doc, trace_id=trace_id, reindex=reindex)


def ingest_path(
    path: Path,
    *,
    trace_id: str,
    reindex: bool,
    metadata: dict[str, Any] | None,
    uploaded_by_id: int | None = None,
    company_id: str = "",
    uploaded_by_employee_id: str = "",
) -> dict[str, Any]:
    data = path.read_bytes()
    return ingest_bytes(
        data=data,
        title=path.stem,
        filename=path.name,
        content_type="application/octet-stream",
        source_path=str(path.resolve()),
        trace_id=trace_id,
        reindex=reindex,
        metadata={**(metadata or {}), "path": str(path)},
        uploaded_by_id=uploaded_by_id,
        company_id=company_id,
        uploaded_by_employee_id=uploaded_by_employee_id,
    )
