"""Keep Qdrant vectors in sync when ORM rows are removed."""

from __future__ import annotations

import logging
import uuid

from django.db.models.signals import pre_delete
from django.dispatch import receiver

from knowledge_base.models import KnowledgeChunk, KnowledgeDocument
from knowledge_base.services.qdrant_service import delete_by_document_id, delete_points_by_ids

logger = logging.getLogger("hr_chatbot")


@receiver(pre_delete, sender=KnowledgeDocument)
def purge_document_vectors(sender, instance: KnowledgeDocument, **kwargs) -> None:
    """Remove tenant-scoped vectors when a document is deleted (admin, API, ORM)."""
    company_id = (instance.company_id or "").strip()
    if not company_id or not instance.pk:
        return
    trace_id = f"kb-delete-doc-{uuid.uuid4().hex[:12]}"
    try:
        delete_by_document_id(instance.pk, company_id=company_id, trace_id=trace_id)
        logger.info(
            "kb_document_qdrant_purged document_id=%s company_id=%s trace_id=%s",
            instance.pk,
            company_id,
            trace_id,
        )
    except Exception:
        logger.exception(
            "kb_document_qdrant_purge_failed document_id=%s company_id=%s trace_id=%s",
            instance.pk,
            company_id,
            trace_id,
        )


@receiver(pre_delete, sender=KnowledgeChunk)
def purge_chunk_vector(sender, instance: KnowledgeChunk, **kwargs) -> None:
    """Remove a single Qdrant point when a chunk row is deleted without deleting the document."""
    point_id = (instance.qdrant_point_id or "").strip()
    if not point_id:
        return
    trace_id = f"kb-delete-chunk-{uuid.uuid4().hex[:12]}"
    try:
        delete_points_by_ids([point_id], trace_id=trace_id)
    except Exception:
        logger.exception(
            "kb_chunk_qdrant_purge_failed chunk_id=%s point_id=%s trace_id=%s",
            instance.pk,
            point_id,
            trace_id,
        )
