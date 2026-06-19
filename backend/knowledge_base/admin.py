import uuid

from django.contrib import admin
from django.contrib import messages

from knowledge_base.models import DocumentStatus, KnowledgeChunk, KnowledgeDocument
from knowledge_base.services.ingestion_service import process_document


class KnowledgeChunkInline(admin.TabularInline):
    model = KnowledgeChunk
    extra = 0
    readonly_fields = ("chunk_index", "token_count", "qdrant_point_id", "language", "created_at")
    fields = ("chunk_index", "token_count", "qdrant_point_id", "language", "created_at")


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(admin.ModelAdmin):
    list_display = (
        "company_id",
        "title",
        "document_type",
        "status",
        "total_chunks",
        "uploaded_at",
    )
    list_filter = ("company_id", "status", "document_type")
    search_fields = ("company_id", "title", "source_path", "checksum")
    readonly_fields = ("uploaded_at", "checksum", "total_chunks")
    inlines = [KnowledgeChunkInline]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if not obj.file or obj.status == DocumentStatus.INDEXED:
            return
        trace_id = f"admin-ingest-{uuid.uuid4().hex}"
        try:
            result = process_document(obj, trace_id=trace_id)
        except Exception as exc:
            self.message_user(
                request,
                f"Document saved, but ingestion failed: {type(exc).__name__}",
                level=messages.ERROR,
            )
            return
        self.message_user(
            request,
            f"Ingestion {result.get('status')}: {result.get('chunks_created', 0)} chunks.",
            level=messages.SUCCESS,
        )


@admin.register(KnowledgeChunk)
class KnowledgeChunkAdmin(admin.ModelAdmin):
    list_display = ("company_id", "document", "chunk_index", "token_count", "language", "created_at")
    list_filter = ("company_id", "language")
    search_fields = ("company_id", "chunk_text", "qdrant_point_id")
