from django.db import models


class DocumentStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    INDEXED = "indexed", "Indexed"
    FAILED = "failed", "Failed"


class DocumentType(models.TextChoices):
    POLICY = "policy", "Policy"
    HANDBOOK = "handbook", "Handbook"
    GENERAL = "general", "General"


class KnowledgeDocument(models.Model):
    company_id = models.CharField(max_length=64, db_index=True)
    title = models.CharField(max_length=512)
    file = models.FileField(upload_to="kb/policies/%Y/%m/", blank=True, null=True)
    source_path = models.CharField(max_length=1024, blank=True, default="")
    checksum = models.CharField(max_length=64, db_index=True, blank=True, default="")
    document_type = models.CharField(
        max_length=32,
        choices=DocumentType.choices,
        default=DocumentType.POLICY,
    )
    uploaded_by_employee_id = models.CharField(max_length=64)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=32,
        choices=DocumentStatus.choices,
        default=DocumentStatus.PENDING,
    )
    total_chunks = models.PositiveIntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ("-uploaded_at",)
        indexes = [
            models.Index(fields=("company_id", "checksum"), name="kb_doc_company_checksum_idx"),
            models.Index(fields=("company_id", "status"), name="kb_doc_company_status_idx"),
            models.Index(fields=("checksum",)),
            models.Index(fields=("status",)),
        ]

    def __str__(self) -> str:
        return f"{self.company_id}:{self.title} ({self.status})"


class KnowledgeChunk(models.Model):
    company_id = models.CharField(max_length=64, db_index=True)
    document = models.ForeignKey(
        KnowledgeDocument,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_index = models.PositiveIntegerField()
    chunk_text = models.TextField()
    token_count = models.PositiveIntegerField(default=0)
    qdrant_point_id = models.CharField(max_length=64, db_index=True)
    language = models.CharField(max_length=8, default="en")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("document", "chunk_index")
        constraints = [
            models.UniqueConstraint(
                fields=("document", "chunk_index"),
                name="kb_chunk_unique_doc_index",
            ),
        ]
        indexes = [
            models.Index(
                fields=("company_id", "document", "chunk_index"),
                name="kb_chunk_company_doc_idx",
            ),
            models.Index(fields=("document", "chunk_index")),
        ]

    def __str__(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"
