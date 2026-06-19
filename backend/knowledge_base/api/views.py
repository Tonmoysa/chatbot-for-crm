import uuid

from django.conf import settings
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from chat.identity import identity_from_validated_data
from knowledge_base.models import DocumentType, KnowledgeDocument
from knowledge_base.serializers import (
    KbPolicyUploadResponseSerializer,
    KbPolicyUploadSerializer,
)
from knowledge_base.services.ingestion_service import process_document


def _trace(request) -> str:
    return getattr(request, "trace_id", None) or str(uuid.uuid4())


@extend_schema(
    summary="Upload HR policy document for indexing",
    tags=["Knowledge base"],
    request=KbPolicyUploadSerializer,
    responses={200: KbPolicyUploadResponseSerializer},
)
class KbUploadPolicyView(APIView):
    def post(self, request):
        ser = KbPolicyUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity = identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        f = ser.validated_data["file"]
        max_b = int(getattr(settings, "KB_MAX_UPLOAD_BYTES", 26_214_400))
        if getattr(f, "size", 0) and int(f.size) > max_b:
            return Response(
                {"detail": "File too large."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        title = (ser.validated_data.get("title") or "").strip() or getattr(
            f, "name", "policy"
        ) or "policy"
        meta = {}
        if ser.validated_data.get("policy_type"):
            meta["policy_type"] = ser.validated_data["policy_type"].strip()
        if ser.validated_data.get("department"):
            meta["department"] = ser.validated_data["department"].strip()
        meta.update(
            {
                "session_id": identity.session_id,
                "idempotency_key": identity.idempotency_key,
                "content_type": getattr(f, "content_type", "") or "",
            }
        )
        if identity.idempotency_key:
            existing = (
                KnowledgeDocument.objects.filter(
                    company_id=identity.company_id,
                    metadata__idempotency_key=identity.idempotency_key,
                )
                .order_by("-uploaded_at")
                .first()
            )
            if existing:
                return Response(
                    {
                        "document_id": str(existing.pk),
                        "chunks_created": int(existing.total_chunks or 0),
                        "status": "idempotent_replay",
                    },
                    status=status.HTTP_200_OK,
                )
        doc = KnowledgeDocument.objects.create(
            company_id=identity.company_id,
            title=title,
            file=f,
            document_type=DocumentType.POLICY,
            uploaded_by_employee_id=identity.employee_id,
            metadata=meta,
        )
        try:
            result = process_document(doc, trace_id=tid, reindex=False)
        except ValueError as exc:
            if str(exc) == "upload_too_large":
                return Response(
                    {"detail": "File too large."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            raise
        out = {
            "document_id": result["document_id"],
            "chunks_created": int(result.get("chunks_created") or 0),
            "status": result.get("status") or "unknown",
        }
        return Response(out, status=status.HTTP_200_OK)
