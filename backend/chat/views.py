import uuid

from django.db import connection
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from chat.serializers import (
    ChatRequestSerializer,
    ChatSessionDetailQuerySerializer,
    ChatSessionsQuerySerializer,
    DecisionRequestSerializer,
    DocumentExtractRequestSerializer,
    ExtractRequestSerializer,
    HrEnvelopeSerializer,
    IntentRequestSerializer,
    MockCreateSerializer,
)
from chat.services.session_history import get_session_messages, list_sessions
from chat.identity import identity_from_request, identity_from_validated_data
from chat.services.crm.factory import get_crm_adapter
from chat.services.document_reader import extract_text_from_upload
from chat.services.llm_client import LLMClient
from chat.services.memory_store import ConversationMemoryStore
from chat.services.observability import log_step
from chat.services.orchestrator import ChatOrchestrator
from django.conf import settings
from knowledge_base.services.qdrant_service import collection_name, get_qdrant_client


def _trace(request) -> str:
    return getattr(request, "trace_id", None) or str(uuid.uuid4())


def _legacy_debug_disabled_response(tid: str, endpoint: str) -> Response:
    """Phase 11 — deprecated debug endpoints."""
    return Response(
        {
            "trace_id": tid,
            "intent": "",
            "entities": {},
            "decision": {"outcome": "DEPRECATED", "reason": f"{endpoint} removed in Phase 11."},
            "response": {
                "message": (
                    f"The `{endpoint}` debug endpoint is deprecated. "
                    "Use POST /chat/ for the full Decision Core pipeline."
                ),
                "status": "deprecated",
                "request_id": "",
            },
            "status": "deprecated",
        },
        status=status.HTTP_410_GONE,
    )


def _legacy_debug_enabled() -> bool:
    return getattr(settings, "ENABLE_LEGACY_DEBUG_ENDPOINTS", False)


@extend_schema(
    summary="Service health",
    tags=["Health"],
    responses={200: HrEnvelopeSerializer},
    auth=[],
)
class HealthView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        tid = _trace(request)
        checks = {
            "db": self._check_db(),
            "qdrant": self._check_qdrant(),
            "embedding_service": self._check_embedding_service(),
            "crm": get_crm_adapter().health(),
        }
        ok = all(bool(v.get("ok")) for v in checks.values())
        return Response(
            {
                "trace_id": tid,
                "intent": "",
                "entities": {},
                "decision": {},
                "response": {
                    "message": "HR chatbot microservice health check complete.",
                    "status": "success" if ok else "degraded",
                    "request_id": "",
                    "checks": checks,
                },
                "status": "success" if ok else "degraded",
            }
        )

    def _check_db(self) -> dict:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}

    def _check_qdrant(self) -> dict:
        try:
            client = get_qdrant_client()
            name = collection_name()
            if hasattr(client, "collection_exists"):
                exists = bool(client.collection_exists(name))
            else:
                cols = client.get_collections().collections
                exists = any(c.name == name for c in cols)
            return {"ok": True, "collection": name, "exists": exists}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}

    def _check_embedding_service(self) -> dict:
        try:
            client = LLMClient()
            configured = bool(client.is_embedding_configured())
            return {"ok": configured, "configured": configured}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}


@extend_schema(
    summary="Chat (full pipeline)",
    tags=["Chat"],
    request=ChatRequestSerializer,
    responses={200: HrEnvelopeSerializer},
)
@extend_schema(
    summary="List recent chat sessions",
    tags=["Chat"],
    parameters=[
        OpenApiParameter("company_id", OpenApiTypes.STR, OpenApiParameter.QUERY, required=True),
        OpenApiParameter("employee_id", OpenApiTypes.STR, OpenApiParameter.QUERY, required=True),
        OpenApiParameter("limit", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False),
    ],
)
class ChatSessionsListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        ser = ChatSessionsQuerySerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)
        company_id = ser.validated_data["company_id"].strip()
        employee_id = ser.validated_data["employee_id"].strip()
        sessions = list_sessions(
            company_id=company_id,
            employee_id=employee_id,
            limit=ser.validated_data.get("limit") or 30,
        )
        return Response(
            {
                "trace_id": _trace(request),
                "status": "success",
                "sessions": sessions,
            }
        )


@extend_schema(
    summary="Load messages for a chat session",
    tags=["Chat"],
    parameters=[
        OpenApiParameter("company_id", OpenApiTypes.STR, OpenApiParameter.QUERY, required=True),
        OpenApiParameter("employee_id", OpenApiTypes.STR, OpenApiParameter.QUERY, required=True),
        OpenApiParameter("session_id", OpenApiTypes.STR, OpenApiParameter.QUERY, required=True),
    ],
)
class ChatSessionDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id: str):
        ser = ChatSessionDetailQuerySerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)
        identity = identity_from_validated_data(
            {
                **ser.validated_data,
                "session_id": session_id,
            }
        )
        messages = get_session_messages(
            company_id=identity.company_id,
            employee_id=identity.employee_id,
            session_id=identity.session_id,
        )
        if messages is None:
            return Response(
                {
                    "trace_id": _trace(request),
                    "status": "not_found",
                    "session_id": identity.session_id,
                    "messages": [],
                },
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(
            {
                "trace_id": _trace(request),
                "status": "success",
                "session_id": identity.session_id,
                "messages": messages,
            }
        )


class ChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = ChatRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity = identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        orch = ChatOrchestrator()
        out = orch.run_chat(
            message=ser.validated_data["message"],
            session_id=identity.session_id,
            company_id=identity.company_id,
            employee_id=identity.employee_id,
            trace_id=tid,
            document_text=(ser.validated_data.get("document_text") or "").strip() or None,
            idempotency_key=identity.idempotency_key,
        )
        sid = out.pop("_session_id", None)
        resp = Response(out, status=status.HTTP_200_OK)
        if sid:
            resp["X-Session-Id"] = sid
        return resp


@extend_schema(
    summary="Extract text from uploaded document (receipt/invoice)",
    tags=["Documents"],
    request=DocumentExtractRequestSerializer,
    responses={200: HrEnvelopeSerializer},
)
class DocumentExtractView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = DocumentExtractRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        f = ser.validated_data["file"]
        data = f.read()
        res = extract_text_from_upload(
            filename=getattr(f, "name", None),
            content_type=getattr(f, "content_type", None),
            data=data,
        )
        msg = "Document processed."
        if res.warnings:
            msg += " Warnings: " + "; ".join(res.warnings)
        return Response(
            {
                "trace_id": tid,
                "intent": "DOCUMENT_EXTRACT",
                "entities": {"source": res.source, "warnings": res.warnings},
                "decision": {"outcome": "INFORMATIONAL"},
                "response": {
                    "message": msg,
                    "status": "success",
                    "request_id": "",
                },
                "status": "success",
                "document_text": res.text,
            },
            status=status.HTTP_200_OK,
        )


@extend_schema(
    summary="[Deprecated] Intent detection only",
    tags=["Chat (deprecated)"],
    request=IntentRequestSerializer,
    responses={410: HrEnvelopeSerializer, 200: HrEnvelopeSerializer},
    deprecated=True,
)
class IntentView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = IntentRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        if not _legacy_debug_enabled():
            return _legacy_debug_disabled_response(tid, "intent")
        from chat.services.intent_detector import IntentDetector

        det = IntentDetector()
        r = det.detect(ser.validated_data["message"], tid)
        log_step(tid, "intent_only", {"intent": r.get("intent"), "deprecated": True})
        return Response(
            {
                "trace_id": tid,
                "intent": r.get("intent", ""),
                "entities": {"confidence": r.get("confidence"), "source": r.get("source")},
                "decision": {"outcome": "DEPRECATED"},
                "response": {
                    "message": "Intent detection complete (deprecated endpoint).",
                    "status": "success",
                    "request_id": "",
                },
                "status": "success",
            }
        )


@extend_schema(
    summary="[Deprecated] Entity extraction only",
    tags=["Chat (deprecated)"],
    request=ExtractRequestSerializer,
    responses={410: HrEnvelopeSerializer, 200: HrEnvelopeSerializer},
    deprecated=True,
)
class ExtractView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = ExtractRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity = identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        if not _legacy_debug_enabled():
            return _legacy_debug_disabled_response(tid, "extract")
        from chat.services.entity_extractor import EntityExtractor

        mem = ConversationMemoryStore()
        session = mem.get_or_create_session(
            company_id=identity.company_id,
            employee_id=identity.employee_id,
            session_id=identity.session_id,
        )
        ctx = mem.recent_context_lines(session)
        ext = EntityExtractor()
        r = ext.extract(
            ser.validated_data["message"],
            ser.validated_data["intent"],
            ctx,
            tid,
        )
        log_step(tid, "extract_only", {"deprecated": True})
        return Response(
            {
                "trace_id": tid,
                "intent": ser.validated_data["intent"],
                "entities": r.get("entities") or {},
                "decision": {"source": r.get("source"), "outcome": "DEPRECATED"},
                "response": {
                    "message": "Entity extraction complete (deprecated endpoint).",
                    "status": "success",
                    "request_id": "",
                },
                "status": "success",
            }
        )


@extend_schema(
    summary="[Deprecated] Decision engine only",
    tags=["Chat (deprecated)"],
    request=DecisionRequestSerializer,
    responses={410: HrEnvelopeSerializer, 200: HrEnvelopeSerializer},
    deprecated=True,
)
class DecisionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = DecisionRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        if not _legacy_debug_enabled():
            return _legacy_debug_disabled_response(tid, "decision")
        from chat.services.decision_engine import DecisionEngine

        crm_context: dict = {}
        intent = ser.validated_data["intent"]
        entities = ser.validated_data["entities"]
        eng = DecisionEngine()
        decision = eng.evaluate(
            intent=intent, entities=entities, crm_context=crm_context
        )
        log_step(tid, "decision_only", {"outcome": decision.get("outcome"), "deprecated": True})
        return Response(
            {
                "trace_id": tid,
                "intent": intent,
                "entities": entities,
                "decision": decision,
                "response": {
                    "message": "Decision evaluation complete (deprecated endpoint).",
                    "status": "success",
                    "request_id": "",
                },
                "status": "success",
            }
        )


@extend_schema(
    summary="Request status by id",
    tags=["CRM"],
    responses={200: HrEnvelopeSerializer},
)
class RequestStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, request_id: str):
        tid = _trace(request)
        identity = identity_from_request(request)
        crm = get_crm_adapter()
        st = crm.get_request_status(
            request_id,
            company_id=identity.company_id,
            employee_id=identity.employee_id,
            session_id=identity.session_id,
        )
        return Response(
            {
                "trace_id": tid,
                "intent": "REQUEST_STATUS",
                "entities": {"request_id": request_id},
                "decision": {"outcome": "INFORMATIONAL", "reason": "Lookup"},
                "response": {
                    "message": f"Status: {st.get('status', 'unknown')}",
                    "status": "success",
                    "request_id": request_id,
                },
                "status": "success",
            }
        )


@extend_schema(
    summary="Mock: create HR request",
    tags=["Mock CRM"],
    request=MockCreateSerializer,
    responses={200: HrEnvelopeSerializer},
)
class MockRequestCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = MockCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity = identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        crm = get_crm_adapter()
        r = crm.create_request(
            company_id=identity.company_id,
            employee_id=identity.employee_id,
            session_id=identity.session_id,
            intent=ser.validated_data["intent"],
            entities=ser.validated_data.get("entities") or {},
            decision=ser.validated_data.get("decision") or {},
            idempotency_key=identity.idempotency_key,
        )
        return Response(
            {
                "trace_id": tid,
                "intent": ser.validated_data["intent"],
                "entities": ser.validated_data.get("entities") or {},
                "decision": ser.validated_data.get("decision") or {},
                "response": {
                    "message": "Mock request created.",
                    "status": "success",
                    "request_id": str(r.get("request_id", "")),
                },
                "status": "success",
            }
        )


@extend_schema(
    summary="Mock: request status",
    tags=["Mock CRM"],
    parameters=[
        OpenApiParameter(
            name="request_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
        ),
    ],
    responses={200: HrEnvelopeSerializer},
)
class MockRequestStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        tid = _trace(request)
        identity = identity_from_request(request)
        rid = request.query_params.get("request_id", "")
        crm = get_crm_adapter()
        st = (
            crm.get_request_status(
                rid,
                company_id=identity.company_id,
                employee_id=identity.employee_id,
                session_id=identity.session_id,
            )
            if rid
            else {"status": "MISSING_ID"}
        )
        return Response(
            {
                "trace_id": tid,
                "intent": "REQUEST_STATUS",
                "entities": {"request_id": rid},
                "decision": {},
                "response": {
                    "message": str(st),
                    "status": "success",
                    "request_id": rid,
                },
                "status": "success",
            }
        )
