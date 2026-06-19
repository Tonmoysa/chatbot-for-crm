import uuid

from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from chat.identity import identity_from_validated_data
from chat.serializers import HrEnvelopeSerializer
from chat.services.observability import log_step
from voice.serializers import VoiceTranscribeRequestSerializer
from voice.services.speech_to_text import transcribe_audio


def _trace(request) -> str:
    return getattr(request, "trace_id", None) or str(uuid.uuid4())


@extend_schema(
    summary="Transcribe audio (OpenAI Whisper)",
    tags=["Voice"],
    request=VoiceTranscribeRequestSerializer,
    responses={200: HrEnvelopeSerializer},
)
class VoiceTranscribeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = VoiceTranscribeRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        identity_from_validated_data(ser.validated_data)
        tid = _trace(request)
        f = ser.validated_data["file"]
        language = (ser.validated_data.get("language") or "").strip() or None
        data = f.read()

        try:
            result = transcribe_audio(
                filename=getattr(f, "name", None),
                content_type=getattr(f, "content_type", None),
                data=data,
                language=language,
                trace_id=tid,
            )
        except ValueError as exc:
            log_step(tid, "voice_transcribe_validation_error", {"error": str(exc)})
            return Response(
                {
                    "trace_id": tid,
                    "intent": "VOICE_TRANSCRIBE",
                    "entities": {},
                    "decision": {"outcome": "REJECTED", "reason": str(exc)},
                    "response": {
                        "message": str(exc),
                        "status": "error",
                        "request_id": "",
                    },
                    "status": "failed",
                    "transcript": "",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            log_step(tid, "voice_transcribe_error", {"error": str(exc)[:200]})
            return Response(
                {
                    "trace_id": tid,
                    "intent": "VOICE_TRANSCRIBE",
                    "entities": {},
                    "decision": {"outcome": "ERROR"},
                    "response": {
                        "message": "Transcription failed. Try again or type your message.",
                        "status": "error",
                        "request_id": "",
                    },
                    "status": "failed",
                    "transcript": "",
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        msg = "Transcription complete."
        if not result.text:
            msg = "No speech detected in the recording."

        return Response(
            {
                "trace_id": tid,
                "intent": "VOICE_TRANSCRIBE",
                "entities": {
                    "provider": result.provider,
                    "language": result.language or "",
                },
                "decision": {"outcome": "INFORMATIONAL"},
                "response": {
                    "message": msg,
                    "status": "success",
                    "request_id": "",
                },
                "status": "success",
                "transcript": result.text,
            }
        )
