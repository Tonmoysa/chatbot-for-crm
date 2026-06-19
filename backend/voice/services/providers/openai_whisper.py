from __future__ import annotations

import logging

import httpx
from django.conf import settings

from chat.services.observability import log_step
from voice.services.providers.base import BaseTranscriptionProvider, TranscribeResult

logger = logging.getLogger("hr_chatbot")


class OpenAIWhisperProvider(BaseTranscriptionProvider):
    name = "openai_whisper"

    def transcribe(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        data: bytes,
        language: str | None,
        trace_id: str,
    ) -> TranscribeResult:
        api_key = (getattr(settings, "OPENAI_WHISPER_API_KEY", None) or "").strip()
        if not api_key:
            raise ValueError("OPENAI_WHISPER_API_KEY is not configured.")

        base = getattr(settings, "OPENAI_WHISPER_API_BASE_URL", "https://api.openai.com/v1").rstrip(
            "/"
        )
        model = getattr(settings, "OPENAI_WHISPER_MODEL", "whisper-1")
        url = f"{base}/audio/transcriptions"

        files = {
            "file": (filename or "audio.webm", data, content_type or "application/octet-stream"),
        }
        form: dict[str, str] = {"model": model}
        if language:
            form["language"] = language[:16]

        timeout = float(getattr(settings, "OPENAI_WHISPER_TIMEOUT_SECONDS", 60))

        log_step(trace_id, "whisper_api_request", {"model": model, "bytes": len(data)})

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=form,
            )
            resp.raise_for_status()
            payload = resp.json()

        text = (payload.get("text") or "").strip()
        log_step(
            trace_id,
            "whisper_api_done",
            {"chars": len(text), "language": payload.get("language")},
        )
        return TranscribeResult(
            text=text,
            language=payload.get("language") or language,
            provider=self.name,
        )
