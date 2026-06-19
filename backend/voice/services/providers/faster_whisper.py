from __future__ import annotations

from voice.services.providers.base import BaseTranscriptionProvider, TranscribeResult


class FasterWhisperProvider(BaseTranscriptionProvider):
    """Optional local STT — not enabled in Phase 1."""

    name = "faster_whisper"

    def transcribe(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        data: bytes,
        language: str | None,
        trace_id: str,
    ) -> TranscribeResult:
        raise NotImplementedError(
            "faster-whisper local STT is not configured. Set VOICE_STT_PROVIDER=openai_whisper."
        )
