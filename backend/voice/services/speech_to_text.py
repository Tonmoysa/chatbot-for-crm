from __future__ import annotations

from django.conf import settings

from chat.services.observability import log_step
from voice.services.audio_utils import estimate_duration_seconds
from voice.services.providers.base import TranscribeResult
from voice.services.providers.faster_whisper import FasterWhisperProvider
from voice.services.providers.openai_whisper import OpenAIWhisperProvider
from voice.services.validators import validate_audio_upload


def get_transcription_provider():
    key = getattr(settings, "VOICE_STT_PROVIDER", "openai_whisper").strip().lower()
    if key == "faster_whisper":
        return FasterWhisperProvider()
    return OpenAIWhisperProvider()


def transcribe_audio(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    language: str | None = None,
    trace_id: str,
) -> TranscribeResult:
    errors = validate_audio_upload(
        filename=filename,
        content_type=content_type,
        size=len(data),
    )
    if errors:
        raise ValueError("; ".join(errors))

    duration = estimate_duration_seconds(data, content_type)
    max_duration = float(getattr(settings, "VOICE_MAX_DURATION_SECONDS", 120))
    if duration is not None and duration > max_duration:
        raise ValueError(f"Audio exceeds maximum duration ({max_duration:.0f}s).")

    log_step(
        trace_id,
        "voice_transcribe_start",
        {
            "filename": filename,
            "content_type": content_type,
            "bytes": len(data),
            "duration_sec": duration,
            "provider": getattr(settings, "VOICE_STT_PROVIDER", "openai_whisper"),
        },
    )

    provider = get_transcription_provider()
    result = provider.transcribe(
        filename=filename,
        content_type=content_type,
        data=data,
        language=language,
        trace_id=trace_id,
    )

    log_step(
        trace_id,
        "voice_transcribe_done",
        {"provider": result.provider, "chars": len(result.text)},
    )
    return result
