from __future__ import annotations

from django.conf import settings

ALLOWED_AUDIO_MIME = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "application/octet-stream",
    }
)

ALLOWED_EXTENSIONS = frozenset({".webm", ".ogg", ".mp3", ".mp4", ".m4a", ".wav", ".mpeg"})


def validate_audio_upload(
    *,
    filename: str | None,
    content_type: str | None,
    size: int,
) -> list[str]:
    errors: list[str] = []
    max_bytes = int(getattr(settings, "VOICE_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    if size <= 0:
        errors.append("Empty audio file.")
    elif size > max_bytes:
        errors.append(f"Audio file exceeds maximum size ({max_bytes} bytes).")

    name = (filename or "").lower()
    ext_ok = any(name.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    ctype = (content_type or "").split(";")[0].strip().lower()
    mime_ok = not ctype or ctype in ALLOWED_AUDIO_MIME
    if not ext_ok and not mime_ok:
        errors.append("Unsupported audio format. Use webm, ogg, mp3, mp4, or wav.")

    return errors
