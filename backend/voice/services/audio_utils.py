from __future__ import annotations

import io
import wave


def estimate_duration_seconds(data: bytes, content_type: str | None) -> float | None:
    """Best-effort duration for WAV; unknown for webm/mp4 without ffprobe."""
    ctype = (content_type or "").lower()
    if "wav" in ctype or data[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                rate = wf.getframerate() or 1
                return wf.getnframes() / float(rate)
        except Exception:
            return None
    return None
