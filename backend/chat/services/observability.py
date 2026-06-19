import json
import logging
import re
from typing import Any

from django.conf import settings

logger = logging.getLogger("hr_chatbot")

_REDACT = re.compile(
    r"(api[_-]?key|password|token|authorization)\s*[:=]\s*[^\s,}\"]+",
    re.I,
)


def _safe_message(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    t = _REDACT.sub(r"\1=<redacted>", text)
    if len(t) > max_len:
        t = t[:max_len] + "…"
    return t


def log_step(trace_id: str, step: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"step": step, **(extra or {})}
    if not settings.DEBUG:
        if "user_message" in payload:
            payload["user_message"] = _safe_message(str(payload["user_message"]))
    logger.info(
        "pipeline_step trace_id=%s %s",
        trace_id,
        json.dumps(payload, default=str),
        extra={"trace_id": trace_id},
    )
