"""Reference / policy entity extraction (Phase 11 SSOT)."""

from __future__ import annotations

import re
from typing import Any

_REF_RE = re.compile(
    r"\b(?:ref|reference|request\s*id|rid)\s*[:#-]?\s*([A-Za-z0-9-]+)\b",
    re.I,
)
_BARE_REF_RE = re.compile(r"\b([A-Z]{2,}-\d{4,}[A-Z0-9-]*)\b")


def extract_reference_entities(message: str) -> dict[str, Any]:
    raw = message or ""
    low = raw.lower()
    out: dict[str, Any] = {}

    match = _REF_RE.search(raw) or _BARE_REF_RE.search(raw)
    if match:
        out["request_id"] = match.group(1)

    topic_match = re.search(
        r"\b(?:about|regarding|on|for)\s+(?:the\s+)?(.{3,80}?)(?:\?|$)",
        low,
    )
    if topic_match:
        out["policy_topic"] = topic_match.group(1).strip(" .")

    return out
