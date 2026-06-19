"""Build API `sources` entries from Qdrant scored points."""

from __future__ import annotations

from typing import Any


def _payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    md = getattr(payload, "model_dump", None)
    if callable(md):
        dumped = md()
        if isinstance(dumped, dict):
            return dumped
    return {}


def snippets_from_payload(payload: dict[str, Any], *, max_len: int = 320) -> str:
    text = str(payload.get("chunk_text") or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def build_sources(hits: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for h in hits:
        payload = _payload_dict(getattr(h, "payload", None) or {})
        if not payload:
            continue
        doc = str(payload.get("document_title") or payload.get("source_document") or "")
        sec = str(payload.get("section_title") or "")
        idx = int(payload.get("chunk_index") or -1)
        key = (doc, sec, idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "document": doc,
                "section": sec,
                "snippet": snippets_from_payload(payload),
                "score": float(getattr(h, "score", 0.0) or 0.0),
            }
        )
    return out
