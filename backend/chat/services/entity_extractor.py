import re
from typing import Any


_REF_RE = re.compile(
    r"\b(?:ref|reference|request\s*id|rid)\s*[:#-]?\s*([A-Za-z0-9-]+)\b",
    re.I,
)
_BARE_REF_RE = re.compile(r"\b([A-Z]{2,}-\d{4,}[A-Z0-9-]*)\b")


class EntityExtractor:
    def extract_rules_only(self, message: str, *, intent: str = "") -> dict[str, Any]:
        return self._extract_rules(message)

    def extract(
        self,
        message: str,
        intent: str,
        context_lines: list[str] | None,
        trace_id: str = "",
    ) -> dict[str, Any]:
        entities = self._extract_rules(message)
        return {"entities": entities, "source": "rules"}

    def _extract_rules(self, message: str) -> dict[str, Any]:
        raw = message or ""
        low = raw.lower()
        out: dict[str, Any] = {}

        m = _REF_RE.search(raw) or _BARE_REF_RE.search(raw)
        if m:
            out["request_id"] = m.group(1)

        topic_match = re.search(
            r"\b(?:about|regarding|on|for)\s+(?:the\s+)?(.{3,80}?)(?:\?|$)",
            low,
        )
        if topic_match:
            out["policy_topic"] = topic_match.group(1).strip(" .")

        return out
