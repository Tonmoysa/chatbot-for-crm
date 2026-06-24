"""Deprecated shim — use chat.services.reference_extractors (Phase 11)."""

from __future__ import annotations

import warnings
from typing import Any

from chat.services.reference_extractors import extract_reference_entities

__all__ = ["EntityExtractor"]


class EntityExtractor:
    """Deprecated. Use extract_reference_entities() directly."""

    def extract_rules_only(self, message: str, *, intent: str = "") -> dict[str, Any]:
        warnings.warn(
            "EntityExtractor is deprecated; use reference_extractors.extract_reference_entities.",
            DeprecationWarning,
            stacklevel=2,
        )
        return extract_reference_entities(message)

    def extract(
        self,
        message: str,
        intent: str,
        context_lines: list[str] | None,
        trace_id: str = "",
    ) -> dict[str, Any]:
        warnings.warn(
            "EntityExtractor is deprecated; use reference_extractors.extract_reference_entities.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {"entities": extract_reference_entities(message), "source": "rules"}
