"""Deterministic amount extraction — Universal Field Engine."""

from __future__ import annotations

import re

_AMOUNT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:taka|tk|taka\.|টাকা|tk\.|bdt)?",
    re.I | re.UNICODE,
)


def parse_amount(message: str) -> float | None:
    m = _AMOUNT_RE.search(message or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None
