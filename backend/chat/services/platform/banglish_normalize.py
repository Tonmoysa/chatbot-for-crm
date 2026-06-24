"""Thin Banglish lexical normalization — not intent classification."""

from __future__ import annotations

import re

_VARIANTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bosusto\b", re.I), "osustho"),
    (re.compile(r"\bosustha\b", re.I), "osustho"),
    (re.compile(r"\bkalke\b", re.I), "kal"),
    (re.compile(r"\bagamikalke\b", re.I), "agamikal"),
    (re.compile(r"\bsummery\b", re.I), "summary"),
    (re.compile(r"\bsaransho\b", re.I), "summary"),
)


def normalize_banglish_message(message: str) -> str:
    """Normalize common spelling variants before extraction / LLM."""
    raw = (message or "").strip()
    if not raw:
        return raw
    out = raw
    for pattern, replacement in _VARIANTS:
        out = pattern.sub(replacement, out)
    return out
