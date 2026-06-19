"""Deterministic modify-request parsing — Modification Engine."""

from __future__ import annotations

import re
from typing import Any

_MODIFY_AMOUNT_VAGUE_RE = re.compile(
    r"(?:amount|taka|tk)\s*(?:ta|টা)?\s*(?:\d+\s*)?(?:kore|kor|dao|de|den|koro)",
    re.I | re.UNICODE,
)

_MODIFY_LUNCH_RE = re.compile(
    r"\b(lunch|lanch|dinner|nasta|nasto|nosto|bus|travel)\b.{0,30}"
    r"(?:amount|er\s*amount|er\s*taka|ta)\s*(?:\d+\s*)?(?:kore|kor|koro|dao)",
    re.I | re.UNICODE,
)

_MODIFY_ORDINAL_RE = re.compile(
    r"\b(prothom|first|1st|pratham|প্রথম)\s*(?:ta|টা)?\b.{0,30}"
    r"(\d+(?:\.\d+)?)\s*(?:taka|tk|tk\.|টাকা)?",
    re.I | re.UNICODE,
)


def is_vague_amount_modify(message: str) -> bool:
    low = (message or "").lower()
    if _MODIFY_LUNCH_RE.search(low) or _MODIFY_ORDINAL_RE.search(message or ""):
        return False
    return bool(_MODIFY_AMOUNT_VAGUE_RE.search(low) or re.search(r"amount\s*ta\s*\d+\s*kore", low))


def parse_modify_request(message: str, items: list[dict]) -> dict[str, Any] | None:
    """Return {item_index, amount, label} or None if vague."""
    raw = message or ""
    low = raw.lower()

    ord_m = _MODIFY_ORDINAL_RE.search(raw)
    if ord_m:
        return {"item_index": 0, "amount": float(ord_m.group(2)), "label": "1st item", "needs_confirm": True}

    amount_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:taka|tk|টাকা)?", low)
    if not amount_m:
        return None
    amount = float(amount_m.group(1))

    for idx, item in enumerate(items):
        desc = str(item.get("description") or "").lower()
        cat = str(item.get("category") or "").lower()
        if re.search(r"\blunch\b", low) and ("lunch" in desc or cat == "meals"):
            return {"item_index": idx, "amount": amount, "label": "lunch"}
        if re.search(r"\bnasta|nasto|nosto|snack\b", low) and ("nasta" in desc or "snack" in desc):
            return {"item_index": idx, "amount": amount, "label": "snack"}
        if re.search(r"\bbus\b", low) and (cat == "travel" or "bus" in desc):
            return {"item_index": idx, "amount": amount, "label": "bus/travel"}

    if is_vague_amount_modify(raw):
        return None
    if len(items) == 1:
        return {"item_index": 0, "amount": amount, "label": "item"}
    return None
