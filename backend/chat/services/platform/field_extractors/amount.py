"""Deterministic amount extraction — Universal Field Engine."""

from __future__ import annotations

import re

_AMOUNT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:taka|tk|taka\.|টাকা|tk\.|bdt)?",
    re.I | re.UNICODE,
)


def parse_amount(message: str) -> float | None:
    """Extract taka amount — skip entry numbers in delete/modify messages (e.g. '3 number bus delete')."""
    raw = message or ""
    from chat.services.platform.field_extractors.modify import _numbered_item_index
    from chat.services.platform.intent_rules import is_delete_request, is_modify_request

    item_num: int | None = None
    if is_delete_request(raw) or is_modify_request(raw):
        numbered = _numbered_item_index(raw, item_count=99)
        if numbered is not None:
            item_num = numbered + 1

    for m in _AMOUNT_RE.finditer(raw):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if item_num is not None and abs(val - item_num) < 0.01:
            continue
        return val

    if item_num is not None:
        nums: list[float] = []
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", raw):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if abs(val - item_num) >= 0.01:
                nums.append(val)
        if len(nums) == 1:
            return nums[0]
        if nums:
            substantial = [n for n in nums if n >= 10]
            if len(substantial) == 1:
                return substantial[0]
    return None
