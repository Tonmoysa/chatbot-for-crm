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

_NUMBERED_ITEM_RE = re.compile(
    r"(?:"
    r"\bexpense\s+(\d+)\b|"
    r"\b(\d+)\s*(?:no|number|nombor|numer|নম্বর)\b|"
    r"\b(?:no|number|numer)\s+(\d+)\b"
    r")",
    re.I | re.UNICODE,
)

_ROUTE_MODIFY_RE = re.compile(
    r"\broute\b",
    re.I | re.UNICODE,
)

_TAKA_AMOUNT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:taka|tk|টাকা)\b",
    re.I | re.UNICODE,
)

_LAST_REF_RE = re.compile(
    r"\b(last|sesh|shesh|sesher|শেষ)\b",
    re.I | re.UNICODE,
)

_FIRST_REF_RE = re.compile(
    r"\b(first|prothom|1st|pratham|প্রথম)\b",
    re.I | re.UNICODE,
)


def build_which_item_clarify(
    *,
    operation: str = "update",
    candidate_indices: list[int],
    target_field: str = "amount",
    pending_patch: dict[str, Any] | None = None,
    proposed_value: Any = None,
    category: str | None = None,
    match_amount: Any = None,
) -> dict[str, Any]:
    clarify: dict[str, Any] = {
        "kind": "which_item",
        "candidate_indices": list(candidate_indices),
        "target_field": target_field,
        "operation": operation,
    }
    if category:
        clarify["category"] = category
    if proposed_value is not None:
        clarify["proposed_value"] = proposed_value
    if match_amount is not None:
        clarify["match_amount"] = match_amount
    if pending_patch:
        clarify["pending_patch"] = dict(pending_patch)
    return clarify


def _parse_modify_request_llm(
    message: str,
    items: list[dict],
    *,
    trace_id: str = "",
) -> dict[str, Any] | None:
    """LLM resolver for modify phrasing — primary path when LLM is configured."""
    raw = (message or "").strip()
    if not raw or not items:
        return None
    try:
        import json

        from chat.services.llm_client import LLMClient
        from chat.services.platform.llm_prompts import EXPENSE_MODIFY_RESOLVER_SYSTEM

        payload = {
            "message": raw,
            "item_count": len(items),
            "items": [
                {
                    "index": i,
                    "line": i + 1,
                    "category": it.get("category"),
                    "amount": it.get("amount"),
                    "from_location": it.get("from_location"),
                    "to_location": it.get("to_location"),
                }
                for i, it in enumerate(items)
            ],
        }
        parsed = LLMClient().chat_json(
            system_prompt=EXPENSE_MODIFY_RESOLVER_SYSTEM,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            trace_id=trace_id or "",
            scope="expense-modify",
        )
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    if parsed.get("needs_clarify"):
        candidate_indices: list[int] = []
        for raw_idx in parsed.get("candidate_indices") or []:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(items):
                candidate_indices.append(idx)
        amount = parsed.get("amount")
        return {
            "needs_clarify": True,
            "candidate_indices": sorted(set(candidate_indices)),
            "amount": float(amount) if amount is not None else None,
            "category": parsed.get("category"),
            "label": str(parsed.get("label") or parsed.get("category") or "item"),
            "match_amount": parsed.get("match_amount"),
        }

    item_index = parsed.get("item_index")
    if item_index is None:
        return None
    try:
        idx = int(item_index)
    except (TypeError, ValueError):
        return None
    if not (0 <= idx < len(items)):
        return None

    result: dict[str, Any] = {
        "item_index": idx,
        "label": str(parsed.get("label") or f"expense {idx + 1}"),
        "needs_confirm": False,
    }
    if parsed.get("amount") is not None:
        result["amount"] = float(parsed["amount"])
    if parsed.get("category"):
        result["category"] = parsed["category"]
    if parsed.get("match_amount") is not None:
        result["match_amount"] = parsed["match_amount"]
    frm = str(parsed.get("from_location") or "").strip()
    to = str(parsed.get("to_location") or "").strip()
    if frm and to:
        result["from_location"] = frm
        result["to_location"] = to
    if result.get("amount") is None and not (frm and to):
        return None
    return result


def _category_from_message(low: str) -> str | None:
    from chat.services.platform.field_extractors.expense import (
        infer_category_from_clause,
        normalize_expense_category,
    )

    cat = infer_category_from_clause(low)
    if cat:
        return cat
    for token in ("bus", "lunch", "lanch", "luch", "snack", "metro", "bike", "rickshaw", "train"):
        if re.search(rf"\b{re.escape(token)}\b", low):
            return normalize_expense_category(token) or token
    return None


def _parse_modify_request_regex(message: str, items: list[dict]) -> dict[str, Any] | None:
    """Regex fallback when LLM is unavailable or returns nothing."""
    raw = message or ""
    low = raw.lower()
    if not items:
        return None

    numbered_idx = _numbered_item_index(raw, item_count=len(items))
    if numbered_idx is not None:
        item_num = numbered_idx + 1
        amount = _extract_modify_amount(raw, item_number_1based=item_num)
        if amount is not None:
            return {
                "item_index": numbered_idx,
                "amount": amount,
                "label": f"expense {item_num}",
                "needs_confirm": False,
            }

    amount_early = _extract_modify_amount(raw, item_number_1based=None)
    if amount_early is not None and (
        _LAST_REF_RE.search(raw)
        or _FIRST_REF_RE.search(raw)
        or _category_from_message(low)
    ):
        resolved = resolve_expense_item_reference(raw, items)
        if resolved.get("needs_clarify"):
            return {
                "needs_clarify": True,
                "candidate_indices": list(resolved.get("candidate_indices") or []),
                "amount": amount_early,
                "category": resolved.get("category"),
                "label": resolved.get("label") or "item",
            }
        if resolved.get("item_index") is not None:
            return {
                "item_index": int(resolved["item_index"]),
                "amount": amount_early,
                "label": str(resolved.get("label") or "item"),
                "needs_confirm": False,
            }

    ord_m = _MODIFY_ORDINAL_RE.search(raw)
    if ord_m:
        return {
            "item_index": 0,
            "amount": float(ord_m.group(2)),
            "label": "1st item",
            "needs_confirm": True,
        }

    item_num = None
    num_m = re.search(r"\b(\d+)\s*(?:no|number|nombor)\b", low)
    if num_m:
        item_num = int(num_m.group(1))

    amount = _extract_modify_amount(raw, item_number_1based=item_num)
    if amount is None:
        return None

    from chat.services.platform.field_extractors.expense import (
        is_travel_category,
        normalize_expense_category,
    )

    if re.search(r"\bbus\b", low):
        bus_indices = []
        for idx, item in enumerate(items):
            desc = str(item.get("description") or "").lower()
            cat = normalize_expense_category(item.get("category")) or ""
            if cat == "bus" or (is_travel_category(cat) and "bus" in desc):
                bus_indices.append(idx)
        if len(bus_indices) > 1:
            return {
                "needs_clarify": True,
                "candidate_indices": bus_indices,
                "amount": amount,
                "category": "bus",
                "label": "bus",
            }
        if len(bus_indices) == 1:
            return {"item_index": bus_indices[0], "amount": amount, "label": "bus"}

    for idx, item in enumerate(items):
        desc = str(item.get("description") or "").lower()
        cat = normalize_expense_category(item.get("category")) or ""
        if re.search(r"\blunch\b", low) and (cat == "lunch" or "lunch" in desc):
            return {"item_index": idx, "amount": amount, "label": "lunch"}
        if re.search(r"\bnasta|nasto|nosto|snack\b", low) and (
            cat == "snack" or "snack" in desc or "nasta" in desc
        ):
            return {"item_index": idx, "amount": amount, "label": "snack"}

    if is_vague_amount_modify(raw):
        return None
    if len(items) == 1:
        return {"item_index": 0, "amount": amount, "label": "item"}
    return None


def _indices_for_category(items: list[dict], category: str) -> list[int]:
    from chat.services.platform.field_extractors.expense import (
        is_travel_category,
        normalize_expense_category,
    )

    cat = normalize_expense_category(category) or category
    indices: list[int] = []
    for idx, item in enumerate(items):
        item_cat = normalize_expense_category(item.get("category")) or ""
        desc = str(item.get("description") or "").lower()
        if item_cat == cat:
            indices.append(idx)
        elif cat == "bus" and is_travel_category(item_cat) and "bus" in desc:
            indices.append(idx)
    return indices


def resolve_expense_item_reference(
    message: str,
    items: list[dict],
) -> dict[str, Any]:
    """
    Resolve which draft line the user means (number, last/first, category).
    Returns item_index, or needs_clarify + candidate_indices when ambiguous.
    """
    raw = message or ""
    low = raw.lower()
    if not items:
        return {"needs_clarify": False}

    numbered = _numbered_item_index(raw, item_count=len(items))
    if numbered is not None:
        return {
            "item_index": numbered,
            "label": f"expense {numbered + 1}",
            "needs_clarify": False,
        }

    del_m = re.search(
        r"\bexpense\s+(\d+)\b",
        low,
    )
    if del_m:
        cand = int(del_m.group(1)) - 1
        if 0 <= cand < len(items):
            return {
                "item_index": cand,
                "label": f"expense {cand + 1}",
                "needs_clarify": False,
            }

    cat = _category_from_message(low)
    pool = _indices_for_category(items, cat) if cat else list(range(len(items)))

    if _LAST_REF_RE.search(low) and pool:
        idx = pool[-1]
        label = f"last {cat}" if cat else f"expense {idx + 1}"
        return {"item_index": idx, "label": label, "needs_clarify": False}

    if _FIRST_REF_RE.search(low) and pool:
        idx = pool[0]
        label = f"first {cat}" if cat else f"expense {idx + 1}"
        return {"item_index": idx, "label": label, "needs_clarify": False}

    if cat and len(pool) == 1:
        return {
            "item_index": pool[0],
            "label": cat,
            "needs_clarify": False,
        }

    if cat and len(pool) > 1:
        return {
            "needs_clarify": True,
            "candidate_indices": pool,
            "category": cat,
            "label": cat,
        }

    return {"needs_clarify": False}


def looks_like_expense_item_modify(message: str) -> bool:
    """Numbered item reference plus a taka amount — not a bare pending slot answer."""
    raw = message or ""
    if not _NUMBERED_ITEM_RE.search(raw):
        return False
    return bool(_TAKA_AMOUNT_RE.search(raw))


def looks_like_expense_route_modify(message: str) -> bool:
    """Numbered expense reference plus a route change — review-stage edit."""
    raw = message or ""
    if not _NUMBERED_ITEM_RE.search(raw) and not re.search(r"\bexpense\s+\d+\b", raw, re.I):
        return False
    from chat.services.platform.field_extractors.route import parse_route

    if parse_route(raw):
        return True
    return bool(_ROUTE_MODIFY_RE.search(raw))


def looks_like_expense_item_delete(message: str) -> bool:
    """Numbered or natural reference delete — e.g. '5 no bad dao', 'last bus delete'."""
    raw = message or ""
    from chat.services.platform.intent_rules import is_delete_request

    if not is_delete_request(raw):
        return False
    if _NUMBERED_ITEM_RE.search(raw) or re.search(r"\bexpense\s+\d+\b", raw, re.I):
        return True
    if _LAST_REF_RE.search(raw) or _FIRST_REF_RE.search(raw):
        return True
    if _category_from_message(raw.lower()):
        return True
    return False


def _numbered_item_index(message: str, *, item_count: int) -> int | None:
    m = _NUMBERED_ITEM_RE.search(message or "")
    if not m:
        return None
    num_s = next((g for g in m.groups() if g), None)
    if not num_s:
        return None
    idx = int(num_s) - 1
    if 0 <= idx < item_count:
        return idx
    return None


def parse_multiple_item_indices(
    message: str,
    *,
    item_count: int,
    candidate_indices: list[int] | None = None,
) -> list[int]:
    """Parse multi-select replies like '2 and 3' or '2, 3' (1-based entry numbers)."""
    raw = (message or "").strip().lower()
    if not raw or item_count <= 0:
        return []
    nums = [int(n) for n in re.findall(r"\b(\d+)\b", raw)]
    if not nums:
        return []

    if candidate_indices:
        ordered = sorted({int(i) for i in candidate_indices})
        # When clarify listed a subset, numbers refer to positions in that list.
        if len(ordered) < item_count:
            indices: list[int] = []
            for n in nums:
                pos = n - 1
                if 0 <= pos < len(ordered):
                    indices.append(ordered[pos])
            return sorted(set(indices))

    indices = []
    for n in nums:
        idx = n - 1
        if 0 <= idx < item_count:
            indices.append(idx)
    if candidate_indices is not None:
        allowed = {int(i) for i in candidate_indices}
        indices = [i for i in indices if i in allowed]
    return sorted(set(indices))


def _extract_modify_amount(message: str, *, item_number_1based: int | None = None) -> float | None:
    """Prefer taka-linked amounts; skip the entry number (e.g. '1 number bus 130 taka')."""
    low = (message or "").lower()
    for m in _TAKA_AMOUNT_RE.finditer(low):
        val = float(m.group(1))
        if item_number_1based is not None and abs(val - item_number_1based) < 0.01:
            continue
        if val > 0:
            return val
    nums = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", low)]
    skip = float(item_number_1based) if item_number_1based is not None else None
    candidates = [n for n in nums if skip is None or abs(n - skip) > 0.01]
    if not candidates:
        return None
    substantial = [n for n in candidates if n >= 10]
    if len(substantial) == 1:
        return substantial[0]
    if substantial:
        return substantial[-1]
    if len(candidates) == 1:
        return candidates[0]
    return None


def is_vague_amount_modify(message: str) -> bool:
    low = (message or "").lower()
    if (
        _MODIFY_LUNCH_RE.search(low)
        or _MODIFY_ORDINAL_RE.search(message or "")
        or looks_like_expense_item_modify(message)
    ):
        return False
    return bool(_MODIFY_AMOUNT_VAGUE_RE.search(low) or re.search(r"amount\s*ta\s*\d+\s*kore", low))


def parse_delete_request(message: str, items: list[dict]) -> dict[str, Any] | None:
    """Return {item_index, label} or {needs_clarify, candidate_indices} for delete."""
    from chat.services.platform.intent_rules import is_delete_request, is_vague_delete

    raw = message or ""
    if not items or not is_delete_request(raw) or is_vague_delete(raw):
        return None

    resolved = resolve_expense_item_reference(raw, items)
    if resolved.get("needs_clarify"):
        return resolved
    if resolved.get("item_index") is not None:
        return resolved

    del_m = re.search(
        r"\bexpense\s+(\d+)\b.{0,40}(?:delete|remove|bad|muche|drop)|"
        r"(?:delete|remove|bad|muche|drop).{0,40}\bexpense\s+(\d+)\b",
        raw.lower(),
    )
    if del_m:
        num_s = next((g for g in del_m.groups() if g), None)
        if num_s:
            cand = int(num_s) - 1
            if 0 <= cand < len(items):
                return {"item_index": cand, "label": f"expense {cand + 1}"}

    return None


def parse_route_modify_request(message: str, items: list[dict]) -> dict[str, Any] | None:
    """Return {item_index, from_location, to_location, label} for route-only edits."""
    raw = message or ""
    if not items:
        return None
    from chat.services.platform.field_extractors.route import parse_route

    route = parse_route(raw)
    if not route:
        return None
    resolved = resolve_expense_item_reference(raw, items)
    if resolved.get("needs_clarify"):
        return resolved
    idx = resolved.get("item_index")
    if idx is None:
        return None
    frm, to = route
    return {
        "item_index": int(idx),
        "from_location": frm,
        "to_location": to,
        "label": str(resolved.get("label") or f"expense {int(idx) + 1}"),
    }


def parse_modify_request(
    message: str,
    items: list[dict],
    *,
    trace_id: str = "",
    prefer_llm: bool = True,
) -> dict[str, Any] | None:
    """Return {item_index, amount, label} or clarify hint — LLM first, regex fallback."""
    if not items:
        return None

    if prefer_llm:
        from chat.services.platform.field_extractors.expense import _llm_client_configured

        if _llm_client_configured():
            llm_result = _parse_modify_request_llm(message, items, trace_id=trace_id)
            if llm_result is not None:
                return llm_result

    return _parse_modify_request_regex(message, items)
