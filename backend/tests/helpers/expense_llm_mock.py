"""Mock LLM responses for expense draft editor unit tests."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


def _pending_item_index(payload: dict[str, Any]) -> int:
    focus = payload.get("pending_focus") or {}
    if focus.get("item_index") is not None:
        return int(focus["item_index"])
    pending = payload.get("pending_question") or {}
    if pending.get("item_index") is not None:
        return int(pending["item_index"])
    return 0


def _route_from_conversation_context(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Test mock — resolve route from conversation_history / recent_user_messages."""
    texts: list[str] = [str(payload.get("message") or "")]
    texts.extend(str(t) for t in (payload.get("recent_user_messages") or []))
    for line in payload.get("conversation_history") or []:
        if isinstance(line, str) and line.startswith("User:"):
            texts.append(line[len("User:") :].strip())
    for text in texts:
        low = text.lower()
        if "dhanmondi" in low and "mirpur" in low:
            return "Dhanmondi", "Mirpur"
        if "mirpur" in low and any(v in low for v in ("motejheel", "motijheel", "motekheel", "motijhil")):
            return "Mirpur", "Motijheel"
    return None


def _mock_expense_draft_interpreter(message: str, payload: dict[str, Any]) -> dict[str, Any]:
    low = (message or "").lower()
    if any(p in low for p in ("summery chai ni", "summary chai ni", "saransho chai ni")):
        return {"intent": "anti_summary", "item_patches": []}
    if "keno abar" in low or ("eta holo nah" in low and "add" in low):
        return {
            "intent": "fix_mistake",
            "item_patches": [{"action": "delete", "match_last": True}],
        }
    if any(w in low for w in ("list", "summary", "dekhao", "summery")) and "chai ni" not in low:
        return {"intent": "show_list", "item_patches": []}
    if "total" in low or "koto" in low:
        return {"intent": "show_total", "item_patches": []}
    if "summery chai ni" in low or "summary chai ni" in low or "present expense chai ni" in low:
        return {"intent": "anti_summary", "item_patches": []}
    if "modify korte bolchi" in low or "ami modify" in low:
        return {"intent": "clarify_modify", "item_patches": [], "clarify": {"kind": "which_item"}}

    del_m = re.search(r"(\d+)\s*(?:no|number).*(?:delete|remove)", low)
    if del_m or ("delete" in low and re.search(r"\b1\b", low)):
        idx = int(del_m.group(1)) - 1 if del_m else 0
        return {
            "intent": "delete",
            "item_patches": [{"action": "delete", "item_index": idx}],
        }

    if "130" in low and "bus" in low:
        items = payload.get("draft_items") or []
        bus_idx = [
            i
            for i, it in enumerate(items)
            if str(it.get("category") or "").lower() == "bus"
        ]
        if len(bus_idx) > 1:
            return {
                "intent": "clarify_modify",
                "item_patches": [],
                "clarify": {
                    "kind": "which_item",
                    "candidate_indices": bus_idx,
                    "category": "bus",
                    "proposed_value": 130,
                },
            }
        if len(bus_idx) == 1:
            return {
                "intent": "update",
                "item_patches": [{"action": "update", "item_index": bus_idx[0], "amount": 130.0}],
            }

    if "expense 5 130" in low or re.search(r"5\s*(?:no|number).*130", low):
        return {
            "intent": "update",
            "item_patches": [{"action": "update", "item_index": 4, "amount": 130.0}],
        }

    if "remove" in low or "delete" in low:
        idx = _pending_item_index(payload)
        return {
            "intent": "delete",
            "item_patches": [{"action": "delete", "item_index": idx}],
        }

    ref = re.search(r"expense\s+(\d+)\s+(\d+(?:\.\d+)?)", low)
    if ref:
        return {
            "intent": "update",
            "item_patches": [
                {
                    "action": "update",
                    "item_index": int(ref.group(1)) - 1,
                    "amount": float(ref.group(2)),
                }
            ],
        }

    focus = payload.get("pending_focus") or {}
    route_hit = _route_from_conversation_context(payload)
    if route_hit and any(
        p in low for p in ("route diyechi", "tomake route", "toy route", "ami toh", "ami to")
    ):
        idx = _pending_item_index(payload)
        frm, to = route_hit
        return {
            "intent": "answer_pending",
            "item_patches": [
                {
                    "action": "update",
                    "item_index": idx,
                    "from_location": frm,
                    "to_location": to,
                }
            ],
        }

    if "add koro" in low or "add kore" in low or "jog koro" in low:
        if route_hit and focus.get("missing_field") == "route":
            idx = _pending_item_index(payload)
            frm, to = route_hit
            return {
                "intent": "answer_pending",
                "item_patches": [
                    {
                        "action": "update",
                        "item_index": idx,
                        "from_location": frm,
                        "to_location": to,
                    }
                ],
            }
        patches: list[dict[str, Any]] = []
        for cat, keyword in (
            ("lunch", "lunch"),
            ("snack", "snack"),
            ("bus", "bus"),
            ("metro", "metro"),
            ("bike", "bike"),
        ):
            if keyword in low:
                amt = _first_amount_in_text(message)
                if amt:
                    patches.append({"action": "append", "category": cat, "amount": amt})
        if patches:
            return {"intent": "add", "item_patches": patches}

    if focus.get("missing_field") == "route":
        route_hit = _route_from_conversation_context(payload)
        if route_hit:
            idx = _pending_item_index(payload)
            frm, to = route_hit
            return {
                "intent": "answer_pending",
                "item_patches": [
                    {
                        "action": "update",
                        "item_index": idx,
                        "from_location": frm,
                        "to_location": to,
                    }
                ],
            }

    if "mirpur" in low and any(v in low for v in ("motejheel", "motijheel", "motekheel", "motijhil")):
        idx = _pending_item_index(payload)
        return {
            "intent": "answer_pending",
            "item_patches": [
                {
                    "action": "update",
                    "item_index": idx,
                    "from_location": "Mirpur",
                    "to_location": "Motijheel",
                }
            ],
        }

    if focus.get("missing_field") == "category" and not any(
        p in low for p in ("add koro", "add kore", "jog koro", "jog kore")
    ):
        low_stripped = low.strip()
        for cat in ("lunch", "snack", "bus", "train", "bike", "metro", "rickshaw"):
            if low_stripped == cat:
                return {
                    "intent": "answer_pending",
                    "item_patches": [
                        {
                            "action": "update",
                            "item_index": int(focus.get("item_index") or 0),
                            "category": cat,
                        }
                    ],
                }

    patches: list[dict[str, Any]] = []
    for cat, keyword in (
        ("lunch", "lunch"),
        ("snack", "snack"),
        ("bus", "bus"),
        ("metro", "metro"),
    ):
        if keyword in low:
            amt = _first_amount_in_text(message)
            item: dict[str, Any] = {"action": "append", "category": cat, "amount": amt or 0}
            if cat == "metro" and "mirpur" in low and "agargaon" in low:
                item["from_location"] = "Mirpur"
                item["to_location"] = "Agargaon"
            if amt:
                patches.append(item)

    if "category mone nei" in low or "category jani na" in low:
        seen = {float(p.get("amount") or 0) for p in patches if p.get("amount")}
        amt = _amount_only_expense_in_text(message, seen)
        if amt:
            patches.append({"action": "append", "amount": amt})

    if "280" in low and "300" in low:
        patches.append({"action": "update", "match_amount": 280, "amount": 300})

    return {"intent": "add" if patches else "conversation", "item_patches": patches}


def _amount_only_expense_in_text(text: str, seen_amounts: set[float]) -> float | None:
    found: list[float] = []
    for token in (text or "").replace(",", " ").split():
        cleaned = "".join(ch for ch in token if ch.isdigit() or ch == ".")
        if cleaned and any(ch.isdigit() for ch in cleaned):
            try:
                val = float(cleaned)
                if val > 0:
                    found.append(val)
            except ValueError:
                continue
    for val in reversed(found):
        if val not in seen_amounts:
            return val
    return None


def _first_amount_in_text(text: str) -> float | None:
    for token in (text or "").replace(",", " ").split():
        cleaned = "".join(ch for ch in token if ch.isdigit() or ch == ".")
        if cleaned and any(ch.isdigit() for ch in cleaned):
            try:
                val = float(cleaned)
                if val > 0:
                    return val
            except ValueError:
                continue
    return None


@contextmanager
def mock_expense_llm():
    with patch("chat.services.llm_client.LLMClient") as mock_cls:
        client = mock_cls.return_value
        client.is_configured.return_value = True

        def chat_json(*, system_prompt: str, user_prompt: str, trace_id: str = "", **kwargs):
            payload = json.loads(user_prompt)
            message = str(payload.get("message") or "")
            if "extract one expense slot" in system_prompt.lower():
                candidate = str(payload.get("candidate_user_message") or "").lower()
                missing = str(payload.get("missing_field") or "")
                if missing == "route" and "dhanmondi" in candidate and "mirpur" in candidate:
                    return {"from_location": "Dhanmondi", "to_location": "Mirpur"}
                if missing == "category" and candidate.strip() in (
                    "bus",
                    "lunch",
                    "snack",
                    "metro",
                ):
                    return {"category": candidate.strip()}
                return {}
            if "expense draft editor" in system_prompt.lower():
                return _mock_expense_draft_interpreter(message, payload)
            if "UNDERSTAND" in system_prompt or "Understanding Layer" in system_prompt:
                draft = _mock_expense_draft_interpreter(message, payload)
                updates = []
                for patch in draft.get("item_patches") or []:
                    if patch.get("action") == "append":
                        body = {k: v for k, v in patch.items() if k != "action"}
                        updates.append({"field": "items", "value": body, "action": "append"})
                return {
                    "goal": "expense",
                    "workflow": "expense",
                    "action": "start" if updates else "clarification_needed",
                    "confidence": 0.9,
                    "field_updates": updates,
                    "answers_pending_field": None,
                }
            return {}

        client.chat_json.side_effect = chat_json
        yield mock_cls
