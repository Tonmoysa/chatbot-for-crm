"""Expense field helpers — LLM-driven extraction; code only validates and coerces."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from chat.services.platform.schemas import FieldUpdate
from chat.services.session_memory import PendingQuestion, SessionMemory, WorkflowDraft

SUPPORTED_CATEGORIES = frozenset({
    "lunch", "snack", "bus", "train", "bike", "metro_rail", "metro", "rickshaw",
})
TRAVEL_CATEGORIES = frozenset({"bus", "train", "bike", "metro_rail", "metro", "rickshaw"})
FOOD_CATEGORIES = frozenset({"lunch", "snack"})

_CATEGORY_ALIASES = {
    "lanch": "lunch",
    "meals": "lunch",
    "meal": "lunch",
    "nasta": "snack",
    "nasto": "snack",
    "nosto": "snack",
    "metro rail": "metro_rail",
    "metrorail": "metro_rail",
    "metro-rail": "metro_rail",
    "cng": "rickshaw",
    "auto": "rickshaw",
    "travel": "bus",
    "transport": "bus",
}


def category_display_name(category: str) -> str:
    mapping = {
        "lunch": "Lunch",
        "snack": "Snack",
        "bus": "Bus",
        "train": "Train",
        "bike": "Bike",
        "metro_rail": "Metro Rail",
        "metro": "Metro",
        "rickshaw": "Rickshaw",
    }
    return mapping.get(str(category or "").strip().lower(), str(category or "?"))


def is_travel_category(category: str | None) -> bool:
    return normalize_expense_category(category) in TRAVEL_CATEGORIES


def normalize_expense_category(value: Any) -> str | None:
    if value in (None, ""):
        return None
    raw = str(value).strip().lower().replace("-", "_")
    compact = raw.replace(" ", "_")
    if compact in _CATEGORY_ALIASES:
        compact = _CATEGORY_ALIASES[compact]
    if compact in SUPPORTED_CATEGORIES:
        return compact
    spaced = str(value).strip().lower()
    if spaced in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[spaced]
    return None


def is_unsupported_category_mention(value: Any) -> bool:
    if value in (None, ""):
        return False
    return normalize_expense_category(value) is None


def _coerce_amount(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        amt = float(value)
        return amt if amt > 0 else None
    except (TypeError, ValueError):
        return None


_ROUTE_JUNK_WORDS = frozenset({
    "office", "jawar", "jawa", "somoy", "somoi", "ajke", "aj", "today", "ferar",
    "dupure", "bikale", "morning", "afternoon", "time", "?", "going", "giye",
    "lagse", "lagbe", "korlam", "korechi", "hoyeche", "hoise", "diyechi", "te", "e",
})


def is_valid_expense_route(from_loc: str | None, to_loc: str | None) -> bool:
    """Reject hallucinated or fragment routes (e.g. office → jawar)."""
    frm = str(from_loc or "").strip().lower()
    to = str(to_loc or "").strip().lower()
    if not frm or not to or frm == to:
        return False
    if frm in _ROUTE_JUNK_WORDS or to in _ROUTE_JUNK_WORDS:
        return False
    if any(part in _ROUTE_JUNK_WORDS for part in frm.split()):
        return False
    if any(part in _ROUTE_JUNK_WORDS for part in to.split()):
        return False
    if "taka" in frm or "taka" in to or "tk" in frm or "tk" in to:
        return False
    if len(frm) < 2 or len(to) < 2:
        return False
    return True


def _coerce_route_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    frm = str(value.get("from_location") or "").strip()[:120]
    to = str(value.get("to_location") or "").strip()[:120]
    if not is_valid_expense_route(frm, to):
        return {}
    return {"from_location": frm, "to_location": to}


def coerce_expense_modify_turn(message: str, memory) -> dict[str, Any] | None:
    """Rules-first item update by number/category — before LLM draft interpreter."""
    from chat.services.platform.field_extractors.modify import (
        looks_like_expense_item_modify,
        parse_modify_request,
    )
    from chat.services.platform.intent_rules import is_modify_request

    draft = memory.active_draft() if memory else None
    if not draft:
        return None
    items = list(draft.fields.get("items") or [])
    if not items:
        return None
    if not is_modify_request(message) and not looks_like_expense_item_modify(message):
        return None
    parsed = parse_modify_request(message, items)
    if not parsed:
        return None
    if parsed.get("needs_clarify"):
        return {
            "intent": "clarify_modify",
            "item_patches": [],
            "clarify": {
                "kind": "which_item",
                "candidate_indices": list(parsed.get("candidate_indices") or []),
                "proposed_value": parsed.get("amount"),
                "category": parsed.get("category"),
            },
        }
    if parsed.get("amount") is None:
        return None
    idx = int(parsed["item_index"])
    if not (0 <= idx < len(items)):
        return None
    patch: dict[str, Any] = {
        "action": "update",
        "item_index": idx,
        "amount": float(parsed["amount"]),
    }
    if parsed.get("category"):
        patch["category"] = parsed["category"]
    return {"intent": "update", "item_patches": [patch]}


def coerce_expense_route_modify_turn(message: str, memory) -> dict[str, Any] | None:
    """Rules-first route update by item number — before LLM draft interpreter."""
    from chat.services.platform.field_extractors.modify import (
        looks_like_expense_route_modify,
        parse_route_modify_request,
    )
    from chat.services.platform.intent_rules import is_modify_request

    draft = memory.active_draft() if memory else None
    if not draft:
        return None
    items = list(draft.fields.get("items") or [])
    if not items:
        return None
    if not is_modify_request(message) and not looks_like_expense_route_modify(message):
        from chat.services.platform.field_extractors.route import parse_route

        if not parse_route(message):
            return None
    parsed = parse_route_modify_request(message, items)
    if not parsed:
        return None
    if parsed.get("needs_clarify"):
        return {
            "intent": "clarify_modify",
            "item_patches": [],
            "clarify": {
                "kind": "which_item",
                "candidate_indices": list(parsed.get("candidate_indices") or []),
                "category": parsed.get("category"),
            },
        }
    idx = int(parsed["item_index"])
    if not (0 <= idx < len(items)):
        return None
    patch: dict[str, Any] = {
        "action": "update",
        "item_index": idx,
        "from_location": parsed["from_location"],
        "to_location": parsed["to_location"],
    }
    intent = "modify_review" if is_expense_review_mode(memory) else "update"
    return {"intent": intent, "item_patches": [patch]}


def _patch_targets_existing_item(
    items: list[dict[str, Any]],
    patch: dict[str, Any],
    *,
    pending_item_index: int | None = None,
) -> bool:
    """True when LLM patch references an existing draft line (not a new append)."""
    if not items or not isinstance(patch, dict):
        return False
    if patch.get("item_index") is not None:
        try:
            idx = int(patch["item_index"])
        except (TypeError, ValueError):
            idx = -1
        return 0 <= idx < len(items)
    if str(patch.get("item_id") or "").strip():
        item_id = str(patch["item_id"]).strip()
        return any(str(it.get("id") or "") == item_id for it in items)
    if patch.get("match_amount") is not None or patch.get("match_last"):
        return resolve_item_index(items, patch, pending_item_index=pending_item_index) is not None
    return False


def empty_expense_turn() -> dict[str, Any]:
    """Fresh expense_turn shell — no patches carried from prior turns."""
    return {
        "intent": None,
        "item_patches": [],
        "delete_indices": [],
        "clarify": {},
    }


def is_expense_regret_remove_message(message: str) -> bool:
    """Banglish regret / undo — item no longer needed or added by mistake."""
    low = (message or "").strip().lower()
    if not low:
        return False
    patterns = (
        r"lagbe\s+nah",
        r"lagbena",
        r"ar\s+lagbe\s+nah",
        r"dorkar\s+nah",
        r"lagbe\s+na\b",
        r"vule\s+add",
        r"vul\s+kore",
        r"bhul\s+kore",
        r"mistake",
        r"vul\s+chilo",
        r"keno\s+add",
        r"add\s+korar\s+dorkar\s+nai",
    )
    return any(re.search(p, low) for p in patterns)


def is_expense_pending_field_value_answer(message: str, memory) -> bool:
    """True only when message is a direct slot answer — not edit/navigation."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.intent_rules import is_workflow_show_request

    pq = memory.pending_question if memory else None
    if not pq or pq.workflow_id != "expense" or pq.item_index is None:
        return False
    if is_expense_draft_mutation_message(message, memory):
        return False
    if is_workflow_show_request(message, workflow_id="expense"):
        return False
    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return False
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    idx = int(pq.item_index)
    if not (0 <= idx < len(items)):
        return False
    from chat.services.platform.field_extractors.modify import _category_from_message

    msg_cat = _category_from_message(raw.lower())
    pending_cat = normalize_expense_category(items[idx].get("category"))
    if msg_cat and pending_cat and msg_cat != pending_cat:
        return False
    if pq.field == "item_route":
        from chat.services.platform.field_extractors.route import parse_route

        route = parse_route(raw) or parse_route(message)
        return bool(route and is_valid_expense_route(route[0], route[1]))
    if pq.field == "item_category":
        cat = normalize_expense_category(raw)
        return bool(cat) and len(raw.split()) <= 4
    if pq.field == "item_amount":
        from chat.services.platform.field_extractors.amount import parse_amount

        return parse_amount(raw) is not None and len(raw.split()) <= 6
    return False


def is_expense_draft_mutation_message(message: str, memory=None) -> bool:
    """True when user is editing draft lines (delete/modify/correct), not navigating."""
    from chat.services.platform.field_extractors.modify import (
        _category_from_message,
        looks_like_expense_item_delete,
        looks_like_expense_route_modify,
        parse_delete_request,
        parse_modify_request,
    )
    from chat.services.platform.intent_rules import is_delete_request, is_modify_request

    if message_has_new_expense_items(message):
        return False

    if is_expense_regret_remove_message(message):
        return True
    if is_delete_request(message) or looks_like_expense_item_delete(message):
        return True
    if is_modify_request(message) or looks_like_expense_route_modify(message):
        return True
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    if items and parse_delete_request(message, items):
        return True
    if items and parse_modify_request(message, items):
        return True
    return False


def sanitize_expense_turn_for_action(
    turn: dict[str, Any] | None,
    *,
    action: str = "",
    expense_intent: str = "",
) -> dict[str, Any]:
    """Drop stale append/update patches when this turn is delete/modify."""
    if not isinstance(turn, dict):
        return empty_expense_turn()
    out = dict(turn)
    act = str(action or "").lower()
    intent = str(expense_intent or out.get("intent") or "").lower()
    if act == "delete" or intent in ("delete", "clarify_delete"):
        patches = [
            p
            for p in (out.get("item_patches") or [])
            if isinstance(p, dict) and str(p.get("action") or "").lower() == "delete"
        ]
        delete_indices = list(out.get("delete_indices") or [])
        if not patches and delete_indices:
            patches = [{"action": "delete", "item_index": i} for i in delete_indices]
        return {
            **out,
            "intent": intent or "delete",
            "item_patches": patches,
            "delete_indices": delete_indices,
        }
    if act == "modify" or intent in ("update", "modify_review", "correct", "fix_mistake"):
        patches = [
            p
            for p in (out.get("item_patches") or [])
            if isinstance(p, dict)
            and str(p.get("action") or "").lower() in ("update", "correct", "delete")
        ]
        return {**out, "item_patches": patches, "delete_indices": []}
    return out


def expense_entities_for_turn(
    entities: dict[str, Any] | None,
    turn: dict[str, Any] | None,
    *,
    expense_intent: str = "",
    action: str = "",
) -> dict[str, Any]:
    """Replace expense_turn in entities — never merge stale patches across turns."""
    out = dict(entities or {})
    if turn is None:
        out["expense_turn"] = empty_expense_turn()
        out.pop("expense_intent", None)
        out.pop("expense_wizard_fallback", None)
        out.pop("expense_llm_degraded", None)
        return out
    sanitized = sanitize_expense_turn_for_action(
        turn,
        action=action,
        expense_intent=expense_intent,
    )
    intent = str(expense_intent or sanitized.get("intent") or "").lower()
    out["expense_turn"] = sanitized
    if intent:
        out["expense_intent"] = intent
    else:
        out.pop("expense_intent", None)
    return out


def normalize_expense_mutation_turn(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """Ground delete/regret to the category in the user message — not pending_question."""
    if not isinstance(turn, dict) or not memory:
        return turn
    if message_has_new_expense_items(message):
        return turn
    from chat.services.platform.field_extractors.modify import (
        _category_from_message,
        _indices_for_category,
    )

    raw = (message or "").strip()
    low = raw.lower()
    msg_cat = _category_from_message(low)
    regret = is_expense_regret_remove_message(message)
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return turn

    pq = memory.pending_question
    pending_cat = None
    if pq and pq.workflow_id == "expense" and pq.item_index is not None:
        pidx = int(pq.item_index)
        if 0 <= pidx < len(items):
            pending_cat = normalize_expense_category(items[pidx].get("category"))

    def _delete_turn(indices: list[int]) -> dict[str, Any]:
        if len(indices) == 1:
            idx = indices[0]
            return {
                **turn,
                "intent": "delete",
                "item_patches": [{"action": "delete", "item_index": idx}],
                "delete_indices": [idx],
                "clarify": {},
            }
        return {
            **turn,
            "intent": "clarify_delete",
            "item_patches": [],
            "delete_indices": [],
            "clarify": {
                "kind": "which_delete",
                "candidate_indices": indices,
                "category": msg_cat or "",
            },
        }

    if msg_cat and pending_cat and msg_cat != pending_cat:
        indices = _indices_for_category(items, msg_cat)
        if regret and indices:
            return _delete_turn(indices)
        intent = str(turn.get("intent") or "").lower()
        clarify = turn.get("clarify") if isinstance(turn.get("clarify"), dict) else {}
        clarify_cat = normalize_expense_category(clarify.get("category") or "")
        if intent in ("clarify_modify", "answer_pending") and (
            not clarify_cat or clarify_cat != normalize_expense_category(msg_cat)
        ):
            if regret and indices:
                return _delete_turn(indices)
            if len(indices) == 1:
                return {
                    **turn,
                    "intent": "modify_review" if is_expense_review_mode(memory) else "update",
                    "item_patches": [{"action": "update", "item_index": indices[0]}],
                    "clarify": {},
                }

    if regret and msg_cat:
        indices = _indices_for_category(items, msg_cat)
        if indices:
            return _delete_turn(indices)

    if regret and not msg_cat:
        from chat.services.platform.field_extractors.modify import resolve_expense_item_reference

        resolved = resolve_expense_item_reference(message, items)
        if resolved.get("item_index") is not None:
            idx = int(resolved["item_index"])
            return _delete_turn([idx])

    intent = str(turn.get("intent") or "").lower()
    if intent == "clarify_modify" and msg_cat:
        clarify = turn.get("clarify") if isinstance(turn.get("clarify"), dict) else {}
        clarify_cat = normalize_expense_category(clarify.get("category") or "")
        if clarify_cat and clarify_cat != normalize_expense_category(msg_cat):
            indices = _indices_for_category(items, msg_cat)
            if regret and indices:
                return _delete_turn(indices)

    return turn


def normalize_expense_correction_turn(
    turn: dict[str, Any],
    memory,
) -> dict[str, Any]:
    """Coerce LLM append+target patches into update — corrections must not add lines."""
    if not isinstance(turn, dict) or not turn.get("item_patches"):
        return turn
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    pq = memory.pending_question if memory else None
    pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None

    intent = str(turn.get("intent") or "").lower()
    normalized: list[dict[str, Any]] = []
    any_correction = False

    for raw in turn.get("item_patches") or []:
        if not isinstance(raw, dict):
            normalized.append(raw)
            continue
        patch = dict(raw)
        action = str(patch.get("action") or "update").strip().lower()
        targets_existing = _patch_targets_existing_item(
            items, patch, pending_item_index=pending_idx
        )

        if action in ("append", "add") and targets_existing:
            any_correction = True
            patch["action"] = "update"
            idx = resolve_item_index(items, patch, pending_item_index=pending_idx)
            if idx is not None:
                patch["item_index"] = idx
            floc = str(patch.get("from_location") or "").strip()
            tloc = str(patch.get("to_location") or "").strip()
            if not is_valid_expense_route(floc, tloc):
                patch.pop("from_location", None)
                patch.pop("to_location", None)
        elif action in ("update", "correct") and targets_existing:
            any_correction = True
            idx = resolve_item_index(items, patch, pending_item_index=pending_idx)
            if idx is not None:
                patch["item_index"] = idx
            floc = str(patch.get("from_location") or "").strip()
            tloc = str(patch.get("to_location") or "").strip()
            if not is_valid_expense_route(floc, tloc):
                patch.pop("from_location", None)
                patch.pop("to_location", None)

        normalized.append(patch)

    if not any_correction:
        return {**turn, "item_patches": normalized}

    if intent in ("add", "answer_pending", "conversation"):
        new_intent = "modify_review" if is_expense_review_mode(memory) else "update"
        return {**turn, "intent": new_intent, "item_patches": normalized}

    return {**turn, "item_patches": normalized}


def expense_turn_has_targeted_patches(
    turn: dict[str, Any] | None,
    memory,
) -> bool:
    """True when LLM patches reference existing draft lines (correction / update / delete)."""
    if not isinstance(turn, dict):
        return False
    intent = str(turn.get("intent") or "").lower()
    if intent == "delete" and (turn.get("delete_indices") or turn.get("item_patches")):
        return True
    if not turn.get("item_patches"):
        return False
    if intent in ("update", "modify_review", "correct", "fix_mistake", "delete"):
        return True
    if intent == "add":
        return False
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return False
    pq = memory.pending_question if memory else None
    pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
    for raw in turn.get("item_patches") or []:
        if not isinstance(raw, dict):
            continue
        patch = dict(raw)
        if str(patch.get("action") or "").lower() == "delete":
            return True
        if _patch_targets_existing_item(items, patch, pending_item_index=pending_idx):
            return True
        try:
            old_amt = float(patch.get("match_amount"))
            new_amt = float(patch.get("amount"))
        except (TypeError, ValueError):
            continue
        if abs(old_amt - new_amt) < 0.01:
            continue
        matches = [
            i
            for i, it in enumerate(items)
            if abs(float(it.get("amount") or 0) - old_amt) < 0.01
        ]
        if matches:
            return True
    return False


def coerce_expense_correction_turn(
    turn: dict[str, Any],
    memory,
) -> dict[str, Any]:
    """Normalize LLM output and force update intent when patches target existing lines."""
    turn = normalize_expense_correction_turn(turn, memory)
    if not expense_turn_has_targeted_patches(turn, memory):
        return turn
    intent = str(turn.get("intent") or "").lower()
    if intent in ("add", "answer_pending", "conversation"):
        turn = {
            **turn,
            "intent": "modify_review" if is_expense_review_mode(memory) else "update",
        }
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    pq = memory.pending_question if memory else None
    pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
    patches: list[dict[str, Any]] = []
    for raw in turn.get("item_patches") or []:
        if not isinstance(raw, dict):
            patches.append(raw)
            continue
        patch = dict(raw)
        action = str(patch.get("action") or "update").strip().lower()
        if action in ("append", "add") and _patch_targets_existing_item(
            items, patch, pending_item_index=pending_idx
        ):
            patch["action"] = "update"
        idx = resolve_item_index(items, patch, pending_item_index=pending_idx)
        if idx is not None:
            patch["item_index"] = idx
        floc = str(patch.get("from_location") or "").strip()
        tloc = str(patch.get("to_location") or "").strip()
        if not is_valid_expense_route(floc, tloc):
            patch.pop("from_location", None)
            patch.pop("to_location", None)
        patches.append(patch)
    return {**turn, "item_patches": patches}


def coerce_compound_expense_add_turn(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """Compound / repeat expense lines always append — never overwrite existing rows."""
    if not isinstance(turn, dict) or not message_has_new_expense_items(message):
        return turn

    patches_out: list[dict[str, Any]] = []
    for raw in turn.get("item_patches") or []:
        if not isinstance(raw, dict):
            continue
        patch = dict(raw)
        action = str(patch.get("action") or "").strip().lower()
        if action == "delete":
            patches_out.append(patch)
            continue
        patch.pop("item_index", None)
        patch.pop("match_amount", None)
        patch.pop("match_last", None)
        patch["action"] = "append"
        body = _patch_body(patch)
        if body:
            patches_out.append(patch)

    if not patches_out:
        wizard = build_wizard_fallback_turn(message, memory)
        if _turn_has_actionable_patches(wizard):
            return {
                **wizard,
                "intent": "add",
                "wizard_fallback": True,
            }
        return {**turn, "intent": "add", "item_patches": []}

    out: dict[str, Any] = {
        **turn,
        "intent": "add",
        "item_patches": patches_out,
        "clarify": {},
    }
    active_id = memory.active_workflow.id if memory and memory.active_workflow else "expense"
    if message_requests_submit_after_edit(message, active_workflow_id=active_id):
        out["submit_after_edit"] = True
    return out


def expense_turn_is_draft_mutation(turn: dict[str, Any] | None, message: str) -> bool:
    """Delete/modify/review turns must never pass through date-add policy."""
    from chat.services.platform.intent_rules import is_delete_request, is_modify_request

    raw = (message or "").strip()
    if raw and (is_delete_request(raw) or is_modify_request(raw)):
        return True
    if not isinstance(turn, dict):
        return False
    intent = str(turn.get("intent") or "").lower()
    if intent in (
        "delete",
        "clarify_delete",
        "update",
        "modify_review",
        "correct",
        "fix_mistake",
        "clarify_modify",
        "confirm",
        "cancel",
        "show_summary",
        "show_list",
        "show_total",
        "conversation",
        "anti_summary",
    ):
        return True
    if turn.get("delete_indices"):
        return True
    patches = turn.get("item_patches") or []
    if patches and all(
        isinstance(p, dict)
        and str(p.get("action") or "").lower() in ("delete", "update", "correct")
        for p in patches
    ):
        return True
    return False


def expense_turn_is_new_add(turn: dict[str, Any] | None, message: str) -> bool:
    """True when the turn is (or should be) blocked by today-only add policy."""
    if expense_turn_is_draft_mutation(turn, message):
        return False
    if not isinstance(turn, dict):
        return False
    intent = str(turn.get("intent") or "").lower()
    if intent in ("replay_blocked_add", "date_correction"):
        return True
    patches = turn.get("item_patches") or []
    if any(
        isinstance(p, dict)
        and str(p.get("action") or "append").lower() in ("append", "add", "")
        for p in patches
    ):
        return True
    if intent in ("add", ""):
        return message_has_new_expense_items(message)
    return False


def finalize_expense_turn_patches(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """Apply compound-add or correction coercion — mutually exclusive."""
    intent = str((turn or {}).get("intent") or "").lower()
    if intent in ("date_not_allowed", "date_correction", "replay_blocked_add") or (turn or {}).get(
        "date_policy_rejected"
    ):
        return dict(turn or {})
    if expense_turn_is_draft_mutation(turn, message):
        return dict(turn or {})
    if message_has_new_expense_items(message):
        return coerce_compound_expense_add_turn(turn, message, memory)
    return coerce_expense_correction_turn(turn, memory)


def normalize_expense_delete_turn(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """Vague delete ('delete koro') must clarify — never guess from pending_question context."""
    from chat.services.platform.intent_rules import is_vague_delete

    intent = str(turn.get("intent") or "").strip().lower()
    if intent != "delete" or not is_vague_delete(message):
        return turn
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    return {
        "intent": "clarify_delete",
        "item_patches": [],
        "clarify": {
            "kind": "which_delete",
            "candidate_indices": list(range(len(items))),
        },
    }


def expense_delete_field_updates(
    fields: dict[str, Any],
    turn: dict[str, Any],
    *,
    pending_item_index: int | None = None,
) -> list:
    """Apply delete patches from expense_turn — used by executor, not understanding."""
    intent = str(turn.get("intent") or "").strip().lower()
    if intent != "delete":
        return []
    return patches_to_field_updates(fields, turn, pending_item_index=pending_item_index)


def coerce_expense_delete_turn(message: str, memory) -> dict[str, Any] | None:
    """Rules-first line delete — before LLM draft interpreter."""
    from chat.services.platform.field_extractors.modify import (
        looks_like_expense_item_delete,
        parse_delete_request,
    )
    from chat.services.platform.intent_rules import is_delete_request, is_vague_delete

    draft = memory.active_draft() if memory else None
    if not draft:
        return None
    items = list(draft.fields.get("items") or [])
    if not items:
        return None
    if is_vague_delete(message):
        return {"intent": "clarify_delete", "item_patches": []}
    if not is_delete_request(message) and not looks_like_expense_item_delete(message):
        return None
    parsed = parse_delete_request(message, items)
    if parsed:
        if parsed.get("needs_clarify"):
            cat = parsed.get("category") or parsed.get("label")
            return {
                "intent": "clarify_delete",
                "item_patches": [],
                "clarify": {
                    "kind": "which_delete",
                    "candidate_indices": list(parsed.get("candidate_indices") or []),
                    "category": cat,
                },
            }
        if parsed.get("item_index") is not None:
            return {
                "intent": "delete",
                "item_patches": [
                    {"action": "delete", "item_index": int(parsed["item_index"])}
                ],
                "delete_indices": [int(parsed["item_index"])],
            }
    if is_delete_request(message):
        return {"intent": "clarify_delete", "item_patches": []}
    return None


def expense_turn_blocks_wizard(
    message: str,
    memory,
    *,
    turn: dict[str, Any] | None = None,
) -> bool:
    """True when message is a draft edit — wizard must not parse amounts from item numbers."""
    intent = str((turn or {}).get("intent") or "").lower()
    if intent in (
        "delete",
        "clarify_delete",
        "update",
        "modify_review",
        "clarify_modify",
        "fix_mistake",
    ):
        return True
    if (turn or {}).get("delete_indices") or any(
        str(p.get("action") or "").lower() == "delete"
        for p in ((turn or {}).get("item_patches") or [])
        if isinstance(p, dict)
    ):
        return True
    if message_has_new_expense_items(message):
        return False
    from chat.services.platform.intent_rules import is_delete_request, is_modify_request

    if is_delete_request(message) or is_modify_request(message):
        return True
    if coerce_expense_delete_turn(message, memory):
        return True
    modify = coerce_expense_modify_turn(message, memory)
    if modify and modify.get("item_patches"):
        return True
    if coerce_expense_route_modify_turn(message, memory):
        return True
    if pending_expense_edit_active(memory):
        return True
    return False


def coerce_pending_expense_turn(
    message: str,
    memory,
) -> dict[str, Any] | None:
    """Obvious pending-slot answers — canonical enum/amount only, not narrative parsing."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    if is_expense_draft_mutation_message(message, memory):
        return None

    route_turn = _coerce_expense_route_answer_turn(message, memory)
    if route_turn:
        return route_turn

    pq = memory.pending_question if memory else None
    if not pq or pq.workflow_id != "expense" or pq.item_index is None:
        return None
    from chat.services.platform.intent_rules import is_expense_add_request

    if is_expense_add_request(message) or message_has_new_expense_items(message):
        return None
    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return None
    idx = int(pq.item_index)
    if pq.field == "item_category":
        cat = normalize_expense_category(raw)
        if cat:
            return {
                "intent": "answer_pending",
                "item_patches": [{"action": "update", "item_index": idx, "category": cat}],
            }
    if pq.field == "item_amount":
        from chat.services.platform.field_extractors.amount import parse_amount

        amt = _coerce_amount(raw) or parse_amount(raw)
        if amt is not None:
            return {
                "intent": "answer_pending",
                "item_patches": [{"action": "update", "item_index": idx, "amount": amt}],
            }
    if pq.field == "item_route":
        from chat.services.platform.field_extractors.route import parse_route

        route = parse_route(raw) or parse_route(message)
        if route:
            frm, to = route
            if is_valid_expense_route(frm, to):
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
    return None


def _coerce_expense_route_answer_turn(message: str, memory) -> dict[str, Any] | None:
    """Route-only replies (e.g. mirpur to badda) for the open travel slot."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.field_extractors.route import parse_route
    from chat.services.platform.intent_rules import is_expense_add_request

    if is_expense_add_request(message) or message_has_new_expense_items(message):
        return None
    pq = memory.pending_question if memory else None
    if pq and pq.workflow_id == "expense" and pq.field not in ("item_route", "route"):
        return None
    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return None
    route = parse_route(raw) or parse_route(message)
    if not route:
        return None
    frm, to = route
    if not is_valid_expense_route(frm, to):
        return None

    draft = memory.active_draft() if memory else None
    if not draft:
        return None
    items = list(draft.fields.get("items") or [])
    queue = build_pending_queue(items)
    route_entries = [e for e in queue if e.field == "route"]
    if not route_entries:
        return None

    pq = memory.pending_question if memory else None
    idx: int | None = None
    if pq and pq.workflow_id == "expense" and pq.field == "item_route" and pq.item_index is not None:
        idx = int(pq.item_index)
    elif len(route_entries) == 1:
        idx = route_entries[0].item_index
    if idx is None or not (0 <= idx < len(items)):
        return None

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


def _pending_missing_field(
    fields: dict[str, Any],
    pending_item_index: int | None,
) -> str | None:
    if pending_item_index is None:
        return None
    items = list(fields.get("items") or [])
    if not (0 <= pending_item_index < len(items)):
        return None
    item = items[pending_item_index]
    if not isinstance(item, dict):
        return None
    missing = list(item.get("missing_fields") or compute_item_missing_fields(item))
    return missing[0] if missing else None


def _filter_patch_body_for_pending(
    body: dict[str, Any],
    *,
    missing_field: str | None,
    intent: str,
) -> dict[str, Any]:
    if intent != "answer_pending" or not missing_field or not body:
        return body
    if missing_field == "category":
        cat = normalize_expense_category(body.get("category"))
        return {"category": cat} if cat else {}
    if missing_field == "route":
        route = _coerce_route_dict(body)
        return route if route else {}
    if missing_field == "amount":
        amt = _coerce_amount(body.get("amount"))
        return {"amount": amt} if amt is not None else {}
    return body


# --- User-signal helpers: structural patterns only (NLU lives in LLM) ---

_AMBIGUOUS_ACK_RE = re.compile(
    r"^(?:ok(?:ay)?|hmm?|ach+ha|thik(?:\s*ache)?|yes|ha+h|hya)\.?$",
    re.I | re.UNICODE,
)
_ANTI_SUMMARY_NEGATION_RE = re.compile(
    r"(?:chai\s*n[ai]|chah[iy]\s*n[ai]|don't\s+want|do\s+not\s+want|lagbe\s*n[ai])",
    re.I | re.UNICODE,
)
_ANTI_SUMMARY_TARGET_RE = re.compile(
    r"(?:summar|saransho|list|present\s+expense|bortoman|current\s+expense|ekhono\s+list)",
    re.I | re.UNICODE,
)
_CATEGORY_DECLINE_RE = re.compile(
    r"^\s*(?:remove|delete|drop|bad\s*d[ai]o)(?:\s+\w+)*\s*\.?$",
    re.I | re.UNICODE,
)


def is_expense_ambiguous_ack(message: str) -> bool:
    """Very short acknowledgement while a slot is open — not full NLU."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    return bool(raw) and bool(_AMBIGUOUS_ACK_RE.match(raw))


def is_expense_anti_summary_request(message: str) -> bool:
    """Structural negation + summary/list rejection — LLM fallback only."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    low = normalize_banglish_message((message or "").strip()).lower()
    if not low:
        return False
    return bool(_ANTI_SUMMARY_NEGATION_RE.search(low) and _ANTI_SUMMARY_TARGET_RE.search(low))


def is_expense_category_unknown_decline(message: str) -> bool:
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    return bool(raw) and bool(_CATEGORY_DECLINE_RE.match(raw))


def is_expense_collect_complaint(message: str) -> bool:
    """Expense-stage complaint — prefer LLM expense_intent; structural fallback only."""
    from chat.services.platform.turn_semantics import is_process_question, is_workflow_meta_complaint

    return (
        is_workflow_meta_complaint(message)
        or is_process_question(message)
        or is_expense_anti_summary_request(message)
    )


def _apply_fix_mistake_undo(memory, turn: dict[str, Any]) -> dict[str, Any]:
    """When LLM says fix_mistake without patches, undo last mistaken append if known."""
    if str(turn.get("intent") or "").lower() != "fix_mistake" or _turn_has_actionable_patches(turn):
        return turn
    draft = memory.active_draft() if memory else None
    items = list(draft.fields.get("items") or []) if draft else []
    last_ops = dict((memory.last_entities or {}).get("expense_last_ops") or {}) if memory else {}
    if items and last_ops.get("appended"):
        return {
            **turn,
            "item_patches": [{"action": "delete", "match_last": True}],
            "reasoning": (turn.get("reasoning") or "Undo mistaken duplicate append.").strip(),
        }
    return turn


def extract_expense_slot_via_llm(
    text: str,
    missing_field: str,
    *,
    trace_id: str = "",
) -> dict[str, Any] | None:
    """LLM-only: pull route/category/amount from one prior user line."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((text or "").strip())
    if not raw or not _llm_client_configured():
        return None
    import json

    from chat.services.llm_client import LLMClient
    from chat.services.platform.llm_prompts import EXPENSE_SLOT_FROM_HISTORY_SYSTEM

    parsed = LLMClient().chat_json(
        system_prompt=EXPENSE_SLOT_FROM_HISTORY_SYSTEM,
        user_prompt=json.dumps(
            {"candidate_user_message": raw, "missing_field": missing_field},
            ensure_ascii=False,
        ),
        trace_id=trace_id or "",
    )
    if not isinstance(parsed, dict):
        return None
    if missing_field == "route":
        route = _coerce_route_dict(parsed)
        return route if route else None
    if missing_field == "category":
        cat = normalize_expense_category(parsed.get("category"))
        return {"category": cat} if cat else None
    if missing_field == "amount":
        amt = _coerce_amount(parsed.get("amount"))
        return {"amount": amt} if amt is not None else None
    return None


def infer_expense_slot_from_history(
    memory,
    conversation_history: list[str] | None,
    *,
    trace_id: str = "",
) -> dict[str, Any] | None:
    """Phase 2 — recover pending slot from earlier user turns when current turn is empty/wrong."""
    focus = build_pending_focus(memory)
    if not focus:
        return None
    missing = str(focus.get("missing_field") or "").strip().lower()
    if missing not in ("route", "category", "amount"):
        return None
    idx = int(focus.get("item_index") or 0)

    from chat.services.platform.turn_semantics import recent_user_messages

    candidates: list[str] = list(recent_user_messages(conversation_history, limit=6))
    for line in reversed(list(conversation_history or ())):
        if not isinstance(line, str) or not line.startswith("User:"):
            continue
        text = line[len("User:") :].strip()
        if text and text not in candidates:
            candidates.append(text)

    seen: set[str] = set()
    for text in reversed(candidates):
        key = text[:100]
        if key in seen:
            continue
        seen.add(key)
        slot = extract_expense_slot_via_llm(text, missing, trace_id=trace_id)
        if not slot:
            continue
        return {
            "intent": "answer_pending",
            "item_patches": [{"action": "update", "item_index": idx, **slot}],
            "reasoning": "Backfilled pending slot from conversation history.",
        }
    return None


def record_expense_last_ops(
    memory,
    *,
    item_count_before: int,
    turn: dict[str, Any] | None,
    applied_notes: list[str] | None,
) -> None:
    """Phase 3 — remember last draft mutation for undo."""
    notes = list(applied_notes or [])
    entities = dict(memory.last_entities or {})
    entities["expense_last_ops"] = {
        "item_count_before": item_count_before,
        "intent": str((turn or {}).get("intent") or ""),
        "notes": notes,
        "appended": any("added item" in n for n in notes),
        "deleted": any("deleted item" in n for n in notes),
    }
    memory.last_entities = entities


def _turn_has_actionable_patches(turn: dict[str, Any]) -> bool:
    intent = str(turn.get("intent") or "").lower()
    if intent in (
        "fix_mistake",
        "answer_pending",
        "add",
        "update",
        "modify_review",
        "delete",
        "correct",
    ):
        return bool(turn.get("item_patches") or turn.get("delete_indices"))
    return False


_CLAUSE_CATEGORY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lunch", ("lunch", "lanch", "dupure", "dinner", "breakfast")),
    ("snack", ("snack", "nasta", "nasto", "nosto", "bikale")),
    ("bus", ("bus", "বাস")),
    ("metro", ("metro",)),
    ("train", ("train", "ট্রেন")),
    ("bike", ("bike", "সাইকেল")),
    ("rickshaw", ("rickshaw", "cng", "auto")),
)


def split_expense_clauses(message: str) -> list[str]:
    """Split a narrative into clauses when LLM is unavailable."""
    text = (message or "").replace("।", ".")
    for sep in ("\n", ";", ","):
        text = text.replace(sep, ".")
    parts = [p.strip() for p in text.split(".") if p.strip()]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def infer_category_from_clause(clause: str) -> str | None:
    low = (clause or "").strip().lower()
    if not low:
        return None
    for cat, hints in _CLAUSE_CATEGORY_HINTS:
        if any(h in low for h in hints):
            return cat
    return None


def _message_looks_like_amount_correction(message: str) -> bool:
    """True when user is correcting an existing line amount — not adding a new one."""
    low = (message or "").strip().lower()
    if not low:
        return False
    patterns = (
        r"vule\s+\d+",
        r"vul\s+kore",
        r"ashole\s+",
        r"\d+\s+na\s+\d+",
        r"er\s+jaygay",
        r"jaygay\s+\d+",
        r"\bmodify\b",
        r"change\s+koro",
        r"update\s+koro",
        r"khorose\s+\d+.*hobe",
    )
    return any(re.search(p, low) for p in patterns)


def _amount_for_category_in_message(message: str, category: str) -> float | None:
    """Parse amount from the clause that mentions this category."""
    from chat.services.platform.field_extractors.amount import parse_amount
    from chat.services.platform.field_extractors.modify import _category_from_message

    want = normalize_expense_category(category)
    if not want:
        return None
    for clause in split_expense_clauses(message):
        clause_cat = infer_category_from_clause(clause)
        if not clause_cat:
            clause_cat = _category_from_message(clause.lower())
        if normalize_expense_category(clause_cat) != want:
            continue
        amt = parse_amount(clause)
        if amt is not None:
            return amt
    return None


def sanitize_expense_llm_patches(
    turn: dict[str, Any],
    message: str,
    memory=None,
) -> dict[str, Any]:
    """Ground add/append amounts from user text; strip mistaken match_amount on append."""
    if not isinstance(turn, dict) or not turn.get("item_patches"):
        return turn
    intent = str(turn.get("intent") or "").lower()
    if intent in ("update", "modify_review", "correct", "fix_mistake", "clarify_modify"):
        return turn
    if _message_looks_like_amount_correction(message):
        return turn

    from chat.services.platform.field_extractors.amount import parse_amount

    raw_patches = list(turn.get("item_patches") or [])
    append_count = sum(
        1
        for p in raw_patches
        if isinstance(p, dict) and str(p.get("action") or "").lower() in ("append", "add")
    )
    single_msg_amt = parse_amount(message) if append_count == 1 else None
    draft = memory.active_draft() if memory else None
    draft_items = list((draft.fields.get("items") or []) if draft else [])

    patches: list[dict[str, Any]] = []
    for raw in raw_patches:
        if not isinstance(raw, dict):
            patches.append(raw)
            continue
        patch = dict(raw)
        action = str(patch.get("action") or "").strip().lower()
        if action not in ("append", "add"):
            patches.append(patch)
            continue

        patch.pop("match_amount", None)
        patch.pop("match_last", None)
        if intent == "add" and patch.get("item_index") is not None:
            try:
                idx = int(patch["item_index"])
            except (TypeError, ValueError):
                idx = -1
            if idx < 0 or idx >= len(draft_items):
                patch.pop("item_index", None)

        grounded = None
        cat = patch.get("category")
        if cat:
            grounded = _amount_for_category_in_message(message, str(cat))
        if grounded is None and append_count == 1:
            grounded = single_msg_amt
        if grounded is not None:
            patch["amount"] = grounded

        patches.append(patch)

    return {**turn, "item_patches": patches}


def build_wizard_fallback_turn(message: str, memory=None) -> dict[str, Any]:
    """Seed draft items from Banglish clauses when LLM cannot run."""
    from chat.services.platform.field_extractors.amount import parse_amount
    from chat.services.platform.field_extractors.route import parse_route
    from chat.services.platform.intent_rules import is_compound_expense_message, is_expense_message

    raw = (message or "").strip()
    if not raw:
        return {"intent": "conversation", "item_patches": []}
    if expense_turn_blocks_wizard(raw, memory):
        return {"intent": "conversation", "item_patches": []}
    if not is_expense_message(raw) and not is_compound_expense_message(raw):
        return {"intent": "conversation", "item_patches": []}

    patches: list[dict[str, Any]] = []
    seen_amounts: set[float] = set()

    for clause in split_expense_clauses(raw):
        amt = parse_amount(clause)
        if amt is None or amt in seen_amounts:
            continue
        seen_amounts.add(amt)
        cat = infer_category_from_clause(clause)
        patch: dict[str, Any] = {"action": "append", "amount": amt}
        if cat:
            patch["category"] = cat
            route = parse_route(clause)
            if route and is_travel_category(cat):
                frm, to = route
                if is_valid_expense_route(frm, to):
                    patch["from_location"] = frm
                    patch["to_location"] = to
        elif normalize_expense_category(clause.strip()):
            patch["category"] = normalize_expense_category(clause.strip())
        patches.append(patch)

    if not patches:
        amt = parse_amount(raw)
        if amt is not None:
            patches.append({"action": "append", "amount": amt})

    if not patches:
        return {"intent": "conversation", "item_patches": []}

    return {
        "intent": "add",
        "item_patches": patches,
        "wizard_fallback": True,
    }


def _try_wizard_fallback_turn(message: str, memory) -> dict[str, Any] | None:
    turn = build_wizard_fallback_turn(message, memory)
    if _turn_has_actionable_patches(turn):
        return turn
    return None


def _expense_llm_unavailable_turn(*, llm_degraded: bool = True) -> dict[str, Any]:
    """Structured turn when semantic LLM is required but unavailable."""
    return {
        "intent": "llm_unavailable",
        "item_patches": [],
        "delete_indices": [],
        "llm_degraded": llm_degraded,
        "llm_used": False,
    }


def message_has_new_expense_items(message: str) -> bool:
    """True when the user is stating new line items — not bare yes/submit/navigation."""
    from chat.services.platform.intent_rules import (
        is_compound_expense_message,
        is_expense_list_request,
        is_expense_message,
        message_has_banglish_submit_phrase,
    )

    raw = (message or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if is_expense_list_request(raw):
        return False
    taka_count = low.count(" taka") + low.count(" tk") + low.count("টাকা")
    if any(tok in low for tok in ("summery", "summry", "summary")) and "expense" in low:
        if taka_count < 1:
            return False
    if message_has_banglish_submit_phrase(raw) and taka_count < 2 and not is_compound_expense_message(raw):
        return False
    if is_compound_expense_message(raw):
        return True
    return bool(is_expense_message(raw))


def message_requests_submit_after_edit(message: str, *, active_workflow_id: str | None = None) -> bool:
    from chat.services.platform.intent_rules import parse_submit_workflow

    if not message_has_new_expense_items(message):
        return False
    return bool(parse_submit_workflow(message, active_workflow_id=active_workflow_id))


def expense_message_requests_submit(message: str, *, active_workflow_id: str | None = None) -> bool:
    """User wants to submit the active expense draft (not answer a pending slot)."""
    from chat.services.platform.intent_rules import parse_submit_workflow

    return bool(parse_submit_workflow(message, active_workflow_id=active_workflow_id))


def _attempt_wizard_expense_turn(
    message: str,
    memory,
    trace_id: str,
    raw: str,
    *,
    llm_degraded: bool = False,
) -> dict[str, Any] | None:
    wizard = _try_wizard_fallback_turn(message, memory)
    if not wizard or not _turn_has_actionable_patches(wizard):
        return None
    if llm_degraded:
        wizard = {**wizard, "llm_degraded": True}
    if message_requests_submit_after_edit(
        message,
        active_workflow_id=(memory.active_workflow.id if memory and memory.active_workflow else None),
    ):
        wizard = {**wizard, "submit_after_edit": True}
    return _log_and_return_expense_turn(
        trace_id,
        raw,
        wizard,
        llm_used=False,
        wizard_fallback=True,
    )


def expense_turn_llm_blocked(turn: dict[str, Any] | None, memory=None) -> bool:
    """True when semantic edit must use LLM — never apply rules-guessed patches."""
    if not isinstance(turn, dict):
        return False
    if turn.get("wizard_fallback") and _turn_has_actionable_patches(turn):
        return False
    if str(turn.get("intent") or "").lower() == "llm_unavailable":
        return True
    if turn.get("llm_used"):
        return False
    intent = str(turn.get("intent") or "").lower()
    if intent not in ("update", "modify_review", "correct", "clarify_modify"):
        return False
    if intent == "answer_pending":
        return False
    if memory and pending_expense_edit_active(memory):
        return False
    return True


def _expense_rules_fallback_turn(
    message: str,
    memory,
    *,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
    llm_degraded: bool = False,
) -> dict[str, Any]:
    """Rules/coerce fallback when expense LLM unavailable or returned no patches."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    low = raw.lower()

    if message_has_new_expense_items(raw):
        wizard_logged = _attempt_wizard_expense_turn(
            message, memory, trace_id, raw, llm_degraded=llm_degraded
        )
        if wizard_logged is not None:
            return wizard_logged

    def _llm_unavailable_turn() -> dict[str, Any]:
        turn = _expense_llm_unavailable_turn(llm_degraded=True)
        return _log_and_return_expense_turn(trace_id, raw, turn, llm_used=False)

    if memory and is_expense_review_mode(memory):
        from chat.services.platform.intent_rules import (
            is_bare_confirmation,
            parse_submit_workflow,
        )

        active_id = memory.active_workflow.id if memory and memory.active_workflow else ""
        if not message_has_new_expense_items(raw) and (
            is_bare_confirmation(raw) or parse_submit_workflow(raw, active_workflow_id=active_id)
        ):
            turn = {"intent": "confirm", "item_patches": []}
            if llm_degraded:
                turn["llm_degraded"] = True
            return _log_and_return_expense_turn(trace_id, raw, turn, llm_used=False)

    if is_expense_anti_summary_request(raw):
        turn = {"intent": "anti_summary", "item_patches": []}
        if llm_degraded:
            turn["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, turn, llm_used=False)

    if "keno abar" in low or ("eta holo nah" in low and "add" in low):
        turn = _apply_fix_mistake_undo(
            memory,
            {
                "intent": "fix_mistake",
                "item_patches": [{"action": "delete", "match_last": True}],
            },
        )
        if llm_degraded:
            turn["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, turn, llm_used=False)

    add_m = re.search(r"\b(?:add|jog)\s+koro\b", low) or "add kore" in low or "jog kore" in low
    if add_m:
        from chat.services.platform.field_extractors.amount import parse_amount

        amt = parse_amount(raw)
        if amt is not None:
            patches: list[dict[str, Any]] = []
            for cat, keyword in (
                ("lunch", "lunch"),
                ("snack", "snack"),
                ("bus", "bus"),
                ("metro", "metro"),
                ("bike", "bike"),
            ):
                if keyword in low:
                    patches.append({"action": "append", "category": cat, "amount": amt})
                    break
            if patches:
                turn = {"intent": "add", "item_patches": patches}
                if llm_degraded:
                    turn["llm_degraded"] = True
                return _log_and_return_expense_turn(trace_id, raw, turn, llm_used=False)

    if is_expense_draft_mutation_message(message, memory):
        regret_turn = normalize_expense_mutation_turn(
            {"intent": "conversation", "item_patches": []},
            message,
            memory,
        )
        if str(regret_turn.get("intent") or "").lower() == "delete" and (
            regret_turn.get("item_patches") or regret_turn.get("delete_indices")
        ):
            if llm_degraded:
                regret_turn["llm_degraded"] = True
            return _log_and_return_expense_turn(trace_id, raw, regret_turn, llm_used=False)

    delete_turn = coerce_expense_delete_turn(message, memory)
    if delete_turn:
        delete_turn = normalize_expense_delete_turn(delete_turn, message, memory)
        if llm_degraded:
            delete_turn["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, delete_turn, llm_used=False)

    if is_expense_draft_mutation_message(message, memory) and not message_has_new_expense_items(raw):
        return _llm_unavailable_turn()

    coerced = coerce_pending_expense_turn(message, memory)
    if coerced:
        if llm_degraded:
            coerced["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, coerced, llm_used=False)

    pending_edit = resolve_pending_expense_edit_turn(message, memory)
    if pending_edit:
        if llm_degraded:
            pending_edit["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, pending_edit, llm_used=False)

    backfill = infer_expense_slot_from_history(
        memory, conversation_history, trace_id=trace_id
    )
    if backfill:
        if llm_degraded:
            backfill["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, backfill, llm_used=False)

    if pending_expense_edit_active(memory):
        resolved = resolve_pending_expense_edit_turn(message, memory)
        if resolved:
            if llm_degraded:
                resolved["llm_degraded"] = True
            return _log_and_return_expense_turn(trace_id, raw, resolved, llm_used=False)
        pending = pending_expense_edit_from_memory(memory)
        kind = str(pending.get("kind") or "modify")
        return _log_and_return_expense_turn(
            trace_id,
            raw,
            {
                "intent": f"clarify_{kind}",
                "item_patches": [],
                "clarify": pending.get("clarify") or {},
            },
            llm_used=False,
        )

    wizard = _try_wizard_fallback_turn(raw, memory)
    if wizard:
        if llm_degraded:
            wizard["llm_degraded"] = True
        return _log_and_return_expense_turn(
            trace_id, raw, wizard, llm_used=False, wizard_fallback=True
        )

    turn = {"intent": "conversation", "item_patches": []}
    if llm_degraded:
        turn["llm_degraded"] = True
    return _log_and_return_expense_turn(trace_id, raw, turn, llm_used=False)


def interpret_expense_draft_turn(
    message: str,
    memory,
    *,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
    fresh_claim: bool = False,
) -> dict[str, Any]:
    """Single LLM interpreter — intent + draft patches (expense draft editor)."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return {"intent": "conversation", "item_patches": []}

    from chat.services.platform.hr_assistant_scope import resolve_hr_assistant_scope

    scope_oos = resolve_hr_assistant_scope(
        raw,
        memory,
        conversation_history=conversation_history,
        trace_id=trace_id,
    )
    if scope_oos is not None and scope_oos.is_out_of_scope:
        return {"intent": "conversation", "item_patches": [], "off_topic": True}

    from chat.services.platform.intent_rules import is_greeting_or_chitchat

    if is_greeting_or_chitchat(raw):
        return _log_and_return_expense_turn(
            trace_id,
            raw,
            {"intent": "conversation", "item_patches": []},
            llm_used=False,
        )

    from chat.services.platform.intent_rules import is_expense_draft_query

    if is_expense_draft_query(raw):
        return _log_and_return_expense_turn(
            trace_id,
            raw,
            {"intent": "show_summary", "item_patches": []},
            llm_used=False,
        )

    if memory and is_expense_review_mode(memory):
        from chat.services.platform.intent_rules import (
            is_bare_confirmation,
            is_bare_rejection,
            is_workflow_show_request,
            parse_submit_workflow,
        )
        from chat.services.platform.turn_semantics import is_process_question

        active_id = memory.active_workflow.id if memory.active_workflow else ""
        if not message_has_new_expense_items(raw) and (
            is_bare_confirmation(raw) or parse_submit_workflow(raw, active_workflow_id=active_id)
        ):
            return {"intent": "confirm", "item_patches": [], "llm_used": False}
        if is_bare_rejection(raw):
            return {"intent": "cancel", "item_patches": [], "llm_used": False}
        if is_workflow_show_request(raw, workflow_id="expense"):
            return {"intent": "show_summary", "item_patches": [], "llm_used": False}
        if is_process_question(raw):
            return {"intent": "conversation", "item_patches": [], "llm_used": False}

    if is_expense_draft_mutation_message(message, memory):
        regret_turn = normalize_expense_mutation_turn(
            {"intent": "conversation", "item_patches": []},
            message,
            memory,
        )
        if str(regret_turn.get("intent") or "").lower() == "delete" and (
            regret_turn.get("item_patches") or regret_turn.get("delete_indices")
        ):
            return _log_and_return_expense_turn(trace_id, raw, regret_turn, llm_used=False)

    coerced_pending = coerce_pending_expense_turn(message, memory)
    if coerced_pending:
        return _log_and_return_expense_turn(trace_id, raw, coerced_pending, llm_used=False)

    active_wf = memory.active_workflow.id if memory and memory.active_workflow else "expense"
    if expense_message_requests_submit(raw, active_workflow_id=active_wf):
        return _log_and_return_expense_turn(
            trace_id,
            raw,
            {"intent": "confirm", "item_patches": [], "llm_used": False},
            llm_used=False,
        )

    from datetime import date

    today = date.today().isoformat()
    if not expense_turn_is_draft_mutation(None, message):
        sem = interpret_expense_turn_semantics(
            message,
            memory,
            None,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
        if (sem.get("replay_blocked_add") or sem.get("date_correction")) and get_expense_blocked_add(
            memory
        ):
            replayed = _replay_blocked_expense_turn(
                {"intent": "add", "item_patches": []},
                memory,
                sem,
                today=today,
            )
            if replayed:
                return _log_and_return_expense_turn(
                    trace_id,
                    raw,
                    replayed,
                    llm_used=True,
                )

    from chat.services.llm_client import (
        expense_llm_done,
        llm_rate_limit_active,
        mark_expense_llm_done,
        peek_expense_turn_cache,
        stash_expense_turn_cache,
    )

    if expense_llm_done(trace_id or ""):
        cached = peek_expense_turn_cache(trace_id or "")
        if isinstance(cached, dict) and cached:
            cached = dict(cached)
            cached["llm_used"] = True
            if not expense_turn_is_draft_mutation(cached, raw):
                cached, _ = coerce_expense_date_policy(
                    cached,
                    raw,
                    memory=memory,
                    trace_id=trace_id,
                    conversation_history=conversation_history,
                )
            return _log_and_return_expense_turn(
                trace_id,
                raw,
                cached,
                llm_used=True,
                wizard_fallback=bool(cached.get("wizard_fallback")),
            )
        blocked = None
        if not expense_turn_is_draft_mutation(None, raw):
            blocked = expense_date_policy_block_turn(
                raw,
                memory=memory,
                trace_id=trace_id,
                conversation_history=conversation_history,
            )
        if blocked:
            blocked = {**blocked, "llm_used": True}
            if (trace_id or "").strip():
                stash_expense_turn_cache(trace_id, blocked)
            return _log_and_return_expense_turn(
                trace_id,
                raw,
                blocked,
                llm_used=True,
            )
        wizard_logged = _attempt_wizard_expense_turn(
            message, memory, trace_id, raw, llm_degraded=True
        )
        if wizard_logged is not None:
            return wizard_logged
        return _expense_rules_fallback_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            llm_degraded=True,
        )

    if not _llm_client_configured() or llm_rate_limit_active(trace_id or "", scope="expense-draft"):
        import logging

        logger = logging.getLogger("hr_chatbot")
        reason = "not_configured" if not _llm_client_configured() else "rate_limit"
        logger.warning(
            "expense_draft_llm_skipped trace_id=%s reason=%s message_len=%s",
            trace_id,
            reason,
            len(raw),
        )
        wizard_logged = _attempt_wizard_expense_turn(
            message, memory, trace_id, raw, llm_degraded=True
        )
        if wizard_logged is not None:
            return wizard_logged
        return _expense_rules_fallback_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            llm_degraded=True,
        )

    import json
    from datetime import date

    from chat.services.llm_client import LLMClient
    from chat.services.platform.llm_prompts import EXPENSE_DRAFT_INTERPRETER_SYSTEM_COMPACT

    today = date.today().isoformat()
    payload = expense_draft_llm_user_payload(
        raw, memory, today_iso=today, fresh_claim=fresh_claim
    )
    from django.conf import settings as django_settings

    expense_model = getattr(django_settings, "LLM_EXPENSE_MODEL", None) or None
    parsed = LLMClient().chat_json(
        system_prompt=EXPENSE_DRAFT_INTERPRETER_SYSTEM_COMPACT,
        user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
        trace_id=trace_id or "",
        scope="expense-draft",
        model=expense_model,
    )
    if not isinstance(parsed, dict):
        return _expense_rules_fallback_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            llm_degraded=True,
        )
    else:
        intent = str(parsed.get("intent") or "add").strip().lower()
        turn = {
            "intent": intent,
            "incurred_date": parsed.get("incurred_date"),
            "item_patches": list(parsed.get("item_patches") or []),
            "delete_indices": list(parsed.get("delete_indices") or []),
            "clarify": parsed.get("clarify") if isinstance(parsed.get("clarify"), dict) else None,
            "reasoning": str(parsed.get("reasoning") or ""),
        }

    turn = sanitize_expense_llm_patches(turn, message, memory)
    turn = _apply_fix_mistake_undo(memory, turn)
    turn = normalize_pending_edit_turn(turn, message, memory)
    turn = normalize_expense_clarify_turn(turn, message, memory)
    turn = normalize_expense_mutation_turn(turn, message, memory)
    turn = _normalize_expense_delete_turn(turn, raw, memory, trace_id=trace_id)
    turn = normalize_expense_delete_turn(turn, raw, memory)
    turn = _merge_clarify_delete_with_rules(turn, raw, memory)
    turn = finalize_expense_turn_patches(turn, message, memory)
    date_rejected = False
    if expense_turn_is_new_add(turn, raw):
        turn, date_rejected = coerce_expense_date_policy(
            turn,
            raw,
            memory=memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
    elif not turn.get("incurred_date"):
        turn = {**turn, "incurred_date": today}
    if date_rejected:
        if (trace_id or "").strip():
            stash_expense_turn_cache(trace_id, turn)
            mark_expense_llm_done(trace_id)
        return _log_and_return_expense_turn(
            trace_id,
            raw,
            turn,
            llm_used=True,
            wizard_fallback=bool(turn.get("wizard_fallback")),
        )
    if turn.get("delete_indices") and str(turn.get("intent") or "").lower() == "clarify_delete":
        turn = {**turn, "intent": "delete"}

    if (
        str(turn.get("intent") or "").lower() == "clarify_delete"
        and not _turn_has_actionable_patches(turn)
    ):
        if (trace_id or "").strip():
            mark_expense_llm_done(trace_id)
        return _log_and_return_expense_turn(
            trace_id,
            raw,
            turn,
            llm_used=True,
            wizard_fallback=bool(turn.get("wizard_fallback")),
        )

    if not _turn_has_actionable_patches(turn):
        wizard_logged = _attempt_wizard_expense_turn(
            message, memory, trace_id, raw, llm_degraded=bool(turn.get("llm_degraded"))
        )
        if wizard_logged is not None:
            return wizard_logged
        return _expense_rules_fallback_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            llm_degraded=bool(turn.get("llm_degraded")),
        )

    if (trace_id or "").strip():
        stash_expense_turn_cache(trace_id, turn)
        mark_expense_llm_done(trace_id)
    return _log_and_return_expense_turn(
        trace_id,
        raw,
        turn,
        llm_used=True,
        wizard_fallback=bool(turn.get("wizard_fallback")),
    )


def interpret_expense_delete_indices_llm(
    message: str,
    items: list[dict[str, Any]],
    *,
    trace_id: str = "",
) -> list[int]:
    """LLM resolves 1-based / range delete phrasing to 0-based indices."""
    raw = (message or "").strip()
    if not raw or not items:
        return []
    payload = {
        "message": raw,
        "item_count": len(items),
        "items": [
            {
                "index": i,
                "line": i + 1,
                "category": it.get("category"),
                "amount": it.get("amount"),
            }
            for i, it in enumerate(items)
        ],
    }
    try:
        import json

        from chat.services.llm_client import LLMClient
        from chat.services.platform.llm_prompts import EXPENSE_DELETE_INDICES_SYSTEM

        parsed = LLMClient().chat_json(
            system_prompt=EXPENSE_DELETE_INDICES_SYSTEM,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            trace_id=trace_id or "",
            scope="expense-delete",
        )
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return []
    out: list[int] = []
    for raw_idx in parsed.get("delete_indices") or []:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items):
            out.append(idx)
    return sorted(set(out))


def _normalize_expense_delete_turn(
    turn: dict[str, Any],
    message: str,
    memory,
    *,
    trace_id: str = "",
) -> dict[str, Any]:
    """Ground delete_indices from user numbering — LLM for ranges and Banglish variants."""
    if str(turn.get("intent") or "").lower() != "delete":
        return turn
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return turn

    llm_indices = interpret_expense_delete_indices_llm(
        message,
        items,
        trace_id=trace_id,
    )
    if llm_indices:
        return {
            **turn,
            "delete_indices": llm_indices,
            "item_patches": [{"action": "delete", "item_index": i} for i in llm_indices],
        }

    from chat.services.platform.field_extractors.modify import (
        _numbered_item_index,
        parse_delete_request,
    )

    parsed = parse_delete_request(message, items)
    idx: int | None = None
    if parsed and parsed.get("item_index") is not None:
        idx = int(parsed["item_index"])
    else:
        numbered = _numbered_item_index(message, item_count=len(items))
        if numbered is not None:
            idx = numbered

    if idx is not None and 0 <= idx < len(items):
        return {
            **turn,
            "delete_indices": [idx],
            "item_patches": [{"action": "delete", "item_index": idx}],
        }

    valid: list[int] = []
    for raw_idx in turn.get("delete_indices") or []:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items):
            valid.append(idx)
    if valid:
        merged = sorted(set(valid))
        patches = [{"action": "delete", "item_index": i} for i in merged]
        return {**turn, "delete_indices": merged, "item_patches": patches}
    return turn


def _ground_expense_delete_field_updates(
    message: str,
    items: list[dict[str, Any]],
) -> list:
    """Rules-grounded delete updates for review / reducer."""
    from chat.services.platform.field_extractors.modify import (
        _numbered_item_index,
        parse_delete_request,
    )
    from chat.services.platform.intent_rules import is_delete_request
    from chat.services.platform.schemas import FieldUpdate

    if not items or not is_delete_request(message):
        return []

    parsed = parse_delete_request(message, items)
    if parsed and parsed.get("item_index") is not None:
        idx = int(parsed["item_index"])
        if 0 <= idx < len(items):
            return [
                FieldUpdate(field="items", value={}, item_index=idx, action="delete")
            ]

    numbered = _numbered_item_index(message, item_count=len(items))
    if numbered is not None:
        return [
            FieldUpdate(
                field="items", value={}, item_index=numbered, action="delete"
            )
        ]
    return []


def _log_and_return_expense_turn(
    trace_id: str,
    message: str,
    turn: dict[str, Any],
    *,
    llm_used: bool,
    wizard_fallback: bool = False,
) -> dict[str, Any]:
    from chat.services.observability import log_expense_draft_turn

    log_expense_draft_turn(
        trace_id,
        message=message,
        turn=turn,
        llm_used=llm_used,
        wizard_fallback=wizard_fallback or bool(turn.get("wizard_fallback")),
    )
    if llm_used:
        turn = {**turn, "llm_used": True}
    return turn


def expense_turn_to_field_updates(
    message: str,
    memory,
    *,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
    expense_turn: dict[str, Any] | None = None,
    fresh_claim: bool = False,
) -> tuple[dict[str, Any], list]:
    if isinstance(expense_turn, dict) and expense_turn:
        turn = dict(expense_turn)
    else:
        turn = interpret_expense_draft_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            fresh_claim=fresh_claim,
        )
    turn = normalize_expense_mutation_turn(turn, message, memory)
    turn = sanitize_expense_llm_patches(turn, message, memory)
    if str(turn.get("intent") or "").lower() != "date_not_allowed" and not turn.get(
        "date_policy_rejected"
    ):
        turn = finalize_expense_turn_patches(turn, message, memory)
    date_rejected = False
    if expense_turn_is_new_add(turn, message):
        turn, date_rejected = coerce_expense_date_policy(
            turn,
            message,
            memory=memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
    intent = str(turn.get("intent") or "").lower()
    if date_rejected or intent == "date_not_allowed":
        return turn, []
    if intent in ("date_correction", "replay_blocked_add"):
        pass
    elif intent == "llm_unavailable":
        return turn, []
    if intent == "clarify_delete" and not turn.get("item_patches") and not turn.get("delete_indices"):
        return turn, []
    if intent == "clarify_modify" and not turn.get("item_patches"):
        return turn, []

    draft = memory.active_draft()
    if draft and draft.workflow_id == "expense":
        fields = dict(draft.fields or {})
    else:
        fields = {"items": []}
    pq = memory.pending_question
    pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
    updates: list[FieldUpdate] = patches_to_field_updates(
        fields,
        turn,
        pending_item_index=pending_idx,
    )
    if intent == "delete":
        updates = [u for u in updates if getattr(u, "action", None) == "delete"]
    return turn, updates


def _coerce_item_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    item: dict[str, Any] = {}
    cat = normalize_expense_category(raw.get("category"))
    if cat:
        item["category"] = cat
    amt = _coerce_amount(raw.get("amount"))
    if amt is not None:
        item["amount"] = amt
    route = _coerce_route_dict(raw)
    item.update(route)
    if cat and is_travel_category(cat) and not is_valid_expense_route(
        item.get("from_location"), item.get("to_location")
    ):
        item.pop("from_location", None)
        item.pop("to_location", None)
    if raw.get("description"):
        item["description"] = str(raw["description"]).strip()[:120]
    if item.get("amount") is not None or item.get("category"):
        return item
    if is_valid_expense_route(item.get("from_location"), item.get("to_location")):
        return item
    return None


# --- Expense draft editor (state sync, pending queue, patch merge) ---

_FIELD_PRIORITY = ("category", "amount", "route")


def ensure_item_ids(items: list[dict[str, Any]]) -> None:
    for item in items:
        if isinstance(item, dict) and not item.get("id"):
            item["id"] = uuid.uuid4().hex[:8]


def compute_item_missing_fields(item: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not normalize_expense_category(item.get("category")):
        missing.append("category")
    try:
        amt = float(item.get("amount") or 0)
    except (TypeError, ValueError):
        amt = 0.0
    if amt <= 0:
        missing.append("amount")
    cat = normalize_expense_category(item.get("category"))
    if cat and is_travel_category(cat):
        if not is_valid_expense_route(item.get("from_location"), item.get("to_location")):
            missing.append("route")
    return missing


def _items_fingerprint(item: dict[str, Any]) -> tuple[Any, ...]:
    cat = normalize_expense_category(item.get("category")) or ""
    try:
        amt = round(float(item.get("amount") or 0), 2)
    except (TypeError, ValueError):
        amt = 0.0
    frm = str(item.get("from_location") or "").strip().lower()
    to = str(item.get("to_location") or "").strip().lower()
    return (cat, amt, frm, to)


_EXPENSE_SEMANTICS_CACHE: dict[str, dict[str, Any]] = {}


def clear_expense_semantics_cache(trace_id: str = "") -> None:
    key = (trace_id or "").strip()
    if not key:
        _EXPENSE_SEMANTICS_CACHE.clear()
        return
    _EXPENSE_SEMANTICS_CACHE.pop(key, None)


def get_expense_blocked_add(memory) -> dict[str, Any] | None:
    """Last compound add blocked by date policy — replay on correction / 'last expense'."""
    if not memory:
        return None
    raw = (memory.last_entities or {}).get("expense_blocked_add")
    return dict(raw) if isinstance(raw, dict) and raw.get("item_patches") else None


def stash_expense_blocked_add(memory, turn: dict[str, Any], message: str) -> None:
    if not memory:
        return
    patches = [
        dict(p)
        for p in (turn.get("item_patches") or [])
        if isinstance(p, dict) and _patch_body(p)
    ]
    if not patches:
        return
    memory.last_entities = {
        **(memory.last_entities or {}),
        "expense_blocked_add": {
            "message": (message or "").strip()[:500],
            "item_patches": patches,
            "submit_after_edit": bool(turn.get("submit_after_edit")),
            "incurred_date": turn.get("incurred_date"),
        },
    }


def clear_expense_blocked_add(memory) -> None:
    if not memory or not memory.last_entities:
        return
    ents = dict(memory.last_entities)
    if "expense_blocked_add" in ents:
        ents.pop("expense_blocked_add", None)
        memory.last_entities = ents


def _blocked_add_summary(blocked: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not blocked:
        return []
    out: list[dict[str, Any]] = []
    for p in blocked.get("item_patches") or []:
        if not isinstance(p, dict):
            continue
        body = _patch_body(p)
        if body:
            out.append(body)
    return out


def interpret_expense_turn_semantics(
    message: str,
    memory,
    turn: dict[str, Any] | None = None,
    *,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
) -> dict[str, Any]:
    """LLM-only date / correction / replay semantics for expense turns."""
    from datetime import date

    key = (trace_id or "").strip()
    if key and key in _EXPENSE_SEMANTICS_CACHE:
        return dict(_EXPENSE_SEMANTICS_CACHE[key])

    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    today = date.today().isoformat()
    blocked = get_expense_blocked_add(memory)
    payload: dict[str, Any] = {
        "message": raw,
        "today_iso": today,
        "draft_incurred_date": None,
        "turn_incurred_date": (turn or {}).get("incurred_date"),
        "blocked_add": _blocked_add_summary(blocked),
        "has_blocked_add": bool(blocked and blocked.get("item_patches")),
        "pending_question": None,
        "recent_user_messages": list(conversation_history or [])[-4:],
    }
    draft = memory.active_draft() if memory else None
    if draft and draft.workflow_id == "expense":
        fields = dict(draft.fields or {})
        if fields.get("incurred_date"):
            payload["draft_incurred_date"] = fields.get("incurred_date")
    pq = memory.pending_question if memory else None
    if pq and pq.workflow_id == "expense":
        payload["pending_question"] = {
            "field": pq.field,
            "item_index": pq.item_index,
        }

    default: dict[str, Any] = {
        "date_effect": "unspecified",
        "date_correction": False,
        "replay_blocked_add": False,
        "incurred_date_iso": None,
        "reasoning": "",
    }
    if not raw:
        if key:
            _EXPENSE_SEMANTICS_CACHE[key] = dict(default)
        return default

    try:
        import json

        from chat.services.llm_client import LLMClient
        from chat.services.platform.llm_prompts import EXPENSE_TURN_SEMANTICS_SYSTEM

        parsed = LLMClient().chat_json(
            system_prompt=EXPENSE_TURN_SEMANTICS_SYSTEM,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            trace_id=trace_id or "",
            scope="expense-semantics",
        )
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        if key:
            _EXPENSE_SEMANTICS_CACHE[key] = dict(default)
        return default

    out = dict(default)
    effect = str(parsed.get("date_effect") or "unspecified").strip().lower()
    if effect in ("today", "non_today", "unspecified"):
        out["date_effect"] = effect
    out["date_correction"] = bool(parsed.get("date_correction"))
    out["replay_blocked_add"] = bool(parsed.get("replay_blocked_add"))
    raw_iso = parsed.get("incurred_date_iso")
    if raw_iso not in (None, ""):
        from chat.services.platform.field_extractors.leave import _coerce_llm_date_output

        iso = _coerce_llm_date_output(raw_iso)
        if iso:
            out["incurred_date_iso"] = iso
    out["reasoning"] = str(parsed.get("reasoning") or "").strip()[:200]
    if key:
        _EXPENSE_SEMANTICS_CACHE[key] = dict(out)
    return out


def expense_message_mentions_non_today_date(message: str) -> bool:
    """Deprecated — use interpret_expense_turn_semantics (LLM). Kept for import compat."""
    return False


def expense_date_policy_block_turn(
    message: str,
    turn: dict[str, Any] | None = None,
    *,
    memory=None,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return a date_not_allowed turn when message/turn requests a non-today date."""
    blocked, rejected = coerce_expense_date_policy(
        dict(turn or {"intent": "add", "item_patches": []}),
        message,
        memory=memory,
        trace_id=trace_id,
        conversation_history=conversation_history,
    )
    return blocked if rejected else None


def expense_requested_date_not_today(turn: dict[str, Any] | None) -> bool:
    """True when LLM parsed incurred_date to a day other than today."""
    from datetime import date

    from chat.services.platform.field_extractors.leave import _coerce_llm_date_output

    if not isinstance(turn, dict):
        return False
    raw = turn.get("incurred_date")
    if raw in (None, ""):
        return False
    iso = _coerce_llm_date_output(raw)
    if not iso:
        return False
    return iso != date.today().isoformat()


def _replay_blocked_expense_turn(
    turn: dict[str, Any],
    memory,
    semantics: dict[str, Any],
    *,
    today: str,
) -> dict[str, Any] | None:
    blocked = get_expense_blocked_add(memory)
    patches: list[dict[str, Any]] = []
    submit_after = False
    if blocked:
        patches = [
            dict(p)
            for p in (blocked.get("item_patches") or [])
            if isinstance(p, dict)
        ]
        submit_after = bool(blocked.get("submit_after_edit"))
    if not patches:
        patches = [
            dict(p)
            for p in (turn.get("item_patches") or [])
            if isinstance(p, dict)
        ]
    if not patches:
        return None
    intent = "replay_blocked_add" if semantics.get("replay_blocked_add") else "date_correction"
    return {
        **turn,
        "intent": intent,
        "item_patches": patches,
        "incurred_date": today,
        "delete_indices": [],
        "clarify": {},
        "date_policy_rejected": False,
        "submit_after_edit": submit_after or bool(turn.get("submit_after_edit")),
    }


def coerce_expense_date_policy(
    turn: dict[str, Any],
    message: str,
    *,
    memory=None,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Block non-today dates (LLM) on new adds only; replay stashed add on correction."""
    from datetime import date

    from chat.services.platform.field_extractors.leave import _coerce_llm_date_output

    out = dict(turn or {})
    today = date.today().isoformat()

    active_wf = memory.active_workflow.id if memory and memory.active_workflow else "expense"
    if expense_message_requests_submit(message, active_workflow_id=active_wf):
        return {**out, "intent": "confirm", "item_patches": [], "delete_indices": []}, False

    if expense_turn_is_draft_mutation(out, message):
        return out, False

    semantics = interpret_expense_turn_semantics(
        message,
        memory,
        out,
        trace_id=trace_id,
        conversation_history=conversation_history,
    )

    if semantics.get("replay_blocked_add") or semantics.get("date_correction"):
        replayed = _replay_blocked_expense_turn(out, memory, semantics, today=today)
        if replayed:
            return replayed, False

    if not expense_turn_is_new_add(out, message):
        if not out.get("incurred_date"):
            out["incurred_date"] = today
        return out, False

    sem_iso = semantics.get("incurred_date_iso")
    turn_iso = _coerce_llm_date_output(out.get("incurred_date")) if out.get("incurred_date") else None
    sem_coerced = _coerce_llm_date_output(sem_iso) if sem_iso else None
    effective_iso = sem_coerced or turn_iso
    date_effect = str(semantics.get("date_effect") or "unspecified").lower()

    rejected = False
    if date_effect == "non_today":
        rejected = True
    elif effective_iso and effective_iso != today:
        rejected = True

    if rejected:
        append_patches = [
            dict(p)
            for p in (out.get("item_patches") or [])
            if isinstance(p, dict)
            and str(p.get("action") or "append").lower() in ("append", "add", "")
            and _patch_body(p)
        ]
        if append_patches:
            stash_expense_blocked_add(
                memory,
                {**out, "item_patches": append_patches},
                message,
            )
        requested = effective_iso
        out = {
            **out,
            "intent": "date_not_allowed",
            "date_policy_rejected": True,
            "past_date_rejected": True,
            "rejected_incurred_date": requested,
            "item_patches": [],
            "delete_indices": [],
            "submit_after_edit": False,
            "clarify": {},
        }
        out.pop("incurred_date", None)
        return out, True

    if not out.get("incurred_date"):
        out["incurred_date"] = today
    else:
        iso = _coerce_llm_date_output(out.get("incurred_date"))
        out["incurred_date"] = iso or today
    return out, False


def expense_turn_fully_redundant(turn: dict[str, Any] | None, memory) -> bool:
    """True when every patch in the turn already exists on the draft (repeat message)."""
    if not isinstance(turn, dict) or not memory:
        return False
    draft = memory.active_draft()
    if not draft or draft.workflow_id != "expense":
        return False
    items = list((draft.fields or {}).get("items") or [])
    if not items:
        return False
    patches = [
        p
        for p in (turn.get("item_patches") or [])
        if isinstance(p, dict)
        and str(p.get("action") or "append").lower() in ("append", "update", "correct", "")
    ]
    if not patches:
        return False
    for patch in patches:
        body = _patch_body(patch)
        if not body:
            continue
        idx = resolve_item_index(items, patch)
        if idx is not None:
            existing = _semantic_item_dict(items[idx])
            proposed = _semantic_item_dict({**items[idx], **body})
            if existing == proposed:
                continue
            return False
        if _find_duplicate_item_index(items, body) is None:
            return False
    return True


def _semantic_item_dict(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in item.items():
        if k in ("id", "missing_fields", "status", "currency"):
            continue
        if k == "amount":
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


def _find_duplicate_item_index(
    items: list[dict[str, Any]],
    body: dict[str, Any],
) -> int | None:
    """Match append candidates to existing rows — avoid duplicate line items."""
    if not body:
        return None
    body_fp = _items_fingerprint(body)
    body_cat, body_amt, body_frm, body_to = body_fp
    matches: list[int] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        fp = _items_fingerprint(item)
        cat, amt, frm, to = fp
        if body_cat and cat and body_cat == cat and abs(body_amt - amt) < 0.01:
            if body_frm and body_to:
                if frm == body_frm and to == body_to:
                    matches.append(idx)
            elif not body_frm and not body_to:
                matches.append(idx)
            continue
        if not body_cat and not cat and body_amt > 0 and abs(body_amt - amt) < 0.01:
            matches.append(idx)
            continue
        if body_amt > 0 and abs(body_amt - amt) < 0.01 and (not body_cat or not cat):
            if body_cat == cat or (not body_cat or not cat):
                matches.append(idx)
    if len(matches) == 1:
        return matches[0]
    return None


def dedupe_expense_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep all line items — duplicates are allowed unless user deletes one."""
    return [dict(i) for i in items if isinstance(i, dict)]


def sync_expense_draft_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Recompute per-item missing_fields and status — never drop existing data."""
    out = dict(fields or {})
    items = dedupe_expense_items([dict(i) for i in (out.get("items") or []) if isinstance(i, dict)])
    ensure_item_ids(items)
    for item in items:
        missing = compute_item_missing_fields(item)
        item["missing_fields"] = missing
        item["status"] = "complete" if not missing else "incomplete"
        item.setdefault("currency", "BDT")
    out["items"] = items
    return out


def sync_expense_draft(draft: WorkflowDraft) -> None:
    draft.fields = sync_expense_draft_fields(draft.fields)
    draft.line_items = list(draft.fields.get("items") or [])


@dataclass
class PendingQueueEntry:
    item_index: int
    field: str
    item_id: str
    item: dict[str, Any]


def build_pending_queue(items: list[dict[str, Any]]) -> list[PendingQueueEntry]:
    """Oldest incomplete expense first; one field per item (required before optional)."""
    queue: list[PendingQueueEntry] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        missing = list(item.get("missing_fields") or compute_item_missing_fields(item))
        if not missing:
            continue
        for field in _FIELD_PRIORITY:
            if field in missing:
                queue.append(
                    PendingQueueEntry(
                        item_index=idx,
                        field=field,
                        item_id=str(item.get("id") or ""),
                        item=item,
                    )
                )
                break
    return queue


def expense_draft_missing_fields(draft: WorkflowDraft) -> list[str]:
    fields = sync_expense_draft_fields(dict(draft.fields or {}))
    missing: list[str] = []
    if fields.get("incurred_date") in (None, ""):
        missing.append("incurred_date")
    items = fields.get("items") or []
    if not items:
        missing.append("items")
    for entry in build_pending_queue(items):
        slot = "item_category" if entry.field == "category" else (
            "item_route" if entry.field == "route" else f"item_{entry.field}"
        )
        missing.append(f"items[{entry.item_index}].{slot}")
    return missing


def resolve_item_index(
    items: list[dict[str, Any]],
    patch: dict[str, Any],
    *,
    pending_item_index: int | None = None,
) -> int | None:
    if patch.get("item_index") is not None:
        idx = int(patch["item_index"])
        if 0 <= idx < len(items):
            return idx
    item_id = str(patch.get("item_id") or "").strip()
    if item_id:
        for i, it in enumerate(items):
            if str(it.get("id") or "") == item_id:
                return i
    if patch.get("match_amount") is not None:
        try:
            target = float(patch["match_amount"])
        except (TypeError, ValueError):
            target = None
        if target is not None:
            matches = [
                i
                for i, it in enumerate(items)
                if abs(float(it.get("amount") or 0) - target) < 0.01
            ]
            if len(matches) == 1:
                return matches[0]
    if patch.get("match_last"):
        return len(items) - 1 if items else None
    return None


def _patch_body(patch: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    coerced = _coerce_item_dict(patch)
    if coerced:
        body.update(coerced)
    for key in ("category", "amount", "from_location", "to_location", "description"):
        if key in patch and patch[key] not in (None, "") and key not in body:
            if key == "category":
                cat = normalize_expense_category(patch[key])
                if cat:
                    body["category"] = cat
            elif key == "amount":
                try:
                    amt = float(patch[key])
                    if amt > 0:
                        body["amount"] = amt
                except (TypeError, ValueError):
                    pass
            else:
                body[key] = str(patch[key]).strip()[:120]
    return body


def apply_expense_patches(
    fields: dict[str, Any],
    turn: dict[str, Any],
    *,
    pending_item_index: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Merge LLM patches into draft fields. Returns (fields, human notes)."""
    out = sync_expense_draft_fields(dict(fields or {}))
    notes: list[str] = []
    intent = str(turn.get("intent") or "").strip().lower()

    if turn.get("incurred_date"):
        from chat.services.platform.field_extractors.leave import _coerce_llm_date_output

        iso = _coerce_llm_date_output(turn.get("incurred_date"))
        if iso:
            out["incurred_date"] = iso
            notes.append("incurred_date")

    items = list(out.get("items") or [])
    missing_field = _pending_missing_field(out, pending_item_index)

    raw_delete_indices = list(turn.get("delete_indices") or [])
    patch_deletes = [
        p
        for p in (turn.get("item_patches") or [])
        if isinstance(p, dict) and str(p.get("action") or "").lower() == "delete"
    ]
    if raw_delete_indices and not patch_deletes:
        for idx in sorted(
            {int(i) for i in raw_delete_indices if str(i).lstrip("-").isdigit()},
            reverse=True,
        ):
            if 0 <= idx < len(items):
                items.pop(idx)
                notes.append(f"deleted item {idx + 1}")

    for raw_patch in turn.get("item_patches") or []:
        if not isinstance(raw_patch, dict):
            continue
        patch = dict(raw_patch)
        action = str(patch.get("action") or "update").strip().lower()

        if intent == "answer_pending":
            if action == "append":
                continue
            if patch.get("item_index") is None and pending_item_index is not None:
                patch["item_index"] = pending_item_index
            if action in ("update", "correct", "") and patch.get("item_index") is not None:
                action = "update"

        body = _filter_patch_body_for_pending(
            _patch_body(patch),
            missing_field=missing_field if intent == "answer_pending" else None,
            intent=intent,
        )
        if action == "delete":
            idx = resolve_item_index(items, patch, pending_item_index=pending_item_index)
            if idx is not None:
                items.pop(idx)
                notes.append(f"deleted item {idx + 1}")
            continue
        if action == "append":
            if intent == "add":
                if body:
                    items.append(body)
                    notes.append(f"added item {len(items)}")
                continue
            idx = resolve_item_index(items, patch, pending_item_index=pending_item_index)
            if idx is not None and body:
                items[idx] = {**items[idx], **body}
                notes.append(f"updated item {idx + 1}")
                continue
            if body:
                items.append(body)
                notes.append(f"added item {len(items)}")
            continue
        idx = resolve_item_index(items, patch, pending_item_index=pending_item_index)
        if idx is not None and body:
            items[idx] = {**items[idx], **body}
            notes.append(f"updated item {idx + 1}")
        elif body and action in ("update", "correct"):
            items.append(body)
            notes.append(f"added item {len(items)}")

    out["items"] = items
    return sync_expense_draft_fields(out), notes


def patches_to_field_updates(
    fields: dict[str, Any],
    turn: dict[str, Any],
    *,
    pending_item_index: int | None = None,
) -> list[FieldUpdate]:
    """Convert LLM turn patches to reducer FieldUpdates."""
    intent = str(turn.get("intent") or "").strip().lower()
    if intent in ("clarify_delete", "clarify_modify") and not turn.get("item_patches"):
        return []

    old_items = list(fields.get("items") or [])

    if intent == "delete":
        delete_indices: list[int] = []
        for raw in turn.get("delete_indices") or []:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(old_items):
                delete_indices.append(idx)
        for patch in turn.get("item_patches") or []:
            if not isinstance(patch, dict):
                continue
            if str(patch.get("action") or "").lower() != "delete":
                continue
            idx = resolve_item_index(old_items, patch, pending_item_index=pending_item_index)
            if idx is not None:
                delete_indices.append(idx)
        if delete_indices:
            return [
                FieldUpdate(field="items", value={}, item_index=idx, action="delete")
                for idx in sorted(set(delete_indices), reverse=True)
            ]
        return []

    merged, _ = apply_expense_patches(fields, turn, pending_item_index=pending_item_index)
    updates: list[FieldUpdate] = []
    if merged.get("incurred_date") and merged.get("incurred_date") != fields.get("incurred_date"):
        updates.append(FieldUpdate(field="incurred_date", value=merged["incurred_date"], action="set"))

    old_items = list(fields.get("items") or [])
    new_items = list(merged.get("items") or [])

    def _semantic_item(item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in item.items():
            if k in ("id", "missing_fields", "status", "currency"):
                continue
            if k == "amount":
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            else:
                out[k] = v
        return out

    patches = list(turn.get("item_patches") or [])
    deletes_only = len(new_items) < len(old_items) and all(
        str(p.get("action") or "").lower() == "delete" for p in patches
    )

    if len(new_items) > len(old_items):
        for item in new_items[len(old_items) :]:
            updates.append(FieldUpdate(field="items", value=dict(item), action="append"))
    elif len(new_items) < len(old_items):
        delete_indices: list[int] = []
        for patch in patches:
            if str(patch.get("action") or "").lower() != "delete":
                continue
            idx = resolve_item_index(old_items, patch, pending_item_index=pending_item_index)
            if idx is not None:
                delete_indices.append(idx)
        if delete_indices:
            for idx in sorted(set(delete_indices), reverse=True):
                updates.append(
                    FieldUpdate(field="items", value={}, item_index=idx, action="delete")
                )
        else:
            for idx in range(len(old_items) - 1, len(new_items) - 1, -1):
                updates.append(
                    FieldUpdate(field="items", value={}, item_index=idx, action="delete")
                )
    if deletes_only:
        return updates
    if intent == "add" and len(new_items) <= len(old_items):
        append_patches = [
            p
            for p in patches
            if isinstance(p, dict)
            and str(p.get("action") or "").lower() in ("append", "add")
        ]
        if append_patches and len(append_patches) == len(
            [p for p in patches if isinstance(p, dict)]
        ):
            add_updates: list[FieldUpdate] = list(updates)
            for patch in append_patches:
                body = _patch_body(patch)
                if body:
                    add_updates.append(FieldUpdate(field="items", value=body, action="append"))
            if add_updates:
                return add_updates
    for idx, item in enumerate(new_items):
        if idx >= len(old_items):
            continue
        if _semantic_item(item) != _semantic_item(old_items[idx]):
            updates.append(
                FieldUpdate(
                    field="items",
                    value=_semantic_item(item),
                    item_index=idx,
                    action="update",
                )
            )
    return updates


def expense_item_gaps(items: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Return (item_index, pending_field) for each incomplete item."""
    gaps: list[tuple[int, str]] = []
    for idx, raw in enumerate(items or []):
        if not isinstance(raw, dict):
            continue
        if not normalize_expense_category(raw.get("category")):
            gaps.append((idx, "item_category"))
            continue
        cat = normalize_expense_category(raw.get("category"))
        if cat and is_travel_category(cat):
            if not is_valid_expense_route(raw.get("from_location"), raw.get("to_location")):
                gaps.append((idx, "item_route"))
    return gaps


def is_expense_collect_mode(memory) -> bool:
    """True while expense draft is being built (not submit review)."""
    if is_expense_review_mode(memory):
        return False
    aw = memory.active_workflow if memory else None
    return bool(aw and aw.id == "expense")


def is_expense_review_mode(memory) -> bool:
    aw = memory.active_workflow
    if not aw or aw.id != "expense":
        return False
    if (memory.pending_confirmation or "") == "submit":
        return True
    return aw.stage == "confirm_submit"


def is_expense_review_edit_turn(
    message: str,
    memory,
    understanding=None,
) -> bool:
    """User is editing expense lines at submit review — not switching workflows."""
    if not is_expense_review_mode(memory):
        return False
    from chat.services.platform.field_extractors.modify import (
        looks_like_expense_item_delete,
        looks_like_expense_route_modify,
        parse_delete_request,
        parse_modify_request,
        parse_route_modify_request,
    )
    from chat.services.platform.intent_rules import is_delete_request, is_modify_request

    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])

    if is_expense_regret_remove_message(message) or is_expense_draft_mutation_message(message):
        return True
    if is_delete_request(message) or looks_like_expense_item_delete(message):
        return True
    if is_modify_request(message) or looks_like_expense_route_modify(message):
        return True
    if items and parse_delete_request(message, items):
        return True
    if items and parse_modify_request(message, items):
        return True
    if items and parse_route_modify_request(message, items):
        return True
    if understanding is not None:
        intent = str((understanding.entities or {}).get("expense_intent") or "").lower()
        expense_turn = (understanding.entities or {}).get("expense_turn")
        from chat.services.platform.field_extractors.expense import (
            expense_turn_has_targeted_patches,
        )

        if expense_turn_has_targeted_patches(expense_turn, memory):
            return True
        if intent in (
            "delete",
            "update",
            "modify_review",
            "answer_pending",
            "clarify_delete",
            "clarify_modify",
            "fix_mistake",
        ):
            if intent != "answer_pending" or understanding.field_updates:
                return True
        if understanding.action in ("delete", "modify"):
            return True
        if understanding.field_updates and understanding.workflow == "expense":
            return True
    return False


def _resolve_review_item_index(
    message: str,
    items: list[dict],
    *,
    fallback: int | None = None,
) -> int | None:
    from chat.services.platform.field_extractors.modify import (
        _numbered_item_index,
        resolve_expense_item_reference,
    )

    numbered = _numbered_item_index(message, item_count=len(items))
    if numbered is not None:
        return numbered
    resolved = resolve_expense_item_reference(message, items)
    if resolved.get("item_index") is not None:
        return int(resolved["item_index"])
    return fallback


def sanitize_expense_review_updates(
    updates: list,
    message: str,
    *,
    memory,
) -> list:
    """Keep only grounded partial item patches during submit review."""
    from chat.services.platform.field_extractors.modify import _extract_modify_amount
    from chat.services.platform.field_extractors.route import parse_route
    from chat.services.platform.schemas import FieldUpdate

    if not is_expense_review_mode(memory):
        return list(updates or [])

    draft = memory.active_draft()
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return []

    append_updates = [
        u
        for u in (updates or [])
        if getattr(u, "field", None) == "items" and getattr(u, "action", None) == "append"
    ]
    if append_updates and not expense_turn_blocks_wizard(message, memory):
        return append_updates

    from chat.services.platform.intent_rules import is_delete_request

    delete_updates = [
        u
        for u in (updates or [])
        if getattr(u, "field", None) == "items" and getattr(u, "action", None) == "delete"
    ]
    if delete_updates or is_delete_request(message):
        grounded = _ground_expense_delete_field_updates(message, items)
        if grounded:
            return grounded

    item_num = None
    num_m = re.search(r"\b(\d+)\s*(?:no|number|nombor|numer)\b", (message or "").lower())
    if num_m:
        item_num = int(num_m.group(1))
    msg_amount = _extract_modify_amount(message, item_number_1based=item_num)
    route = parse_route(message)
    route_only = bool(route) and msg_amount is None

    sanitized: list[FieldUpdate] = []
    for upd in updates or []:
        if upd.field != "items" or upd.action not in ("update", "update_last"):
            continue
        idx = _resolve_review_item_index(message, items, fallback=upd.item_index)
        if idx is None or not (0 <= idx < len(items)):
            continue
        body = dict(upd.value) if isinstance(upd.value, dict) else {}
        body.pop("description", None)
        if route_only and route:
            body = {"from_location": route[0], "to_location": route[1]}
        else:
            if msg_amount is not None:
                body["amount"] = msg_amount
            elif "amount" in body and not re.search(
                r"\b(\d+(?:\.\d+)?)\s*(?:taka|tk|টাকা)\b", (message or "").lower()
            ):
                body.pop("amount", None)
            if route:
                body["from_location"] = route[0]
                body["to_location"] = route[1]
        if not body:
            continue
        sanitized.append(
            FieldUpdate(field="items", value=body, item_index=idx, action="update")
        )
    return sanitized


def review_field_updates_from_message(
    message: str,
    memory,
    *,
    trace_id: str = "",
    understanding_updates: list | None = None,
    expense_turn: dict[str, Any] | None = None,
) -> list:
    """Primary expense review edit path — sanitize LLM patches, then rules-first parsers."""
    from chat.services.platform.intent_rules import (
        is_compound_expense_message,
        is_expense_message,
    )

    turn_hint = expense_turn if isinstance(expense_turn, dict) else {}
    llm_turn = turn_hint if expense_turn_blocks_wizard(message, memory, turn=turn_hint) else None

    if llm_turn and expense_turn_has_targeted_patches(llm_turn, memory):
        corrected = coerce_expense_correction_turn(llm_turn, memory)
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        pq = memory.pending_question if memory else None
        pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
        targeted = patches_to_field_updates(
            fields,
            corrected,
            pending_item_index=pending_idx,
        )
        if targeted:
            return targeted

    delete_turn = coerce_expense_delete_turn(message, memory)
    if not delete_turn and llm_turn and str(llm_turn.get("intent") or "").lower() == "delete":
        delete_turn = llm_turn
    if delete_turn and delete_turn.get("item_patches"):
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        return patches_to_field_updates(fields, delete_turn)

    route_turn = coerce_expense_route_modify_turn(message, memory)
    if route_turn and route_turn.get("item_patches") and not _llm_client_configured():
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        return patches_to_field_updates(fields, route_turn)

    modify_turn = coerce_expense_modify_turn(message, memory)
    if not modify_turn and llm_turn and str(llm_turn.get("intent") or "").lower() in (
        "update",
        "modify_review",
    ):
        modify_turn = llm_turn
    if modify_turn and modify_turn.get("item_patches") and not _llm_client_configured():
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        return patches_to_field_updates(fields, modify_turn)

    if not expense_turn_blocks_wizard(message, memory, turn=turn_hint):
        if is_expense_message(message) or is_compound_expense_message(message):
            wizard = _try_wizard_fallback_turn(message, memory)
            if wizard and _turn_has_actionable_patches(wizard):
                draft = memory.active_draft()
                fields = dict(draft.fields or {}) if draft else {"items": []}
                return patches_to_field_updates(fields, wizard)

    sanitized = sanitize_expense_review_updates(
        list(understanding_updates or []),
        message,
        memory=memory,
    )
    if sanitized:
        return sanitized

    from chat.services.platform.intent_rules import is_delete_request

    draft = memory.active_draft()
    items = list((draft.fields.get("items") or []) if draft else [])
    if is_delete_request(message) and items:
        grounded = _ground_expense_delete_field_updates(message, items)
        if grounded:
            return grounded

    _, updates = expense_turn_to_field_updates(
        message,
        memory,
        trace_id=trace_id,
    )
    return sanitize_expense_review_updates(updates, message, memory=memory)


def filter_expense_updates_for_review(
    updates: list,
    message: str,
    *,
    memory,
    trace_id: str = "",
    expense_turn: dict[str, Any] | None = None,
) -> list:
    if not is_expense_review_mode(memory):
        return list(updates or [])

    turn_hint = expense_turn if isinstance(expense_turn, dict) else {}
    turn_intent = str(turn_hint.get("intent") or "").lower()
    if turn_intent == "clarify_delete":
        return []

    if expense_turn_llm_blocked(turn_hint, memory):
        return []

    if expense_turn_has_targeted_patches(turn_hint, memory):
        corrected = coerce_expense_correction_turn(turn_hint, memory)
        draft = memory.active_draft() if memory else None
        fields = dict(draft.fields or {}) if draft else {"items": []}
        pq = memory.pending_question if memory else None
        pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
        targeted = patches_to_field_updates(
            fields,
            corrected,
            pending_item_index=pending_idx,
        )
        if targeted:
            return targeted

    if turn_intent in ("update", "modify_review", "correct", "fix_mistake"):
        modify_updates = [
            u
            for u in (updates or [])
            if getattr(u, "field", None) == "items"
            and getattr(u, "action", None) in ("update", "update_last")
        ]
        if modify_updates:
            return modify_updates
        draft = memory.active_draft() if memory else None
        fields = dict(draft.fields or {}) if draft else {"items": []}
        pq = memory.pending_question if memory else None
        pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
        return patches_to_field_updates(
            fields,
            turn_hint,
            pending_item_index=pending_idx,
        )

    delete_updates = [
        u
        for u in (updates or [])
        if getattr(u, "field", None) == "items" and getattr(u, "action", None) == "delete"
    ]
    if delete_updates:
        return delete_updates

    modify_updates = [
        u
        for u in (updates or [])
        if getattr(u, "field", None) == "items"
        and getattr(u, "action", None) in ("update", "update_last")
    ]
    if modify_updates and expense_turn_blocks_wizard(message, memory, turn=expense_turn):
        return modify_updates

    if expense_turn_blocks_wizard(message, memory, turn=expense_turn):
        return review_field_updates_from_message(
            message,
            memory,
            trace_id=trace_id,
            understanding_updates=updates,
            expense_turn=expense_turn,
        )

    append_updates = [
        u
        for u in (updates or [])
        if getattr(u, "field", None) == "items" and getattr(u, "action", None) == "append"
    ]
    if append_updates:
        return append_updates

    return review_field_updates_from_message(
        message,
        memory,
        trace_id=trace_id,
        understanding_updates=updates,
        expense_turn=expense_turn,
    )


def expense_pending_interrupted_by_updates(updates, memory) -> bool:
    """True when applied item ops target a line other than the open pending slot."""
    pq = memory.pending_question if memory else None
    if not pq or pq.workflow_id != "expense" or pq.item_index is None:
        return False
    pending_idx = int(pq.item_index)
    for u in updates or []:
        if getattr(u, "field", None) != "items":
            continue
        action = str(getattr(u, "action", None) or "").lower()
        idx = getattr(u, "item_index", None)
        if action == "delete":
            return True
        if action in ("update", "update_last") and idx is not None:
            if int(idx) != pending_idx:
                return True
    return False


def normalize_expense_clarify_turn(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """When user names a single category + route, skip ambiguous clarify_modify."""
    if str(turn.get("intent") or "").lower() != "clarify_modify":
        return turn
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.field_extractors.modify import (
        _category_from_message,
        _indices_for_category,
    )
    from chat.services.platform.field_extractors.route import parse_route

    raw = normalize_banglish_message((message or "").strip())
    msg_cat = _category_from_message(raw.lower())
    if not msg_cat:
        return turn
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return turn
    indices = _indices_for_category(items, msg_cat)
    clarify = turn.get("clarify") if isinstance(turn.get("clarify"), dict) else {}
    candidates = list(clarify.get("candidate_indices") or [])
    if candidates:
        indices = [i for i in indices if i in candidates]
    if len(indices) != 1:
        return turn
    route = parse_route(raw) or parse_route(message)
    if not route:
        return turn
    frm, to = route
    if not is_valid_expense_route(frm, to):
        return turn
    idx = indices[0]
    intent = "modify_review" if is_expense_review_mode(memory) else "update"
    return {
        **turn,
        "intent": intent,
        "item_patches": [
            {
                "action": "update",
                "item_index": idx,
                "from_location": frm,
                "to_location": to,
            }
        ],
        "clarify": {},
        "delete_indices": [],
    }


def normalize_pending_edit_turn(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """After LLM: explicit add wins; otherwise resolve pending clarify selection."""
    if not memory or not pending_expense_edit_active(memory):
        return turn
    intent = str(turn.get("intent") or "").lower()
    patches = list(turn.get("item_patches") or [])
    if intent == "add" and any(
        str(p.get("action") or "").lower() in ("append", "add")
        for p in patches
        if isinstance(p, dict)
    ):
        return turn
    if intent in ("modify_review", "update", "delete", "correct") and _turn_has_actionable_patches(
        turn
    ):
        return turn
    resolved = resolve_pending_expense_edit_turn(message, memory)
    if resolved:
        return resolved
    return turn


def pending_expense_edit_active(memory) -> bool:
    pending = pending_expense_edit_from_memory(memory)
    return str(pending.get("kind") or "").strip().lower() in ("delete", "modify")


def pending_expense_edit_from_memory(memory) -> dict[str, Any]:
    """Active clarify-delete/modify context from session state."""
    if not memory:
        return {}
    pending = dict((memory.last_entities or {}).get("expense_pending_edit") or {})
    if pending.get("kind"):
        return pending
    last_intent = str((memory.last_entities or {}).get("expense_intent") or "").lower()
    if last_intent in ("clarify_delete", "clarify_modify"):
        turn = dict((memory.last_entities or {}).get("expense_turn") or {})
        built = build_expense_pending_edit_from_turn(turn)
        if built:
            return built
    return {}


def _apply_stored_pending_modify(
    stored_message: str,
    idx: int,
    memory,
    *,
    clarify: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Apply the original modify request to a user-selected item index."""
    from chat.services.platform.field_extractors.modify import parse_modify_request
    from chat.services.platform.field_extractors.route import parse_route

    stored = (stored_message or "").strip()
    if not stored:
        return None
    intent = "modify_review" if is_expense_review_mode(memory) else "update"
    patch: dict[str, Any] = {"action": "update", "item_index": idx}
    clarify = clarify or {}

    route = parse_route(stored)
    if route:
        frm, to = route
        if is_valid_expense_route(frm, to):
            patch["from_location"] = frm
            patch["to_location"] = to
            return {"intent": intent, "item_patches": [patch]}

    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    parsed = parse_modify_request(stored, items)
    if parsed and parsed.get("item_index") is not None and not parsed.get("needs_clarify"):
        if parsed.get("amount") is not None:
            patch["amount"] = float(parsed["amount"])
        if parsed.get("category"):
            patch["category"] = parsed["category"]
        return {"intent": intent, "item_patches": [patch]}

    if clarify.get("proposed_value") is not None:
        try:
            patch["amount"] = float(clarify["proposed_value"])
        except (TypeError, ValueError):
            return None
        if clarify.get("category"):
            patch["category"] = clarify["category"]
        return {"intent": intent, "item_patches": [patch]}
    return None


def _merge_clarify_delete_with_rules(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """Prefer rules-grounded category candidates over LLM listing every line."""
    intent = str(turn.get("intent") or "").strip().lower()
    if intent != "clarify_delete":
        return turn
    coerced = coerce_expense_delete_turn(message, memory)
    if not coerced or coerced.get("intent") != "clarify_delete":
        return turn
    coerce_clarify = coerced.get("clarify") if isinstance(coerced.get("clarify"), dict) else {}
    coerce_candidates = list(coerce_clarify.get("candidate_indices") or [])
    if not coerce_candidates:
        return turn
    clarify = dict(turn.get("clarify") or {})
    clarify["candidate_indices"] = coerce_candidates
    if coerce_clarify.get("category"):
        clarify["category"] = coerce_clarify["category"]
    return {**turn, "clarify": clarify}


def resolve_pending_expense_edit_turn(message: str, memory) -> dict[str, Any] | None:
    """Apply numbered reply after clarify_delete / clarify_modify."""
    if not memory:
        return None
    pending = pending_expense_edit_from_memory(memory)
    kind = str(pending.get("kind") or "").strip().lower()
    if kind not in ("delete", "modify"):
        return None
    draft = memory.active_draft()
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return None

    from chat.services.platform.field_extractors.modify import (
        _numbered_item_index,
        _extract_modify_amount,
        parse_multiple_item_indices,
        parse_route_modify_request,
    )

    raw = (message or "").strip()
    if not raw:
        return None

    clarify = dict(pending.get("clarify") or {})
    candidates = [
        int(i)
        for i in (clarify.get("candidate_indices") or [])
        if isinstance(i, (int, float)) or str(i).isdigit()
    ]
    candidate_arg = candidates if candidates else None

    indices = parse_multiple_item_indices(
        raw,
        item_count=len(items),
        candidate_indices=candidate_arg,
    )
    idx: int | None = indices[0] if len(indices) == 1 else None
    if idx is None and not indices:
        idx = _numbered_item_index(raw, item_count=len(items))
        if idx is None:
            m = re.search(r"^\s*(\d+)\s*\.?\s*$", raw)
            if m:
                cand = int(m.group(1)) - 1
                if 0 <= cand < len(items):
                    idx = cand
        if idx is None:
            return None
        if candidates and idx not in candidates:
            return None

    if kind == "delete":
        if len(indices) > 1:
            return {
                "intent": "delete",
                "item_patches": [
                    {"action": "delete", "item_index": i} for i in indices
                ],
                "delete_indices": indices,
            }
        assert idx is not None
        return {
            "intent": "delete",
            "item_patches": [{"action": "delete", "item_index": idx}],
            "delete_indices": [idx],
        }

    assert idx is not None
    stored_msg = str(pending.get("message") or "").strip()
    if stored_msg:
        applied = _apply_stored_pending_modify(
            stored_msg,
            idx,
            memory,
            clarify=clarify,
        )
        if applied:
            return applied

    route_turn = parse_route_modify_request(raw, items)
    if route_turn and int(route_turn.get("item_index", -1)) == idx:
        return {
            "intent": "modify_review",
            "item_patches": [
                {
                    "action": "update",
                    "item_index": idx,
                    "from_location": route_turn["from_location"],
                    "to_location": route_turn["to_location"],
                }
            ],
        }
    amount = _extract_modify_amount(raw, item_number_1based=idx + 1)
    if amount is None and clarify.get("proposed_value") is not None:
        try:
            amount = float(clarify["proposed_value"])
        except (TypeError, ValueError):
            amount = None
    if amount is None:
        return None
    patch: dict[str, Any] = {"action": "update", "item_index": idx, "amount": amount}
    if clarify.get("category"):
        patch["category"] = clarify["category"]
    return {"intent": "modify_review", "item_patches": [patch]}


def build_expense_pending_edit_from_turn(
    turn: dict[str, Any],
    *,
    message: str = "",
    memory=None,
) -> dict[str, Any] | None:
    """Remember an in-flight clarify-modify/delete turn for the next user reply."""
    intent = str(turn.get("intent") or "").strip().lower()
    if intent not in ("clarify_modify", "clarify_delete"):
        return None
    clarify = turn.get("clarify") if isinstance(turn.get("clarify"), dict) else {}
    new_msg = (message or "").strip()
    existing = (
        dict((memory.last_entities or {}).get("expense_pending_edit") or {})
        if memory
        else {}
    )
    prev_msg = str(existing.get("message") or "").strip()
    stored_msg = prev_msg or new_msg
    payload: dict[str, Any] = {
        "kind": intent.replace("clarify_", ""),
        "message": stored_msg,
    }
    if clarify:
        payload["clarify"] = dict(clarify)
    return payload


def _llm_client_configured() -> bool:
    from chat.services.llm_client import LLMClient

    return LLMClient().is_configured()


def expense_field_updates_from_message(
    message: str,
    *,
    memory=None,
    trace_id: str = "",
    conversation_history: list[str] | None = None,
) -> list:
    from chat.services.platform.schemas import FieldUpdate

    if memory is None:
        return []
    _, updates = expense_turn_to_field_updates(
        message,
        memory,
        trace_id=trace_id,
        conversation_history=conversation_history,
    )
    return updates


def expense_fields_from_message(
    message: str,
    memory=None,
    *,
    trace_id: str = "",
) -> dict[str, Any]:
    """Build expense field dict from the unified draft interpreter."""
    if memory is None:
        from chat.services.session_memory import SessionMemory

        memory = SessionMemory()
    turn, updates = expense_turn_to_field_updates(message, memory, trace_id=trace_id)
    out: dict[str, Any] = {}
    if turn.get("incurred_date"):
        from chat.services.platform.field_extractors.leave import _coerce_llm_date_output

        iso = _coerce_llm_date_output(turn.get("incurred_date"))
        if iso:
            out["incurred_date"] = iso
    items: list[dict[str, Any]] = []
    for upd in updates:
        if upd.field == "items" and upd.action == "append" and isinstance(upd.value, dict):
            coerced = _coerce_item_dict(upd.value)
            if coerced:
                items.append(coerced)
    if items:
        out["items"] = items
    return out


def expense_fields_for_submit(fields: dict[str, Any]) -> dict[str, Any]:
    return dict(fields or {})


def expense_item_label(item: dict[str, Any], *, index: int) -> str:
    """Human label for LLM context and focused prompts."""
    cat = category_display_name(normalize_expense_category(item.get("category")) or "?")
    amt = item.get("amount")
    amt_part = f"{amt} taka" if amt is not None else "? taka"
    line = f"Expense {index + 1} — {cat} — {amt_part}"
    if is_travel_category(item.get("category")) and is_valid_expense_route(
        item.get("from_location"), item.get("to_location")
    ):
        line += f" ({item.get('from_location')} → {item.get('to_location')})"
    return line


def build_pending_focus(memory) -> dict[str, Any] | None:
    """First incomplete expense item — anchor for LLM pending answers."""
    draft = memory.active_draft() if memory else None
    if not draft:
        return None
    fields = sync_expense_draft_fields(dict(draft.fields or {}))
    queue = build_pending_queue(fields.get("items") or [])
    if not queue:
        return None
    entry = queue[0]
    item = entry.item
    pq = memory.pending_question if memory else None
    return {
        "item_index": entry.item_index,
        "item_id": entry.item_id,
        "category": normalize_expense_category(item.get("category")),
        "amount": item.get("amount"),
        "missing": list(item.get("missing_fields") or compute_item_missing_fields(item)),
        "missing_field": entry.field,
        "label": expense_item_label(item, index=entry.item_index),
        "last_bot_question": pq.prompt if pq and pq.workflow_id == "expense" else "",
    }


def expense_focus_prompt(
    entry: PendingQueueEntry,
    *,
    lang: str = "en",
) -> str:
    """One focused follow-up tied to a specific draft line."""
    item = entry.item
    label = expense_item_label(item, index=entry.item_index)
    if entry.field == "category":
        if lang == "bn":
            return f"{label}: category ki chilo? (Lunch, Snack, Bus, …) মনে না থাকলে **remove** বলুন।"
        return f"{label}: what category was it? (Lunch, Snack, Bus, …) Say **remove** to drop it."
    if entry.field == "route":
        if lang == "bn":
            return f"{label}: kothay theke kothay travel korlen?"
        return f"{label}: where did you travel from and to?"
    if entry.field == "amount":
        if lang == "bn":
            return f"{label}: amount koto taka?"
        return f"{label}: what was the amount in taka?"
    return expense_item_prompt(
        "item_category" if entry.field == "category" else (
            "item_route" if entry.field == "route" else "item_amount"
        ),
        item_index=entry.item_index,
        lang=lang,
        item=item,
    )


def expense_item_prompt(
    field: str,
    *,
    item_index: int = 0,
    lang: str = "en",
    item: dict[str, Any] | None = None,
) -> str:
    n = item_index + 1
    ordinal = {1: "first", 2: "second", 3: "third"}.get(n, f"{n}th")
    amt = (item or {}).get("amount")
    amt_note = f" ({amt} taka)" if amt is not None else ""
    if field == "item_category":
        if lang == "bn":
            base = f"খরচের ধরন কী ছিল{amt_note}? (Lunch, Snack, Bus, Train, Metro, Rickshaw, Bike)"
            return f"{base} মনে না থাকলে **remove** বলুন।"
        base = f"What type of expense was it{amt_note}? (Lunch, Snack, Bus, Train, Metro, Rickshaw, Bike)"
        return f"{base} Say **remove** if you want to drop it."
    if field == "item_route":
        if lang == "bn":
            return f"{ordinal} expense-এর route কোথা থেকে কোথায়?"
        return f"Where did you travel from and to for the {ordinal} expense?"
    if field == "item_amount":
        if lang == "bn":
            return f"{ordinal} expense-এর amount কত taka?"
        return f"What is the amount for the {ordinal} expense?"
    return field.replace("_", " ")


def next_pending_question(
    memory: SessionMemory,
    *,
    lang: str = "en",
) -> PendingQuestion | None:
    draft = memory.active_draft()
    if not draft:
        return None
    fields = sync_expense_draft_fields(dict(draft.fields or {}))
    if fields.get("incurred_date") in (None, ""):
        prompt = (
            "খরচ কখন হয়েছে?"
            if lang == "bn"
            else "When was this expense incurred?"
        )
        return PendingQuestion(
            field="incurred_date",
            prompt=prompt,
            workflow_id="expense",
            asked_at_turn=memory.turn_count,
        )
    items = fields.get("items") or []
    if not items:
        prompt = (
            "খরচের বিবরণ দিন — category ও amount।"
            if lang == "bn"
            else "Tell me about the expense — category and amount."
        )
        return PendingQuestion(
            field="items",
            prompt=prompt,
            workflow_id="expense",
            asked_at_turn=memory.turn_count,
        )
    queue = build_pending_queue(items)
    if not queue:
        return None
    entry = queue[0]
    slot = "item_category" if entry.field == "category" else (
        "item_route" if entry.field == "route" else "item_amount"
    )
    prompt = expense_focus_prompt(entry, lang=lang)
    return PendingQuestion(
        field=slot,
        prompt=prompt,
        workflow_id="expense",
        asked_at_turn=memory.turn_count,
        item_index=entry.item_index,
    )


def compact_draft_items_for_llm(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Token-light item rows for expense LLM (no long descriptions)."""
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {"i": idx}
        cat = normalize_expense_category(item.get("category"))
        if cat:
            row["cat"] = cat
        try:
            amt = float(item.get("amount") or 0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt > 0:
            row["amt"] = round(amt, 2)
        frm = str(item.get("from_location") or "").strip()
        to = str(item.get("to_location") or "").strip()
        if frm:
            row["from"] = frm
        if to:
            row["to"] = to
        missing = list(item.get("missing_fields") or compute_item_missing_fields(item))
        if missing:
            row["missing"] = missing
        item_id = item.get("id")
        if item_id:
            row["id"] = item_id
        rows.append(row)
    return rows


def expense_draft_llm_user_payload(
    message: str,
    memory: SessionMemory,
    *,
    today_iso: str,
    fresh_claim: bool = False,
) -> dict[str, Any]:
    """Minimal JSON for expense-draft LLM — keeps input tokens low."""
    aw = memory.active_workflow.id if memory.active_workflow else ""
    if fresh_claim or aw != "expense":
        return {
            "message": message,
            "today_iso": today_iso,
            "stage": "collecting",
            "pending_confirmation": "",
            "items": [],
        }
    draft = draft_context_payload(memory, compact=True)
    stage = str(draft.get("stage") or "")
    items = list(draft.get("items") or [])
    if stage == "submitted":
        items = []
    payload: dict[str, Any] = {
        "message": message,
        "today_iso": today_iso,
        "stage": stage,
        "pending_confirmation": draft.get("pending_confirmation") or "",
        "items": items,
    }
    if draft.get("incurred_date"):
        payload["incurred_date"] = draft["incurred_date"]
    if draft.get("pending_question"):
        payload["pending_question"] = draft["pending_question"]
    if draft.get("expense_pending_edit"):
        payload["expense_pending_edit"] = draft["expense_pending_edit"]
    if draft.get("pending_focus"):
        payload["pending_focus"] = draft["pending_focus"]
    blocked = get_expense_blocked_add(memory)
    if blocked:
        payload["blocked_add"] = _blocked_add_summary(blocked)
    return payload


def draft_context_payload(memory: SessionMemory, *, compact: bool = True) -> dict[str, Any]:
    draft = memory.active_draft()
    fields = sync_expense_draft_fields(dict(draft.fields if draft else {}))
    items = list(fields.get("items") or [])
    pq = memory.pending_question
    focus = build_pending_focus(memory)
    aw = memory.active_workflow
    stage = (aw.stage if aw and aw.id == "expense" else "") or ""
    if (memory.pending_confirmation or "") == "submit":
        stage = "confirm_submit"
    pending_edit = pending_expense_edit_from_memory(memory)
    pending_edit_ctx = None
    if pending_edit.get("kind"):
        pending_edit_ctx = {
            "kind": pending_edit.get("kind"),
            "message": pending_edit.get("message"),
            "clarify": pending_edit.get("clarify"),
        }

    if compact:
        pending_q = None
        if pq and pq.workflow_id == "expense":
            pending_q = {
                "field": pq.field,
                "item_index": pq.item_index,
            }
        return {
            "stage": stage,
            "pending_confirmation": memory.pending_confirmation or "",
            "incurred_date": fields.get("incurred_date"),
            "items": compact_draft_items_for_llm(items),
            "pending_focus": focus,
            "pending_question": pending_q,
            "expense_pending_edit": pending_edit_ctx,
            "pending_queue": [
                {"i": e.item_index, "field": e.field}
                for e in build_pending_queue(items)
            ],
        }

    labeled_items = [
        {
            **item,
            "label": expense_item_label(item, index=idx),
        }
        for idx, item in enumerate(items)
        if isinstance(item, dict)
    ]
    focus = build_pending_focus(memory)
    return {
        "draft_fields": fields,
        "draft_items": labeled_items,
        "pending_queue": [
            {
                "item_index": e.item_index,
                "field": e.field,
                "item_id": e.item_id,
                "label": expense_item_label(e.item, index=e.item_index),
            }
            for e in build_pending_queue(items)
        ],
        "pending_focus": focus,
        "expense_pending_edit": pending_edit_ctx,
        "pending_question": {
            "field": pq.field,
            "item_index": pq.item_index,
            "prompt": pq.prompt,
        }
        if pq and pq.workflow_id == "expense"
        else None,
    }
