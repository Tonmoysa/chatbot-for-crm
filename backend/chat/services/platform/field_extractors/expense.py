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
            return {
                "intent": "clarify_delete",
                "item_patches": [],
                "candidate_indices": list(parsed.get("candidate_indices") or []),
                "category": parsed.get("category") or parsed.get("label"),
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


def coerce_pending_expense_turn(
    message: str,
    memory,
) -> dict[str, Any] | None:
    """Obvious pending-slot answers — canonical enum/amount only, not narrative parsing."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message

    pq = memory.pending_question if memory else None
    if not pq or pq.workflow_id != "expense" or pq.item_index is None:
        return None
    from chat.services.platform.intent_rules import is_expense_add_request

    if is_expense_add_request(message):
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
    if intent in ("fix_mistake", "answer_pending", "add", "update", "delete", "correct"):
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
    for sep in ("\n", ";"):
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


def build_wizard_fallback_turn(message: str, memory=None) -> dict[str, Any]:
    """Seed draft items from Banglish clauses when LLM cannot run."""
    from chat.services.platform.field_extractors.amount import parse_amount
    from chat.services.platform.field_extractors.route import parse_route
    from chat.services.platform.intent_rules import is_compound_expense_message, is_expense_message

    _ = memory
    raw = (message or "").strip()
    if not raw:
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

    if memory and is_expense_review_mode(memory):
        from chat.services.platform.intent_rules import (
            is_bare_confirmation,
            parse_submit_workflow,
        )

        active_id = memory.active_workflow.id if memory and memory.active_workflow else ""
        if is_bare_confirmation(raw) or parse_submit_workflow(raw, active_workflow_id=active_id):
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

    coerced = coerce_pending_expense_turn(message, memory)
    if coerced:
        if llm_degraded:
            coerced["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, coerced, llm_used=False)

    modify_turn = coerce_expense_modify_turn(message, memory)
    if modify_turn:
        if llm_degraded:
            modify_turn["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, modify_turn, llm_used=False)

    route_turn = coerce_expense_route_modify_turn(message, memory)
    if route_turn:
        if llm_degraded:
            route_turn["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, route_turn, llm_used=False)

    delete_turn = coerce_expense_delete_turn(message, memory)
    if delete_turn:
        if llm_degraded:
            delete_turn["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, delete_turn, llm_used=False)

    backfill = infer_expense_slot_from_history(
        memory, conversation_history, trace_id=trace_id
    )
    if backfill:
        if llm_degraded:
            backfill["llm_degraded"] = True
        return _log_and_return_expense_turn(trace_id, raw, backfill, llm_used=False)

    if not (memory and is_expense_review_mode(memory)):
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
) -> dict[str, Any]:
    """Single LLM interpreter — intent + draft patches (expense draft editor)."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.turn_semantics import (
        expense_conversation_payload,
        understanding_session_context,
    )

    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return {"intent": "conversation", "item_patches": []}

    from chat.services.platform.intent_rules import (
        is_clearly_off_hr_question,
        is_off_hr_topic_message,
        is_programming_question,
        is_workflow_turn_message,
    )

    active_id = memory.active_workflow.id if memory and memory.active_workflow else ""
    if is_programming_question(raw):
        off_topic = True
    elif active_id and is_workflow_turn_message(raw, memory=memory):
        off_topic = False
    elif is_clearly_off_hr_question(raw) or is_off_hr_topic_message(raw, memory=memory):
        off_topic = True
    else:
        off_topic = False

    if off_topic:
        return {"intent": "conversation", "item_patches": [], "off_topic": True}

    if memory and is_expense_review_mode(memory):
        from chat.services.platform.intent_rules import (
            is_bare_confirmation,
            is_bare_rejection,
            is_workflow_show_request,
            parse_submit_workflow,
        )
        from chat.services.platform.turn_semantics import is_process_question

        active_id = memory.active_workflow.id if memory.active_workflow else ""
        if is_bare_confirmation(raw) or parse_submit_workflow(raw, active_workflow_id=active_id):
            return {"intent": "confirm", "item_patches": [], "llm_used": False}
        if is_bare_rejection(raw):
            return {"intent": "cancel", "item_patches": [], "llm_used": False}
        if is_workflow_show_request(raw, workflow_id="expense"):
            return {"intent": "show_summary", "item_patches": [], "llm_used": False}
        if is_process_question(raw):
            return {"intent": "conversation", "item_patches": [], "llm_used": False}

    coerced_pending = coerce_pending_expense_turn(message, memory)
    if coerced_pending:
        return _log_and_return_expense_turn(trace_id, raw, coerced_pending, llm_used=False)

    from chat.services.llm_client import llm_rate_limit_active

    if not _llm_client_configured() or llm_rate_limit_active(trace_id or "", scope="expense-draft"):
        return _expense_rules_fallback_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            llm_degraded=_llm_client_configured(),
        )

    import json
    from datetime import date

    from chat.services.llm_client import LLMClient
    from chat.services.platform.llm_prompts import EXPENSE_DRAFT_INTERPRETER_SYSTEM

    history = list(conversation_history or ())
    ctx = understanding_session_context(memory, history)
    payload = {
        "message": raw,
        "today_iso": date.today().isoformat(),
        **draft_context_payload(memory),
        **expense_conversation_payload(conversation_history, limit=3),
        **ctx,
    }
    from django.conf import settings as django_settings

    expense_model = getattr(django_settings, "LLM_EXPENSE_MODEL", None) or None
    parsed = LLMClient().chat_json(
        system_prompt=EXPENSE_DRAFT_INTERPRETER_SYSTEM,
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

    turn = _apply_fix_mistake_undo(memory, turn)
    turn = _normalize_expense_delete_turn(turn, raw, memory)

    if not _turn_has_actionable_patches(turn):
        return _expense_rules_fallback_turn(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history,
            llm_degraded=bool(turn.get("llm_degraded")),
        )

    return _log_and_return_expense_turn(
        trace_id,
        raw,
        turn,
        llm_used=True,
        wizard_fallback=bool(turn.get("wizard_fallback")),
    )


def _normalize_expense_delete_turn(
    turn: dict[str, Any],
    message: str,
    memory,
) -> dict[str, Any]:
    """Ground delete_indices from user numbering — LLM often returns 0 for '4 number'."""
    if str(turn.get("intent") or "").lower() != "delete":
        return turn
    draft = memory.active_draft() if memory else None
    items = list((draft.fields.get("items") or []) if draft else [])
    if not items:
        return turn

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
    for raw in turn.get("delete_indices") or []:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items):
            valid.append(idx)
    if valid:
        patches = [{"action": "delete", "item_index": i} for i in sorted(set(valid))]
        return {**turn, "delete_indices": sorted(set(valid)), "item_patches": patches}
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
) -> tuple[dict[str, Any], list]:
    turn = interpret_expense_draft_turn(
        message,
        memory,
        trace_id=trace_id,
        conversation_history=conversation_history,
    )
    draft = memory.active_draft()
    fields = dict(draft.fields or {}) if draft else {"items": []}
    pq = memory.pending_question
    pending_idx = pq.item_index if pq and pq.workflow_id == "expense" else None
    updates: list[FieldUpdate] = patches_to_field_updates(
        fields,
        turn,
        pending_item_index=pending_idx,
    )
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
    if pending_item_index is not None and 0 <= pending_item_index < len(items):
        return pending_item_index
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
) -> list:
    """Primary expense review edit path — sanitize LLM patches, then rules-first parsers."""
    delete_turn = coerce_expense_delete_turn(message, memory)
    if delete_turn and delete_turn.get("item_patches"):
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        return patches_to_field_updates(fields, delete_turn)

    sanitized = sanitize_expense_review_updates(
        list(understanding_updates or []),
        message,
        memory=memory,
    )
    if sanitized:
        return sanitized

    route_turn = coerce_expense_route_modify_turn(message, memory)
    if route_turn and route_turn.get("item_patches"):
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        return patches_to_field_updates(fields, route_turn)

    modify_turn = coerce_expense_modify_turn(message, memory)
    if modify_turn and modify_turn.get("item_patches"):
        draft = memory.active_draft()
        fields = dict(draft.fields or {}) if draft else {"items": []}
        return patches_to_field_updates(fields, modify_turn)

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
) -> list:
    if not is_expense_review_mode(memory):
        return list(updates or [])
    return review_field_updates_from_message(
        message,
        memory,
        trace_id=trace_id,
        understanding_updates=updates,
    )


def build_expense_pending_edit_from_turn(
    turn: dict[str, Any],
    *,
    message: str = "",
) -> dict[str, Any] | None:
    """Remember an in-flight clarify-modify/delete turn for the next user reply."""
    intent = str(turn.get("intent") or "").strip().lower()
    if intent not in ("clarify_modify", "clarify_delete"):
        return None
    clarify = turn.get("clarify") if isinstance(turn.get("clarify"), dict) else {}
    payload: dict[str, Any] = {
        "kind": intent.replace("clarify_", ""),
        "message": (message or "").strip(),
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
        "pending_question": {
            "field": pq.field,
            "item_index": pq.item_index,
            "prompt": pq.prompt,
        }
        if pq and pq.workflow_id == "expense"
        else None,
    }
