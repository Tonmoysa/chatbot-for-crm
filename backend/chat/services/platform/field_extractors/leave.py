"""Leave field helpers — LLM-driven extraction; code only validates and coerces."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

LEAVE_INTERNAL_DRAFT_FIELDS = frozenset({"reason_skipped", "medical_document_skipped"})
CANONICAL_LEAVE_TYPES = frozenset({"annual", "sick", "lwop"})
_SKIP_TOKENS = frozenset({"skip", "none", "na", "n/a", "no", "no reason"})
_DECLINE_TOKENS = frozenset({"false", "no", "na", "nai", "nei", "n/a"})


def _coerce_iso_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return None


def _coerce_llm_date_output(value: Any) -> str | None:
    """Normalize date strings returned by the LLM (not user-message parsing)."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    iso = _coerce_iso_date(s)
    if iso:
        return iso
    from datetime import datetime

    for fmt in (
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_leave_type_value(value: Any) -> str | None:
    """Map provider values to canonical leave_type enum."""
    if value is None:
        return None
    lt = str(value).strip().lower()
    if lt in ("unpaid",):
        return "lwop"
    if lt in CANONICAL_LEAVE_TYPES:
        return lt
    return None


def sanitize_leave_type_value(
    message: str,
    leave_type: str | None,
    *,
    context: str = "",
) -> str | None:
    """Normalize leave_type to canonical enum."""
    _ = message, context
    return normalize_leave_type_value(leave_type)


def is_reason_skip_message(message: str) -> bool:
    """User declines to provide an optional leave reason."""
    low = (message or "").strip().lower().strip(".")
    if not low:
        return False
    if low in _SKIP_TOKENS:
        return True
    if "skip" in low.split() and len(low.split()) <= 3:
        return True
    if "lagbe na" in low or "lagi na" in low:
        return True
    return False


def leave_fields_for_submit(fields: dict[str, Any]) -> dict[str, Any]:
    """CRM payload — omit internal draft-only flags."""
    return {k: v for k, v in (fields or {}).items() if k not in LEAVE_INTERNAL_DRAFT_FIELDS}


def is_medical_document_unavailable(message: str) -> bool:
    low = (message or "").lower().strip()
    if not low:
        return False
    if low in _DECLINE_TOKENS:
        return True
    if any(tok in low for tok in ("nai", "nei", "n/a", "don't have", "do not have", "document nai")):
        return True
    return False


def is_medical_document_skip_message(message: str) -> bool:
    """User declines or defers medical document — optional field may be skipped."""
    low = (message or "").strip().lower()
    if not low:
        return False
    if is_medical_document_unavailable(message):
        return True
    if any(tok in low for tok in ("pore", "later", "parbo", "pari")) and any(
        w in low for w in ("upload", "debo", "provide", "document", "prescription")
    ):
        return True
    return is_reason_skip_message(message)


_HALF_DAY_RE = re.compile(
    r"\b(?:half[\s-]?day|ordho\s*din|half\s*din|adha\s*din)\b",
    re.I | re.UNICODE,
)


def message_implies_half_day(message: str) -> bool:
    return bool(_HALF_DAY_RE.search(message or ""))


def leave_span_days(fields: dict[str, Any]) -> int | None:
    start = fields.get("start_date")
    end = fields.get("end_date") or start
    if not start:
        return None
    try:
        s = date.fromisoformat(str(start)[:10])
        e = date.fromisoformat(str(end)[:10])
    except ValueError:
        return None
    return (e - s).days + 1


def apply_multi_day_scope_to_fields(fields: dict[str, Any], message: str = "") -> dict[str, Any]:
    """Multi-day leave spans default to full_day unless user asked for half day."""
    out = dict(fields or {})
    if out.get("day_scope") in ("full_day", "half_day"):
        return out
    if message_implies_half_day(message):
        return out
    span = leave_span_days(out)
    if span is not None and span >= 2:
        out["day_scope"] = "full_day"
    return out


def apply_leave_derived_fields(draft: Any, *, message: str = "") -> None:
    if not draft or draft.workflow_id != "leave" or draft.locked:
        return
    if scrub_invalid_leave_reason_from_fields(draft.fields):
        draft.version += 1
    before = draft.fields.get("day_scope")
    updated = apply_multi_day_scope_to_fields(draft.fields, message)
    if updated.get("day_scope") and not before:
        draft.fields["day_scope"] = updated["day_scope"]
        draft.version += 1


def parse_medical_document_field(message: str) -> Any:
    if is_medical_document_unavailable(message):
        return None
    val = (message or "").strip()
    return val[:500] if val else None


def is_leave_review_mode(memory) -> bool:
    """Draft is at review/submit — reason and other fields are modify-only."""
    if (memory.pending_confirmation or "") == "submit":
        return True
    aw = memory.active_workflow
    return bool(aw and aw.id == "leave" and aw.stage == "confirm_submit")


def is_leave_review_complaint_or_question(message: str) -> bool:
    """User questions draft/review — not a field value or modify command."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.turn_semantics import is_process_question, is_workflow_meta_complaint

    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return False
    if is_workflow_meta_complaint(raw) or is_process_question(raw):
        return True
    low = raw.lower()
    if any(
        p in low
        for p in (
            "dekhchi nah",
            "dekhte parch",
            "dekha jacche nah",
            "show kore nah",
            "kothay",
            "where is",
            "keno",
            "why",
        )
    ):
        return True
    if any(w in low for w in ("but", "kintu", "tobu")) and any(
        w in low for w in ("update", "review", "draft", "dekh", "show", "end date", "shesh", "3 din", "tin din")
    ):
        return True
    return False


def is_leave_complaint_reason_value(text: str) -> bool:
    """Reject complaint/question sentences stored as reason."""
    raw = (text or "").strip()
    if not raw:
        return False
    if is_leave_review_complaint_or_question(raw):
        return True
    low = raw.lower()
    if any(w in low for w in ("kothay", "keno", "why", "dekhchi nah", "dekhte parch")):
        return True
    # Long draft/review meta-complaints — not leave narratives stored as reason.
    if len(raw.split()) > 12:
        if any(w in low for w in ("update", "review", "draft", "dekh", "dekha", "summary")):
            if any(w in low for w in ("kothay", "keno", "nah", "dekhchi", "parch", "bujhte")):
                return True
    return False


def is_temporal_or_request_like_leave_reason(text: str) -> bool:
    """Reason value that is really a date or leave request — not a narrative why."""
    raw = (text or "").strip()
    if not raw:
        return False
    low = raw.lower().strip(".")
    if low in (
        "tomorrow",
        "today",
        "kalke",
        "kal",
        "agamikal",
        "ajke",
        "leave tomorrow",
        "leave today",
        "chuti kalke",
        "chuti lagbe",
        "leave lagbe",
        "amar leave lagbe",
        "need leave",
        "want leave",
    ):
        return True
    if "kal theke" in low or "kalke" in low or "agamikal" in low:
        if len(low.split()) <= 6:
            return True
    if low.startswith("leave ") and any(w in low for w in ("tomorrow", "today", "kalke", "kal")):
        return True
    if "lagbe" in low and any(w in low for w in ("leave", "chuti", "chhuti")) and len(low.split()) <= 8:
        return True
    return False


def scrub_invalid_leave_reason_from_fields(fields: dict[str, Any]) -> bool:
    """Drop bogus reason already on draft — returns True if removed."""
    reason = fields.get("reason")
    if not reason:
        return False
    if is_garbage_leave_reason_value(str(reason)) or is_temporal_or_request_like_leave_reason(str(reason)):
        fields.pop("reason", None)
        return True
    return False


def is_garbage_leave_reason_value(text: str) -> bool:
    """Reject candidate reason VALUES."""
    raw = (text or "").strip()
    if not raw:
        return False
    low = raw.lower().strip(".")
    if is_temporal_or_request_like_leave_reason(raw):
        return True
    if low in ("leave", "chuti", "chhuti", "modify", "submit", "cancel", "summary", "review", "yes", "no", "ha"):
        return True
    if low in ("full day", "half day"):
        return True
    if any(h in low for h in ("date ta", "tarikh ta", "leave type", "end date", "start date", "last date")):
        return True
    if any(
        h in low
        for h in (
            "attend korte parbo na",
            "aste parbo na",
            "office e attend",
            "apply korte hobe",
        )
    ):
        return True
    if "modify korbo" in low or "back koro" in low:
        return True
    from chat.services.platform.turn_semantics import is_process_question

    if is_process_question(raw):
        return True
    if is_leave_complaint_reason_value(raw):
        return True
    return False


def is_garbage_leave_reason(text: str) -> bool:
    return is_garbage_leave_reason_value(text)


def extract_leave_reason_via_llm(
    message: str,
    memory=None,
    *,
    trace_id: str = "",
) -> str | None:
    """LLM-only: extract a concise leave reason from narrative text."""
    if not _llm_client_configured():
        return None

    from chat.services.platform.banglish_normalize import normalize_banglish_message

    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return None

    import json

    from chat.services.llm_client import LLMClient
    from chat.services.platform.llm_prompts import LEAVE_REASON_EXTRACT_SYSTEM
    from chat.services.platform.turn_semantics import understanding_session_context

    ctx = understanding_session_context(memory, None) if memory is not None else {}
    payload = {
        "message": raw,
        "draft_fields": dict((memory.active_draft().fields if memory and memory.active_draft() else {}) or {}),
        "today_iso": date.today().isoformat(),
        **ctx,
    }
    parsed = LLMClient().chat_json(
        system_prompt=LEAVE_REASON_EXTRACT_SYSTEM,
        user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
        trace_id=trace_id or "",
    )
    if not isinstance(parsed, dict):
        return None
    reason = str(parsed.get("reason") or "").strip()
    if reason and not is_garbage_leave_reason_value(reason):
        return reason[:200]
    return None


def infer_leave_reason_from_history(
    memory,
    conversation_history: list[str] | None = None,
    *,
    trace_id: str = "",
) -> str | None:
    """LLM-only: recover reason from an earlier user turn when the slot was skipped."""
    draft = memory.active_draft() if memory else None
    if not draft or draft.fields.get("reason") or draft.fields.get("reason_skipped"):
        return None

    candidates: list[str] = []
    seed = str((memory.last_entities or {}).get("leave_narrative_seed") or "").strip()
    if seed:
        candidates.append(seed)

    for line in reversed(list(conversation_history or ())):
        text = (line or "").strip()
        if not text:
            continue
        if text.lower().startswith("assistant:"):
            continue
        if text.lower().startswith("user:"):
            text = text[5:].strip()
        if len(text) >= 40:
            candidates.append(text)

    seen: set[str] = set()
    for text in candidates:
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        reason = extract_leave_reason_via_llm(text, memory, trace_id=trace_id)
        if reason:
            return reason
    return None


def remember_leave_narrative_seed(memory, message: str) -> None:
    """Store a long opening message so LLM can recover reason later on skip."""
    if not memory:
        return
    aw = memory.active_workflow
    if not aw or aw.id != "leave":
        return
    text = (message or "").strip()
    if len(text) < 80:
        return
    entities = dict(memory.last_entities or {})
    entities["leave_narrative_seed"] = text[:800]
    memory.last_entities = entities


def _coerce_review_delta(
    updates: dict[str, Any],
    *,
    message: str = "",
    draft_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize LLM review patches — trust LLM for dates and semantics."""
    _ = message, draft_fields
    out: dict[str, Any] = {}

    if updates.get("leave_type") is not None:
        clean = normalize_leave_type_value(updates["leave_type"])
        if clean:
            out["leave_type"] = clean

    if updates.get("day_scope") in ("full_day", "half_day"):
        out["day_scope"] = updates["day_scope"]

    if updates.get("half_day_period") in ("morning", "afternoon"):
        out["half_day_period"] = updates["half_day_period"]

    for key in ("start_date", "end_date"):
        iso = _coerce_llm_date_output(updates.get(key))
        if iso:
            out[key] = iso

    reason_raw = updates.get("reason")
    if reason_raw is not None:
        reason = str(reason_raw).strip()
        if reason and not is_garbage_leave_reason_value(reason):
            out["reason"] = reason[:200]

    return out


def _rules_leave_review_delta(
    message: str,
    *,
    draft_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structural review edits when leave review LLM is unavailable."""
    from chat.services.platform.field_extractors.date import parse_leave_dates, parse_relative_date

    low = (message or "").lower()
    dates = parse_leave_dates(message)
    single = parse_relative_date(message)
    updates: dict[str, Any] = {}

    if "reason ta" in low or "karon ta" in low or "kar ta" in low:
        val = message
        for prefix in ("reason ta ", "karon ta ", "kar ta "):
            if low.startswith(prefix):
                val = message[len(prefix) :].strip()
                break
        for suffix in (" koro", " kor", " kore", " dao", " daw"):
            if low.endswith(suffix):
                val = val[: -len(suffix)].strip()
        if val:
            updates["reason"] = val

    if "suru hobe" in low or "surur din" in low:
        day_m = re.search(r"\b(\d{1,2})\s+tarik\b", low)
        if day_m and draft_fields:
            day = int(day_m.group(1))
            base = str(draft_fields.get("start_date") or draft_fields.get("end_date") or "")[:10]
            if base:
                try:
                    year_s, month_s, _ = base.split("-")
                    from datetime import date as date_cls

                    updates["start_date"] = date_cls(int(year_s), int(month_s), day).isoformat()
                except ValueError:
                    pass

    if any(h in low for h in ("end date", "last date", "shesh", "sesh tarikh", "sesh din")):
        end = dates.get("end_date") or single
        if end:
            updates["end_date"] = end
    elif any(h in low for h in ("start date", "surur", "suru hobe", "surur din")):
        start = dates.get("start_date") or single
        if start:
            updates["start_date"] = start
    elif dates:
        updates.update({k: v for k, v in dates.items() if v})

    if not updates:
        return {}
    return _coerce_review_delta(updates, message=message, draft_fields=draft_fields)


def interpret_leave_review_turn(
    message: str,
    memory,
    *,
    trace_id: str = "",
) -> dict[str, Any]:
    """Review turn — intent + optional field delta (LLM only)."""
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.intent_rules import (
        is_bare_confirmation,
        is_bare_rejection,
        is_workflow_show_request,
    )
    from chat.services.platform.turn_semantics import is_process_question

    raw = normalize_banglish_message((message or "").strip())
    if not raw or not is_leave_review_mode(memory):
        return {"intent": "none", "field_updates": {}}

    if is_bare_confirmation(raw):
        return {"intent": "confirm", "field_updates": {}}
    if is_bare_rejection(raw):
        return {"intent": "cancel", "field_updates": {}}
    if is_workflow_show_request(raw, workflow_id="leave"):
        return {"intent": "navigation", "field_updates": {}}
    if is_process_question(raw) or is_leave_review_complaint_or_question(raw):
        return {"intent": "question", "field_updates": {}}

    draft = memory.active_draft()
    if not draft:
        return {"intent": "none", "field_updates": {}}

    if not _llm_client_configured():
        coerced = _rules_leave_review_delta(
            raw,
            draft_fields=dict(draft.fields or {}),
        )
        if coerced:
            return {"intent": "modify", "field_updates": coerced}
        return {"intent": "unclear", "field_updates": {}}

    import json

    from chat.services.llm_client import LLMClient
    from chat.services.platform.llm_prompts import LEAVE_REVIEW_EDIT_SYSTEM
    from chat.services.platform.turn_semantics import understanding_session_context

    draft_fields = dict(draft.fields or {})
    ctx = understanding_session_context(memory, None)
    payload = {
        "message": raw,
        "draft_fields": draft_fields,
        "draft_start_date": draft_fields.get("start_date"),
        "draft_end_date": draft_fields.get("end_date"),
        "last_assistant_message": ctx.get("last_assistant_message"),
        "today_iso": date.today().isoformat(),
        "instructions": (
            "Return all dates as ISO YYYY-MM-DD only. "
            "When the user gives only a day (e.g. 13 tarik) without a month, "
            "infer month and year from draft_start_date unless they name another month."
        ),
    }
    parsed = LLMClient().chat_json(
        system_prompt=LEAVE_REVIEW_EDIT_SYSTEM,
        user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
        trace_id=trace_id or "",
    )
    if not isinstance(parsed, dict):
        return {"intent": "unclear", "field_updates": {}}

    intent = str(parsed.get("intent") or "none").strip().lower()
    if intent not in ("modify", "question", "unclear", "navigation", "none"):
        intent = "modify" if parsed.get("field_updates") else "unclear"

    if intent != "modify":
        return {"intent": intent, "field_updates": {}}

    updates: dict[str, Any] = {}
    for item in parsed.get("field_updates") or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        value = item.get("value")
        if field and value not in (None, ""):
            updates[field] = value

    coerced = _coerce_review_delta(updates, message=raw, draft_fields=draft_fields)
    if coerced:
        return {"intent": "modify", "field_updates": coerced}
    return {"intent": "unclear", "field_updates": {}}


def interpret_leave_review_message(
    message: str,
    memory,
    *,
    trace_id: str = "",
) -> dict[str, Any]:
    """Semantic review-stage field edits — modify intent only."""
    turn = interpret_leave_review_turn(message, memory, trace_id=trace_id)
    if turn.get("intent") == "modify":
        return dict(turn.get("field_updates") or {})
    return {}


def review_delta_to_field_updates(delta: dict[str, Any]) -> list:
    from chat.services.platform.schemas import FieldUpdate

    return [
        FieldUpdate(field=str(k), value=v, action="set")
        for k, v in (delta or {}).items()
        if v not in (None, "")
    ]


def _llm_client_configured() -> bool:
    from chat.services.llm_client import LLMClient

    return LLMClient().is_configured()


LEAVE_COLLECT_PREFILL_FIELDS = frozenset({"start_date", "end_date"})
LEAVE_COLLECT_FIELDS = frozenset(
    {
        "leave_type",
        "day_scope",
        "half_day_period",
        "start_date",
        "end_date",
        "reason",
        "medical_document",
    }
)


def _pending_collect_allowed_fields(field: str) -> set[str]:
    allowed = {(field or "").strip()}
    if field == "day_scope":
        allowed.add("half_day_period")
    allowed.update(LEAVE_COLLECT_PREFILL_FIELDS)
    return {f for f in allowed if f}


def _leave_collect_correction_fields(draft_fields: dict[str, Any], pending_field: str) -> set[str]:
    """Already-filled leave fields the user may correct while another slot is pending."""
    filled = {
        k
        for k, v in (draft_fields or {}).items()
        if k in LEAVE_COLLECT_FIELDS and v not in (None, "")
    }
    filled.discard((pending_field or "").strip())
    return filled


def _parse_collect_slot_llm_turn(parsed_llm: dict[str, Any], *, pending_field: str) -> dict[str, Any]:
    """Normalize collect-slot LLM JSON into {field, value, answers_pending_field}."""
    if not isinstance(parsed_llm, dict):
        return {"answers_pending_field": None, "field": "", "value": None}

    apf_raw = parsed_llm.get("answers_pending_field")
    if apf_raw is True:
        answers_pending_field = True
    elif apf_raw is False:
        answers_pending_field = False
    else:
        answers_pending_field = None

    out_field = str(parsed_llm.get("field") or "").strip()
    value = parsed_llm.get("value")

    if not out_field and value in (None, ""):
        return {"answers_pending_field": answers_pending_field or False, "field": "", "value": None}

    if answers_pending_field is not False and not out_field:
        out_field = pending_field

    if answers_pending_field is True and out_field and out_field != pending_field:
        out_field = pending_field

    return {
        "answers_pending_field": answers_pending_field,
        "field": out_field,
        "value": value,
    }


def _coerce_collect_slot_value(
    field: str,
    value: Any,
    *,
    message: str,
) -> Any:
    """Normalize a single collect-slot value from LLM."""
    if field == "reason" and is_reason_skip_message(message):
        return None
    if field == "medical_document" and is_medical_document_skip_message(message):
        return None
    if value in (None, ""):
        return None
    if field == "leave_type":
        return normalize_leave_type_value(value) or sanitize_leave_type_value(message, str(value))
    if field == "day_scope" and str(value) in ("full_day", "half_day"):
        return str(value)
    if field == "half_day_period" and str(value) in ("morning", "afternoon"):
        return str(value)
    if field in ("start_date", "end_date"):
        return _coerce_llm_date_output(value)
    if field == "reason":
        reason = str(value).strip()
        if reason and not is_garbage_leave_reason_value(reason):
            return reason[:200]
        return None
    if field == "medical_document":
        return parse_medical_document_field(str(value))
    return value


def interpret_leave_collect_message(
    message: str,
    memory,
    *,
    trace_id: str = "",
) -> dict[str, Any]:
    """Semantic collect-slot fill — LLM only. Returns field deltas (legacy flat dict)."""
    turn = resolve_leave_collect_turn(message, memory, trace_id=trace_id)
    return dict(turn.get("updates") or {})


def resolve_leave_collect_turn(
    message: str,
    memory,
    *,
    trace_id: str = "",
    understanding_updates: list | None = None,
) -> dict[str, Any]:
    """Collect-slot turn: pending answer OR correction of a prior field.

    Returns {
        "updates": {field: value, ...},
        "answers_pending_field": bool | None,
        "is_correction": bool,
    }
    """
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.schemas import FieldUpdate
    from chat.services.platform.turn_semantics import understanding_session_context

    empty: dict[str, Any] = {
        "updates": {},
        "answers_pending_field": None,
        "is_correction": False,
    }

    pq = memory.pending_question
    if not pq or pq.workflow_id != "leave" or not pq.field:
        return empty
    if is_leave_review_mode(memory):
        return empty

    pending_field = str(pq.field)
    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return empty

    if pending_field == "reason" and is_reason_skip_message(raw):
        return {"updates": {"reason_skipped": True}, "answers_pending_field": True, "is_correction": False}
    if pending_field == "medical_document" and is_medical_document_skip_message(raw):
        return {
            "updates": {"medical_document_skipped": True},
            "answers_pending_field": True,
            "is_correction": False,
        }

    draft = memory.active_draft()
    draft_fields = dict((draft.fields if draft else {}) or {})
    correction_fields = _leave_collect_correction_fields(draft_fields, pending_field)
    allowed = _pending_collect_allowed_fields(pending_field)

    if understanding_updates:
        pending_updates: dict[str, Any] = {}
        correction_updates: dict[str, Any] = {}
        for upd in understanding_updates:
            if upd.value in (None, ""):
                continue
            fname = str(upd.field)
            coerced = _coerce_collect_slot_value(fname, upd.value, message=raw)
            if coerced is None:
                continue
            if fname == pending_field:
                pending_updates[fname] = coerced
            elif fname in correction_fields:
                correction_updates[fname] = coerced

        if correction_updates:
            return {
                "updates": correction_updates,
                "answers_pending_field": False,
                "is_correction": True,
            }
        if pending_updates:
            return {
                "updates": pending_updates,
                "answers_pending_field": True,
                "is_correction": False,
            }

    if not _llm_client_configured():
        return empty

    import json

    from chat.services.llm_client import LLMClient
    from chat.services.platform.llm_prompts import LEAVE_COLLECT_SLOT_SYSTEM

    ctx = understanding_session_context(memory, None)
    payload = {
        "message": raw,
        "pending_field": pending_field,
        "pending_prompt": pq.prompt,
        "draft_fields": draft_fields,
        "today_iso": date.today().isoformat(),
        **ctx,
    }
    parsed_llm = LLMClient().chat_json(
        system_prompt=LEAVE_COLLECT_SLOT_SYSTEM,
        user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
        trace_id=trace_id or "",
    )
    if not isinstance(parsed_llm, dict):
        return empty

    slot = _parse_collect_slot_llm_turn(parsed_llm, pending_field=pending_field)
    out_field = str(slot.get("field") or "").strip()
    value = slot.get("value")
    answers_pending_field = slot.get("answers_pending_field")

    if not out_field:
        return {
            "updates": {},
            "answers_pending_field": answers_pending_field if answers_pending_field is not None else False,
            "is_correction": False,
        }

    is_correction = bool(
        answers_pending_field is False
        or (out_field != pending_field and out_field in correction_fields)
    )
    if out_field != pending_field and not is_correction:
        return empty

    target_field = out_field if is_correction else pending_field
    if target_field == "reason" and value in (None, "") and is_reason_skip_message(raw):
        return {"updates": {"reason_skipped": True}, "answers_pending_field": True, "is_correction": False}

    coerced = _coerce_collect_slot_value(target_field, value, message=raw)
    if coerced is None:
        return empty

    return {
        "updates": {target_field: coerced},
        "answers_pending_field": False if is_correction else True,
        "is_correction": is_correction,
    }


def collect_slot_field_updates(
    message: str,
    memory,
    *,
    trace_id: str = "",
    understanding_updates: list | None = None,
) -> list:
    """Primary collect-slot path — pending answer or prior-field correction."""
    from chat.services.platform.schemas import FieldUpdate

    turn = resolve_leave_collect_turn(
        message,
        memory,
        trace_id=trace_id,
        understanding_updates=understanding_updates,
    )
    return [
        FieldUpdate(field=str(k), value=v, action="set")
        for k, v in (turn.get("updates") or {}).items()
        if v not in (None, "")
    ]


def leave_collect_turn_to_field_updates(turn: dict[str, Any]) -> list:
    from chat.services.platform.schemas import FieldUpdate

    return [
        FieldUpdate(field=str(k), value=v, action="set")
        for k, v in (turn.get("updates") or {}).items()
        if v not in (None, "")
    ]


def sanitize_leave_review_updates(
    updates: list,
    message: str,
    *,
    memory,
    trace_id: str = "",
) -> list:
    """Keep only valid partial review patches."""
    from chat.services.platform.schemas import FieldUpdate

    _ = trace_id
    if not is_leave_review_mode(memory):
        return list(updates or [])

    draft = memory.active_draft()
    draft_fields = dict(draft.fields or {}) if draft else {}
    merged: dict[str, Any] = {}
    for upd in updates or []:
        if upd.field and upd.value not in (None, ""):
            merged[str(upd.field)] = upd.value

    coerced = _coerce_review_delta(merged, message=message, draft_fields=draft_fields)
    allowed = {
        "leave_type",
        "day_scope",
        "half_day_period",
        "start_date",
        "end_date",
        "reason",
    }
    return [
        FieldUpdate(field=str(k), value=v, action="set")
        for k, v in coerced.items()
        if k in allowed
        and v not in (None, "")
        and str(draft_fields.get(k)) != str(v)
    ]


def review_field_updates_from_message(
    message: str,
    memory,
    *,
    trace_id: str = "",
    understanding_updates: list | None = None,
) -> list:
    """Primary review edit path — understanding patches, then semantic interpreter."""
    if is_leave_review_complaint_or_question(message):
        return []

    sanitized = sanitize_leave_review_updates(
        list(understanding_updates or []),
        message,
        memory=memory,
    )
    if sanitized:
        return sanitized
    turn = interpret_leave_review_turn(message, memory, trace_id=trace_id)
    if turn.get("intent") == "modify" and turn.get("field_updates"):
        return review_delta_to_field_updates(turn["field_updates"])
    return []


def leave_modify_updates_as_dict(
    message: str,
    *,
    memory=None,
    trace_id: str = "",
) -> dict[str, Any]:
    """Review-edit via LLM — requires review-mode memory."""
    if memory is None:
        return {}
    return interpret_leave_review_message(message, memory, trace_id=trace_id)


def parse_leave_modify_command(
    message: str,
    *,
    memory=None,
    trace_id: str = "",
) -> dict[str, Any] | None:
    updates = leave_modify_updates_as_dict(message, memory=memory, trace_id=trace_id)
    if not updates:
        return None
    return {"updates": updates}


def is_leave_modify_message(message: str, *, memory=None, trace_id: str = "") -> bool:
    return bool(leave_modify_updates_as_dict(message, memory=memory, trace_id=trace_id))


def leave_field_updates_from_modify(message: str, *, memory=None, trace_id: str = "") -> list:
    from chat.services.platform.schemas import FieldUpdate

    mod = leave_modify_updates_as_dict(message, memory=memory, trace_id=trace_id)
    return [
        FieldUpdate(field=str(k), value=v, action="set")
        for k, v in mod.items()
        if v not in (None, "")
    ]


def filter_leave_updates_for_review(
    updates: list,
    message: str,
    *,
    memory,
    trace_id: str = "",
) -> list:
    if not is_leave_review_mode(memory):
        return list(updates or [])
    return review_field_updates_from_message(
        message,
        memory,
        trace_id=trace_id,
        understanding_updates=updates,
    )


def extract_leave_fields_via_llm(
    message: str,
    memory=None,
    *,
    trace_id: str = "",
) -> dict[str, Any]:
    """One-shot LLM field extraction when understanding layer returned no fields."""
    if not _llm_client_configured():
        return {}
    import json

    from chat.services.llm_client import LLMClient
    from chat.services.platform.banglish_normalize import normalize_banglish_message
    from chat.services.platform.llm_prompts import LEAVE_FIELD_EXTRACT_SYSTEM
    from chat.services.platform.turn_semantics import understanding_session_context

    raw = normalize_banglish_message((message or "").strip())
    if not raw:
        return {}

    ctx = understanding_session_context(memory, None) if memory is not None else {}
    payload = {
        "message": raw,
        "draft_fields": dict((memory.active_draft().fields if memory and memory.active_draft() else {}) or {}),
        "today_iso": date.today().isoformat(),
        **ctx,
    }
    parsed = LLMClient().chat_json(
        system_prompt=LEAVE_FIELD_EXTRACT_SYSTEM,
        user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
        trace_id=trace_id or "",
    )
    if not isinstance(parsed, dict):
        return {}
    return merge_leave_field_dicts(
        {},
        {str(u.get("field")): u.get("value") for u in (parsed.get("field_updates") or []) if isinstance(u, dict) and u.get("field")},
        raw,
        memory=memory,
    )


def extract_leave_fields(message: str) -> dict[str, Any]:
    """Legacy API — use extract_leave_fields_via_llm with memory for real extraction."""
    _ = message
    return {}


def parse_leave_field(message: str, field: str, *, context: str = "") -> Any:
    """Legacy API — single-field parse is LLM-driven; returns None."""
    _ = message, field, context
    return None


def merge_deterministic_leave_dates(fields: dict[str, Any], message: str) -> dict[str, Any]:
    """Legacy API — dates come from LLM; passthrough only."""
    _ = message
    return dict(fields or {})


def merge_leave_field_dicts(
    rules_fields: dict[str, Any],
    llm_fields: dict[str, Any],
    message: str,
    *,
    memory=None,
) -> dict[str, Any]:
    """Validate and coerce LLM field patches only."""
    _ = rules_fields
    out: dict[str, Any] = {}
    llm = dict(llm_fields or {})

    for key in ("start_date", "end_date"):
        iso = _coerce_llm_date_output(llm.get(key))
        if iso:
            out[key] = iso

    if llm.get("leave_type"):
        clean = normalize_leave_type_value(llm["leave_type"])
        if clean:
            out["leave_type"] = clean

    if llm.get("day_scope") in ("full_day", "half_day"):
        out["day_scope"] = llm["day_scope"]

    if llm.get("half_day_period") in ("morning", "afternoon"):
        out["half_day_period"] = llm["half_day_period"]

    if memory is not None and is_leave_review_mode(memory):
        pass
    elif llm.get("reason"):
        reason = str(llm["reason"]).strip()
        if reason and not is_garbage_leave_reason_value(reason):
            out["reason"] = reason[:200]

    if llm.get("medical_document"):
        doc = parse_medical_document_field(str(llm["medical_document"]))
        if doc:
            out["medical_document"] = doc

    return apply_multi_day_scope_to_fields(out, message)


# Backward-compatible stubs — callers should use LLM entities / grounding instead.
def has_explicit_leave_type(text: str, leave_type: str) -> bool:
    _ = text, leave_type
    return False


def extract_unrecognized_leave_type_mention(text: str) -> str | None:
    _ = text
    return None


def infer_leave_type_from_text(text: str) -> str | None:
    _ = text
    return None


def extract_leave_reason(message: str) -> str | None:
    _ = message
    return None


def is_acceptable_leave_reason(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or len(raw) > 200:
        return False
    return not is_garbage_leave_reason_value(raw)


def leave_range_from_fields(fields: dict[str, Any]) -> tuple[str, str] | None:
    start = fields.get("start_date")
    if not start:
        return None
    end = fields.get("end_date") or start
    return str(start)[:10], str(end)[:10]


def leave_ranges_overlap(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    try:
        sa = date.fromisoformat(start_a[:10])
        ea = date.fromisoformat(end_a[:10])
        sb = date.fromisoformat(start_b[:10])
        eb = date.fromisoformat(end_b[:10])
    except ValueError:
        return False
    return sa <= eb and sb <= ea


def leave_date_ranges_match(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    """True when two leave ranges have identical start and end dates."""
    try:
        sa = date.fromisoformat(start_a[:10])
        ea = date.fromisoformat(end_a[:10])
        sb = date.fromisoformat(start_b[:10])
        eb = date.fromisoformat(end_b[:10])
    except ValueError:
        return False
    return sa == sb and ea == eb


def find_submitted_leave_overlap(
    memory, start_date: str, end_date: str | None = None
) -> dict[str, Any] | None:
    end_date = end_date or start_date
    for entry in (memory.conversation_facts or {}).get("submitted_leave_ranges") or []:
        if not isinstance(entry, dict):
            continue
        s = entry.get("start_date")
        e = entry.get("end_date") or s
        if s and leave_date_ranges_match(str(start_date)[:10], str(end_date)[:10], str(s)[:10], str(e)[:10]):
            return entry
    return None


def draft_overlaps_submitted_leave(memory, draft: Any | None = None) -> dict[str, Any] | None:
    """True when active/passed draft dates collide with a submitted leave range."""
    if not memory:
        return None
    d = draft
    if d is None:
        d = memory.active_draft()
    if not d or getattr(d, "workflow_id", None) != "leave":
        return None
    fields = dict(getattr(d, "fields", None) or {})
    start = fields.get("start_date")
    if not start:
        return None
    end = fields.get("end_date") or start
    return find_submitted_leave_overlap(memory, str(start), str(end) if end else None)


def record_submitted_leave_range(
    memory,
    fields: dict[str, Any],
    *,
    request_id: str = "",
    state: Any | None = None,
) -> None:
    patch = {
        "op": "record_submitted_leave_range",
        "fields": dict(fields),
        "request_id": request_id,
    }
    if state is not None:
        state.push("record_submitted_leave_range", fields=dict(fields), request_id=request_id)
    else:
        from chat.services.session_memory import apply_state_patches

        apply_state_patches(memory, [patch])
