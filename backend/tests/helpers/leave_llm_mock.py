"""Mock LLM responses for leave review/collect unit tests."""

from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any
from unittest.mock import patch


def _tomorrow_iso() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _mock_review_response(message: str, payload: dict[str, Any]) -> dict[str, Any]:
    low = (message or "").lower()
    updates: list[dict[str, Any]] = []

    if "reason ta" in low or "karon ta" in low or "kar ta" in low or "reason change" in low:
        val = message
        for prefix in ("reason ta ", "karon ta ", "kar ta ", "reason change koro ", "reason change "):
            if low.startswith(prefix):
                val = message[len(prefix) :]
                break
        for suffix in (" koro", " kor", " kore", " dao", " daw"):
            if low.endswith(suffix):
                val = val[: -len(suffix)]
                break
        if "poriborton kore" in low:
            val = low.split("poriborton kore", 1)[-1].strip()
            for suffix in (" koro", " kor", " kore"):
                if val.endswith(suffix):
                    val = val[: -len(suffix)]
        updates.append({"field": "reason", "value": val.strip()})

    if "leave type" in low and "lwop" in low:
        updates.append({"field": "leave_type", "value": "lwop"})
    elif "type ta sick" in low or "type sick" in low:
        updates.append({"field": "leave_type", "value": "sick"})

    if "surur" in low or "suru hobe" in low or "surur din" in low:
        if "13" in low and ("september" in low or "sep" in low or "tarik" in low):
            updates.append({"field": "start_date", "value": "2026-09-13"})
        elif "13 tarik" in low:
            updates.append({"field": "start_date", "value": "2026-09-13"})

    if "end date" in low or "last date" in low or "shesh" in low or "sesh" in low:
        if "7 july" in low:
            updates.append({"field": "end_date", "value": "2026-07-07"})
        elif "3 july" in low:
            updates.append({"field": "end_date", "value": "2026-07-03"})
        elif "5 july" in low:
            updates.append({"field": "end_date", "value": "2026-07-05"})
    elif "date ta" in low or "tarikh" in low or re.search(r"\b\d{1,2}\s+july\b", low):
        if "23 august" in low and "25" in low:
            updates.extend(
                [
                    {"field": "start_date", "value": "2026-08-23"},
                    {"field": "end_date", "value": "2026-08-25"},
                ]
            )
        elif "23 august" in low:
            updates.append({"field": "start_date", "value": "2026-08-23"})
        elif "3 july" in low:
            updates.append({"field": "start_date", "value": "2026-07-03"})

    if updates:
        return {"intent": "modify", "field_updates": updates}
    if any(w in low for w in ("kothay", "dekhchi nah", "keno")):
        return {"intent": "question", "field_updates": []}
    return {"intent": "unclear", "field_updates": []}


def _mock_collect_response(message: str, payload: dict[str, Any]) -> dict[str, Any]:
    field = str(payload.get("pending_field") or "")
    low = (message or "").lower().strip()
    draft = dict(payload.get("draft_fields") or {})
    if field == "start_date" and low in ("kalke", "kal", "tomorrow"):
        return {"answers_pending_field": True, "field": "start_date", "value": _tomorrow_iso()}
    if field == "leave_type" and low in ("sick", "annual", "lwop"):
        return {"answers_pending_field": True, "field": "leave_type", "value": low}
    if field == "reason" and low == "skip":
        return {"answers_pending_field": True, "field": "reason", "value": ""}
    if field == "day_scope" and "sick" in low and draft.get("leave_type"):
        return {"answers_pending_field": False, "field": "leave_type", "value": "sick"}
    return {"answers_pending_field": True, "field": field, "value": message}


def _mock_field_extract_response(message: str, payload: dict[str, Any]) -> dict[str, Any]:
    low = (message or "").lower()
    updates: list[dict[str, Any]] = []
    if "dadi" in low and "osustho" in low:
        updates.append(
            {
                "field": "reason",
                "value": "Grandfather unwell; family traveling to village",
            }
        )
    if "biye" in low or "wedding" in low:
        updates.append({"field": "reason", "value": "Younger sister's wedding"})
    if "baba" in low and "operation" in low:
        updates.append({"field": "reason", "value": "Father's operation; hospital stay"})
    if "14" in low and "september" in low:
        updates.append({"field": "start_date", "value": "2026-09-14"})
        if "17" in low:
            updates.append({"field": "end_date", "value": "2026-09-17"})
    if "annual" in low:
        updates.append({"field": "leave_type", "value": "annual"})
        updates.append({"field": "day_scope", "value": "full_day"})
    if "5 august" in low and "9 august" in low:
        updates.append({"field": "start_date", "value": "2026-08-05"})
        updates.append({"field": "end_date", "value": "2026-08-09"})
    return {"field_updates": updates, "entities": {}}


def _mock_reason_extract_response(message: str, payload: dict[str, Any]) -> dict[str, Any]:
    low = (message or "").lower()
    if "dadi" in low and "osustho" in low:
        return {"reason": "Grandfather unwell; family traveling to village"}
    if "biye" in low or "wedding" in low:
        return {"reason": "Younger sister's wedding"}
    if "baba" in low and "operation" in low:
        return {"reason": "Father's operation; hospital stay"}
    if "mama" in low and "osustho" in low:
        return {"reason": "Mama osustho"}
    return {"reason": ""}


@contextmanager
def mock_leave_llm():
    with patch("chat.services.llm_client.LLMClient") as mock_cls:
        client = mock_cls.return_value
        client.is_configured.return_value = True

        def chat_json(*, system_prompt: str, user_prompt: str, trace_id: str = ""):
            import json

            payload = json.loads(user_prompt)
            message = str(payload.get("message") or "")
            if "Extract ONLY the leave reason" in system_prompt:
                return _mock_reason_extract_response(message, payload)
            if "field_updates" in system_prompt and "Extract leave workflow fields" in system_prompt:
                return _mock_field_extract_response(message, payload)
            if "REVIEW" in system_prompt or "review" in system_prompt.lower()[:80]:
                return _mock_review_response(message, payload)
            return _mock_collect_response(message, payload)

        client.chat_json.side_effect = chat_json
        yield mock_cls


def review_memory():
    from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft

    return SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="confirm_submit"),
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="leave",
                fields={
                    "leave_type": "annual",
                    "day_scope": "full_day",
                    "start_date": "2026-06-29",
                    "end_date": "2026-07-02",
                    "reason": "Father unwell; Hospital/treatment visit",
                },
            )
        },
        pending_confirmation="submit",
    )
