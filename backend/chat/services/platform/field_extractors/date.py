"""Deterministic date extraction — Universal Field Engine."""

from __future__ import annotations

import re
from datetime import date, timedelta

from chat.services.platform.field_extractors.common import MONTHS


def parse_relative_date(message: str, *, today: date | None = None) -> str | None:
    today = today or date.today()
    raw = message or ""
    low = raw.lower()

    if re.search(r"\b(ajke|ajker|today|aaj)\b", low) or "আজ" in raw:
        return today.isoformat()
    if re.search(r"\b(kal|agamikal|tomorrow)\b", low) or "আগামীকাল" in raw:
        return (today + timedelta(days=1)).isoformat()
    if re.search(r"\b(goto|yesterday)\b", low) or "গতকাল" in raw:
        return (today - timedelta(days=1)).isoformat()

    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", raw)
    if iso:
        return iso.group(1)

    m = re.search(
        r"(?:agami|next|upcoming)?\s*(\d{1,2})\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
        low,
    )
    if m:
        day = int(m.group(1))
        month = MONTHS.get(m.group(2)[:3], MONTHS.get(m.group(2)))
        if month:
            year = today.year
            try:
                candidate = date(year, month, day)
                if candidate < today and re.search(r"\bagami\b", low):
                    candidate = date(year + 1, month, day)
                elif candidate < today:
                    candidate = date(year + 1, month, day)
                return candidate.isoformat()
            except ValueError:
                pass

    m2 = re.search(
        r"(\d{1,2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r".{0,20}(?:leave|chuti|chhuti|ছুটি)",
        low,
    )
    if m2:
        day = int(m2.group(1))
        month_key = m2.group(2)[:3]
        month = MONTHS.get(month_key, MONTHS.get(m2.group(2)))
        if month:
            year = today.year
            try:
                candidate = date(year, month, day)
                if candidate < today:
                    candidate = date(year + 1, month, day)
                return candidate.isoformat()
            except ValueError:
                pass
    return None


def parse_leave_dates(message: str, *, today: date | None = None) -> dict[str, str]:
    """Extract start_date and end_date as ISO strings from leave narratives."""
    today = today or date.today()
    low = (message or "").lower()
    out: dict[str, str] = {}

    def _iso(day: int, month: int) -> str | None:
        try:
            d = date(today.year, month, day)
            if d < today:
                d = date(today.year + 1, month, day)
            return d.isoformat()
        except ValueError:
            return None

    found: list[str] = []
    for m in re.finditer(
        r"(?:\(\s*)?(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
        low,
    ):
        day = int(m.group(1))
        month_key = m.group(2)[:3]
        month = MONTHS.get(month_key, MONTHS.get(m.group(2)))
        if month:
            iso = _iso(day, month)
            if iso:
                found.append(iso)

    if found:
        out["start_date"] = found[0]
        if len(found) > 1:
            out["end_date"] = found[-1]
    else:
        single = parse_relative_date(message, today=today)
        if single:
            out["start_date"] = single

    return out


def format_iso_date_display(iso: str) -> str:
    try:
        d = date.fromisoformat(str(iso)[:10])
        return d.strftime("%d %B %Y")
    except ValueError:
        return str(iso)
