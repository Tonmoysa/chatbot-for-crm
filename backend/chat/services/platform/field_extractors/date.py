"""Deterministic date extraction — Universal Field Engine."""

from __future__ import annotations

import re
from datetime import date, timedelta

from chat.services.platform.field_extractors.common import MONTHS

_WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

_WDAY_TOKEN = (
    r"monday|mon|tuesday|tue(?:s)?|wednesday|wed|thursday|thu(?:r(?:s)?)?|"
    r"friday|fri|saturday|sat|sunday|sun"
)
_WDAY_MOD = r"(?:next|this|coming|upcoming|agami|agamik|poroborti)"


def _weekday_index(name: str) -> int | None:
    key = (name or "").strip().lower()
    if key in _WEEKDAY_NAMES:
        return _WEEKDAY_NAMES[key]
    for token, idx in _WEEKDAY_NAMES.items():
        if token.startswith(key[:3]):
            return idx
    return None


def _resolve_weekday(
    name: str,
    today: date,
    *,
    modifier: str = "",
    not_before: date | None = None,
) -> date | None:
    dow = _weekday_index(name)
    if dow is None:
        return None
    anchor = not_before or today
    days_ahead = dow - anchor.weekday()
    mod = (modifier or "").strip().lower()
    if not_before is not None:
        if days_ahead <= 0:
            days_ahead += 7
    elif mod in ("next", "agami", "agamik", "upcoming", "coming", "poroborti"):
        if days_ahead <= 0:
            days_ahead += 7
    elif days_ahead < 0:
        days_ahead += 7
    return anchor + timedelta(days=days_ahead)


def _parse_weekday_span(message: str, *, today: date) -> dict[str, str]:
    """Parse weekday ranges like 'next Wednesday theke Friday'."""
    low = (message or "").lower()
    out: dict[str, str] = {}

    range_re = re.compile(
        rf"(?P<smod>{_WDAY_MOD})?\s*(?P<sday>{_WDAY_TOKEN})\b"
        rf".{{0,30}}?(?:theke|to|through|until|thru|porjonto|-|–)\s*"
        rf"(?P<emod>{_WDAY_MOD})?\s*(?P<eday>{_WDAY_TOKEN})\b",
        re.I | re.UNICODE,
    )
    m = range_re.search(low)
    if m:
        start = _resolve_weekday(
            m.group("sday"),
            today,
            modifier=m.group("smod") or "",
        )
        if start:
            end = _resolve_weekday(
                m.group("eday"),
                today,
                modifier=m.group("emod") or "",
                not_before=start,
            )
            if end:
                out["start_date"] = start.isoformat()
                out["end_date"] = end.isoformat()
                return out

    single_re = re.compile(
        rf"(?P<mod>{_WDAY_MOD})?\s*(?P<day>{_WDAY_TOKEN})\b",
        re.I | re.UNICODE,
    )
    sm = single_re.search(low)
    if sm:
        start = _resolve_weekday(sm.group("day"), today, modifier=sm.group("mod") or "")
        if start:
            out["start_date"] = start.isoformat()
    return out


def parse_relative_date(message: str, *, today: date | None = None) -> str | None:
    today = today or date.today()
    raw = message or ""
    low = raw.lower()

    if re.search(r"\b(ajke|ajker|today|aaj)\b", low) or "আজ" in raw:
        return today.isoformat()
    if (
        re.search(r"\b(kalke|kal|agamikal|agamikalke|tomorrow)\b", low)
        or "আগামীকাল" in raw
        or "কালকে" in raw
    ):
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
        weekday_span = _parse_weekday_span(message, today=today)
        if weekday_span:
            out.update(weekday_span)
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
