"""Deterministic route extraction — Universal Field Engine."""



from __future__ import annotations



import re



_ROUTE_RE = re.compile(

    r"(\w+(?:\s+\w+)?)\s+(?:theke|theke|from|থেকে)\s+(\w+(?:\s+\w+)?)\s+"

    r"(?:bus|train|e|এ|ye|y'e|jete|য)?",

    re.I | re.UNICODE,

)



_TO_RE = re.compile(

    r"^([A-Za-z\u0980-\u09FF]+(?:\s+[A-Za-z\u0980-\u09FF]+)?)\s+to\s+"

    r"([A-Za-z\u0980-\u09FF]+(?:\s+[A-Za-z\u0980-\u09FF]+)?)\s*\.?$",

    re.I | re.UNICODE,

)





_ROUTE_STOPWORDS = frozenset(
    {
        "hobe", "habe", "route", "expense", "er", "theke", "from", "to",
        "koro", "kor", "dao", "de", "den", "hoye", "hoy", "no", "number",
        "nombor", "numer",
    }
)


def _pick_place_to_place_route(raw: str) -> tuple[str, str] | None:
    """Prefer the last clean ``Place to Place`` pair, skipping Banglish filler words."""
    best: tuple[str, str] | None = None
    for m in re.finditer(
        r"\b([A-Za-z\u0980-\u09FF]+)\s+to\s+([A-Za-z\u0980-\u09FF]+)\b",
        raw,
        re.I,
    ):
        frm, to = m.group(1).strip().lower(), m.group(2).strip().lower()
        if frm in _ROUTE_STOPWORDS or to in _ROUTE_STOPWORDS:
            continue
        best = (m.group(1).strip().title(), m.group(2).strip().title())
    return best


def parse_route(message: str) -> tuple[str, str] | None:

    raw = (message or "").strip()

    if not raw:

        return None



    m = _TO_RE.match(raw)

    if m:

        return m.group(1).strip().title(), m.group(2).strip().title()

    picked = _pick_place_to_place_route(raw)
    if picked:
        return picked

    m = _ROUTE_RE.search(raw)

    if m:

        return m.group(1).strip().title(), m.group(2).strip().title()



    m2 = re.search(

        r"([A-Za-z\u0980-\u09FF]+(?:\s+[A-Za-z\u0980-\u09FF]+)?)\s+theke\s+"

        r"([A-Za-z\u0980-\u09FF]+(?:\s+[A-Za-z\u0980-\u09FF]+)?)",

        raw,

        re.I,

    )

    if m2:

        return m2.group(1).strip().title(), m2.group(2).strip().title()

    return None


