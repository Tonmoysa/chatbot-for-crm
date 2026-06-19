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





def parse_route(message: str) -> tuple[str, str] | None:

    raw = (message or "").strip()

    if not raw:

        return None



    m = _TO_RE.match(raw)

    if m:

        return m.group(1).strip().title(), m.group(2).strip().title()



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


