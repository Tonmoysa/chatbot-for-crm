"""Deterministic expense item extraction — Universal Field Engine."""



from __future__ import annotations



import re

from typing import Any



from chat.services.platform.field_extractors.amount import parse_amount

from chat.services.platform.field_extractors.common import EXPENSE_LABELS

from chat.services.platform.field_extractors.route import parse_route



_TRAVEL_WORDS = re.compile(

    r"\b(bus|train|bike|uber|taxi|cng|metro|transport|travel|office jete|commute)\b|"

    r"(বাস|ট্রেন|যাতায়|যাত্রা|অফিস\s*য)",

    re.I | re.UNICODE,

)



_MEAL_WORDS = re.compile(

    r"\b(lunch|dinner|breakfast|meal|snack|nasta|nasto|nosto|lanch|tiffin|khabar|"

    r"khawa|khete|korechi|khabar)\b|"

    r"(লাঞ্চ|দুপুর|নasta|নাস্তা|খাবার|খেয়েছি|খেয়েছি|হালকা\s*নasta|হালকা\s*নাস্তা|বিকাল)",

    re.I | re.UNICODE,

)



_BUS_WORD = re.compile(r"\b(bus|বাস)\b", re.I | re.UNICODE)



_AMOUNT_FOR_LABEL = re.compile(

    rf"(\d+(?:\.\d+)?)\s*(?:taka|tk|টাকা)\s+for\s+({EXPENSE_LABELS})\b",

    re.I | re.UNICODE,

)



_LABEL_AMOUNT = re.compile(

    rf"\b({EXPENSE_LABELS})\s+(\d+(?:\.\d+)?)\s*(?:taka|tk|টাকা)?\b",

    re.I | re.UNICODE,

)





def detect_expense_category(message: str) -> str:

    low = (message or "").lower()

    if _TRAVEL_WORDS.search(low) or _BUS_WORD.search(message or ""):

        return "travel"

    if _MEAL_WORDS.search(low) or re.search(r"\b(nasta|nasto|nosto|snack)\b", low):

        return "meals"

    if re.search(r"\b(supplies|stationery|pen|paper)\b", low):

        return "supplies"

    return "meals"





def extract_expense_items(message: str) -> list[dict[str, Any]]:

    """Parse one or many expense items from a compound message."""

    raw = message or ""

    low = raw.lower()

    items: list[dict[str, Any]] = []

    seen: set[tuple[str, float]] = set()



    def _add(label: str, amount: float) -> None:

        label = label.lower().strip()

        key = (label, amount)

        if key in seen:

            return

        seen.add(key)

        cat = "travel" if label in ("bus", "bike", "train", "uber", "taxi", "cng") else "meals"

        items.append({

            "category": cat,

            "amount": amount,

            "description": f"{label} {amount:.0f} taka",

        })



    segments = re.split(r"\bthen\b|\band then\b|,", low)

    for seg in segments:

        for m in _AMOUNT_FOR_LABEL.finditer(seg):

            _add(m.group(2), float(m.group(1)))

        for m in _LABEL_AMOUNT.finditer(seg):

            _add(m.group(1), float(m.group(2)))



    if items:

        return items



    single = extract_expense_item(message)

    return [single] if single else []





def extract_expense_item(message: str) -> dict[str, Any] | None:

    amount = parse_amount(message)

    if amount is None:

        return None

    category = detect_expense_category(message)

    item: dict[str, Any] = {

        "category": category,

        "amount": amount,

        "description": (message or "").strip()[:120],

    }

    route = parse_route(message)

    if route:

        item["from_location"] = route[0]

        item["to_location"] = route[1]

    return item


