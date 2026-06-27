"""Expense amount association — nearest amount per category in run-on messages."""

from __future__ import annotations

from chat.services.platform.field_extractors.expense import (
    _amount_for_category_in_message,
    build_wizard_fallback_turn,
    sanitize_expense_llm_patches,
    split_expense_clauses,
)
from chat.services.session_memory import SessionMemory

RUN_ON_EXPENSE = (
    "amar ajke expense hoyeche 20 taka bus mirpur to baridhara "
    "lunch 100 taka bike 150 taka mirpur to motejheel"
)


def test_split_expense_clauses_splits_run_on_multi_category_message():
    parts = split_expense_clauses(RUN_ON_EXPENSE)
    assert len(parts) == 3
    assert any("bus" in p and "20" in p for p in parts)
    assert any("lunch" in p and "100" in p for p in parts)
    assert any("bike" in p and "150" in p for p in parts)


def test_amount_for_category_picks_nearest_amount_not_first_in_message():
    assert _amount_for_category_in_message(RUN_ON_EXPENSE, "bus") == 20.0
    assert _amount_for_category_in_message(RUN_ON_EXPENSE, "lunch") == 100.0
    assert _amount_for_category_in_message(RUN_ON_EXPENSE, "bike") == 150.0


def test_sanitize_expense_llm_patches_corrects_wrong_lunch_amount():
    llm_turn = {
        "intent": "add",
        "item_patches": [
            {"action": "append", "category": "bus", "amount": 20},
            {"action": "append", "category": "lunch", "amount": 20.0},
            {"action": "append", "category": "bike", "amount": 150},
        ],
    }
    fixed = sanitize_expense_llm_patches(llm_turn, RUN_ON_EXPENSE, SessionMemory())
    patches = fixed["item_patches"]
    by_cat = {p["category"]: p["amount"] for p in patches}
    assert by_cat["bus"] == 20.0
    assert by_cat["lunch"] == 100.0
    assert by_cat["bike"] == 150.0


def test_wizard_fallback_run_on_message_amounts():
    turn = build_wizard_fallback_turn(RUN_ON_EXPENSE)
    patches = turn.get("item_patches") or []
    by_cat = {p.get("category"): p.get("amount") for p in patches if p.get("category")}
    assert by_cat.get("bus") == 20.0
    assert by_cat.get("lunch") == 100.0
    assert by_cat.get("bike") == 150.0
