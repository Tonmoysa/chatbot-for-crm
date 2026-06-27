"""Ambiguous category delete — must clarify, never delete wrong item."""

from __future__ import annotations

from chat.services.platform.field_extractors.expense import (
    normalize_expense_mutation_turn,
    review_field_updates_from_message,
    sanitize_expense_review_updates,
)
from chat.services.platform.schemas import FieldUpdate
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft


def _four_item_review_memory() -> SessionMemory:
    return SessionMemory(
        active_workflow=ActiveWorkflow(id="expense", stage="confirm_submit", draft_id="default"),
        pending_confirmation="submit",
        workflow_drafts={
            "default": WorkflowDraft(
                workflow_id="expense",
                fields={
                    "incurred_date": "2026-06-27",
                    "items": [
                        {
                            "category": "bus",
                            "amount": 30.0,
                            "from_location": "kamlapur",
                            "to_location": "dhanmondi",
                            "id": "eb7ded46",
                        },
                        {"category": "lunch", "amount": 120.0, "id": "f6b8634d"},
                        {
                            "category": "bike",
                            "amount": 130.0,
                            "from_location": "dhanmondi",
                            "to_location": "mirpur",
                            "id": "32f85204",
                        },
                        {
                            "category": "bus",
                            "amount": 30.0,
                            "from_location": "Mirpur",
                            "to_location": "Office",
                            "id": "dbf4b54b",
                        },
                    ],
                },
            )
        },
    )


def test_bus_delete_with_two_buses_clarifies_not_bike():
    memory = _four_item_review_memory()
    llm_turn = {
        "intent": "delete",
        "item_patches": [{"action": "delete", "item_index": 2}],
        "delete_indices": [2],
    }
    grounded = normalize_expense_mutation_turn(llm_turn, "bus delete koro", memory)
    assert grounded["intent"] == "clarify_delete"
    assert grounded.get("item_patches") == []
    assert set(grounded["clarify"]["candidate_indices"]) == {0, 3}
    assert grounded["clarify"]["category"] == "bus"


def test_review_path_blocks_wrong_llm_delete():
    memory = _four_item_review_memory()
    llm_turn = {
        "intent": "delete",
        "item_patches": [{"action": "delete", "item_index": 2}],
        "delete_indices": [2],
    }
    updates = review_field_updates_from_message(
        "bus delete koro",
        memory,
        expense_turn=llm_turn,
    )
    assert updates == []


def test_sanitize_rejects_category_mismatch_delete():
    memory = _four_item_review_memory()
    bad_updates = [
        FieldUpdate(field="items", value={}, item_index=2, action="delete"),
    ]
    sanitized = sanitize_expense_review_updates(
        bad_updates,
        "bus delete koro",
        memory=memory,
    )
    assert sanitized == []


def test_single_bus_delete_uses_correct_index():
    memory = _four_item_review_memory()
    items = memory.active_draft().fields["items"]
    items.pop(3)
    memory.active_draft().fields["items"] = items

    llm_turn = {
        "intent": "delete",
        "item_patches": [{"action": "delete", "item_index": 2}],
        "delete_indices": [2],
    }
    grounded = normalize_expense_mutation_turn(llm_turn, "bus delete koro", memory)
    assert grounded["intent"] == "delete"
    assert grounded["delete_indices"] == [0]
