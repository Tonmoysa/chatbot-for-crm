"""Expense draft editor — state machine unit tests."""

from __future__ import annotations

from chat.services.platform.field_extractors.expense import (
    apply_expense_patches,
    build_pending_queue,
    compute_item_missing_fields,
    patches_to_field_updates,
    sync_expense_draft_fields,
)
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft


def test_sync_assigns_missing_fields_per_item():
    fields = sync_expense_draft_fields(
        {
            "items": [
                {"category": "lunch", "amount": 280},
                {"amount": 150},
                {"category": "bus", "amount": 120},
            ]
        }
    )
    items = fields["items"]
    assert items[0]["missing_fields"] == []
    assert items[0]["status"] == "complete"
    assert "category" in items[1]["missing_fields"]
    assert "route" in items[2]["missing_fields"]


def test_pending_queue_oldest_incomplete_first():
    fields = sync_expense_draft_fields(
        {
            "items": [
                {"category": "lunch", "amount": 280},
                {"amount": 150},
                {"category": "bus", "amount": 120},
            ]
        }
    )
    queue = build_pending_queue(fields["items"])
    assert queue[0].item_index == 1
    assert queue[0].field == "category"
    assert queue[1].item_index == 2
    assert queue[1].field == "route"


def test_correction_updates_existing_item_not_duplicate():
    fields = sync_expense_draft_fields(
        {"items": [{"category": "lunch", "amount": 280}]}
    )
    merged, notes = apply_expense_patches(
        fields,
        {
            "item_patches": [
                {"action": "update", "match_amount": 280, "amount": 300},
            ]
        },
    )
    assert len(merged["items"]) == 1
    assert merged["items"][0]["amount"] == 300
    assert "updated" in notes[0]


def test_append_adds_new_item():
    fields = sync_expense_draft_fields(
        {"items": [{"category": "lunch", "amount": 280}]}
    )
    merged, _ = apply_expense_patches(
        fields,
        {"item_patches": [{"action": "append", "category": "snack", "amount": 70}]},
    )
    assert len(merged["items"]) == 2
    assert merged["items"][1]["category"] == "snack"


def test_patches_to_field_updates_append():
    fields = {"items": []}
    updates = patches_to_field_updates(
        fields,
        {"item_patches": [{"action": "append", "category": "lunch", "amount": 250}]},
    )
    assert any(u.field == "items" and u.action == "append" for u in updates)


def test_draft_never_drops_existing_items_on_partial_patch():
    fields = sync_expense_draft_fields(
        {
            "items": [
                {"category": "lunch", "amount": 280},
                {"category": "snack", "amount": 70},
            ]
        }
    )
    merged, _ = apply_expense_patches(
        fields,
        {"item_patches": [{"action": "update", "item_index": 0, "amount": 300}]},
    )
    assert len(merged["items"]) == 2
    assert merged["items"][0]["amount"] == 300
    assert merged["items"][1]["category"] == "snack"
