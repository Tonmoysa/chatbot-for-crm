"""Generic modification and delete target resolution."""

from __future__ import annotations

from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.schemas import FieldUpdate, TargetRef, UnderstandingResult
from chat.services.session_memory import SessionMemory, StatePatchBuffer, apply_state_patches


class ModificationEngine:
    def __init__(self, fields: FieldEngine | None = None) -> None:
        self.fields = FieldEngine() if fields is None else fields

    def apply_understanding(
        self,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        *,
        state: StatePatchBuffer | None = None,
    ) -> list[str]:
        draft = memory.active_draft()
        if not draft or draft.locked:
            return ["Cannot modify a submitted or missing draft."]

        applied: list[str] = []
        if understanding.field_updates:
            if state is not None:
                state.apply_field_updates(understanding.field_updates)
            else:
                self.fields.apply_updates(draft, understanding.field_updates)
            applied.extend(u.field for u in understanding.field_updates)

        for target in understanding.targets:
            msg = self._delete_target(draft, target, state=state)
            if msg:
                applied.append(msg)

        if applied and understanding.entities:
            patch = {"op": "merge_last_entities", "value": dict(understanding.entities)}
            if state is not None:
                state.push("merge_last_entities", value=dict(understanding.entities))
            else:
                apply_state_patches(memory, [patch])
        return applied

    def _delete_target(
        self,
        draft,
        target: TargetRef,
        *,
        state: StatePatchBuffer | None = None,
    ) -> str:
        if target.field == "items" and target.item_index is not None:
            if state is not None:
                state.apply_field_updates(
                    [
                        FieldUpdate(
                            field="items",
                            value=None,
                            item_index=target.item_index,
                            action="delete",
                        )
                    ]
                )
                return f"items[{target.item_index}]"
            items = list(draft.fields.get("items") or draft.line_items or [])
            if 0 <= target.item_index < len(items):
                items.pop(target.item_index)
                draft.fields["items"] = items
                draft.line_items = items
                draft.version += 1
                return f"items[{target.item_index}]"
        elif target.field in draft.fields:
            if state is not None:
                state.remove_draft_field(target.field)
            else:
                draft.fields.pop(target.field, None)
                draft.version += 1
            return target.field
        return ""
