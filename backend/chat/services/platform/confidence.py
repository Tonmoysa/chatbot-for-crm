"""Confidence guard — block irreversible actions when uncertain."""

from __future__ import annotations

from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult

HIGH = 0.85
MEDIUM = 0.70

_IRREVERSIBLE = frozenset(
    {
        UnderstandingAction.START.value,
        UnderstandingAction.SUBMIT.value,
        UnderstandingAction.DELETE.value,
        UnderstandingAction.SWITCH.value,
        UnderstandingAction.MODIFY.value,
    }
)


def apply_confidence_guard(understanding: UnderstandingResult) -> UnderstandingResult:
    action = understanding.action
    conf = understanding.confidence

    if action == UnderstandingAction.CLARIFICATION_NEEDED.value:
        return understanding

    if conf >= HIGH:
        return understanding

    if conf >= MEDIUM:
        understanding.reasoning = (
            f"{understanding.reasoning} (moderate confidence — please confirm if this looks right.)"
        ).strip()
        return understanding

    if action in _IRREVERSIBLE or action == UnderstandingAction.CONFIRM.value:
        understanding.action = UnderstandingAction.CLARIFICATION_NEEDED.value
        understanding.reasoning = (
            understanding.reasoning
            or f"Not confident enough ({conf:.2f}) to perform '{action}'."
        )
    return understanding


def should_block_workflow_action(action: str, confidence: float) -> bool:
    if confidence >= MEDIUM:
        return False
    return action in _IRREVERSIBLE or action == UnderstandingAction.CONFIRM.value
