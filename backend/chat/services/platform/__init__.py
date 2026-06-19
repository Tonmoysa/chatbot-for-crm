"""Universal conversational workflow platform."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat.services.platform.pipeline import WorkflowPipeline

__all__ = ["WorkflowPipeline"]


def __getattr__(name: str):
    if name == "WorkflowPipeline":
        from chat.services.platform.pipeline import WorkflowPipeline

        return WorkflowPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
