"""Load workflow definitions from YAML configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from chat.services.platform.schemas import FieldDefinition, ValidationRule, WorkflowDefinition

_DEFINITIONS_DIR = Path(__file__).resolve().parent / "workflow_definitions"


def _parse_workflow(raw: dict[str, Any]) -> WorkflowDefinition:
    fields = [FieldDefinition.from_dict(f) for f in (raw.get("fields") or [])]
    rules = [ValidationRule.from_dict(r) for r in (raw.get("validation_rules") or [])]
    return WorkflowDefinition(
        workflow_id=str(raw["id"]),
        name=str(raw.get("name") or raw["id"]),
        fields=fields,
        validation_rules=rules,
        crm_intent=str(raw.get("crm_intent") or raw["id"].upper()),
        requires_review=bool(raw.get("requires_review", True)),
        requires_confirmation=bool(raw.get("requires_confirmation", True)),
    )


@lru_cache(maxsize=16)
def get_workflow_definition(workflow_id: str) -> WorkflowDefinition | None:
    wf_id = (workflow_id or "").strip().lower()
    path = _DEFINITIONS_DIR / f"{wf_id}.yaml"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        return None
    return _parse_workflow(raw)


def list_workflow_ids() -> list[str]:
    return sorted(p.stem for p in _DEFINITIONS_DIR.glob("*.yaml"))


def reload_definitions() -> None:
    get_workflow_definition.cache_clear()
