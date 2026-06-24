"""Phase 9 — YAML conversation scenario loader and runner."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml
from django.db import transaction

from chat.models import ConversationSession
from chat.services.orchestrator import ChatOrchestrator
from chat.services.session_memory import load_session_memory

_SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "scenarios"


@contextmanager
def llm_disabled() -> Iterator[None]:
    """Disable LLM for deterministic rules-only scenario runs."""
    from unittest.mock import patch

    with patch("chat.services.pending_question_engine.LLMClient") as pq_llm, patch(
        "chat.services.platform.ai_understanding.LLMClient"
    ) as ai_llm, patch("chat.services.conversational.LLMClient") as conv_llm, patch(
        "chat.services.llm_client.LLMClient"
    ) as root_llm:
        for mock in (pq_llm, ai_llm, conv_llm, root_llm):
            mock.return_value.is_configured.return_value = False
            mock.return_value.chat_json.return_value = None
        yield


def scenarios_dir() -> Path:
    return _SCENARIOS_DIR


def load_yaml_scenarios() -> list[dict[str, Any]]:
    """Load all scenarios from ``tests/scenarios/*.yaml``."""
    out: list[dict[str, Any]] = []
    for path in sorted(scenarios_dir().glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            continue
        source = path.name
        for item in raw.get("scenarios") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            scenario = dict(item)
            scenario["_source"] = source
            out.append(scenario)
    return out


def load_paraphrase_cases() -> list[dict[str, Any]]:
    """Expand ``paraphrase_groups`` from YAML into individual test cases."""
    out: list[dict[str, Any]] = []
    for path in sorted(scenarios_dir().glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            continue
        source = path.name
        for group in raw.get("paraphrase_groups") or []:
            if not isinstance(group, dict) or not group.get("id"):
                continue
            group_id = str(group["id"])
            users = list(group.get("users") or [])
            for index, user_msg in enumerate(users):
                user_text = str(user_msg or "").strip()
                if not user_text:
                    continue
                out.append(
                    {
                        "id": f"{group_id}__{index + 1}",
                        "group_id": group_id,
                        "user": user_text,
                        "seed_turns": list(group.get("seed_turns") or []),
                        "seed_memory": dict(group["seed_memory"]) if group.get("seed_memory") else None,
                        "expect_any": list(group.get("expect_any") or []),
                        "expect_none": list(group.get("expect_none") or []),
                        "expect_all": list(group.get("expect_all") or []),
                        "state_after": dict(group["state_after"]) if group.get("state_after") else None,
                        "company_id": group.get("company_id"),
                        "employee_id": group.get("employee_id"),
                        "_source": source,
                    }
                )
    return out


def llm_tests_enabled() -> bool:
    """True when Phase 3 LLM regression tests should run (env + API key)."""
    import os

    from django.conf import settings

    if os.environ.get("RUN_LLM_TESTS", "").strip().lower() not in ("1", "true", "yes"):
        return False
    key = (getattr(settings, "LLM_API_KEY", None) or os.environ.get("LLM_API_KEY") or "").strip()
    return bool(key)


def _match_patterns(msg: str, patterns: list[str], *, mode: str) -> bool:
    low = (msg or "").lower()
    if not patterns:
        return True
    if mode == "all":
        return all(re.search(p, low, re.I) for p in patterns)
    if mode == "none":
        return not any(re.search(p, low, re.I) for p in patterns)
    return any(re.search(p, low, re.I) for p in patterns)


def _assert_turn_expectations(
    *,
    scenario_id: str,
    turn_index: int,
    message: str,
    reply: str,
    turn: dict[str, Any],
) -> None:
    label = f"{scenario_id} turn {turn_index + 1}"
    if turn.get("expect_all"):
        assert _match_patterns(reply, list(turn["expect_all"]), mode="all"), (
            f"{label}: reply missing required patterns {turn['expect_all']!r}\n"
            f"user: {message!r}\nreply: {reply!r}"
        )
    if turn.get("expect_any"):
        assert _match_patterns(reply, list(turn["expect_any"]), mode="any"), (
            f"{label}: reply matched none of {turn['expect_any']!r}\n"
            f"user: {message!r}\nreply: {reply!r}"
        )
    if turn.get("expect_none"):
        assert _match_patterns(reply, list(turn["expect_none"]), mode="none"), (
            f"{label}: reply must not match {turn['expect_none']!r}\n"
            f"user: {message!r}\nreply: {reply!r}"
        )


def _draft_fields(memory) -> dict[str, Any]:
    draft = memory.active_draft()
    return dict(draft.fields) if draft else {}


def _assert_state_expectations(
    *,
    scenario_id: str,
    turn_index: int,
    memory,
    expected: dict[str, Any],
) -> None:
    label = f"{scenario_id} turn {turn_index + 1} state"
    if "active_workflow_id" in expected:
        aw = memory.active_workflow
        actual = aw.id if aw else None
        assert actual == expected["active_workflow_id"], (
            f"{label}: active_workflow_id expected {expected['active_workflow_id']!r}, got {actual!r}"
        )
    if "pending_confirmation" in expected:
        assert memory.pending_confirmation == expected["pending_confirmation"], (
            f"{label}: pending_confirmation expected {expected['pending_confirmation']!r}, "
            f"got {memory.pending_confirmation!r}"
        )
    if "pending_question_field" in expected:
        pq = memory.pending_question
        actual = pq.field if pq else None
        assert actual == expected["pending_question_field"], (
            f"{label}: pending_question_field expected {expected['pending_question_field']!r}, got {actual!r}"
        )
    if "pending_question_field_not" in expected:
        pq = memory.pending_question
        actual = pq.field if pq else None
        assert actual != expected["pending_question_field_not"], (
            f"{label}: pending_question_field must not be {expected['pending_question_field_not']!r}, got {actual!r}"
        )
    fields = _draft_fields(memory)
    if "draft_fields" in expected:
        for key, value in dict(expected["draft_fields"]).items():
            assert fields.get(key) == value, (
                f"{label}: draft field {key!r} expected {value!r}, got {fields.get(key)!r}"
            )
    for key in list(expected.get("draft_field_present") or []):
        assert fields.get(key) not in (None, ""), (
            f"{label}: draft field {key!r} expected to be set, got {fields.get(key)!r}"
        )
    for key in list(expected.get("draft_field_absent") or []):
        assert fields.get(key) in (None, ""), (
            f"{label}: draft field {key!r} expected absent, got {fields.get(key)!r}"
        )
    for key, value in dict(expected.get("draft_field_not_equals") or {}).items():
        assert fields.get(key) != value, (
            f"{label}: draft field {key!r} must not equal {value!r}, got {fields.get(key)!r}"
        )


def _apply_seed_memory(
    orch: ChatOrchestrator,
    *,
    company_id: str,
    employee_id: str,
    seed: dict[str, Any],
) -> str:
    """Pre-load session memory before the first turn (e.g. submitted leave ranges)."""
    from chat.services.session_memory import save_session_memory

    bundle = orch.session_store.open(company_id=company_id, employee_id=employee_id)
    memory = bundle.memory
    facts = dict(memory.conversation_facts or {})
    facts.update(dict(seed.get("conversation_facts") or {}))
    memory.conversation_facts = facts
    save_session_memory(bundle.session, memory)
    return bundle.session.session_id


def run_yaml_scenario(scenario: dict[str, Any], *, use_llm: bool = False) -> None:
    """Execute one YAML scenario through ChatOrchestrator (production path)."""
    scenario_id = str(scenario["id"])
    company_id = str(scenario.get("company_id") or f"co-{scenario_id}")
    employee_id = str(scenario.get("employee_id") or f"emp-{scenario_id}")
    trace_id = str(scenario.get("trace_id") or f"yaml-{scenario_id}")
    turns = list(scenario.get("turns") or [])
    assert turns, f"{scenario_id}: no turns"

    ctx = llm_disabled() if not use_llm else _nullcontext()
    with ctx:
        orch = ChatOrchestrator()
        session_id = ""
        seed = scenario.get("seed_memory")
        if seed:
            with transaction.atomic():
                session_id = _apply_seed_memory(
                    orch,
                    company_id=company_id,
                    employee_id=employee_id,
                    seed=dict(seed),
                )

        for index, turn in enumerate(turns):
            message = str(turn.get("user") or "").strip()
            assert message, f"{scenario_id} turn {index + 1}: empty user message"

            with transaction.atomic():
                result = orch.run_chat(
                    message=message,
                    session_id=session_id or None,
                    company_id=company_id,
                    employee_id=employee_id,
                    trace_id=trace_id,
                )
            session_id = result.get("_session_id") or session_id
            reply = (result.get("response") or {}).get("message") or ""

            _assert_turn_expectations(
                scenario_id=scenario_id,
                turn_index=index,
                message=message,
                reply=reply,
                turn=turn,
            )

            state_after = turn.get("state_after")
            if state_after and session_id:
                session = ConversationSession.objects.get(
                    company_id=company_id,
                    employee_id=employee_id,
                    session_id=session_id,
                )
                memory = load_session_memory(session)
                _assert_state_expectations(
                    scenario_id=scenario_id,
                    turn_index=index,
                    memory=memory,
                    expected=dict(state_after),
                )


def run_paraphrase_case(case: dict[str, Any], *, use_llm: bool = False) -> None:
    """Run seed turns then one paraphrased user message; assert group expectations."""
    case_id = str(case["id"])
    group_id = str(case.get("group_id") or case_id)
    company_id = str(case.get("company_id") or f"co-{group_id}")
    employee_id = str(case.get("employee_id") or f"emp-{group_id}")
    trace_id = f"para-{case_id}"
    user_message = str(case.get("user") or "").strip()
    assert user_message, f"{case_id}: empty user paraphrase"

    turn_expectations = {
        "expect_any": case.get("expect_any"),
        "expect_none": case.get("expect_none"),
        "expect_all": case.get("expect_all"),
    }

    ctx = llm_disabled() if not use_llm else _nullcontext()
    with ctx:
        orch = ChatOrchestrator()
        session_id = ""
        seed = case.get("seed_memory")
        if seed:
            with transaction.atomic():
                session_id = _apply_seed_memory(
                    orch,
                    company_id=company_id,
                    employee_id=employee_id,
                    seed=dict(seed),
                )

        for index, seed_turn in enumerate(list(case.get("seed_turns") or [])):
            message = str(seed_turn.get("user") or "").strip()
            assert message, f"{case_id} seed {index + 1}: empty user"
            with transaction.atomic():
                result = orch.run_chat(
                    message=message,
                    session_id=session_id or None,
                    company_id=company_id,
                    employee_id=employee_id,
                    trace_id=trace_id,
                )
            session_id = result.get("_session_id") or session_id

        with transaction.atomic():
            result = orch.run_chat(
                message=user_message,
                session_id=session_id or None,
                company_id=company_id,
                employee_id=employee_id,
                trace_id=trace_id,
            )
        reply = (result.get("response") or {}).get("message") or ""
        assert reply, f"{case_id}: empty bot reply for paraphrase {user_message!r}"

        _assert_turn_expectations(
            scenario_id=case_id,
            turn_index=0,
            message=user_message,
            reply=reply,
            turn=turn_expectations,
        )

        state_after = case.get("state_after")
        session_id = result.get("_session_id") or session_id
        if state_after and session_id:
            session = ConversationSession.objects.get(
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
            )
            memory = load_session_memory(session)
            _assert_state_expectations(
                scenario_id=case_id,
                turn_index=0,
                memory=memory,
                expected=dict(state_after),
            )


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield
