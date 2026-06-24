"""Leave field extraction — LLM merge and validation."""

from __future__ import annotations

from chat.services.platform.field_extractors.leave import (
    extract_leave_fields,
    extract_leave_reason_via_llm,
    infer_leave_reason_from_history,
    merge_leave_field_dicts,
    normalize_leave_type_value,
    remember_leave_narrative_seed,
)
from chat.services.session_memory import ActiveWorkflow, SessionMemory, WorkflowDraft
from tests.helpers.leave_llm_mock import mock_leave_llm
from chat.services.platform.field_engine import FieldEngine
from chat.services.platform.schemas import FieldUpdate, UnderstandingResult


NARRATIVE = (
    "Hello, amar ekta leave apply korte hobe. Next Wednesday theke Friday porjonto "
    "ami office e aste parbo na. Basay ekta family function ache ebong kichu personal "
    "responsibility o handle korte hobe. Motamoti 3 diner leave lagbe. Eta Annual Leave "
    "hisebe consider korle bhalo hoy."
)

WEDDING_NARRATIVE = (
    "Hi, amar ekta leave apply korte hobe. Amar choto boner biye fix hoyeche, tai family-r "
    "shob arrangement amakei dekhte hocche. Ei karone ami 5 August 2026 theke 9 August 2026 "
    "porjonto office e attend korte parbo na. Mot 5 diner leave lagbe. Eta Casual Leave hisebe "
    "apply korte chai."
)

HEALTH_SICK_NARRATIVE = (
    "Hey, amar ekta leave apply korte hobe. Kichudin dhore amar health ta bhalo jacche na, "
    "doctor dekhanor por uni kichudin complete rest nite bolechen. Tai ami 18 August 2026 theke "
    "21 August 2026 porjonto office e aste parbo na. Mot 4 diner leave lagbe. Eta Sick Leave "
    "hisebe apply korte chai."
)

FATHER_OPERATION_NARRATIVE = (
    "baba-r operation hoyeche, tar sathe hospital e thakte hobe. Sick Leave hisebe apply korte chai. "
    "Monday theke Thursday porjonto office e aste parbo na."
)

DADI_VILLAGE_NARRATIVE = (
    "Hello, amar ekta leave apply korte hobe. Ashole amar dadi onekdin dhore osustho chilen, "
    "kal rate abar tar obostha kharap hoyeche. Family theke shobai gram e jacche, amakeo jete hobe. "
    "Tai ami 14 September 2026 theke 17 September 2026 porjonto office attend korte parbo na. "
    "Mot 4 diner leave lagbe. Eta Annual Leave hisebe apply korte chai."
)


def test_rules_extract_leave_fields_empty():
    assert extract_leave_fields(NARRATIVE) == {}
    assert extract_leave_fields(WEDDING_NARRATIVE) == {}


def test_wedding_narrative_reason_from_llm_merge():
    merged = merge_leave_field_dicts(
        {},
        {
            "leave_type": "annual",
            "start_date": "2026-08-05",
            "end_date": "2026-08-09",
            "day_scope": "full_day",
            "reason": "Younger sister's wedding",
        },
        WEDDING_NARRATIVE,
    )
    assert merged.get("start_date") == "2026-08-05"
    assert merged.get("end_date") == "2026-08-09"
    assert "wedding" in (merged.get("reason") or "").lower()


def test_merge_rejects_garbage_llm_reason():
    merged = merge_leave_field_dicts(
        {},
        {"reason": "ami 5 August theke 9 August porjonto office e attend korte parbo na"},
        WEDDING_NARRATIVE,
    )
    assert "reason" not in merged


def test_dadi_village_narrative_reason_via_llm():
    with mock_leave_llm():
        reason = extract_leave_reason_via_llm(DADI_VILLAGE_NARRATIVE)
    assert reason
    low = reason.lower()
    assert "grandfather" in low or "village" in low or "family" in low


def test_grounding_always_calls_llm_even_when_rules_return_dates():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={})},
    )
    remember_leave_narrative_seed(memory, DADI_VILLAGE_NARRATIVE)
    engine = FieldEngine()
    from chat.services.platform.schemas import UnderstandingAction

    result = UnderstandingResult(
        goal="Start leave",
        workflow="leave",
        action=UnderstandingAction.START.value,
        confidence=0.84,
        field_updates=[
            FieldUpdate(field="start_date", value="2026-09-14", action="set"),
            FieldUpdate(field="end_date", value="2026-09-17", action="set"),
            FieldUpdate(field="leave_type", value="annual", action="set"),
            FieldUpdate(field="day_scope", value="full_day", action="set"),
        ],
        source="rules",
    )
    with mock_leave_llm():
        grounded = engine.ground_leave_understanding(
            DADI_VILLAGE_NARRATIVE,
            result,
            memory=memory,
            trace_id="test-dadi-ground",
        )
    by_field = {u.field: u.value for u in (grounded.field_updates or [])}
    assert by_field.get("reason")
    assert "grandfather" in str(by_field.get("reason") or "").lower() or "village" in str(
        by_field.get("reason") or ""
    ).lower()


def test_skip_recovers_reason_from_narrative_seed_via_llm():
    memory = SessionMemory(
        active_workflow=ActiveWorkflow(id="leave", stage="collecting"),
        workflow_drafts={"default": WorkflowDraft(workflow_id="leave", fields={"leave_type": "annual"})},
    )
    remember_leave_narrative_seed(memory, DADI_VILLAGE_NARRATIVE)
    with mock_leave_llm():
        reason = infer_leave_reason_from_history(memory, conversation_history=[], trace_id="test-skip-seed")
    assert reason
    assert "grandfather" in reason.lower() or "village" in reason.lower()


def test_long_llm_reason_not_rejected_as_complaint():
    merged = merge_leave_field_dicts(
        {},
        {
            "reason": (
                "Grandfather has been unwell for a long time; family traveling to village "
                "after condition worsened"
            )
        },
        DADI_VILLAGE_NARRATIVE,
    )
    assert merged.get("reason")
    assert "grandfather" in merged["reason"].lower() or "village" in merged["reason"].lower()


def test_merge_leave_field_dicts_llm_dates():
    merged = merge_leave_field_dicts(
        {},
        {"start_date": "2026-06-24", "end_date": "2026-06-26", "leave_type": "annual"},
        NARRATIVE,
    )
    assert merged["start_date"] == "2026-06-24"
    assert merged["end_date"] == "2026-06-26"
    assert merged["leave_type"] == "annual"


def test_ground_leave_understanding_uses_llm_fields_only():
    engine = FieldEngine()
    result = UnderstandingResult(
        workflow="leave",
        action="start",
        field_updates=[
            FieldUpdate(field="start_date", value="2026-06-24"),
            FieldUpdate(field="end_date", value="2026-06-26"),
            FieldUpdate(field="leave_type", value="annual"),
            FieldUpdate(
                field="reason",
                value="Family function; Personal responsibilities",
            ),
        ],
        source="llm",
    )
    grounded = engine.ground_leave_understanding(NARRATIVE, result, memory=SessionMemory())
    by_field = {u.field: u.value for u in grounded.field_updates}
    assert by_field.get("start_date") == "2026-06-24"
    assert by_field.get("end_date") == "2026-06-26"
    assert by_field.get("leave_type") == "annual"
    assert "Family function" in (by_field.get("reason") or "")


def test_health_sick_narrative_from_llm_merge():
    merged = merge_leave_field_dicts(
        {},
        {
            "leave_type": "sick",
            "start_date": "2026-08-18",
            "end_date": "2026-08-21",
            "day_scope": "full_day",
        },
        HEALTH_SICK_NARRATIVE,
    )
    assert merged.get("leave_type") == "sick"
    assert merged.get("start_date") == "2026-08-18"


def test_family_narrative_llm_omits_sick_type():
    merged = merge_leave_field_dicts(
        {},
        {"reason": "Mama osustho", "start_date": "2026-07-11", "end_date": "2026-07-13"},
        "mama osustho, agami theke 3 din leave lagbe",
    )
    assert merged.get("leave_type") is None
    assert merged.get("reason") == "Mama osustho"


def test_father_operation_llm_merge():
    merged = merge_leave_field_dicts(
        {},
        {"reason": "Father's operation; hospital stay", "start_date": "2026-07-14", "end_date": "2026-07-17"},
        FATHER_OPERATION_NARRATIVE,
    )
    assert merged.get("leave_type") is None
    assert "operation" in (merged.get("reason") or "").lower()


def test_coerce_llm_date_output_accepts_natural_llm_strings():
    from chat.services.platform.field_extractors.leave import _coerce_llm_date_output

    assert _coerce_llm_date_output("2026-09-13") == "2026-09-13"
    assert _coerce_llm_date_output("13 September 2026") == "2026-09-13"


def test_ground_leave_canonical_requested_type_maps_to_leave_type():
    engine = FieldEngine()
    result = UnderstandingResult(
        workflow="leave",
        action="start",
        field_updates=[
            FieldUpdate(field="start_date", value="2026-08-18"),
            FieldUpdate(field="end_date", value="2026-08-21"),
            FieldUpdate(field="leave_type", value="sick"),
            FieldUpdate(field="day_scope", value="full_day"),
        ],
        entities={"requested_leave_type": "sick"},
        source="llm",
    )
    grounded = engine.ground_leave_understanding(
        HEALTH_SICK_NARRATIVE, result, memory=SessionMemory()
    )
    by_field = {u.field: u.value for u in grounded.field_updates}
    assert by_field.get("leave_type") == "sick"
    assert (grounded.entities or {}).get("requested_leave_type") is None


def test_ground_leave_noncanonical_requested_type_strips_leave_type():
    engine = FieldEngine()
    result = UnderstandingResult(
        workflow="leave",
        action="start",
        field_updates=[
            FieldUpdate(field="start_date", value="2026-08-05"),
            FieldUpdate(field="end_date", value="2026-08-09"),
            FieldUpdate(field="leave_type", value="annual"),
        ],
        entities={"requested_leave_type": "casual"},
        source="llm",
    )
    grounded = engine.ground_leave_understanding(
        WEDDING_NARRATIVE, result, memory=SessionMemory()
    )
    by_field = {u.field: u.value for u in grounded.field_updates}
    assert "leave_type" not in by_field
    assert (grounded.entities or {}).get("requested_leave_type") == "casual"
