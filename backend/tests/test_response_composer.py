"""Phase 8 — response composer copy and summary vs review routing."""

from chat.services.platform.response_composer import ResponseComposer, leave_field_prompt, leave_validation_message
from chat.services.platform.schemas import WorkflowDefinition
from chat.services.session_memory import WorkflowDraft


def test_workflow_summary_expense_with_total():
    composer = ResponseComposer()
    draft = WorkflowDraft(
        workflow_id="expense",
        fields={"items": [{"category": "meals", "amount": 150}]},
    )
    defn = WorkflowDefinition(workflow_id="expense", name="Expense", fields=[])
    msg = composer.workflow_summary(defn, draft, emphasize_total=True)
    assert "150" in msg
    assert "Total" in msg


def test_review_ready_message_joins_prefix():
    composer = ResponseComposer()
    msg = composer.review_ready_message("Saved **date**.", "**Leave — Review**\n\n_Reply yes_")
    assert "Saved **date**." in msg
    assert "Review" in msg


def test_slot_still_needed_bilingual():
    composer = ResponseComposer()
    en = composer.slot_still_needed("leave_type", "Which type?", lang="en")
    bn = composer.slot_still_needed("leave_type", "Which type?", lang="bn")
    assert "leave type" in en.lower()
    assert "leave_type" not in bn
    assert "দরকার" in bn


def test_general_help_and_platform_continue_clarify():
    composer = ResponseComposer()
    assert "leave" in composer.general_help(lang="en").lower()
    assert "দরকার" in composer.platform_continue_clarify("leave", lang="bn")
    assert composer.platform_continue_clarify("expense", reasoning="Need dates.", lang="en") == "Need dates."


def test_workflow_continuation_hint_open_draft():
    from chat.services.session_memory import ActiveWorkflow, SessionMemory

    composer = ResponseComposer()
    memory = SessionMemory()
    memory.active_workflow = ActiveWorkflow(id="leave", stage="collecting")
    hint = composer.workflow_continuation_hint(memory)
    assert "leave" in hint
    assert "continue" in hint.lower() or "saved" in hint.lower()


def test_policy_rules_footer():
    from chat.services.informational_responses import policy_rules_footer

    assert "policies" in policy_rules_footer(lang="en")
    assert "পলিসি" in policy_rules_footer(lang="bn")


def test_composer_facade_rules_footer_and_continuation():
    from chat.services.informational_responses import policy_rules_footer
    from chat.services.session_memory import ActiveWorkflow, SessionMemory

    composer = ResponseComposer()
    assert composer.rules_footer(lang="en") == policy_rules_footer(lang="en")
    memory = SessionMemory(active_workflow=ActiveWorkflow(id="leave", stage="collecting"))
    msg = composer.with_continuation_hint("Hello", memory)
    assert msg.startswith("Hello")
    assert "leave" in msg


def test_composer_today_date():
    composer = ResponseComposer()
    en = composer.today_date(today_iso="2026-06-21", lang="en")
    bn = composer.today_date(today_iso="2026-06-21", lang="bn")
    assert "2026-06-21" in en
    assert "2026-06-21" in bn


def test_leave_field_prompts_centralized():
    en = leave_field_prompt("leave_type", lang="en")
    bn = leave_field_prompt("leave_type", lang="bn")
    assert "annual" in en and "sick" in en and "lwop" in en
    assert "annual" in bn and "ছুটি" in bn
    assert leave_validation_message("day_scope_required", lang="bn") == "পুরো দিন নাকি অর্ধ দিন জানান।"


def test_leave_review_and_summary_bilingual():
    from chat.services.platform.registry import get_workflow_definition

    composer = ResponseComposer()
    draft = WorkflowDraft(
        workflow_id="leave",
        fields={
            "leave_type": "annual",
            "day_scope": "full_day",
            "start_date": "2026-08-15",
            "reason_skipped": True,
        },
    )
    defn = get_workflow_definition("leave")
    review_bn = composer.leave_review(draft, defn, lang="bn")
    assert "পর্যালোচনা" in review_bn
    assert "ছুটির ধরন" in review_bn
    assert "ha" in review_bn

    summary_en = composer.leave_summary(draft, lang="en")
    summary_bn = composer.leave_summary(draft, lang="bn")
    assert "Leave Summary" in summary_en
    assert "ছুটি সারাংশ" in summary_bn
    assert "annual" in summary_en
    assert "ছুটির ধরন" in summary_bn


def test_field_engine_leave_next_question_uses_centralized_prompts():
    from chat.services.platform.field_engine import FieldEngine
    from chat.services.platform.registry import get_workflow_definition
    from chat.services.session_memory import SessionMemory, WorkflowDraft

    engine = FieldEngine()
    defn = get_workflow_definition("leave")
    memory = SessionMemory()
    draft = WorkflowDraft(workflow_id="leave", fields={"start_date": "2026-08-15"})
    pq = engine.next_question(memory, draft, defn, lang="bn")
    assert pq is not None
    assert pq.field == "leave_type"
    assert pq.prompt == leave_field_prompt("leave_type", lang="bn")
