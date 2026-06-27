"""Leave workflow policy interrupts — runtime patches (disk-safe via apps.ready)."""

from __future__ import annotations


def _leave_navigation_shortcut(message: str, memory) -> object | None:
    from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
    from chat.services.platform.intent_rules import should_resume_suspended_leave

    raw = (message or "").strip()
    if not raw:
        return None
    aw = memory.active_workflow
    if not should_resume_suspended_leave(
        message=raw,
        active_workflow_id=aw.id if aw else None,
        suspended_workflows=memory.suspended_workflows,
        memory=memory,
    ):
        return None
    return PendingQuestionDecision(
        kind=MessageIntentKind.SWITCH_WORKFLOW,
        confidence=0.94,
        reasoning="Navigate to leave draft in session.",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="leave",
    )


def _patch_intent_priority_policy_guard() -> None:
    import chat.services.pending_question_engine as pqe

    original = pqe.intent_priority_decision

    def intent_priority_decision(message, *, memory, understanding):
        from chat.services._policy_interrupt import is_informational_interrupt_message

        if is_informational_interrupt_message(message):
            return None
        return original(message, memory=memory, understanding=understanding)

    pqe.intent_priority_decision = intent_priority_decision


def _patch_informational_priority_leave() -> None:
    import chat.services.pending_question_engine as pqe

    original = pqe.informational_priority_decision

    def informational_priority_decision(message, *, memory, **kwargs):
        from chat.services._policy_interrupt import is_informational_interrupt_message
        from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
        from chat.services.platform.intent_rules import should_resume_suspended_leave

        nav = _leave_navigation_shortcut(message, memory)
        if nav is not None:
            return nav

        raw = (message or "").strip()
        aw = memory.active_workflow
        if should_resume_suspended_leave(
            message=raw,
            active_workflow_id=aw.id if aw else None,
            suspended_workflows=memory.suspended_workflows,
            memory=memory,
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=0.94,
                reasoning="Navigate to suspended leave draft.",
                source="rules",
                blocks_new_workflow=True,
                target_workflow="leave",
            )

        if raw and is_informational_interrupt_message(raw):
            kwargs = {**kwargs, "include_policy_status": True}

        return original(message, memory=memory, **kwargs)

    pqe.informational_priority_decision = informational_priority_decision


def _patch_session_switch_to_workflow_leave() -> None:
    import chat.services.session_memory as sm

    original = sm.reduce_switch_to_workflow

    def reduce_switch_to_workflow(memory, target_workflow_id: str):
        target = target_workflow_id.strip().lower()
        if target == "leave":
            from chat.services.platform.field_engine import leave_draft_in_progress

            for draft_id in ("leave", "default"):
                draft = (memory.workflow_drafts or {}).get(draft_id)
                if (
                    draft
                    and getattr(draft, "workflow_id", None) == "leave"
                    and leave_draft_in_progress(draft)
                    and not getattr(draft, "locked", False)
                ):
                    active = memory.active_workflow
                    if active and active.id == "leave" and active.draft_id == draft_id:
                        return draft
                    sm.reduce_set_active_workflow(
                        memory,
                        sm.ActiveWorkflow(id="leave", stage="collecting", draft_id=draft_id),
                    )
                    sm.reduce_clear_pending_question(memory)
                    sm.reduce_clear_pending_confirmation(memory)
                    return draft
        return original(memory, target_workflow_id)

    sm.reduce_switch_to_workflow = reduce_switch_to_workflow


def apply() -> None:
    _patch_intent_priority_policy_guard()
    _patch_informational_priority_leave()
    _patch_session_switch_to_workflow_leave()
