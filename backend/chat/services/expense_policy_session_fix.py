"""Preserve workflow drafts across policy interrupts — runtime patches (disk-safe via apps.ready)."""

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


def _expense_navigation_shortcut(message: str, memory) -> object | None:
    from chat.services.pending_question_engine import MessageIntentKind, PendingQuestionDecision
    from chat.services.platform.intent_rules import expense_navigation_kind, should_resume_suspended_expense

    raw = (message or "").strip()
    if not raw:
        return None
    aw = memory.active_workflow
    if not should_resume_suspended_expense(
        message=raw,
        active_workflow_id=aw.id if aw else None,
        suspended_workflows=memory.suspended_workflows,
        memory=memory,
    ):
        return None
    nav = expense_navigation_kind(raw)
    return PendingQuestionDecision(
        kind=MessageIntentKind.SWITCH_WORKFLOW,
        confidence=0.94,
        reasoning="Navigate to expense draft in session.",
        source="rules",
        blocks_new_workflow=True,
        target_workflow="expense",
        extracted_entities={"expense_navigation": nav},
    )


def _patch_plan_shortcut_expense_nav() -> None:
    import chat.services.plan_shortcut_router as psr

    original = psr.detect_plan_shortcut

    def detect_plan_shortcut(message, *, memory, conversation_history):
        nav = _expense_navigation_shortcut(message, memory)
        if nav is not None:
            return nav
        return original(message, memory=memory, conversation_history=conversation_history)

    psr.detect_plan_shortcut = detect_plan_shortcut


def _patch_session_switch_to_workflow() -> None:
    import chat.services.session_memory as sm

    original = sm.reduce_switch_to_workflow

    def reduce_switch_to_workflow(memory, target_workflow_id: str):
        target = target_workflow_id.strip().lower()
        if target == "expense":
            from chat.services.platform.field_extractors.expense import _expense_draft_has_items

            for draft_id in ("expense", "default"):
                draft = (memory.workflow_drafts or {}).get(draft_id)
                if _expense_draft_has_items(draft):
                    active = memory.active_workflow
                    if active and active.id == "expense" and active.draft_id == draft_id:
                        return draft
                    sm.reduce_set_active_workflow(
                        memory,
                        sm.ActiveWorkflow(id="expense", stage="collecting", draft_id=draft_id),
                    )
                    sm.reduce_clear_pending_question(memory)
                    sm.reduce_clear_pending_confirmation(memory)
                    return draft
        return original(memory, target_workflow_id)

    sm.reduce_switch_to_workflow = reduce_switch_to_workflow


def _patch_skip_session_context_for_expense_nav() -> None:
    import chat.services.platform.turn_semantics as ts

    original = ts.should_skip_session_context_llm

    def should_skip_session_context_llm(message: str, memory):
        from chat.services.platform.field_extractors.expense import memory_has_expense_draft
        from chat.services.platform.intent_rules import is_expense_navigation_message

        active_id = (memory.active_workflow.id if memory.active_workflow else "").strip().lower()
        if is_expense_navigation_message(message) and (
            memory_has_expense_draft(memory) or active_id == "expense"
        ):
            return True
        return original(message, memory)

    ts.should_skip_session_context_llm = should_skip_session_context_llm


def _patch_workflow_show_review_wf_id() -> None:
    from chat.services.platform.pipeline import WorkflowPipeline
    from chat.services.platform.schemas import PlanOp, UnderstandingResult

    original = WorkflowPipeline._run_workflow_plan_op

    def patched_run(self, op, *, ctx, message, memory, understanding, pq_decision, **rest):
        if op == PlanOp.WORKFLOW_SHOW_REVIEW:
            valid = ("leave", "expense")
            wf_id = ""
            if pq_decision and (pq_decision.target_workflow or "").strip().lower() in valid:
                wf_id = str(pq_decision.target_workflow).strip().lower()
            else:
                show_target = str((understanding.entities or {}).get("show_workflow_target") or "").strip().lower()
                if show_target in valid:
                    wf_id = show_target
                elif memory.active_workflow and memory.active_workflow.id in valid:
                    wf_id = memory.active_workflow.id
                else:
                    wf = (understanding.workflow or "").strip().lower()
                    wf_id = wf if wf in valid else ""

            if wf_id in valid and (understanding.workflow or "").strip().lower() not in valid:
                understanding = UnderstandingResult(
                    goal=understanding.goal,
                    workflow=wf_id,
                    action=understanding.action,
                    confidence=understanding.confidence,
                    entities={
                        **dict(understanding.entities or {}),
                        "show_workflow_target": wf_id,
                    },
                    field_updates=understanding.field_updates,
                    targets=understanding.targets,
                    missing_fields=understanding.missing_fields,
                    is_out_of_scope=understanding.is_out_of_scope,
                    is_greeting=understanding.is_greeting,
                    interrupt_workflow=understanding.interrupt_workflow,
                    reasoning=understanding.reasoning,
                    source=understanding.source,
                    answers_pending_field=understanding.answers_pending_field,
                )

        return original(
            self,
            op,
            ctx=ctx,
            message=message,
            memory=memory,
            understanding=understanding,
            pq_decision=pq_decision,
            **rest,
        )

    WorkflowPipeline._run_workflow_plan_op = patched_run


def _patch_informational_priority_memory() -> None:
    import chat.services.pending_question_engine as pqe

    original = pqe.informational_priority_decision

    def informational_priority_decision(message, *, memory, **kwargs):
        nav = _expense_navigation_shortcut(message, memory)
        if nav is not None:
            return nav
        import chat.services.platform.intent_rules as ir

        orig_should = ir.should_resume_suspended_expense

        def should_with_memory(**kw):
            if kw.get("memory") is None:
                kw = {**kw, "memory": memory}
            return orig_should(**kw)

        ir.should_resume_suspended_expense = should_with_memory
        try:
            return original(message, memory=memory, **kwargs)
        finally:
            ir.should_resume_suspended_expense = orig_should

    pqe.informational_priority_decision = informational_priority_decision


def apply() -> None:
    _patch_plan_shortcut_expense_nav()
    _patch_informational_priority_memory()
    _patch_session_switch_to_workflow()
    _patch_skip_session_context_for_expense_nav()
    _patch_workflow_show_review_wf_id()
