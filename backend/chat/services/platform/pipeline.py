"""Main workflow pipeline — ties platform modules together."""

from __future__ import annotations

import re
from typing import Any

from chat.services.platform.field_engine import (
    FieldEngine,
    duplicate_leave_arm_entities,
    is_duplicate_leave_attempt,
    leave_draft_in_progress,
)
from chat.services.platform.modification_engine import ModificationEngine
from chat.services.platform.field_extractors.leave import is_medical_document_skip_message, is_reason_skip_message
from chat.services.platform.field_extractors import (
    looks_like_expense_item_delete,
    parse_delete_request,
    parse_modify_request,
    parse_relative_date,
)
from chat.services.platform.intent_rules import (
    find_submitted_leave_overlap_from_message,
    is_bare_confirmation,
    is_bare_rejection,
    is_cancel_workflow_message,
    is_expense_list_request,
    is_compound_expense_message,
    is_expense_message,
    is_pure_expense_navigation,
    is_greeting_or_chitchat,
    is_leave_message,
    is_programming_question,
    message_has_new_leave_date_range,
    should_resume_expense_for_list,
    should_resume_suspended_expense,
    should_route_expense_after_submitted_leave,
    is_resume_workflow_request,
    is_strong_new_workflow_message,
    is_summary_request,
    is_vague_delete,
    is_workflow_interrupt_expense,
    parse_submit_workflow,
)
from chat.services.platform.registry import get_workflow_definition
from chat.services.platform.response_composer import ResponseComposer
from chat.services.platform.review_engine import ReviewEngine
from chat.services.platform.schemas import (
    ExecutionPlan,
    FieldUpdate,
    PlanOp,
    TurnContext,
    TurnDecision,
    UnderstandingAction,
    UnderstandingResult,
    WORKFLOW_PLAN_OPS,
    WorkflowStage,
)
from chat.services.platform.submission_engine import SubmissionEngine
from chat.services.platform.validation_engine import ValidationEngine
from chat.services.platform.workflow_manager import WorkflowManager
from chat.services.pending_question_engine import (
    MessageIntentKind,
    PendingQuestionDecision,
    _assistant_text_for_translation,
)
from chat.services.session_memory import (
    PendingQuestion,
    SessionMemory,
    StatePatchBuffer,
    WorkflowDraft,
    apply_state_patches,
)
from chat.services.translator import (
    is_translation_request,
    strip_policy_footer,
    translate_text,
)
from chat.services.platform.response_composer import normalize_reply_lang
from chat.services.translator import detect_user_language
from django.conf import settings

INFORMATIONAL_WORKFLOW_ID = "informational"

_INFORMATIONAL_PQ_KINDS = frozenset(
    {
        MessageIntentKind.ASK_POLICY,
        MessageIntentKind.ASK_STATUS,
        MessageIntentKind.ASK_TODAY_DATE,
        MessageIntentKind.ASK_TRANSLATION,
        MessageIntentKind.OUT_OF_SCOPE,
    }
)


class PlanBuilder:
    """Build execution plans from immutable turn context + decision (leave + expense)."""

    @staticmethod
    def _is_expense_review_edit(
        ctx: TurnContext,
        u: UnderstandingResult | None,
        message: str | None = None,
    ) -> bool:
        from chat.services.platform.field_extractors.modify import (
            looks_like_expense_item_delete,
            looks_like_expense_route_modify,
            parse_delete_request,
            parse_modify_request,
            parse_route_modify_request,
        )
        from chat.services.platform.intent_rules import is_delete_request, is_modify_request

        if (ctx.active_workflow_id or "").strip().lower() != "expense":
            return False
        if (ctx.pending_confirmation or "") != "submit":
            return False

        msg = message if message is not None else (ctx.user_message or "")
        draft = ctx.draft_snapshot if isinstance(ctx.draft_snapshot, dict) else {}
        items = list(draft.get("items") or [])

        if is_delete_request(msg) or looks_like_expense_item_delete(msg):
            return True
        if is_modify_request(msg) or looks_like_expense_route_modify(msg):
            return True
        if items and parse_delete_request(msg, items):
            return True
        if items and parse_modify_request(msg, items):
            return True
        if items and parse_route_modify_request(msg, items):
            return True
        if u is not None:
            intent = str((u.entities or {}).get("expense_intent") or "").lower()
            if intent in (
                "delete",
                "update",
                "modify_review",
                "answer_pending",
                "clarify_delete",
                "clarify_modify",
                "fix_mistake",
            ):
                if intent != "answer_pending" or u.field_updates:
                    return True
            if u.action in (
                UnderstandingAction.DELETE.value,
                UnderstandingAction.MODIFY.value,
            ):
                return True
            if u.field_updates and (u.workflow or "").strip().lower() == "expense":
                return True
        return False

    @staticmethod
    def _is_cross_workflow_interrupt(ctx: TurnContext, u: UnderstandingResult) -> bool:
        active = (ctx.active_workflow_id or "").strip().lower()
        target = (u.workflow or "").strip().lower()
        if not active or not target or target in ("none", ""):
            return False
        return target != active

    @staticmethod
    def _interrupts_submit_confirmation(
        u: UnderstandingResult,
        *,
        active_workflow_id: str | None = None,
    ) -> bool:
        """User is changing the draft instead of answering yes/no at submit review."""
        active = (active_workflow_id or "").strip().lower()
        target = (u.workflow or "").strip().lower()
        if active and target and target not in ("none", "") and target != active:
            return False
        if u.action in (UnderstandingAction.MODIFY.value, UnderstandingAction.DELETE.value):
            return True
        if u.field_updates and u.action in (
            UnderstandingAction.COLLECT.value,
            UnderstandingAction.START.value,
        ):
            return True
        return False

    @staticmethod
    def is_leave_turn(ctx: TurnContext, decision: TurnDecision) -> bool:
        u = decision.understanding
        pq = decision.pq
        message = ctx.user_message or ""
        from chat.services.platform.turn_semantics import is_expense_review_request

        if is_expense_review_request(message, u):
            return False
        if ctx.active_workflow_id == "leave":
            return True
        if ctx.pending_question_workflow_id == "leave":
            return True
        if ctx.pending_confirmation == "duplicate_leave":
            return True
        if ctx.pending_confirmation == "submit" and ctx.active_workflow_id == "leave":
            return True
        if u and u.workflow == "leave":
            return True
        if pq and pq.target_workflow == "leave":
            return True
        return False

    @staticmethod
    def is_expense_turn(ctx: TurnContext, decision: TurnDecision) -> bool:
        u = decision.understanding
        pq = decision.pq
        if PlanBuilder.is_leave_turn(ctx, decision):
            return False
        if u and u.interrupt_workflow == "leave":
            return False
        if pq and pq.kind == MessageIntentKind.SWITCH_WORKFLOW and pq.target_workflow == "leave":
            return False
        if ctx.active_workflow_id == "expense":
            return True
        if ctx.pending_question_workflow_id == "expense":
            return True
        if ctx.pending_confirmation == "submit" and ctx.active_workflow_id == "expense":
            return True
        if ctx.pending_confirmation and ctx.pending_confirmation.startswith("modify:"):
            if ctx.active_workflow_id == "expense":
                return True
        if u and u.workflow == "expense":
            return True
        if pq and pq.target_workflow == "expense":
            return True
        return False

    @staticmethod
    def is_informational_turn(ctx: TurnContext, decision: TurnDecision) -> bool:
        """Non leave/expense turns — policy, status, OOS, greeting, fallbacks."""
        pq = decision.pq
        if pq and pq.kind in _INFORMATIONAL_PQ_KINDS:
            return True
        if PlanBuilder.is_leave_turn(ctx, decision):
            return False
        if PlanBuilder.is_expense_turn(ctx, decision):
            return False
        return True

    @staticmethod
    def _is_new_expense_claim(
        message: str,
        understanding: UnderstandingResult | None = None,
        *,
        active_workflow_id: str | None = None,
        memory=None,
        trace_id: str = "",
    ) -> bool:
        from chat.services.platform.field_extractors.expense import (
            user_requests_fresh_expense_draft,
        )
        from chat.services.platform.intent_rules import (
            is_compound_expense_message,
            is_expense_draft_query,
            is_expense_message,
        )

        if user_requests_fresh_expense_draft(message, memory, trace_id=trace_id):
            return True
        active = (active_workflow_id or "").strip().lower()
        if active == "expense":
            return False
        if is_expense_draft_query(message):
            return False
        u = understanding
        if u and (u.entities or {}).get("expense_new_claim"):
            return True
        if u and u.field_updates:
            return True
        expense_turn = dict((u.entities or {}).get("expense_turn") or {}) if u else {}
        if expense_turn.get("item_patches"):
            return True
        return is_compound_expense_message(message) or (
            is_expense_message(message) and not is_expense_draft_query(message)
        )

    @staticmethod
    def _prepare_fresh_expense_draft(state: StatePatchBuffer, memory: SessionMemory) -> None:
        for draft_id in ("expense", "default"):
            draft = (memory.workflow_drafts or {}).get(draft_id)
            if draft and draft.workflow_id == "expense":
                state.push("clear_draft_fields", draft_id=draft_id)
        state.push("drop_suspended_workflow", value="expense")
        state.push("clear_pending_confirmation")
        state.push("clear_pending_question")

    @classmethod
    def _suspended_expense_list_plan(cls, ctx: TurnContext, *, reason: str) -> ExecutionPlan:
        return ExecutionPlan(
            ops=[PlanOp.WORKFLOW_SWITCH],
            reason=reason,
            workflow_id="expense",
        )

    @staticmethod
    def _wants_suspended_expense(ctx: TurnContext, message: str) -> bool:
        return should_resume_suspended_expense(
            message=message,
            active_workflow_id=ctx.active_workflow_id,
            suspended_workflows=ctx.suspended_workflows,
        )

    @staticmethod
    def _wants_suspended_expense_list(ctx: TurnContext, message: str) -> bool:
        return should_resume_suspended_expense(
            message=message,
            active_workflow_id=ctx.active_workflow_id,
            suspended_workflows=ctx.suspended_workflows,
        )

    @classmethod
    def build(cls, ctx: TurnContext, decision: TurnDecision) -> ExecutionPlan | None:
        pq = decision.pq
        u = decision.understanding

        if pq and pq.kind in _INFORMATIONAL_PQ_KINDS:
            if not getattr(settings, "USE_INFORMATIONAL_PLAN", True):
                return None
            return cls._build_informational_plan(ctx, decision)

        if u and u.is_greeting and getattr(settings, "USE_INFORMATIONAL_PLAN", True):
            return ExecutionPlan(
                ops=[PlanOp.REPLY_GREETING],
                reason="greeting during any session state",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )

        if cls._wants_suspended_expense(ctx, ctx.user_message or ""):
            return cls._suspended_expense_list_plan(
                ctx,
                reason="expense navigation while another workflow active",
            )

        pq_target = (pq.target_workflow if pq else None) or ""
        if should_route_expense_after_submitted_leave(
            draft_locked=ctx.draft_locked,
            active_workflow_id=ctx.active_workflow_id,
            message=ctx.user_message or "",
            understanding=u,
            pq_target_workflow=pq_target,
        ):
            if not getattr(settings, "EXPENSE_NEW_ARCH", False):
                return cls._platform_clarify_plan("expense plan disabled")
            return cls._build_expense_plan(ctx, decision)

        if cls.is_leave_turn(ctx, decision):
            if not getattr(settings, "USE_LEAVE_PLAN", True):
                return cls._platform_clarify_plan("leave plan disabled")
            return cls._build_leave_plan(ctx, decision)

        if cls.is_expense_turn(ctx, decision):
            if not getattr(settings, "EXPENSE_NEW_ARCH", False):
                return cls._platform_clarify_plan("expense plan disabled")
            return cls._build_expense_plan(ctx, decision)

        if cls.is_informational_turn(ctx, decision):
            if not getattr(settings, "USE_INFORMATIONAL_PLAN", True):
                return None
            return cls._build_informational_plan(ctx, decision)

        return None

    @staticmethod
    def _platform_clarify_plan(reason: str) -> ExecutionPlan:
        return ExecutionPlan(
            ops=[PlanOp.REPLY_PLATFORM_CLARIFY],
            reason=reason,
            workflow_id=INFORMATIONAL_WORKFLOW_ID,
        )

    @classmethod
    def informational_fallback_plan(
        cls,
        ctx: TurnContext,
        decision: TurnDecision,
    ) -> ExecutionPlan:
        """Last-resort informational chain when no workflow plan matches."""
        if ctx.has_active_workflow and ctx.active_workflow_id:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CLARIFICATION],
                reason="active workflow — avoid conversational disconnect",
                workflow_id=ctx.active_workflow_id,
            )
        _ = decision
        return ExecutionPlan(
            ops=[PlanOp.REPLY_CONVERSATIONAL, PlanOp.REPLY_GENERAL_HELP],
            reason="informational fallback",
            workflow_id=INFORMATIONAL_WORKFLOW_ID,
        )

    @classmethod
    def _build_informational_plan(
        cls,
        ctx: TurnContext,
        decision: TurnDecision,
    ) -> ExecutionPlan:
        u = decision.understanding
        pq = decision.pq
        kind = pq.kind if pq else MessageIntentKind.CLARIFICATION_NEEDED

        if u is None:
            return ExecutionPlan(
                ops=[PlanOp.NONE],
                reason="missing understanding",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )

        if kind == MessageIntentKind.ASK_POLICY:
            return ExecutionPlan(
                ops=[PlanOp.REPLY_POLICY],
                reason="policy query",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )
        if kind == MessageIntentKind.ASK_STATUS:
            return ExecutionPlan(
                ops=[PlanOp.REPLY_STATUS],
                reason="status query",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )
        if kind == MessageIntentKind.ASK_TODAY_DATE:
            return ExecutionPlan(
                ops=[PlanOp.REPLY_TODAY_DATE],
                reason="today date query",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )
        if kind == MessageIntentKind.ASK_TRANSLATION:
            return ExecutionPlan(
                ops=[PlanOp.REPLY_TRANSLATION],
                reason="translation follow-up",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )
        if kind == MessageIntentKind.OUT_OF_SCOPE:
            return ExecutionPlan(
                ops=[PlanOp.REPLY_OOS],
                reason="out of scope",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )

        if kind == MessageIntentKind.NEW_WORKFLOW:
            if is_programming_question(ctx.user_message):
                return ExecutionPlan(
                    ops=[PlanOp.REPLY_OOS],
                    reason="programming out of scope",
                    workflow_id=INFORMATIONAL_WORKFLOW_ID,
                )
            if u.is_out_of_scope and u.action not in (
                UnderstandingAction.REVIEW.value,
                UnderstandingAction.SUBMIT.value,
                UnderstandingAction.CONFIRM.value,
            ):
                return ExecutionPlan(
                    ops=[PlanOp.REPLY_OOS],
                    reason="understanding out of scope",
                    workflow_id=INFORMATIONAL_WORKFLOW_ID,
                )
            if u.is_greeting:
                return ExecutionPlan(
                    ops=[PlanOp.REPLY_GREETING],
                    reason="greeting during any session state",
                    workflow_id=INFORMATIONAL_WORKFLOW_ID,
                )

        if kind == MessageIntentKind.CLARIFICATION_NEEDED:
            if ctx.has_active_workflow and ctx.active_workflow_id:
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_CLARIFICATION],
                    reason="clarify within active workflow",
                    workflow_id=ctx.active_workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.REPLY_GENERAL_HELP],
                reason="needs clarification",
                workflow_id=INFORMATIONAL_WORKFLOW_ID,
            )

        return ExecutionPlan(
            ops=[PlanOp.REPLY_CONVERSATIONAL, PlanOp.REPLY_GENERAL_HELP],
            reason="informational fallback",
            workflow_id=INFORMATIONAL_WORKFLOW_ID,
        )

    @classmethod
    def _build_leave_plan(cls, ctx: TurnContext, decision: TurnDecision) -> ExecutionPlan | None:
        return cls._build_workflow_plan("leave", ctx, decision)

    @classmethod
    def _build_expense_plan(cls, ctx: TurnContext, decision: TurnDecision) -> ExecutionPlan | None:
        return cls._build_workflow_plan("expense", ctx, decision)

    @classmethod
    def _build_workflow_plan(
        cls,
        workflow_id: str,
        ctx: TurnContext,
        decision: TurnDecision,
    ) -> ExecutionPlan | None:
        u = decision.understanding
        if u is None:
            return ExecutionPlan(
                ops=[PlanOp.NONE],
                reason="missing understanding",
                workflow_id=workflow_id,
            )

        if u.is_out_of_scope:
            return ExecutionPlan(
                ops=[PlanOp.REJECT_OOS],
                reason="out of scope",
                workflow_id=workflow_id,
            )

        if ctx.draft_locked:
            active = (ctx.active_workflow_id or "").strip().lower()
            if workflow_id == "expense" and active == "leave":
                return cls._build_post_submit_expense_start_plan(ctx, decision, u)
            if workflow_id == "leave":
                return cls._build_locked_leave_plan(ctx, decision, workflow_id)
            if workflow_id == "expense":
                return cls._build_locked_expense_plan(ctx, decision, workflow_id)
            return ExecutionPlan(
                ops=[PlanOp.LOCKED_RESPONSE],
                reason="post-submit lock",
                workflow_id=workflow_id,
            )

        pending = ctx.pending_confirmation or ""
        if pending.startswith("switch:"):
            return ExecutionPlan(
                ops=[PlanOp.RESOLVE_WORKFLOW_SWITCH],
                reason="workflow switch pending",
                workflow_id=workflow_id,
            )
        if workflow_id == "leave" and pending == "duplicate_leave":
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CLARIFICATION],
                reason="legacy duplicate_leave cleared — parallel leave is hard-blocked",
                workflow_id=workflow_id,
            )
        if pending == "submit" and ctx.active_workflow_id == workflow_id:
            submit_plan = cls._build_submit_pending_plan(workflow_id, ctx, decision, u)
            if submit_plan is not None:
                return submit_plan

        if decision.route_source == "active":
            return cls._build_workflow_active(workflow_id, ctx, decision, u)

        return cls._build_workflow_pending(workflow_id, ctx, decision, u)

    @classmethod
    def _build_locked_leave_plan(
        cls,
        ctx: TurnContext,
        decision: TurnDecision,
        workflow_id: str,
    ) -> ExecutionPlan:
        """After submit — allow summary and new non-overlapping leave; block edits."""
        u = decision.understanding
        message = ctx.user_message or ""
        pq = decision.pq
        submitted = list((ctx.conversation_facts or {}).get("submitted_leave_ranges") or [])

        if should_route_expense_after_submitted_leave(
            draft_locked=True,
            active_workflow_id=ctx.active_workflow_id,
            message=message,
            understanding=u,
            pq_target_workflow=(pq.target_workflow if pq else None),
        ):
            if cls._wants_suspended_expense(ctx, message):
                return cls._suspended_expense_list_plan(
                    ctx,
                    reason="expense navigation after submitted leave",
                )
            return cls._build_post_submit_expense_start_plan(ctx, decision, u)

        if cls._wants_suspended_expense(ctx, message):
            return cls._suspended_expense_list_plan(
                ctx,
                reason="expense navigation while leave locked",
            )

        overlap = find_submitted_leave_overlap_from_message(message, submitted)
        if overlap and (is_leave_message(message) or (u and u.workflow == "leave")):
            return ExecutionPlan(
                ops=[PlanOp.SUBMITTED_LEAVE_OVERLAP],
                reason="submitted leave date overlap",
                workflow_id=workflow_id,
            )

        if cls._is_post_submit_new_leave(ctx, decision):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="new leave after submit",
                workflow_id=workflow_id,
            )

        if message_has_new_leave_date_range(message, submitted_ranges=submitted) and (
            is_leave_message(message) or is_strong_new_workflow_message(message)
        ):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="new leave dates after submit",
                workflow_id=workflow_id,
            )

        if is_summary_request(message) or (u and u.action == UnderstandingAction.REVIEW.value):
            if message_has_new_leave_date_range(message, submitted_ranges=submitted):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_NEW],
                    reason="new leave dates despite review phrasing",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                reason="leave summary on submitted draft",
                workflow_id=workflow_id,
            )

        if u and u.action in (
            UnderstandingAction.MODIFY.value,
            UnderstandingAction.DELETE.value,
            UnderstandingAction.SUBMIT.value,
            UnderstandingAction.CONFIRM.value,
        ):
            return ExecutionPlan(
                ops=[PlanOp.LOCKED_RESPONSE],
                reason="post-submit lock",
                workflow_id=workflow_id,
            )

        if u and u.action == UnderstandingAction.COLLECT.value and cls._is_post_submit_new_leave(ctx, decision):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="new leave field collection after submit",
                workflow_id=workflow_id,
            )

        if is_leave_message(message) or is_strong_new_workflow_message(message):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="leave phrasing after submit",
                workflow_id=workflow_id,
            )

        if pq and pq.kind == MessageIntentKind.NEW_WORKFLOW:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="new workflow after submit",
                workflow_id=workflow_id,
            )

        return ExecutionPlan(
            ops=[PlanOp.LOCKED_RESPONSE],
            reason="post-submit lock",
            workflow_id=workflow_id,
        )

    @classmethod
    def _build_locked_expense_plan(
        cls,
        ctx: TurnContext,
        decision: TurnDecision,
        workflow_id: str,
    ) -> ExecutionPlan:
        """After submit — allow summary and new expense; block edits to submitted draft."""
        u = decision.understanding
        message = ctx.user_message or ""
        pq = decision.pq
        from chat.services.platform.turn_semantics import is_expense_review_request

        if is_expense_review_request(message, u) or (
            pq and pq.kind == MessageIntentKind.SHOW_REVIEW and pq.target_workflow == "expense"
        ):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                reason="expense summary on submitted draft",
                workflow_id=workflow_id,
            )

        if u and u.action in (
            UnderstandingAction.MODIFY.value,
            UnderstandingAction.DELETE.value,
            UnderstandingAction.SUBMIT.value,
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.CANCEL.value,
        ):
            return ExecutionPlan(
                ops=[PlanOp.LOCKED_RESPONSE],
                reason="post-submit lock",
                workflow_id=workflow_id,
            )

        if (
            (pq and pq.kind == MessageIntentKind.CANCEL_WORKFLOW)
            or is_cancel_workflow_message(message, workflow_id="expense")
        ):
            return ExecutionPlan(
                ops=[PlanOp.LOCKED_RESPONSE],
                reason="post-submit cancel blocked",
                workflow_id=workflow_id,
            )

        if is_expense_message(message) or (u and u.workflow == "expense" and u.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
        )):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="new expense after submit",
                workflow_id=workflow_id,
            )

        if is_summary_request(message) or (u and u.action == UnderstandingAction.REVIEW.value):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                reason="expense summary on submitted draft",
                workflow_id=workflow_id,
            )

        if pq and pq.kind == MessageIntentKind.NEW_WORKFLOW:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="new workflow after expense submit",
                workflow_id=workflow_id,
            )

        if is_expense_message(message):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_NEW],
                reason="expense phrasing after submit",
                workflow_id=workflow_id,
            )

        return ExecutionPlan(
            ops=[PlanOp.LOCKED_RESPONSE],
            reason="post-submit lock",
            workflow_id=workflow_id,
        )

    @classmethod
    def _build_post_submit_expense_start_plan(
        cls,
        ctx: TurnContext,
        decision: TurnDecision,
        u: UnderstandingResult,
    ) -> ExecutionPlan:
        """Start a fresh expense workflow after a submitted leave draft is still active."""
        ops: list[PlanOp] = [PlanOp.WORKFLOW_NEW]
        if u and u.field_updates and u.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
            UnderstandingAction.MODIFY.value,
        ):
            ops.append(PlanOp.WORKFLOW_APPLY_UPDATES)
        return ExecutionPlan(
            ops=ops,
            reason="new expense after submitted leave",
            workflow_id="expense",
        )

    @staticmethod
    def _is_post_submit_new_leave(ctx: TurnContext, decision: TurnDecision) -> bool:
        u = decision.understanding
        pq = decision.pq
        message = ctx.user_message or ""
        submitted = list((ctx.conversation_facts or {}).get("submitted_leave_ranges") or [])

        if pq and pq.kind == MessageIntentKind.NEW_WORKFLOW:
            return True
        if message_has_new_leave_date_range(message, submitted_ranges=submitted):
            if is_leave_message(message) or is_strong_new_workflow_message(message):
                return True
            if u and u.workflow == "leave" and u.action in (
                UnderstandingAction.START.value,
                UnderstandingAction.COLLECT.value,
            ):
                return True
        if is_strong_new_workflow_message(message) or is_leave_message(message):
            if message_has_new_leave_date_range(message, submitted_ranges=submitted):
                return True
        if u and u.workflow == "leave" and u.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
        ):
            if u.field_updates or message_has_new_leave_date_range(message, submitted_ranges=submitted):
                return True
        return False

    @classmethod
    def _build_submit_pending_plan(
        cls,
        workflow_id: str,
        ctx: TurnContext,
        decision: TurnDecision,
        u: UnderstandingResult,
    ) -> ExecutionPlan | None:
        """Phase 2 — priority routing while pending_confirmation=submit."""
        message = ctx.user_message or ""
        pq = decision.pq

        if workflow_id == "leave" and (
            is_workflow_interrupt_expense(message, active_workflow="leave")
            or (u and (
                u.interrupt_workflow == "expense"
                or (u.workflow == "expense" and u.action in (
                    UnderstandingAction.START.value,
                    UnderstandingAction.COLLECT.value,
                ))
            ))
        ):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SWITCH],
                reason="expense interrupt during leave submit review",
                workflow_id="expense",
            )

        submit_wf_early = parse_submit_workflow(message, active_workflow_id=workflow_id)
        if submit_wf_early == workflow_id or (
            u.action in (
                UnderstandingAction.CONFIRM.value,
                UnderstandingAction.SUBMIT.value,
            )
            and (u.workflow or workflow_id) == workflow_id
        ):
            return ExecutionPlan(
                ops=[PlanOp.RESOLVE_SUBMIT_CONFIRMATION],
                reason="confirm during submit review",
                workflow_id=workflow_id,
            )

        if cls._is_cross_workflow_interrupt(ctx, u) and not cls._is_expense_review_edit(
            ctx, u, message
        ):
            return ExecutionPlan(
                ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
                reason=f"cross-workflow during {workflow_id} submit",
                workflow_id=workflow_id,
            )

        if (
            (pq and pq.kind == MessageIntentKind.CANCEL_WORKFLOW)
            or u.action == UnderstandingAction.CANCEL.value
            or is_cancel_workflow_message(message, workflow_id=workflow_id)
        ):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CANCEL],
                reason="cancel during submit review",
                workflow_id=workflow_id,
            )

        if (pq and pq.kind == MessageIntentKind.DELETE_DATA) or u.action == UnderstandingAction.DELETE.value:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_DELETE],
                reason="delete during submit review",
                workflow_id=workflow_id,
            )

        submit_wf = parse_submit_workflow(message, active_workflow_id=workflow_id)
        confirm_on_active = u.action in (
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SUBMIT.value,
        ) and (u.workflow or workflow_id) == workflow_id
        if submit_wf == workflow_id or confirm_on_active:
            return ExecutionPlan(
                ops=[PlanOp.RESOLVE_SUBMIT_CONFIRMATION],
                reason="submit phrase during pending review",
                workflow_id=workflow_id,
            )

        if workflow_id == "leave":
            if u.action == UnderstandingAction.MODIFY.value or (
                u.action == UnderstandingAction.COLLECT.value and u.field_updates
            ):
                return cls._build_workflow_active(workflow_id, ctx, decision, u)

        if workflow_id == "expense":
            intent = str((u.entities or {}).get("expense_intent") or "").lower()
            from chat.services.platform.intent_rules import (
                is_compound_expense_message,
                is_expense_message,
            )

            if intent in (
                "add",
                "update",
                "delete",
                "modify_review",
                "answer_pending",
                "fix_mistake",
                "anti_summary",
                "clarify_modify",
                "clarify_delete",
            ) or (
                u.action
                in (
                    UnderstandingAction.MODIFY.value,
                    UnderstandingAction.COLLECT.value,
                    UnderstandingAction.DELETE.value,
                    UnderstandingAction.CLARIFICATION_NEEDED.value,
                )
                and (u.field_updates or intent == "anti_summary")
            ) or (
                intent == "conversation"
                and u.field_updates
            ) or (
                is_expense_message(message) or is_compound_expense_message(message)
            ):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_COLLECT],
                    reason="expense draft editor during submit review",
                    workflow_id=workflow_id,
                )

        if cls._interrupts_submit_confirmation(u, active_workflow_id=ctx.active_workflow_id):
            return cls._build_workflow_active(workflow_id, ctx, decision, u)

        if cls._wants_suspended_expense_list(ctx, message):
            return cls._suspended_expense_list_plan(
                ctx,
                reason="expense list during submit review",
            )

        from chat.services.platform.intent_rules import is_clearly_off_hr_question, is_off_hr_topic_message

        if (
            is_programming_question(message)
            or is_clearly_off_hr_question(message)
            or is_off_hr_topic_message(message)
        ):
            return ExecutionPlan(
                ops=[PlanOp.REJECT_OOS],
                reason="off-hr topic during submit review",
                workflow_id=workflow_id,
            )

        if (
            is_summary_request(message)
            or is_bare_rejection(message)
            or u.action == UnderstandingAction.REVIEW.value
            or is_resume_workflow_request(message, workflow_id=workflow_id)
        ) and not is_clearly_off_hr_question(message):
            from chat.services.platform.workflow_show import resolve_workflow_show_target

            show_target = str((u.entities or {}).get("show_workflow_target") or "").lower()
            if not show_target:
                show_target = resolve_workflow_show_target(
                    message,
                    None,
                    active_workflow_id=workflow_id,
                ) or ""
            if show_target == "leave":
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                    reason="leave summary during expense submit review",
                    workflow_id="leave",
                )
            if is_bare_rejection(message):
                return ExecutionPlan(
                    ops=[PlanOp.RESOLVE_SUBMIT_CONFIRMATION],
                    reason="decline submit confirmation",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                reason="summary/review during submit",
                workflow_id=workflow_id,
            )

        if (
            (pq and pq.kind == MessageIntentKind.MODIFY_DATA)
            or (
                u.action == UnderstandingAction.MODIFY.value
                and u.field_updates
            )
        ):
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_MODIFY],
                reason="modify during submit review",
                workflow_id=workflow_id,
            )

        from chat.services.platform.intent_rules import is_bare_confirmation

        if (
            is_bare_confirmation(message)
            or u.action in (
                UnderstandingAction.CONFIRM.value,
                UnderstandingAction.SUBMIT.value,
            )
            or (
                pq
                and pq.kind == MessageIntentKind.ANSWER_PENDING
                and is_bare_confirmation(message)
            )
        ):
            return ExecutionPlan(
                ops=[PlanOp.RESOLVE_SUBMIT_CONFIRMATION],
                reason="submit confirmation yes",
                workflow_id=workflow_id,
            )

        if (
            (pq and pq.kind == MessageIntentKind.CLARIFICATION_NEEDED)
            or u.action == UnderstandingAction.CLARIFICATION_NEEDED.value
        ):
            expense_intent = str((u.entities or {}).get("expense_intent") or "").lower()
            if workflow_id == "expense" and expense_intent in ("clarify_delete", "clarify_modify"):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_COLLECT],
                    reason="expense edit clarification during submit review",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CLARIFICATION],
                reason="clarify during submit review",
                workflow_id=workflow_id,
            )

        return ExecutionPlan(
            ops=[PlanOp.RESOLVE_SUBMIT_CONFIRMATION],
            reason="submit confirmation",
            workflow_id=workflow_id,
        )

    @staticmethod
    def _cross_workflow_confirm_plan(
        workflow_id: str,
        ctx: TurnContext,
        u: UnderstandingResult,
        *,
        message: str | None = None,
    ) -> ExecutionPlan | None:
        msg = message if message is not None else (ctx.user_message or "")
        if not PlanBuilder._is_cross_workflow_interrupt(ctx, u):
            return None
        if PlanBuilder._is_expense_review_edit(ctx, u, msg):
            return None
        return ExecutionPlan(
            ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
            reason=f"confirm cross-workflow switch from {workflow_id}",
            workflow_id=workflow_id,
        )

    @classmethod
    def _build_workflow_pending(
        cls,
        workflow_id: str,
        ctx: TurnContext,
        decision: TurnDecision,
        u: UnderstandingResult,
    ) -> ExecutionPlan:
        pq = decision.pq
        kind = pq.kind if pq else MessageIntentKind.CLARIFICATION_NEEDED

        if u.action == UnderstandingAction.REVIEW.value:
            target = (u.workflow or workflow_id or "").strip().lower()
            if target in ("leave", "expense"):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                    reason="summary/review without active workflow",
                    workflow_id=target,
                )

        target = (
            (pq.target_workflow if pq else None)
            or u.interrupt_workflow
            or u.workflow
            or ""
        ).strip().lower()
        active_id = (ctx.active_workflow_id or "").strip().lower()
        suspended_ids = {
            str(sw.get("workflow_id") or "").strip().lower()
            for sw in (ctx.suspended_workflows or ())
            if isinstance(sw, dict)
        }

        if kind == MessageIntentKind.CANCEL_WORKFLOW:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CANCEL],
                reason="cancel workflow",
                workflow_id=workflow_id,
            )

        if kind == MessageIntentKind.SHOW_REVIEW:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                reason="show workflow draft / summary",
                workflow_id=workflow_id,
            )

        if kind == MessageIntentKind.ANSWER_PENDING:
            expense_intent = str((u.entities or {}).get("expense_intent") or "").lower()
            if (
                workflow_id == "expense"
                and expense_intent == "add"
                and u.field_updates
                and not ctx.pending_question_field
                and active_id != "expense"
            ):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_NEW, PlanOp.WORKFLOW_APPLY_UPDATES],
                    reason="new expense claim with line items",
                    workflow_id="expense",
                )
            ops: list[PlanOp] = [PlanOp.MAYBE_WORKFLOW_SWITCH, PlanOp.WORKFLOW_COLLECT]
            if workflow_id == "leave":
                ops = [PlanOp.MAYBE_DUPLICATE_LEAVE, *ops]
            return ExecutionPlan(
                ops=ops,
                reason=f"answer pending {workflow_id} slot",
                workflow_id=workflow_id,
            )
        switch_plan = cls._cross_workflow_confirm_plan(workflow_id, ctx, u)
        if switch_plan is not None:
            return switch_plan
        if kind == MessageIntentKind.MODIFY_DATA:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_MODIFY],
                reason=f"modify {workflow_id} draft",
                workflow_id=workflow_id,
            )
        if kind == MessageIntentKind.DELETE_DATA:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_DELETE],
                reason=f"delete from {workflow_id} draft",
                workflow_id=workflow_id,
            )
        if kind == MessageIntentKind.SWITCH_WORKFLOW:
            if active_id and target and target != active_id:
                return ExecutionPlan(
                    ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
                    reason="confirm switch away from active workflow",
                    workflow_id=workflow_id,
                )
            if (
                target
                and active_id
                and target != active_id
                and target in suspended_ids
                and target == "expense"
                and cls._is_new_expense_claim(
                    ctx.user_message or "",
                    u,
                    active_workflow_id=active_id,
                    memory=ctx.memory,
                    trace_id=ctx.trace_id or "",
                )
            ):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_NEW, PlanOp.WORKFLOW_APPLY_UPDATES],
                    reason="new expense claim replaces suspended draft",
                    workflow_id="expense",
                )
            if target and active_id and target != active_id and target in suspended_ids:
                return ExecutionPlan(
                    ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
                    reason=f"confirm resume suspended {target}",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
                reason="switch workflow",
                workflow_id=workflow_id,
            )
        if kind == MessageIntentKind.NEW_WORKFLOW and pq and not pq.blocks_new_workflow:
            message = ctx.user_message or ""
            from chat.services.platform.turn_semantics import cross_workflow_switch_target

            switch_target = cross_workflow_switch_target(u, active_id or None)
            if active_id and switch_target and switch_target != active_id:
                return ExecutionPlan(
                    ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
                    reason="confirm switch before starting other workflow",
                    workflow_id=workflow_id,
                )
            if workflow_id == "expense" and (
                is_cancel_workflow_message(message, workflow_id="expense")
                or u.action == UnderstandingAction.CANCEL.value
            ):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_CANCEL],
                    reason="cancel expense without active draft",
                    workflow_id="expense",
                )
            ops = [PlanOp.WORKFLOW_NEW]
            if u.field_updates and workflow_id in ("expense", "leave"):
                ops.append(PlanOp.WORKFLOW_APPLY_UPDATES)
            if workflow_id == "leave":
                ops = [PlanOp.MAYBE_DUPLICATE_LEAVE, *ops]
            return ExecutionPlan(
                ops=ops,
                reason=f"start {workflow_id} workflow",
                workflow_id=workflow_id,
            )
        if kind == MessageIntentKind.CLARIFICATION_NEEDED:
            expense_intent = str((u.entities or {}).get("expense_intent") or "").lower()
            if workflow_id == "expense" and expense_intent in ("clarify_delete", "clarify_modify"):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_COLLECT],
                    reason="expense edit clarification",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CLARIFICATION],
                reason="needs clarification",
                workflow_id=workflow_id,
            )
        return ExecutionPlan(
            ops=[PlanOp.NONE],
            reason=f"unhandled pq kind {kind.value}",
            workflow_id=workflow_id,
        )

    @classmethod
    def _build_workflow_active(
        cls,
        workflow_id: str,
        ctx: TurnContext,
        decision: TurnDecision,
        u: UnderstandingResult,
    ) -> ExecutionPlan:
        pending = ctx.pending_confirmation or ""
        message = ctx.user_message
        pq = decision.pq

        if (pq and pq.kind == MessageIntentKind.CANCEL_WORKFLOW) or u.action == UnderstandingAction.CANCEL.value:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_CANCEL],
                reason="cancel active workflow",
                workflow_id=workflow_id,
            )

        if (pq and pq.kind == MessageIntentKind.DELETE_DATA) or u.action == UnderstandingAction.DELETE.value:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_DELETE],
                reason="delete from active draft",
                workflow_id=workflow_id,
            )

        if (pq and pq.kind == MessageIntentKind.SHOW_REVIEW) or u.action == UnderstandingAction.REVIEW.value:
            from chat.services.platform.intent_rules import is_clearly_off_hr_question, is_programming_question
            from chat.services.platform.workflow_show import resolve_workflow_show_target

            if is_programming_question(message or "") or is_clearly_off_hr_question(message or ""):
                return ExecutionPlan(
                    ops=[PlanOp.REJECT_OOS],
                    reason="off-hr topic during active workflow",
                    workflow_id=workflow_id,
                )
            show_target = str((u.entities or {}).get("show_workflow_target") or "").lower()
            if not show_target:
                show_target = resolve_workflow_show_target(
                    message,
                    None,
                    active_workflow_id=workflow_id,
                ) or ""
            if show_target == "leave":
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                    reason="leave summary during active expense",
                    workflow_id="leave",
                )
            if cls._wants_suspended_expense_list(ctx, message or ""):
                return cls._suspended_expense_list_plan(
                    ctx,
                    reason="expense list during active workflow",
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                reason=f"{workflow_id} review / navigation",
                workflow_id=workflow_id,
            )

        if u.action == UnderstandingAction.SWITCH.value:
            switch_target = (
                (pq.target_workflow if pq else None)
                or u.interrupt_workflow
                or u.workflow
                or ""
            ).strip().lower()
            active_id = (ctx.active_workflow_id or "").strip().lower()
            if switch_target and switch_target == active_id:
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_SHOW_REVIEW],
                    reason="same-workflow navigation",
                    workflow_id=workflow_id,
                )

        if pending.startswith("modify:"):
            if u.action == UnderstandingAction.CONFIRM.value or is_bare_confirmation(message):
                return ExecutionPlan(
                    ops=[PlanOp.APPLY_PENDING_MODIFY],
                    reason="confirm modify",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.CLEAR_PENDING_MODIFY],
                reason="cancel modify confirm",
                workflow_id=workflow_id,
            )

        submit_wf = parse_submit_workflow(message, active_workflow_id=ctx.active_workflow_id)
        if u.action == UnderstandingAction.SUBMIT.value:
            submit_wf = u.workflow or submit_wf
        if submit_wf == workflow_id or (submit_wf and ctx.active_workflow_id == workflow_id):
            active_id = (ctx.active_workflow_id or "").strip().lower()
            suspended_ids = {
                str(sw.get("workflow_id") or "").strip().lower()
                for sw in (ctx.suspended_workflows or ())
                if isinstance(sw, dict)
            }
            ops: list[PlanOp] = [PlanOp.WORKFLOW_REQUEST_SUBMIT]
            if (
                workflow_id == "expense"
                and active_id != "expense"
                and workflow_id in suspended_ids
            ):
                ops = [PlanOp.WORKFLOW_SWITCH, PlanOp.WORKFLOW_REQUEST_SUBMIT]
            return ExecutionPlan(
                ops=ops,
                reason="explicit submit",
                workflow_id=workflow_id,
            )

        if u.action == UnderstandingAction.MODIFY.value:
            if workflow_id == "expense":
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_COLLECT],
                    reason="expense draft editor modify",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_MODIFY],
                reason=f"modify {workflow_id} draft",
                workflow_id=workflow_id,
            )

        if pending == "submit" and workflow_id == "leave":
            if u.action in (
                UnderstandingAction.MODIFY.value,
                UnderstandingAction.COLLECT.value,
            ) and u.field_updates:
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_MODIFY],
                    reason="modify leave during submit review",
                    workflow_id=workflow_id,
                )

        if pending == "submit" and workflow_id == "expense":
            intent = str((u.entities or {}).get("expense_intent") or "").lower()

            if intent in (
                "add",
                "update",
                "delete",
                "modify_review",
                "answer_pending",
                "fix_mistake",
                "anti_summary",
                "clarify_modify",
                "clarify_delete",
            ) or (
                u.action
                in (
                    UnderstandingAction.MODIFY.value,
                    UnderstandingAction.COLLECT.value,
                    UnderstandingAction.DELETE.value,
                    UnderstandingAction.CLARIFICATION_NEEDED.value,
                )
                and (u.field_updates or intent == "anti_summary")
            ) or (
                intent == "conversation"
                and u.field_updates
            ) or (
                is_expense_message(message) or is_compound_expense_message(message)
            ):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_COLLECT],
                    reason="expense draft editor during submit review",
                    workflow_id=workflow_id,
                )

        if u.action == UnderstandingAction.CONFIRM.value:
            if ctx.pending_confirmation == "submit":
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_CONFIRM_SUBMIT],
                    reason="confirm submit",
                    workflow_id=workflow_id,
                )
            return ExecutionPlan(
                ops=[PlanOp.SHOW_SESSION_CONTEXT],
                reason="confirm without pending submit",
                workflow_id=workflow_id,
            )

        if u.action == UnderstandingAction.SUBMIT.value:
            active_id = (ctx.active_workflow_id or "").strip().lower()
            suspended_ids = {
                str(sw.get("workflow_id") or "").strip().lower()
                for sw in (ctx.suspended_workflows or ())
                if isinstance(sw, dict)
            }
            ops: list[PlanOp] = [PlanOp.WORKFLOW_REQUEST_SUBMIT]
            if (
                workflow_id == "expense"
                and active_id != "expense"
                and workflow_id in suspended_ids
            ):
                ops = [PlanOp.WORKFLOW_SWITCH, PlanOp.WORKFLOW_REQUEST_SUBMIT]
            return ExecutionPlan(
                ops=ops,
                reason="submit action",
                workflow_id=workflow_id,
            )

        active_id = (ctx.active_workflow_id or "").strip().lower()
        switch_target = (u.interrupt_workflow or u.workflow or "").strip().lower()
        suspended_ids = {
            str(sw.get("workflow_id") or "").strip().lower()
            for sw in (ctx.suspended_workflows or ())
            if isinstance(sw, dict)
        }
        if (
            switch_target
            and active_id
            and switch_target != active_id
            and switch_target in suspended_ids
            and not cls._is_expense_review_edit(ctx, u, message)
        ):
            if switch_target == "expense" and cls._is_new_expense_claim(
                message or "",
                u,
                active_workflow_id=active_id,
                memory=ctx.memory,
                trace_id=ctx.trace_id or "",
            ):
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_NEW, PlanOp.WORKFLOW_APPLY_UPDATES],
                    reason="new expense claim replaces suspended draft",
                    workflow_id="expense",
                )
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_SWITCH],
                reason=f"resume suspended {switch_target}",
                workflow_id=active_id,
            )

        if cls._is_cross_workflow_interrupt(ctx, u) and not cls._is_expense_review_edit(
            ctx, u, message
        ):
            return ExecutionPlan(
                ops=[PlanOp.MAYBE_WORKFLOW_SWITCH],
                reason=f"cross-workflow interrupt during {workflow_id}",
                workflow_id=workflow_id,
            )

        if u.action in (UnderstandingAction.START.value, UnderstandingAction.COLLECT.value) and u.field_updates:
            skip_expense_apply = False
            if workflow_id == "expense":
                from chat.services.platform.field_extractors.expense import (
                    expense_turn_blocks_wizard,
                    pending_expense_edit_active,
                )
                from chat.services.platform.intent_rules import is_delete_request

                expense_intent = str((u.entities or {}).get("expense_intent") or "").lower()
                expense_turn = dict((u.entities or {}).get("expense_turn") or {})
                skip_expense_apply = (
                    expense_intent in ("clarify_delete", "delete")
                    or pending_expense_edit_active(memory)
                    or is_delete_request(message)
                    or expense_turn_blocks_wizard(message, memory, turn=expense_turn)
                )
            if not skip_expense_apply:
                ops = [PlanOp.WORKFLOW_APPLY_UPDATES]
                if workflow_id == "leave":
                    ops = cls._with_duplicate_leave_guard(ops)
                return ExecutionPlan(
                    ops=ops,
                    reason=f"apply {workflow_id} field updates",
                    workflow_id=workflow_id,
                )

        if workflow_id == "expense":
            expense_intent = str((u.entities or {}).get("expense_intent") or "").lower()
            expense_signal = (
                is_expense_message(message)
                or is_compound_expense_message(message)
                or is_workflow_interrupt_expense(message, active_workflow=workflow_id)
                or expense_intent
                in (
                    "add",
                    "update",
                    "delete",
                    "answer_pending",
                    "fix_mistake",
                    "clarify_modify",
                    "clarify_delete",
                    "conversation",
                )
            )
            if expense_signal and not is_pure_expense_navigation(message):
                if u.field_updates and expense_intent not in ("delete", "clarify_delete"):
                    return ExecutionPlan(
                        ops=[PlanOp.WORKFLOW_APPLY_UPDATES],
                        reason="apply expense field updates",
                        workflow_id=workflow_id,
                    )
                return ExecutionPlan(
                    ops=[PlanOp.WORKFLOW_COLLECT],
                    reason="expense draft editor collect",
                    workflow_id=workflow_id,
                )

        pq = decision.pq
        switch_plan = cls._cross_workflow_confirm_plan(workflow_id, ctx, u)
        if switch_plan is not None:
            return switch_plan
        if pq and pq.kind == MessageIntentKind.MODIFY_DATA:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_MODIFY],
                reason=f"modify {workflow_id} draft",
                workflow_id=workflow_id,
            )
        if pq and pq.kind == MessageIntentKind.DELETE_DATA:
            return ExecutionPlan(
                ops=[PlanOp.WORKFLOW_DELETE],
                reason=f"delete from {workflow_id} draft",
                workflow_id=workflow_id,
            )

        ops = [PlanOp.WORKFLOW_CLARIFICATION]
        if workflow_id == "leave":
            ops = cls._with_duplicate_leave_guard(ops)
        return ExecutionPlan(
            ops=ops,
            reason=f"active {workflow_id} default",
            workflow_id=workflow_id,
        )

    @staticmethod
    def _with_duplicate_leave_guard(ops: list[PlanOp]) -> list[PlanOp]:
        if ops and ops[0] == PlanOp.MAYBE_DUPLICATE_LEAVE:
            return ops
        return [PlanOp.MAYBE_DUPLICATE_LEAVE, *ops]


class WorkflowPipeline:
    """Plan-first workflow executor: PlanBuilder → execute_plan → StatePatchBuffer → apply_state_patches."""

    _is_new_expense_claim = staticmethod(PlanBuilder._is_new_expense_claim)
    _prepare_fresh_expense_draft = staticmethod(PlanBuilder._prepare_fresh_expense_draft)

    @staticmethod
    def _require_turn_context(
        turn_context: TurnContext | None,
        *,
        trace_id: str = "",
    ) -> TurnContext:
        if turn_context is None:
            from chat.services.observability import log_step

            log_step(
                trace_id,
                "turn_context_missing",
                {"reason": "TurnContext required; only ChatOrchestrator builds it per turn"},
            )
            raise AssertionError(
                "TurnContext is required; only ChatOrchestrator may build TurnContext per turn"
            )
        return turn_context

    def __init__(self) -> None:
        self.manager = WorkflowManager()
        self.fields = FieldEngine()
        self.validator = ValidationEngine()
        self.modifier = ModificationEngine(self.fields)
        self.review = ReviewEngine(self.fields, self.validator, self.manager)
        self.submission = SubmissionEngine(self.validator, self.manager)
        self.composer = ResponseComposer()

    def execute_platform_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        turn_context: TurnContext | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Deprecated Phase 5 — prefer execute_workflow_turn; attaches PQ metadata."""
        u = self._require_understanding(understanding, trace_id=trace_id)
        if u is None:
            return None
        ctx = self._require_turn_context(turn_context, trace_id=trace_id)
        result = self.execute_workflow_turn(
            message,
            memory=memory,
            understanding=u,
            pq_decision=pq_decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=ctx,
            route_source="pending",
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        if not result:
            return None
        msg, envelope = result
        envelope["pending_question_decision"] = pq_decision.to_log_dict()
        envelope.setdefault("rules_applied", []).append("PENDING_QUESTION_ENGINE")
        envelope["rules_applied"].append(pq_decision.kind.value.upper())
        return msg, envelope

    def build_platform_response(
        self,
        pq_decision: PendingQuestionDecision,
        *,
        memory: SessionMemory,
        user_message: str,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
        understanding: UnderstandingResult | None = None,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        turn_context: TurnContext | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Deprecated Phase 5 — thin wrapper around execute_platform_turn."""
        handled = self.execute_platform_turn(
            user_message,
            memory=memory,
            pq_decision=pq_decision,
            understanding=understanding,
            conversation_history=conversation_history or [],
            trace_id=trace_id,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            turn_context=turn_context,
        )
        if handled:
            return handled
        return "", {}

    def execute_plan(
        self,
        plan: ExecutionPlan,
        ctx: TurnContext,
        *,
        message: str,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        """Run an ExecutionPlan — first op that returns a response wins."""
        from chat.services.observability import log_step

        log_step(trace_id, "execution_plan", plan.to_log_dict())
        lang = normalize_reply_lang(ctx.reply_language or detect_user_language(message))

        for op in plan.ops:
            if op == PlanOp.NONE:
                continue
            result = self._run_plan_op(
                op,
                ctx=ctx,
                message=message,
                memory=memory,
                understanding=understanding,
                pq_decision=pq_decision,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
                idempotency_key=idempotency_key,
            )
            if result is not None:
                msg, envelope = result
                envelope.setdefault("rules_applied", []).append(f"PLAN_{op.value.upper()}")
                envelope["execution_plan"] = plan.to_log_dict()
                return msg, envelope
        return None

    def _run_workflow_plan_op(
        self,
        op: PlanOp,
        *,
        ctx: TurnContext,
        message: str,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]] | None:
        """Phase 5 — single dispatch for all workflow-scoped PlanOps."""
        wf_id = (
            (memory.active_workflow.id if memory.active_workflow else "")
            or (ctx.active_workflow_id or "")
            or (
                (understanding.workflow or "").strip().lower()
                if understanding and understanding.workflow not in ("none", "")
                else ""
            )
            or (
                (pq_decision.target_workflow or "").strip().lower()
                if pq_decision and pq_decision.target_workflow
                else ""
            )
        )

        if op == PlanOp.WORKFLOW_COLLECT:
            if pq_decision is None and wf_id == "expense" and understanding:
                defn = self.manager.ensure_definition(wf_id)
                draft = state.ensure_active_draft(wf_id)
                if not draft:
                    return self.composer.clarification(understanding, lang=lang), {
                        "outcome": "NEEDS_CLARIFICATION",
                    }
                return self._handle_expense_draft_collect(
                    message,
                    memory=memory,
                    defn=defn,
                    understanding=understanding,
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    lang=lang,
                    state=state,
                )
            if pq_decision is None:
                return None
            return self._handle_collect(
                message,
                memory=memory,
                pq_decision=pq_decision,
                understanding=understanding,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                state=state,
            )
        if op == PlanOp.WORKFLOW_NEW:
            return self._handle_new(
                message,
                memory=memory,
                understanding=understanding,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
                idempotency_key=idempotency_key,
                state=state,
            )
        if op == PlanOp.WORKFLOW_MODIFY:
            return self._handle_modify(
                message,
                memory=memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                understanding=understanding,
                state=state,
            )
        if op == PlanOp.WORKFLOW_DELETE:
            return self._handle_delete(
                message,
                memory=memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                understanding=understanding,
                state=state,
            )
        if op == PlanOp.WORKFLOW_SWITCH:
            pq = pq_decision or self._synthesize_switch_pq_decision(understanding)
            if pq is None:
                return None
            return self._handle_switch(
                message,
                memory=memory,
                pq_decision=pq,
                understanding=understanding,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                state=state,
            )
        if op == PlanOp.WORKFLOW_CLARIFICATION:
            draft = memory.active_draft()
            if understanding and understanding.workflow == "leave" and not memory.active_workflow:
                state.push("merge_last_entities", value={"leave_start_clarify": True})
            elif memory.active_workflow and draft:
                state.push("merge_last_entities", value={"leave_start_clarify": False})
            expense_intent = str((understanding.entities or {}).get("expense_intent") or "").lower()
            if (
                memory.active_workflow
                and memory.active_workflow.id == "expense"
                and expense_intent in ("clarify_delete", "clarify_modify")
            ):
                turn = dict((understanding.entities or {}).get("expense_turn") or {})
                turn.setdefault("intent", expense_intent)
                body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "reason": understanding.reasoning if understanding else "",
                    "understanding": understanding.to_dict() if understanding else {},
                    "rules_applied": ["EXPENSE_EDIT_CLARIFY"],
                }
            return self.composer.clarification(
                understanding, lang=lang, draft=draft, memory=memory
            ), {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": understanding.reasoning if understanding else "",
                "understanding": understanding.to_dict() if understanding else {},
            }
        if op == PlanOp.WORKFLOW_SHOW_REVIEW:
            show_target = str((understanding.entities or {}).get("show_workflow_target") or wf_id or "").lower()
            defn = get_workflow_definition(wf_id) or self._active_defn(memory)
            if show_target == "leave" or (defn and defn.workflow_id == "leave"):
                body = self.composer.leave_status_report(memory, lang=lang)
                return body, {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["LEAVE_SESSION_SUMMARY"],
                }
            if defn:
                return self._show_review(
                    memory,
                    defn,
                    lang=lang,
                    state=state,
                    message=message,
                    trace_id=trace_id,
                )
            return None
        if op == PlanOp.WORKFLOW_REQUEST_SUBMIT:
            defn = get_workflow_definition(wf_id) or self._active_defn(memory)
            if defn:
                return self._request_submit(memory, defn, lang=lang, state=state)
            return None
        if op == PlanOp.WORKFLOW_CONFIRM_SUBMIT:
            defn = self._active_defn(memory)
            if defn:
                return self._confirm_submit(
                    memory,
                    defn,
                    lang=lang,
                    company_id=company_id,
                    employee_id=employee_id,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    state=state,
                )
            return None
        if op == PlanOp.WORKFLOW_APPLY_UPDATES:
            defn = get_workflow_definition(wf_id) or self._active_defn(memory)
            draft = memory.active_draft()
            if defn and draft and memory.active_workflow:
                was_submit = memory.pending_confirmation == "submit"
                updates = list(understanding.field_updates or [])
                if wf_id == "leave":
                    from chat.services.platform.field_extractors.leave import (
                        filter_leave_updates_for_review,
                    )

                    updates = filter_leave_updates_for_review(
                        updates, message, memory=memory, trace_id=trace_id
                    )
                if wf_id == "expense":
                    from chat.services.platform.field_extractors.expense import (
                        filter_expense_updates_for_review,
                    )

                    expense_turn = dict((understanding.entities or {}).get("expense_turn") or {})
                    expense_intent = str((understanding.entities or {}).get("expense_intent") or "").lower()
                    if expense_intent == "clarify_delete":
                        updates = []
                    else:
                        updates = filter_expense_updates_for_review(
                            updates,
                            message,
                            memory=memory,
                            trace_id=trace_id,
                            expense_turn=expense_turn,
                        )
                if not updates:
                    return None
                if not was_submit:
                    state.push("clear_pending_confirmation")
                if wf_id == "leave":
                    return self._finish_leave_update_turn(
                        memory,
                        defn,
                        updates=updates,
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=(
                            ["LEAVE_MODIFY_REVIEW"] if was_submit else ["LEAVE_COLLECT_APPLY"]
                        ),
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                    )
                if wf_id == "expense":
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=updates,
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=(
                            ["EXPENSE_MODIFY_REVIEW"] if was_submit else ["EXPENSE_COLLECT_APPLY"]
                        ),
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                        understanding=understanding,
                    )
                state.apply_field_updates(updates, message=message)
                if wf_id == "expense":
                    self._push_default_expense_date(state, memory)
                prefix = self.composer.item_prefix_from_updates(updates, lang=lang)
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)
            return None
        if op == PlanOp.WORKFLOW_CANCEL:
            return self._handle_workflow_cancel(
                memory,
                lang=lang,
                state=state,
                understanding=understanding,
                pq_decision=pq_decision,
            )
        return None

    def _handle_workflow_cancel(
        self,
        memory: SessionMemory,
        *,
        lang: str,
        state: StatePatchBuffer,
        understanding: UnderstandingResult | None = None,
        pq_decision=None,
    ) -> tuple[str, dict[str, Any]]:
        from chat.services.platform.field_extractors.expense import expense_has_cancellable_draft

        draft = memory.active_draft()
        if draft and (draft.locked or str(draft.status or "").lower() == "submitted"):
            rid = str(draft.submitted_request_id or "")
            if draft.workflow_id == "expense":
                msg = self.composer.expense_submitted_locked(rid, lang=lang)
            else:
                msg = self.composer.locked_with_reference(rid, lang=lang)
            return msg, {
                "outcome": "INFORMATIONAL",
                "rules_applied": ["POST_SUBMIT_LOCK"],
            }

        target = ""
        if understanding is not None:
            target = str((understanding.entities or {}).get("cancel_workflow_target") or "").strip().lower()
            if not target:
                target = str(understanding.workflow or "").strip().lower()
        if not target and pq_decision is not None:
            target = str(getattr(pq_decision, "target_workflow", None) or "").strip().lower()

        aw = memory.active_workflow
        wf_id = ""
        if aw and (not target or target == aw.id):
            wf_id = aw.id
        elif target in ("leave", "expense"):
            wf_id = target
        else:
            wf_id = aw.id if aw else target or "workflow"

        if wf_id == "expense" and not expense_has_cancellable_draft(memory):
            submitted = list((memory.conversation_facts or {}).get("submitted_expenses") or [])
            rid = ""
            if submitted and isinstance(submitted[-1], dict):
                rid = str(submitted[-1].get("request_id") or "")
            msg = self.composer.expense_no_draft_to_cancel(lang=lang, request_id=rid)
            return msg, {
                "outcome": "INFORMATIONAL",
                "rules_applied": ["EXPENSE_NO_DRAFT_CANCEL"],
            }

        if aw and (not target or target == aw.id):
            state.push("cancel_active_workflow")
            wf_id = aw.id
        elif target in ("leave", "expense"):
            state.push("cancel_workflow_draft", value=target)
            wf_id = target
        else:
            state.push("cancel_active_workflow")
            wf_id = aw.id if aw else "workflow"

        msg = self.composer.workflow_cancelled(wf_id, lang=lang)
        return msg, {
            "outcome": "CANCELLED",
            "reason": "workflow cancelled by user",
            "rules_applied": ["WORKFLOW_CANCEL"],
        }

    def _run_plan_op(
        self,
        op: PlanOp,
        *,
        ctx: TurnContext,
        message: str,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
    ) -> tuple[str, dict[str, Any]] | None:
        state = StatePatchBuffer(memory, trace_id=trace_id)
        result: tuple[str, dict[str, Any]] | None = None

        if op == PlanOp.REJECT_OOS:
            self._pause_for_oos(state)
            result = self._reject_oos(lang)
        elif op == PlanOp.LOCKED_RESPONSE:
            result = self._locked_response(memory, lang)
        elif op == PlanOp.SUBMITTED_LEAVE_OVERLAP:
            result = self._submitted_leave_overlap_response(
                ctx.user_message or "",
                memory,
                lang=lang,
            )
        elif op == PlanOp.RESOLVE_WORKFLOW_SWITCH:
            result = self._resolve_workflow_switch(
                message,
                memory,
                lang,
                state=state,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
        elif op == PlanOp.RESOLVE_DUPLICATE_LEAVE:
            result = self._resolve_duplicate_leave(
                message, memory, lang, understanding=understanding, state=state
            )
        elif op == PlanOp.RESOLVE_SUBMIT_CONFIRMATION:
            result = self._resolve_submit_confirmation(
                message,
                memory,
                understanding,
                lang=lang,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
                idempotency_key=idempotency_key,
                state=state,
            )
        elif op == PlanOp.APPLY_PENDING_MODIFY:
            result = self._apply_pending_modify(memory, lang, state=state)
        elif op == PlanOp.CLEAR_PENDING_MODIFY:
            state.push("clear_pending_confirmation")
        elif op == PlanOp.SHOW_SESSION_CONTEXT:
            result = self.composer.session_context(memory, lang=lang), {
                "outcome": "INFORMATIONAL",
                "reason": "No pending confirmation — showing session context.",
                "rules_applied": ["SESSION_CONTEXT"],
            }
        elif op == PlanOp.MAYBE_DUPLICATE_LEAVE:
            result = self._block_parallel_leave(message, memory, understanding, lang, state=state)
        elif op == PlanOp.MAYBE_WORKFLOW_SWITCH:
            result = self._maybe_workflow_switch_confirm(
                message, memory, lang, understanding=understanding, state=state
            )
        elif op in WORKFLOW_PLAN_OPS:
            result = self._run_workflow_plan_op(
                op,
                ctx=ctx,
                message=message,
                memory=memory,
                understanding=understanding,
                pq_decision=pq_decision,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
                idempotency_key=idempotency_key,
                state=state,
            )
        elif op == PlanOp.REPLY_POLICY:
            self._pause_active_workflow_for_interrupt(state)
            result = self._reply_policy(
                message,
                memory=memory,
                ctx=ctx,
                pq_decision=pq_decision,
                conversation_history=conversation_history,
                company_id=company_id,
                trace_id=trace_id,
                state=state,
            )
        elif op == PlanOp.REPLY_STATUS:
            self._pause_active_workflow_for_interrupt(state)
            result = self._reply_status(
                message,
                memory=memory,
                pq_decision=pq_decision,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session_id,
                state=state,
            )
        elif op == PlanOp.REPLY_OOS:
            result = self._reply_oos(
                message,
                memory=memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                state=state,
            )
        elif op == PlanOp.REPLY_GREETING:
            result = self._reply_greeting(
                message,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
        elif op == PlanOp.REPLY_CONVERSATIONAL:
            result = self._reply_conversational(
                message,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
            if (
                result
                and understanding
                and understanding.workflow == "leave"
                and not memory.active_workflow
            ):
                state.push("merge_last_entities", value={"leave_start_clarify": True})
        elif op == PlanOp.REPLY_PLATFORM_CLARIFY:
            result = self._reply_platform_clarify(
                memory=memory,
                understanding=understanding,
                lang=lang,
            )
        elif op == PlanOp.REPLY_GENERAL_HELP:
            result = self._reply_general_help(understanding=understanding, lang=lang)
        elif op == PlanOp.REPLY_TODAY_DATE:
            self._pause_active_workflow_for_interrupt(state)
            result = self._reply_today_date(
                memory=memory,
                ctx=ctx,
                lang=lang,
            )
        elif op == PlanOp.REPLY_TRANSLATION:
            result = self._reply_translation(
                message,
                memory=memory,
                pq_decision=pq_decision,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
            )

        state.flush()
        return result

    def _maybe_execute_turn_plan(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        turn_context: TurnContext | None,
        route_source: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        ctx = self._require_turn_context(turn_context, trace_id=trace_id)
        from chat.services.observability import log_turn_context_layer

        log_turn_context_layer(trace_id, "plan", ctx)
        decision = TurnDecision(
            pq=pq_decision,
            understanding=understanding,
            route_source=route_source,
        )
        plan = PlanBuilder.build(ctx, decision)
        from chat.services.observability import patch_turn_trace

        patch_turn_trace(
            trace_id,
            turn_decision=decision.to_log_dict(),
            execution_plan=plan.to_log_dict() if plan else None,
            plan_skipped=plan is None or plan.primary_op == PlanOp.NONE,
            plan_skip_reason=(plan.reason if plan else "no plan"),
        )
        if plan is None or plan.primary_op == PlanOp.NONE:
            if getattr(settings, "USE_INFORMATIONAL_PLAN", True):
                plan = PlanBuilder.informational_fallback_plan(ctx, decision)
            else:
                return None
        return self.execute_plan(
            plan,
            ctx,
            message=message,
            memory=memory,
            understanding=understanding,
            pq_decision=pq_decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )

    def execute_planned_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        turn_context: TurnContext,
        route_source: str = "pending",
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        """Deprecated Phase 5 — alias for execute_workflow_turn."""
        return self.execute_workflow_turn(
            message,
            memory=memory,
            understanding=understanding,
            pq_decision=pq_decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=turn_context,
            route_source=route_source,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _require_understanding(
        understanding: UnderstandingResult | None,
        *,
        trace_id: str = "",
    ) -> UnderstandingResult | None:
        if understanding is None:
            from chat.services.observability import log_step

            log_step(trace_id, "understanding_missing", {"reason": "orchestrator must supply understanding"})
            return None
        return understanding

    def execute_workflow_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        turn_context: TurnContext,
        route_source: str = "pending",
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        pre_patches: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Single workflow entry: PlanBuilder → execute_plan → clarify on miss."""
        u = self._require_understanding(understanding, trace_id=trace_id)
        if u is None:
            return None
        from chat.services.observability import log_turn_context_layer

        log_turn_context_layer(trace_id, "executor", turn_context)

        if pre_patches:
            apply_state_patches(memory, pre_patches)

        plan_result = self._maybe_execute_turn_plan(
            message,
            memory=memory,
            understanding=u,
            pq_decision=pq_decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=turn_context,
            route_source=route_source,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        if plan_result is not None:
            return plan_result

        lang = normalize_reply_lang(turn_context.reply_language or detect_user_language(message))
        draft = memory.active_draft()
        if (
            (u.workflow or "").strip().lower() == "expense"
            and memory.active_workflow
            and memory.active_workflow.id == "leave"
        ):
            apply_state_patches(
                memory,
                [
                    {
                        "op": "merge_last_entities",
                        "value": {
                            "last_expense_clarify_message": message,
                            "expense_new_claim": True,
                        },
                    }
                ],
            )
        return self.composer.clarification(u, lang=lang, draft=draft, memory=memory), {
            "outcome": "NEEDS_CLARIFICATION",
            "reason": u.reasoning or "Could not execute any plan op for this turn.",
            "understanding": u.to_dict(),
            "rules_applied": ["PLAN_EXEC_MISS_CLARIFY"],
        }

    def execute_turn(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        turn_context: TurnContext | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Deprecated Phase 5 — alias for handle → execute_workflow_turn."""
        return self.handle(
            message,
            memory=memory,
            pq_decision=pq_decision,
            understanding=understanding,
            conversation_history=conversation_history,
            trace_id=trace_id,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            turn_context=turn_context,
        )

    def handle(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        turn_context: TurnContext | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Deprecated Phase 5 — alias for execute_workflow_turn with pending PQ routing."""
        u = self._require_understanding(understanding, trace_id=trace_id)
        if u is None:
            return None
        ctx = self._require_turn_context(turn_context, trace_id=trace_id)
        return self.execute_workflow_turn(
            message,
            memory=memory,
            understanding=u,
            pq_decision=pq_decision,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=ctx,
            route_source="pending",
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )

    def try_handle_active_workflow(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult | None = None,
        conversation_history: list[str],
        trace_id: str,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        turn_context: TurnContext | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Deprecated Phase 5 — alias for execute_workflow_turn with route_source=active."""
        if not memory.active_workflow:
            return None
        u = self._require_understanding(understanding, trace_id=trace_id)
        if u is None:
            return None
        ctx = self._require_turn_context(turn_context, trace_id=trace_id)
        return self.execute_workflow_turn(
            message,
            memory=memory,
            understanding=u,
            pq_decision=None,
            conversation_history=conversation_history,
            trace_id=trace_id,
            turn_context=ctx,
            route_source="active",
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )

    def _handle_new(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        u = understanding
        from chat.services.platform.field_extractors.expense import expense_entities_for_turn

        wf_id_early = (u.workflow or "").strip().lower()
        if wf_id_early == "expense" and self._is_new_expense_claim(
            message,
            u,
            active_workflow_id=(memory.active_workflow.id if memory.active_workflow else ""),
            memory=memory,
            trace_id=trace_id,
        ):
            self._prepare_fresh_expense_draft(state, memory)

        expense_entity_patch: dict[str, Any] = {}
        if (u.workflow or "").strip().lower() == "expense":
            if isinstance((u.entities or {}).get("expense_turn"), dict):
                expense_entity_patch = expense_entities_for_turn(
                    dict(memory.last_entities or {}),
                    (u.entities or {}).get("expense_turn"),
                    expense_intent=str((u.entities or {}).get("expense_intent") or ""),
                    action=str(u.action or ""),
                )
            else:
                expense_entity_patch = expense_entities_for_turn(
                    dict(memory.last_entities or {}),
                    None,
                )
        state.push(
            "merge_last_entities",
            value={
                "turn_understanding": u.to_dict(),
                "leave_start_clarify": False,
                **expense_entity_patch,
            },
        )

        if u.is_out_of_scope:
            self._pause_for_oos(state)
            return self._reject_oos(lang)

        if u.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
            draft = memory.active_draft()
            wf_id = (u.workflow or "").strip().lower()
            if (
                wf_id == "expense"
                and draft
                and draft.locked
                and draft.workflow_id == "leave"
            ):
                state.push(
                    "merge_last_entities",
                    value={
                        "last_expense_clarify_message": message,
                        "leave_start_clarify": False,
                    },
                )
                interrupted = self._maybe_workflow_switch_confirm(
                    message,
                    memory,
                    lang,
                    understanding=u,
                    state=state,
                )
                if interrupted:
                    return interrupted
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": u.reasoning,
                "understanding": u.to_dict(),
            }

        wf_id = (u.workflow or "").strip().lower()
        if wf_id in ("none", "policy", "status", ""):
            return self.composer.clarification(u, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        defn = get_workflow_definition(wf_id)
        if not defn:
            return self.composer.clarification(u, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        draft = memory.active_draft()
        if u.action in (UnderstandingAction.SUBMIT.value, UnderstandingAction.CONFIRM.value):
            if draft and memory.active_workflow and memory.active_workflow.id == wf_id:
                return self._maybe_submit(
                    memory, defn, u, lang=lang,
                    company_id=company_id, employee_id=employee_id,
                    session_id=session_id, idempotency_key=idempotency_key,
                    state=state,
                )
            # Misclassified submit on first message — start collecting instead.
            u = UnderstandingResult(
                goal=u.goal,
                workflow=wf_id,
                action=UnderstandingAction.START.value,
                confidence=u.confidence,
                field_updates=u.field_updates,
                reasoning=u.reasoning,
                source=u.source,
            )

        draft = memory.active_draft()
        unlocking_submitted = bool(draft and draft.locked and draft.workflow_id == wf_id)
        if unlocking_submitted:
            draft_id = memory.active_workflow.draft_id if memory.active_workflow else "default"
            state.push("ensure_draft", draft_id=draft_id, workflow_id=wf_id)
            state.push("set_active_stage", value=WorkflowStage.COLLECTING.value)
            state.flush()
            if u.action == UnderstandingAction.REVIEW.value:
                u = UnderstandingResult(
                    goal=u.goal,
                    workflow=wf_id,
                    action=UnderstandingAction.START.value,
                    confidence=u.confidence,
                    field_updates=u.field_updates,
                    reasoning=u.reasoning,
                    source=u.source,
                )
        elif u.action == UnderstandingAction.REVIEW.value:
            return self._show_review(
                memory,
                defn,
                lang=lang,
                state=state,
                message=message,
                trace_id=trace_id,
            )

        draft = memory.active_draft()
        if not memory.active_workflow or memory.active_workflow.id != wf_id:
            state.push("start_workflow", value=wf_id)
            state.flush()
        draft = memory.active_draft()
        if not draft:
            return self.composer.clarification(u, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        blocked = self._block_submitted_leave_overlap(memory, u, lang, message=message)
        if blocked:
            return blocked

        if u.field_updates:
            if wf_id == "expense":
                from chat.services.platform.intent_rules import parse_submit_workflow

                was_submit = bool(
                    parse_submit_workflow(message, active_workflow_id=wf_id)
                    or (u.entities or {}).get("submit_after_edit")
                )
                return self._finish_expense_update_turn(
                    memory,
                    defn,
                    updates=list(u.field_updates),
                    message=message,
                    lang=lang,
                    state=state,
                    was_submit=was_submit,
                    rules_applied=["EXPENSE_NEW_CLAIM"],
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    understanding=u,
                )
            state.apply_field_updates(u.field_updates, message=message)
            self._push_default_expense_date(state, memory)
            state.push("clear_pending_confirmation")
            self.manager.events.emit(memory, "field_collected", wf_id, {})

        prefix = self.composer.item_prefix_from_updates(u.field_updates, lang=lang) or self.composer.workflow_started(defn.name, lang=lang)

        return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

    def _handle_expense_draft_collect(
        self,
        message: str,
        *,
        memory: SessionMemory,
        defn,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        """Expense draft editor — intent-first, LLM patches, pending queue."""
        from chat.services.platform.field_extractors.expense import (
            expense_turn_has_targeted_patches,
            expense_turn_llm_blocked,
            is_expense_ambiguous_ack,
            is_expense_category_unknown_decline,
            is_expense_collect_complaint,
            is_expense_draft_mutation_message,
            sync_expense_draft,
        )
        from chat.services.platform.response_composer import localized
        from chat.services.platform.summary import (
            expense_total,
            format_expense_collect_recap,
            format_expense_summary,
        )

        u = understanding
        intent = str((u.entities or {}).get("expense_intent") or "").lower()
        expense_turn = dict((u.entities or {}).get("expense_turn") or {})
        draft = memory.active_draft()

        active_id = memory.active_workflow.id if memory.active_workflow else "expense"
        from chat.services.platform.field_extractors.expense import (
            expense_message_requests_submit,
            clear_expense_blocked_add,
        )

        if expense_message_requests_submit(message, active_workflow_id=active_id):
            from chat.services.platform.field_extractors.expense import message_has_new_expense_items

            if not u.field_updates and not message_has_new_expense_items(message):
                clear_expense_blocked_add(memory)
                return self._request_submit(memory, defn, lang=lang, state=state)

        from chat.services.platform.field_extractors.expense import (
            interpret_expense_turn_semantics,
        )

        sem = interpret_expense_turn_semantics(
            message,
            memory,
            expense_turn,
            trace_id=trace_id,
            conversation_history=conversation_history,
        )
        if (
            not sem.get("replay_blocked_add")
            and not sem.get("date_correction")
            and (
                intent == "date_not_allowed"
                or expense_turn.get("date_policy_rejected")
            )
        ):
            requested = expense_turn.get("rejected_incurred_date")
            body = self.composer.expense_date_policy_blocked(
                lang=lang,
                requested_date=str(requested) if requested else None,
            )
            return self._continue_collection(
                memory,
                defn,
                lang=lang,
                prefix=body,
                state=state,
                extra_rules=["EXPENSE_DATE_TODAY_ONLY"],
            )

        pq_slot = memory.pending_question
        from chat.services.platform.intent_rules import parse_submit_workflow
        from chat.services.platform.field_extractors.expense import message_requests_submit_after_edit

        active_id = memory.active_workflow.id if memory.active_workflow else "expense"
        was_submit = memory.pending_confirmation == "submit" or bool(
            parse_submit_workflow(message, active_workflow_id=active_id)
            or message_requests_submit_after_edit(message, active_workflow_id=active_id)
        )

        from chat.services.platform.intent_rules import is_bare_confirmation

        if (
            pq_slot
            and pq_slot.workflow_id == "expense"
            and is_bare_confirmation(message)
            and pq_slot.field == "item_route"
        ):
            body = self.composer.slot_still_needed(
                pq_slot.field,
                pq_slot.prompt or "",
                lang=lang,
            )
            return body, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["EXPENSE_ROUTE_NEEDED"],
            }

        if intent == "repeat_ack":
            prefix_parts = [self.composer.expense_already_recorded(lang=lang)]
            if expense_turn.get("past_date_rejected"):
                prefix_parts.insert(0, self.composer.expense_past_date_policy(lang=lang))
            prefix = "\n\n".join(prefix_parts)
            return self._continue_collection(
                memory, defn, lang=lang, prefix=prefix, state=state
            )

        has_understood_review_edit = bool(u.field_updates) and intent in (
            "update",
            "modify_review",
            "correct",
            "fix_mistake",
            "delete",
        )
        if intent == "llm_unavailable" or (
            expense_turn_llm_blocked(expense_turn, memory) and not has_understood_review_edit
        ):
            from chat.services.platform.field_extractors.expense import (
                _try_wizard_fallback_turn,
                coerce_pending_expense_turn,
                expense_turn_to_field_updates,
            )

            coerced_pending = coerce_pending_expense_turn(message, memory, trace_id=trace_id)
            if coerced_pending and coerced_pending.get("item_patches"):
                _, pending_updates = expense_turn_to_field_updates(
                    message,
                    memory,
                    trace_id=trace_id,
                    conversation_history=conversation_history,
                    expense_turn=coerced_pending,
                )
                if pending_updates:
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=list(pending_updates),
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=["EXPENSE_PENDING_ROUTE"],
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                        understanding=u,
                    )

            wizard = _try_wizard_fallback_turn(message, memory)
            if wizard and wizard.get("item_patches"):
                _, wizard_updates = expense_turn_to_field_updates(
                    message,
                    memory,
                    trace_id=trace_id,
                    conversation_history=conversation_history,
                    expense_turn=wizard,
                )
                if wizard_updates:
                    wizard_updates = _review_safe_updates(wizard_updates)
                    if wizard_updates:
                        state.apply_field_updates(wizard_updates, message=message)
                        self._push_default_expense_date(state, memory)
                        sync_expense_draft(memory.active_draft())
                        from chat.services.platform.intent_rules import parse_submit_workflow

                        active_id = memory.active_workflow.id if memory.active_workflow else "expense"
                        if parse_submit_workflow(message, active_workflow_id=active_id):
                            return self._maybe_submit(
                                memory,
                                defn,
                                UnderstandingResult(
                                    goal="Submit expense",
                                    workflow="expense",
                                    action=UnderstandingAction.CONFIRM.value,
                                    confidence=0.9,
                                    reasoning="Submit after wizard expense edit.",
                                    source="wizard",
                                ),
                                lang=lang,
                                company_id="",
                                employee_id="",
                                session_id="",
                                idempotency_key="",
                                state=state,
                            )
                        prefix = self.composer.item_prefix_from_updates(wizard_updates, lang=lang)
                        return self._continue_collection(
                            memory, defn, lang=lang, prefix=prefix, state=state
                        )

            body = self.composer.expense_llm_unavailable(lang=lang, for_edit=True)
            if was_submit and draft:
                sync_expense_draft(draft)
                review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
                if review_text:
                    body = f"{body}\n\n{review_text}".strip()
            return body, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["EXPENSE_LLM_UNAVAILABLE"],
                "awaiting_confirmation": was_submit,
            }

        draft = memory.active_draft()

        def _review_safe_updates(raw_updates: list) -> list:
            if not was_submit:
                return list(raw_updates)
            from chat.services.platform.field_extractors.expense import (
                filter_expense_updates_for_review,
            )

            return filter_expense_updates_for_review(
                raw_updates,
                message,
                memory=memory,
                trace_id=trace_id,
                expense_turn=dict((u.entities or {}).get("expense_turn") or {}),
            )

        from chat.services.platform.intent_rules import (
            is_cancel_workflow_message,
            is_delete_request,
            is_leave_navigation_from_expense,
            is_modify_request,
        )
        from chat.services.platform.field_extractors.route import parse_route
        from chat.services.platform.schemas import FieldUpdate, UnderstandingAction

        if is_cancel_workflow_message(message, workflow_id="expense") or (
            u.action == UnderstandingAction.CANCEL.value or intent == "cancel"
        ):
            draft = memory.active_draft()
            if draft and (draft.locked or str(draft.status or "").lower() == "submitted"):
                rid = str(draft.submitted_request_id or "")
                return self.composer.expense_submitted_locked(rid, lang=lang), {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["POST_SUBMIT_LOCK"],
                }
            from chat.services.platform.field_extractors.expense import expense_has_cancellable_draft

            if not expense_has_cancellable_draft(memory):
                submitted = list((memory.conversation_facts or {}).get("submitted_expenses") or [])
                rid = ""
                if submitted and isinstance(submitted[-1], dict):
                    rid = str(submitted[-1].get("request_id") or "")
                return self.composer.expense_no_draft_to_cancel(lang=lang, request_id=rid), {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["EXPENSE_NO_DRAFT_CANCEL"],
                }
            state.push("cancel_active_workflow")
            return self.composer.workflow_cancelled("expense", lang=lang), {
                "outcome": "CANCELLED",
                "rules_applied": ["EXPENSE_CANCEL"],
            }

        from chat.services.platform.field_extractors.expense import (
            expense_turn_to_field_updates,
            pending_expense_edit_active,
            resolve_pending_expense_edit_turn,
        )

        if pending_expense_edit_active(memory):
            resolved = resolve_pending_expense_edit_turn(message, memory)
            if resolved and (
                resolved.get("item_patches") or resolved.get("delete_indices")
            ):
                state.push("merge_last_entities", value={"expense_pending_edit": None})
                state.push("clear_pending_question")
                _, patch_updates = expense_turn_to_field_updates(
                    message,
                    memory,
                    trace_id=trace_id,
                    conversation_history=conversation_history,
                    expense_turn=resolved,
                )
                if patch_updates:
                    resolved_intent = str(resolved.get("intent") or "").lower()
                    rules = ["EXPENSE_PENDING_EDIT_RESOLVE"]
                    if resolved_intent == "delete":
                        rules = ["EXPENSE_RULES_DELETE"]
                    elif resolved_intent in ("modify_review", "update", "correct"):
                        rules = ["EXPENSE_RULES_MODIFY"]
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=_review_safe_updates(list(patch_updates)),
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=rules,
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                        understanding=u,
                    )

        from chat.services.platform.field_extractors.expense import coerce_pending_expense_turn

        coerced_pending = coerce_pending_expense_turn(message, memory, trace_id=trace_id)
        if coerced_pending and coerced_pending.get("item_patches"):
            from chat.services.platform.field_extractors.expense import (
                expense_turn_to_field_updates,
            )

            _, pending_updates = expense_turn_to_field_updates(
                message,
                memory,
                trace_id=trace_id,
                conversation_history=conversation_history,
            )
            if pending_updates:
                return self._finish_expense_update_turn(
                    memory,
                    defn,
                    updates=_review_safe_updates(list(pending_updates)),
                    message=message,
                    lang=lang,
                    state=state,
                    was_submit=was_submit,
                    rules_applied=["EXPENSE_PENDING_ROUTE"],
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    understanding=u,
                )

        if u.action == UnderstandingAction.DELETE.value and u.targets:
            delete_updates = [
                FieldUpdate(
                    field="items",
                    value={},
                    item_index=t.item_index,
                    action="delete",
                )
                for t in u.targets
                if t.item_index is not None
            ]
            if not delete_updates:
                delete_updates = [
                    FieldUpdate(
                        field="items",
                        value={},
                        item_index=u.targets[0].item_index,
                        action="delete",
                    )
                ]
            from chat.services.platform.field_extractors.expense import clear_expense_blocked_add

            clear_expense_blocked_add(memory)
            return self._finish_expense_update_turn(
                memory,
                defn,
                updates=delete_updates,
                message=message,
                lang=lang,
                state=state,
                rules_applied=["EXPENSE_RULES_DELETE"],
                conversation_history=conversation_history,
                trace_id=trace_id,
                understanding=u,
            )

        if draft and is_delete_request(message) and not parse_route(message):
            parsed_del = parse_delete_request(message, list(draft.fields.get("items") or []))
            if parsed_del and parsed_del.get("item_index") is not None:
                return self._finish_expense_update_turn(
                    memory,
                    defn,
                    updates=[
                        FieldUpdate(
                            field="items",
                            value={},
                            item_index=int(parsed_del["item_index"]),
                            action="delete",
                        )
                    ],
                    message=message,
                    lang=lang,
                    state=state,
                    rules_applied=["EXPENSE_RULES_DELETE"],
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    understanding=u,
                )
            if parsed_del and parsed_del.get("needs_clarify"):
                turn = {
                    "intent": "clarify_delete",
                    "item_patches": [],
                    "clarify": {
                        "kind": "which_delete",
                        "candidate_indices": list(parsed_del.get("candidate_indices") or []),
                        "category": parsed_del.get("category") or parsed_del.get("label"),
                    },
                }
                from chat.services.platform.field_extractors.expense import (
                    build_expense_pending_edit_from_turn,
                )

                pending_edit = build_expense_pending_edit_from_turn(turn, message=message, memory=memory)
                if pending_edit:
                    state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
                body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "rules_applied": ["EXPENSE_DELETE_CLARIFY"],
                }
            if is_vague_delete(message) or (
                is_delete_request(message) and not looks_like_expense_item_delete(message)
            ):
                turn = {"intent": "clarify_delete", "item_patches": []}
                from chat.services.platform.field_extractors.expense import (
                    build_expense_pending_edit_from_turn,
                )

                pending_edit = build_expense_pending_edit_from_turn(turn, message=message, memory=memory)
                if pending_edit:
                    state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
                body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "rules_applied": ["EXPENSE_DELETE_CLARIFY"],
                }

        if u.action == UnderstandingAction.MODIFY.value and u.field_updates:
            return self._finish_expense_update_turn(
                memory,
                defn,
                updates=_review_safe_updates(list(u.field_updates)),
                message=message,
                lang=lang,
                state=state,
                was_submit=was_submit,
                rules_applied=["EXPENSE_RULES_MODIFY"],
                conversation_history=conversation_history,
                trace_id=trace_id,
                understanding=u,
            )

        expense_turn_ready = bool(
            u.field_updates
            and (
                expense_turn.get("llm_used")
                or u.source == "llm_expense"
                or (u.entities or {}).get("expense_domain_llm")
            )
            and intent in ("add", "answer_pending", "")
        )
        from chat.services.platform.field_extractors.expense import message_has_new_expense_items

        if (
            draft
            and is_modify_request(message)
            and not expense_turn_ready
            and not message_has_new_expense_items(message)
        ):
            from chat.services.platform.field_extractors.expense import _llm_client_configured

            items = list(draft.fields.get("items") or [])
            prefer_llm = bool(_llm_client_configured())
            parsed = parse_modify_request(
                message,
                items,
                trace_id=trace_id,
                prefer_llm=prefer_llm,
            )
            if parsed and parsed.get("needs_clarify"):
                turn = {
                    "intent": "clarify_modify",
                    "item_patches": [],
                    "clarify": {
                        "kind": "which_item",
                        "candidate_indices": list(parsed.get("candidate_indices") or []),
                        "proposed_value": parsed.get("amount"),
                        "category": parsed.get("category"),
                    },
                }
                body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "rules_applied": ["EXPENSE_MODIFY_CLARIFY"],
                }
            if parsed and parsed.get("item_index") is not None:
                from chat.services.platform.field_extractors.expense import _coerce_route_dict

                body: dict[str, Any] = {}
                if parsed.get("amount") is not None:
                    body["amount"] = float(parsed["amount"])
                if parsed.get("category"):
                    body["category"] = parsed["category"]
                route = _coerce_route_dict(
                    {
                        "from_location": parsed.get("from_location"),
                        "to_location": parsed.get("to_location"),
                    }
                )
                if route:
                    body.update(route)
                if body:
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=[
                            FieldUpdate(
                                field="items",
                                value=body,
                                item_index=int(parsed["item_index"]),
                                action="update",
                            )
                        ],
                        message=message,
                        lang=lang,
                        state=state,
                        rules_applied=["EXPENSE_RULES_MODIFY"],
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                        understanding=u,
                    )
            if prefer_llm or is_expense_draft_mutation_message(message, memory):
                body = self.composer.expense_llm_unavailable(lang=lang, for_edit=True)
                if was_submit and draft:
                    sync_expense_draft(draft)
                    review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
                    if review_text:
                        body = f"{body}\n\n{review_text}".strip()
                return body, {
                    "outcome": "NEEDS_INPUT",
                    "rules_applied": ["EXPENSE_LLM_UNAVAILABLE"],
                    "awaiting_confirmation": was_submit,
                }

        if not u.field_updates and (
            is_expense_message(message) or is_compound_expense_message(message)
        ) and not (pending_expense_edit_active(memory) and intent != "add"):
            from chat.services.platform.field_extractors.expense import (
                coerce_expense_correction_turn,
                expense_turn_to_field_updates,
            )

            expense_turn = dict((u.entities or {}).get("expense_turn") or {})
            if expense_turn.get("item_patches") and str(
                expense_turn.get("intent") or ""
            ).lower() != "add":
                expense_turn = coerce_expense_correction_turn(expense_turn, memory)
            _, retry_updates = expense_turn_to_field_updates(
                message,
                memory,
                trace_id=trace_id,
                conversation_history=conversation_history,
                expense_turn=expense_turn if expense_turn.get("item_patches") else None,
            )
            if retry_updates:
                return self._finish_expense_update_turn(
                    memory,
                    defn,
                    updates=retry_updates,
                    message=message,
                    lang=lang,
                    state=state,
                    rules_applied=["EXPENSE_DRAFT_COLLECT_RETRY"],
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    understanding=understanding,
                )

        if intent == "anti_summary" or (u.entities or {}).get("anti_summary"):
            if draft:
                sync_expense_draft(draft)
            body = self.composer.expense_frustration_reply(
                memory,
                message=message,
                lang=lang,
            )
            return body, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["EXPENSE_ANTI_SUMMARY"],
            }

        if intent == "fix_mistake" and not u.field_updates:
            body = self.composer.expense_repair_ack(
                memory,
                message=message,
                lang=lang,
                repaired=False,
            )
            return body, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["EXPENSE_REPAIR_ACK"],
            }

        if intent in ("clarify_modify", "clarify_delete"):
            from chat.services.platform.field_extractors.expense import (
                build_expense_pending_edit_from_turn,
                sync_expense_draft,
            )

            if draft:
                sync_expense_draft(draft)
            turn = dict(u.entities.get("expense_turn") or {})
            turn.setdefault("intent", intent)
            pending_edit = build_expense_pending_edit_from_turn(
                turn, message=message, memory=memory
            )
            if pending_edit:
                state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
            body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
            return body, {
                "outcome": "NEEDS_CLARIFICATION",
                "rules_applied": ["EXPENSE_EDIT_CLARIFY"],
            }

        if (
            memory.pending_question
            and memory.pending_question.workflow_id == "expense"
            and not is_modify_request(message)
            and not is_delete_request(message)
            and not is_leave_navigation_from_expense(message)
        ):
            from chat.services.platform.field_extractors.expense import expense_turn_to_field_updates

            _, retry_updates = expense_turn_to_field_updates(
                message,
                memory,
                trace_id=trace_id,
                conversation_history=conversation_history,
            )
            if retry_updates:
                return self._finish_expense_update_turn(
                    memory,
                    defn,
                    updates=retry_updates,
                    message=message,
                    lang=lang,
                    state=state,
                    rules_applied=["EXPENSE_DRAFT_RETRY"],
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                )

        if (
            pq_slot
            and pq_slot.workflow_id == "expense"
            and pq_slot.field == "item_category"
            and pq_slot.item_index is not None
            and is_expense_category_unknown_decline(message)
        ):
            from chat.services.platform.schemas import FieldUpdate

            return self._finish_expense_update_turn(
                memory,
                defn,
                updates=[
                    FieldUpdate(
                        field="items",
                        value={},
                        item_index=pq_slot.item_index,
                        action="delete",
                    )
                ],
                message=message,
                lang=lang,
                state=state,
                rules_applied=["EXPENSE_CATEGORY_REMOVE"],
                conversation_history=conversation_history,
                trace_id=trace_id,
            )

        if u.action == UnderstandingAction.REVIEW.value or intent in (
            "show_summary",
            "show_list",
            "show_total",
        ):
            if draft:
                sync_expense_draft(draft)
            if intent == "show_total" and draft:
                total = expense_total(draft)
                prefix = localized(
                    lang,
                    en=f"**Total: {total:.0f} taka**",
                    bn=f"**মোট: {total:.0f} taka**",
                    banglish=f"**Total: {total:.0f} taka**",
                )
                body = format_expense_summary(draft, lang=lang, memory=memory)
                return f"{prefix}\n\n{body}".strip(), {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["EXPENSE_TOTAL"],
                }
            if draft:
                return self.composer.workflow_summary(defn, draft, lang=lang), {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["EXPENSE_SUMMARY"],
                }

        if u.action == UnderstandingAction.CONFIRM.value or intent == "confirm":
            return self._request_submit(memory, defn, lang=lang, state=state)

        from chat.services.platform.field_extractors.expense import pending_expense_edit_active

        if (
            u.field_updates
            and pending_expense_edit_active(memory)
            and intent not in ("delete",)
        ):
            pass
        elif u.field_updates and (
            intent in ("date_correction", "replay_blocked_add")
            or intent not in ("delete", "clarify_delete", "date_not_allowed")
        ):
            entities = dict(memory.last_entities or {})
            entities["expense_pending_intent"] = intent
            memory.last_entities = entities
            rules = ["EXPENSE_DRAFT_PATCH"]
            if intent == "delete":
                rules = ["EXPENSE_RULES_DELETE"]
            elif intent in ("update", "modify_review", "correct") or expense_turn_has_targeted_patches(
                (u.entities or {}).get("expense_turn"), memory
            ):
                rules = ["EXPENSE_RULES_MODIFY"]
            return self._finish_expense_update_turn(
                memory,
                defn,
                updates=_review_safe_updates(list(u.field_updates)),
                message=message,
                lang=lang,
                state=state,
                was_submit=was_submit,
                rules_applied=rules,
                conversation_history=conversation_history,
                trace_id=trace_id,
                understanding=u,
            )

        if (
            not u.field_updates
            and (
                intent in ("update", "modify_review", "correct")
                or expense_turn_has_targeted_patches((u.entities or {}).get("expense_turn"), memory)
            )
        ):
            from chat.services.platform.field_extractors.expense import (
                coerce_expense_correction_turn,
                expense_turn_has_targeted_patches,
                expense_turn_to_field_updates,
            )

            expense_turn = dict((u.entities or {}).get("expense_turn") or {})
            if expense_turn_llm_blocked(expense_turn, memory):
                body = self.composer.expense_llm_unavailable(lang=lang, for_edit=True)
                if was_submit and draft:
                    sync_expense_draft(draft)
                    review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
                    if review_text:
                        body = f"{body}\n\n{review_text}".strip()
                return body, {
                    "outcome": "NEEDS_INPUT",
                    "rules_applied": ["EXPENSE_LLM_UNAVAILABLE"],
                    "awaiting_confirmation": was_submit,
                }
            if expense_turn_has_targeted_patches(expense_turn, memory):
                expense_turn = coerce_expense_correction_turn(expense_turn, memory)
            if expense_turn.get("item_patches"):
                _, patch_updates = expense_turn_to_field_updates(
                    message,
                    memory,
                    trace_id=trace_id,
                    conversation_history=conversation_history,
                    expense_turn=expense_turn,
                )
                if patch_updates:
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=_review_safe_updates(list(patch_updates)),
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=["EXPENSE_RULES_MODIFY"],
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                        understanding=u,
                    )

        if intent == "conversation" or is_expense_ambiguous_ack(message):
            from chat.services.platform.intent_rules import is_delete_request, is_modify_request

            if (u.entities or {}).get("expense_llm_degraded") and not u.field_updates:
                pq_degraded = memory.pending_question
                answering_pending_slot = bool(
                    u.answers_pending_field
                    or (
                        pq_degraded
                        and pq_degraded.workflow_id == "expense"
                        and not is_modify_request(message)
                        and not is_delete_request(message)
                    )
                )
                if answering_pending_slot:
                    from chat.services.platform.field_extractors.expense import (
                        coerce_pending_expense_turn,
                        expense_turn_to_field_updates,
                    )

                    coerced_pending = coerce_pending_expense_turn(message, memory, trace_id=trace_id)
                    if coerced_pending and coerced_pending.get("item_patches"):
                        _, pending_updates = expense_turn_to_field_updates(
                            message,
                            memory,
                            trace_id=trace_id,
                            conversation_history=conversation_history,
                            expense_turn=coerced_pending,
                        )
                        if pending_updates:
                            return self._finish_expense_update_turn(
                                memory,
                                defn,
                                updates=_review_safe_updates(list(pending_updates)),
                                message=message,
                                lang=lang,
                                state=state,
                                was_submit=was_submit,
                                rules_applied=["EXPENSE_PENDING_ROUTE"],
                                conversation_history=conversation_history,
                                trace_id=trace_id,
                                understanding=u,
                            )
                    if pq_degraded:
                        body = self.composer.slot_still_needed(
                            pq_degraded.field,
                            pq_degraded.prompt or "",
                            lang=lang,
                        )
                        return body, {
                            "outcome": "NEEDS_INPUT",
                            "rules_applied": ["EXPENSE_PENDING_SLOT_RETRY"],
                        }

                body = self.composer.expense_llm_unavailable(
                    lang=lang,
                    for_edit=bool(
                        is_modify_request(message)
                        or is_delete_request(message)
                        or expense_turn_llm_blocked(expense_turn, memory)
                    ),
                )
                return body, {
                    "outcome": "NEEDS_INPUT",
                    "rules_applied": ["EXPENSE_LLM_DEGRADED"],
                }

            if (
                memory.pending_confirmation == "submit"
                and (is_modify_request(message) or is_delete_request(message) or is_expense_collect_complaint(message))
            ):
                turn = dict(u.entities.get("expense_turn") or {})
                turn.setdefault("intent", "clarify_modify" if is_modify_request(message) else "clarify_delete")
                from chat.services.platform.field_extractors.expense import (
                    build_expense_pending_edit_from_turn,
                )

                pending_edit = build_expense_pending_edit_from_turn(turn, message=message, memory=memory)
                if pending_edit:
                    state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
                body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "rules_applied": ["EXPENSE_REVIEW_EDIT_FALLBACK"],
                }
            if draft:
                sync_expense_draft(draft)
                from chat.services.platform.field_extractors.expense import (
                    build_pending_queue,
                    expense_focus_prompt,
                )

                focus_q = None
                pq = memory.pending_question
                if pq and pq.workflow_id == "expense":
                    queue = build_pending_queue(list(draft.fields.get("items") or []))
                    if queue:
                        focus_q = expense_focus_prompt(queue[0], lang=lang)
                body = format_expense_collect_recap(
                    draft,
                    lang=lang,
                    include_focus_question=focus_q,
                )
                if focus_q:
                    return body, {
                        "outcome": "NEEDS_INPUT",
                        "rules_applied": ["EXPENSE_AMBIGUOUS_RECAP"],
                    }
                return body, {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["EXPENSE_SUMMARY"],
                }

        if intent == "conversation":
            if memory.pending_confirmation == "submit" and draft:
                review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
                msg = self.composer.clarification(u, lang=lang, draft=draft, memory=memory)
                if review_text:
                    msg = f"{msg}\n\n{review_text}".strip()
                return msg, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "awaiting_confirmation": True,
                    "rules_applied": ["EXPENSE_REVIEW_CLARIFY"],
                }
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "rules_applied": ["EXPENSE_CONVERSATION"],
            }

        if memory.pending_confirmation == "submit" and not u.field_updates:
            review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
            msg = self.composer.clarification(u, lang=lang, draft=draft, memory=memory)
            if review_text:
                msg = f"{msg}\n\n{review_text}".strip()
            return msg, {
                "outcome": "NEEDS_CLARIFICATION",
                "awaiting_confirmation": True,
                "rules_applied": ["EXPENSE_REVIEW_CLARIFY"],
            }

        return self._continue_collection(
            memory,
            defn,
            lang=lang,
            prefix="",
            state=state,
        )

    def _handle_collect(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        u = understanding
        wf_id = memory.active_workflow.id if memory.active_workflow else u.workflow
        defn = self.manager.ensure_definition(wf_id)
        draft = state.ensure_active_draft(wf_id)
        if not draft:
            return self.composer.clarification(u, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        if wf_id == "expense":
            return self._handle_expense_draft_collect(
                message,
                memory=memory,
                defn=defn,
                understanding=u,
                conversation_history=conversation_history,
                trace_id=trace_id,
                lang=lang,
                state=state,
            )

        if u.action == UnderstandingAction.REVIEW.value:
            if defn.workflow_id == "leave" and draft:
                return self.composer.workflow_summary(defn, draft, lang=lang), {
                    "outcome": "INFORMATIONAL",
                    "rules_applied": ["SUMMARY"],
                }

        saved_field = memory.pending_question.field if memory.pending_question else "field"

        if (
            memory.pending_question
            and memory.pending_question.field == "medical_document"
            and is_medical_document_skip_message(message)
        ):
            return self._handle_medical_document_skip(memory, defn, lang=lang, state=state)

        if memory.pending_question and memory.pending_question.field == "reason" and is_reason_skip_message(message):
            from chat.services.platform.field_extractors.leave import (
                apply_leave_derived_fields,
                infer_leave_reason_from_history,
            )

            backfill = infer_leave_reason_from_history(memory, conversation_history, trace_id=trace_id)
            if backfill:
                state.apply_field_updates(
                    [FieldUpdate(field="reason", value=backfill, action="set")],
                    message=message,
                )
                state.push("clear_pending_question")
                state.push("merge_last_entities", value={"leave_start_clarify": False})
                draft = memory.active_draft()
                if draft:
                    apply_leave_derived_fields(draft, message=message)
                prefix = self.composer.item_prefix_from_updates(
                    [FieldUpdate(field="reason", value=backfill, action="set")],
                    lang=lang,
                )
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

            state.apply_field_updates(
                [FieldUpdate(field="reason_skipped", value=True, action="set")],
                message=message,
            )
            state.push("clear_pending_question")
            state.push("merge_last_entities", value={"leave_start_clarify": False})

            draft = memory.active_draft()
            if draft:
                apply_leave_derived_fields(draft, message=message)
            prefix = self.composer.leave_reason_skipped(lang=lang)
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        if memory.pending_question and memory.pending_question.field == "delete_which_item":
            m = re.search(r"\b(\d+)\b", message)
            if m:
                idx = int(m.group(1)) - 1
                from chat.services.platform.schemas import TargetRef, UnderstandingResult

                self.modifier.apply_understanding(
                    memory,
                    UnderstandingResult(
                        targets=[TargetRef(field="items", item_index=idx)],
                    ),
                    state=state,
                )
                state.push("clear_pending_question")
                prefix = self.composer.item_removed_by_index(idx, lang=lang)
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        if u.field_updates:
            if wf_id == "expense":
                return self._finish_expense_update_turn(
                    memory,
                    defn,
                    updates=list(u.field_updates),
                    message=message,
                    lang=lang,
                    state=state,
                    rules_applied=["EXPENSE_COLLECT_APPLY"],
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    understanding=u,
                )
            state.apply_field_updates(u.field_updates, message=message)
            saved_field = u.field_updates[0].field
        elif u.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "reason": u.reasoning,
                "understanding": u.to_dict(),
            }

        if not u.field_updates and memory.pending_question:
            pending_field = memory.pending_question.field
            wf_id_for_field = memory.pending_question.workflow_id or wf_id
            if wf_id_for_field == "expense":
                return self._handle_expense_draft_collect(
                    message,
                    memory=memory,
                    defn=defn,
                    understanding=u,
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    lang=lang,
                    state=state,
                )
            parsed_val = self.fields.parse_pending_field(
                wf_id_for_field,
                pending_field,
                message,
                memory=memory,
            )
            if parsed_val is not None:
                state.apply_field_updates(
                    [FieldUpdate(field=pending_field, value=parsed_val)],
                    message=message,
                )
                saved_field = pending_field
            else:
                pq = memory.pending_question
                still_need = self.composer.slot_still_needed(
                    pending_field,
                    pq.prompt if pq else "",
                    lang=lang,
                )
                return still_need, {
                    "outcome": "NEEDS_INPUT",
                    "rules_applied": ["SLOT_MISSING"],
                }

        state.push("clear_pending_question")
        state.push("clear_pending_confirmation")
        self._push_default_expense_date(state, memory)
        self.manager.events.emit(memory, "field_collected", wf_id, {"field": saved_field})

        prefix = self.composer.field_saved(str(saved_field), lang=lang, workflow_id=wf_id)
        if u.field_updates and u.field_updates[0].field == "items":
            val = u.field_updates[0].value
            if isinstance(val, dict):
                prefix = self.composer.item_added(val, lang=lang)

        return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

    def _handle_modify(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        understanding: UnderstandingResult | None = None,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        u = understanding
        draft = memory.active_draft()
        if u is None:
            return self.composer.clarification(
                UnderstandingResult(reasoning="Missing understanding for modify."),
                lang=lang,
                draft=draft,
            ), {"outcome": "NEEDS_CLARIFICATION"}
        if u.action == UnderstandingAction.CLARIFICATION_NEEDED.value and not (
            understanding.field_updates or []
        ):
            from chat.services.platform.field_extractors.expense import (
                pending_expense_edit_active,
                resolve_pending_expense_edit_turn,
            )

            if (
                draft
                and draft.workflow_id == "expense"
                and pending_expense_edit_active(memory)
                and resolve_pending_expense_edit_turn(message, memory)
            ):
                defn = self.manager.ensure_definition("expense")
                return self._handle_expense_draft_collect(
                    message,
                    memory=memory,
                    defn=defn,
                    understanding=u,
                    conversation_history=conversation_history or [],
                    trace_id=trace_id,
                    lang=lang,
                    state=state,
                )
            return self.composer.clarification(u, lang=lang, draft=draft), {
                "outcome": "NEEDS_CLARIFICATION",
                "understanding": u.to_dict(),
            }

        if not draft:
            return self.composer.clarification(u, lang=lang, draft=draft), {"outcome": "NEEDS_CLARIFICATION"}

        if draft.workflow_id == "leave":
            return self._handle_leave_modify(
                message,
                memory=memory,
                understanding=u,
                lang=lang,
                state=state,
                trace_id=trace_id,
                conversation_history=conversation_history,
            )

        if draft.workflow_id == "expense":
            defn = self.manager.ensure_definition("expense")
            return self._handle_expense_draft_collect(
                message,
                memory=memory,
                defn=defn,
                understanding=u,
                conversation_history=conversation_history or [],
                trace_id=trace_id,
                lang=lang,
                state=state,
            )

        items = draft.fields.get("items") or []
        updates = list(u.field_updates or [])
        if not updates and items:
            parsed = parse_modify_request(message, items)
            if parsed:
                updates = [
                    FieldUpdate(
                        field="items",
                        value={"amount": parsed["amount"]},
                        item_index=parsed["item_index"],
                        action="update",
                    )
                ]

        if updates:
            upd = updates[0]
            idx = upd.item_index if upd.item_index is not None else 0
            parsed = parse_modify_request(message, items)
            needs_confirm = parsed.get("needs_confirm") if parsed else len(items) > 1
            if 0 <= idx < len(items):
                old_amt = float(items[idx].get("amount") or 0)
                new_amt = float((upd.value or {}).get("amount") or 0)
                label = str(
                    u.entities.get("modify_label")
                    or (parsed.get("label") if parsed else None)
                    or f"item {idx + 1}"
                )
                if needs_confirm or (parsed and parsed.get("needs_confirm")):
                    state.push("set_pending_confirmation", value=f"modify:{idx}:{new_amt}")
                    return self.composer.modify_confirm(
                        label=label, old=old_amt, new=new_amt, draft=draft, lang=lang
                    ), {"outcome": "NEEDS_INPUT", "awaiting_confirmation": True}
                state.apply_field_updates(updates)
                prefix = self.composer.item_updated(label, new_amt, lang=lang)
                defn = self.manager.ensure_definition(draft.workflow_id)
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        return self.composer.clarification(u, lang=lang, draft=draft), {"outcome": "NEEDS_CLARIFICATION"}

    def _finish_leave_update_turn(
        self,
        memory: SessionMemory,
        defn,
        *,
        updates: list,
        message: str,
        lang: str,
        state: StatePatchBuffer,
        was_submit: bool = False,
        rules_applied: list[str] | None = None,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """Shared leave field-apply path for collect, modify, and review edits."""
        from chat.services.platform.field_engine import deserialize_field_updates

        applied_raw = state.apply_field_updates(
            updates,
            message=message,
            review_validated=was_submit,
        )
        applied = deserialize_field_updates(applied_raw)
        draft = memory.active_draft()
        prefix = self.composer.item_prefix_from_updates(applied, lang=lang) if applied else ""

        if was_submit and not applied:
            msg = self.composer.leave_review_natural_reply(
                message,
                memory,
                intent="unclear",
                lang=lang,
                trace_id=trace_id,
                conversation_history=conversation_history or [],
            )
            review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
            if review_text:
                msg = f"{msg}\n\n{review_text}".strip()
            return msg, {
                "outcome": "NEEDS_CLARIFICATION",
                "awaiting_confirmation": True,
                "rules_applied": rules_applied or ["LEAVE_REVIEW_NO_APPLY"],
            }

        errors = self.validator.validate(draft, defn, lang=lang)
        if errors:
            return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}

        if was_submit or self.fields.missing_fields(draft, defn) == []:
            blocked = self._leave_submitted_overlap_block(memory, lang, draft=draft)
            if blocked:
                return blocked
            state.push("clear_pending_question")
            state.push("set_pending_confirmation", value="submit")
            state.push("set_active_stage", value=WorkflowStage.CONFIRM_SUBMIT.value)
            review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
            msg = self.composer.review_ready_message(prefix, review_text or "", lang=lang, memory=memory)
            return msg, {
                "outcome": "NEEDS_INPUT",
                "awaiting_confirmation": True,
                "rules_applied": rules_applied or ["LEAVE_MODIFY_REVIEW"],
            }

        return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

    def _handle_leave_modify(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        lang: str,
        state: StatePatchBuffer,
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        from chat.services.platform.field_extractors.leave import (
            filter_leave_updates_for_review,
            interpret_leave_review_turn,
            review_field_updates_from_message,
        )

        draft = memory.active_draft()
        if not draft:
            return self.composer.clarification(understanding, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        was_submit = memory.pending_confirmation == "submit"

        updates = filter_leave_updates_for_review(
            list(understanding.field_updates or []),
            message,
            memory=memory,
            trace_id=trace_id,
        )
        if not updates:
            updates = review_field_updates_from_message(
                message, memory, trace_id=trace_id
            )
        if not updates:
            turn = interpret_leave_review_turn(message, memory, trace_id=trace_id)
            intent = str(turn.get("intent") or "unclear")
            if intent == "navigation":
                defn = self.manager.ensure_definition("leave")
                return self._show_review(memory, defn, lang=lang, state=state)
            msg = self.composer.leave_review_natural_reply(
                message,
                memory,
                intent=intent,
                lang=lang,
                trace_id=trace_id,
                conversation_history=conversation_history or [],
            )
            if was_submit:
                defn = self.manager.ensure_definition("leave")
                review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
                if review_text:
                    msg = f"{msg}\n\n{review_text}".strip()
            return msg, {
                "outcome": "NEEDS_CLARIFICATION",
                "awaiting_confirmation": was_submit,
                "reason": "Leave review — no field change applied.",
            }

        defn = self.manager.ensure_definition("leave")
        return self._finish_leave_update_turn(
            memory,
            defn,
            updates=updates,
            message=message,
            lang=lang,
            state=state,
            was_submit=was_submit,
            rules_applied=["LEAVE_MODIFY_REVIEW"],
            conversation_history=conversation_history,
            trace_id=trace_id,
        )

    @staticmethod
    def _expense_wizard_flags(
        memory: SessionMemory,
        understanding: UnderstandingResult | None = None,
    ) -> tuple[bool, bool]:
        u_ent = (understanding.entities or {}) if understanding else {}
        if understanding and understanding.source == "llm_expense":
            return (
                bool(u_ent.get("expense_wizard_fallback")),
                bool(u_ent.get("expense_llm_degraded")),
            )
        entities = dict(memory.last_entities or {})
        expense_turn = u_ent.get("expense_turn") or entities.get("expense_turn") or {}
        wizard = bool(
            u_ent.get("expense_wizard_fallback")
            or entities.get("expense_wizard_fallback")
            or expense_turn.get("wizard_fallback")
        )
        degraded = bool(
            u_ent.get("expense_llm_degraded") or entities.get("expense_llm_degraded")
        )
        return wizard, degraded

    def _finish_expense_update_turn(
        self,
        memory: SessionMemory,
        defn,
        *,
        updates: list,
        message: str,
        lang: str,
        state: StatePatchBuffer,
        was_submit: bool = False,
        rules_applied: list[str] | None = None,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
        understanding: UnderstandingResult | None = None,
    ) -> tuple[str, dict[str, Any]]:
        from chat.services.platform.field_engine import deserialize_field_updates
        from chat.services.platform.field_extractors.expense import (
            expense_pending_interrupted_by_updates,
            expense_turn_llm_blocked,
            clear_expense_blocked_add,
            record_expense_last_ops,
            expense_understanding_patch_from_applied_deletes,
            refresh_expense_pending_question_after_draft_change,
        )

        draft_before = memory.active_draft()
        item_count_before = len((draft_before.fields.get("items") or []) if draft_before else [])
        prev_ops = dict((memory.last_entities or {}).get("expense_last_ops") or {})
        expense_turn = dict((understanding.entities or {}).get("expense_turn") or {}) if understanding else {}

        if not updates and expense_turn.get("intent") == "repeat_ack":
            prefix_parts = [self.composer.expense_already_recorded(lang=lang)]
            if expense_turn.get("past_date_rejected"):
                prefix_parts.insert(0, self.composer.expense_past_date_policy(lang=lang))
            return self._continue_collection(
                memory,
                defn,
                lang=lang,
                prefix="\n\n".join(prefix_parts),
                state=state,
                extra_rules=rules_applied,
            )

        applied_raw = state.apply_field_updates(
            updates,
            message=message,
            review_validated=was_submit,
        )
        applied = deserialize_field_updates(applied_raw)
        draft = memory.active_draft()
        delete_applied = any(
            getattr(u, "field", None) == "items"
            and str(getattr(u, "action", "") or "").lower() == "delete"
            for u in applied
        )
        if defn.workflow_id == "expense" and delete_applied:
            patch = expense_understanding_patch_from_applied_deletes(memory, applied)
            if patch:
                state.push("merge_last_entities", value=patch)
            pq = refresh_expense_pending_question_after_draft_change(memory, lang=lang)
            state.push("set_pending_question", value=pq.to_dict() if pq else None)
        elif defn.workflow_id == "expense" and expense_pending_interrupted_by_updates(applied, memory):
            state.push("clear_pending_question")

        items_before = list((draft_before.fields.get("items") or []) if draft_before else [])
        prefix = self.composer.item_prefix_from_updates(applied, lang=lang) if applied else ""
        expense_intent = str(
            expense_turn.get("intent")
            or ((understanding.entities or {}).get("expense_intent") if understanding else None)
            or ""
        ).lower()
        if expense_intent in ("replay_blocked_add", "date_correction") and applied:
            clear_expense_blocked_add(memory)
            notice = self.composer.expense_replay_blocked_ack(lang=lang)
            prefix = "\n\n".join(p for p in [notice, prefix] if p).strip()
        if expense_turn.get("past_date_rejected") and not expense_turn.get("date_policy_rejected"):
            notice = self.composer.expense_past_date_policy(lang=lang)
            prefix = "\n\n".join(p for p in [notice, prefix] if p).strip()
        wizard_fallback, llm_degraded = self._expense_wizard_flags(memory, understanding)
        turn_wizard = bool(expense_turn.get("wizard_fallback"))
        items_after = list((draft.fields.get("items") or []) if draft else [])
        modify_no_op = (
            defn.workflow_id == "expense"
            and items_before == items_after
            and "EXPENSE_RULES_MODIFY" in (rules_applied or [])
            and expense_turn_llm_blocked(expense_turn, memory)
        )
        if modify_no_op:
            body = self.composer.expense_llm_unavailable(lang=lang, for_edit=True)
            if was_submit or memory.pending_confirmation == "submit":
                state.push("set_active_stage", value=WorkflowStage.CONFIRM_SUBMIT.value)
                review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
                if review_text:
                    body = f"{body}\n\n{review_text}".strip()
            return body, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["EXPENSE_LLM_UNAVAILABLE"],
                "awaiting_confirmation": bool(memory.pending_confirmation == "submit"),
            }
        if wizard_fallback and turn_wizard:
            notice = self.composer.expense_wizard_fallback_notice(
                lang=lang,
                llm_degraded=llm_degraded,
            )
            prefix = "\n\n".join(p for p in [notice, prefix] if p).strip()
            rules_applied = list(rules_applied or [])
            if "EXPENSE_WIZARD_FALLBACK" not in rules_applied:
                rules_applied.append("EXPENSE_WIZARD_FALLBACK")

        if defn.workflow_id == "expense":
            new_count = len((draft.fields.get("items") or []) if draft else [])
            notes: list[str] = []
            if new_count > item_count_before:
                notes.append(f"added item {new_count}")
            if new_count < item_count_before:
                notes.append("deleted item")
            delete_requested = bool(
                rules_applied
                and "EXPENSE_RULES_DELETE" in (rules_applied or [])
            )
            if delete_requested and new_count >= item_count_before and item_count_before > 0:
                defn = self.manager.ensure_definition("expense")
                clarify_turn = {
                    "intent": "clarify_delete",
                    "item_patches": [],
                    "clarify": {
                        "kind": "which_delete",
                        "candidate_indices": list(range(item_count_before)),
                    },
                }
                from chat.services.platform.field_extractors.expense import (
                    build_expense_pending_edit_from_turn,
                )

                pending_edit = build_expense_pending_edit_from_turn(clarify_turn, message=message, memory=memory)
                if pending_edit:
                    state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
                body = self.composer.expense_edit_clarify(memory, clarify_turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "rules_applied": ["EXPENSE_DELETE_CLARIFY"],
                }
            intent_hint = str((memory.last_entities or {}).get("expense_pending_intent") or "")
            record_expense_last_ops(
                memory,
                item_count_before=item_count_before,
                turn={"intent": intent_hint},
                applied_notes=notes,
            )
            state.push(
                "merge_last_entities",
                value={
                    "expense_last_ops": dict(memory.last_entities.get("expense_last_ops") or {}),
                    "expense_pending_intent": None,
                    "expense_pending_edit": None,
                },
            )
            if new_count < item_count_before and prev_ops.get("appended"):
                ack = self.composer.expense_repair_ack(
                    memory,
                    message=message,
                    lang=lang,
                    repaired=True,
                ).split("\n\n")[0]
                prefix = "\n\n".join(p for p in [ack, prefix] if p).strip()

        expense_collect = defn.workflow_id == "expense" and not was_submit
        errors = self.validator.validate(
            draft,
            defn,
            lang=lang,
            collect_mode=expense_collect,
        )
        missing = self.fields.missing_fields(draft, defn)
        if errors:
            if defn.workflow_id == "expense" and missing:
                if memory.pending_confirmation == "submit":
                    state.push("clear_pending_confirmation")
                return self._continue_collection(
                    memory,
                    defn,
                    lang=lang,
                    prefix=prefix,
                    state=state,
                    extra_rules=rules_applied,
                )
            if not (expense_collect and missing):
                return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}

        if missing:
            if memory.pending_confirmation == "submit":
                state.push("clear_pending_confirmation")
            return self._continue_collection(
                memory,
                defn,
                lang=lang,
                prefix=prefix,
                state=state,
                extra_rules=rules_applied,
            )

        if was_submit or not missing:
            state.push("clear_pending_question")
            state.push("set_pending_confirmation", value="submit")
            state.push("set_active_stage", value=WorkflowStage.CONFIRM_SUBMIT.value)
            review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
            msg = self.composer.review_ready_message(prefix, review_text or "", lang=lang, memory=memory)
            return msg, {
                "outcome": "NEEDS_INPUT",
                "awaiting_confirmation": True,
                "rules_applied": rules_applied or ["EXPENSE_MODIFY_REVIEW"],
            }

        return self._continue_collection(
            memory,
            defn,
            lang=lang,
            prefix=prefix,
            state=state,
            extra_rules=rules_applied,
        )

    def _handle_delete(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        understanding: UnderstandingResult | None = None,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        u = understanding
        draft = memory.active_draft()
        if u is None:
            return self.composer.no_draft_to_delete(lang=lang), {"outcome": "NEEDS_CLARIFICATION"}
        if not draft:
            return self.composer.no_draft_to_delete(lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        if draft.workflow_id == "expense":
            from chat.services.platform.field_extractors.modify import parse_delete_request
            from chat.services.platform.field_extractors.expense import (
                expense_delete_field_updates,
                normalize_expense_delete_turn,
                sync_expense_draft,
            )

            sync_expense_draft(draft)
            items = list(draft.fields.get("items") or [])
            was_submit = memory.pending_confirmation == "submit"
            from chat.services.platform.field_extractors.expense import (
                is_expense_draft_mutation_message,
                sanitize_expense_turn_for_action,
            )

            expense_turn = dict((u.entities or {}).get("expense_turn") or {}) if u else {}
            if is_expense_draft_mutation_message(message, memory) and str(
                expense_turn.get("intent") or ""
            ).lower() in ("add", "answer_pending", "conversation", ""):
                expense_turn = {}
            expense_turn = sanitize_expense_turn_for_action(
                expense_turn,
                action=str(u.action or "") if u else "",
                expense_intent=str((u.entities or {}).get("expense_intent") or "") if u else "",
            )
            expense_intent = str(
                (u.entities or {}).get("expense_intent") or expense_turn.get("intent") or ""
            ).lower()

            if u and u.action == UnderstandingAction.MODIFY.value and (
                "delete" in (u.goal or "").lower() or "delete" in (u.reasoning or "").lower()
            ):
                u = UnderstandingResult(
                    goal=u.goal,
                    workflow=u.workflow,
                    action=UnderstandingAction.DELETE.value,
                    confidence=u.confidence,
                    field_updates=u.field_updates,
                    targets=u.targets,
                    entities=u.entities,
                    reasoning=u.reasoning,
                    source=u.source,
                    answers_pending_field=u.answers_pending_field,
                )

            if expense_intent == "clarify_delete" or is_vague_delete(message):
                clarify_turn = expense_turn if expense_intent == "clarify_delete" else {
                    "intent": "clarify_delete",
                    "item_patches": [],
                    "clarify": {
                        "kind": "which_delete",
                        "candidate_indices": list(range(len(items))),
                    },
                }
                from chat.services.platform.field_extractors.expense import (
                    build_expense_pending_edit_from_turn,
                )

                pending_edit = build_expense_pending_edit_from_turn(clarify_turn, message=message, memory=memory)
                if pending_edit:
                    state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
                defn = self.manager.ensure_definition("expense")
                body = self.composer.expense_edit_clarify(memory, clarify_turn, lang=lang)
                return body, {
                    "outcome": "NEEDS_CLARIFICATION",
                    "rules_applied": ["EXPENSE_DELETE_CLARIFY"],
                }

            expense_turn = normalize_expense_delete_turn(expense_turn, message, memory)
            if str(expense_turn.get("intent") or "").lower() == "delete" and (
                expense_turn.get("item_patches") or expense_turn.get("delete_indices")
            ):
                fields = dict(draft.fields or {})
                retry_updates = expense_delete_field_updates(fields, expense_turn)
                if retry_updates:
                    defn = self.manager.ensure_definition("expense")
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=retry_updates,
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=["EXPENSE_RULES_DELETE"],
                        trace_id=trace_id,
                        understanding=understanding,
                    )

            if u and u.action == UnderstandingAction.DELETE.value and u.targets:
                delete_updates = [
                    FieldUpdate(
                        field="items",
                        value={},
                        item_index=int(t.item_index),
                        action="delete",
                    )
                    for t in u.targets
                    if t.item_index is not None
                    and 0 <= int(t.item_index) < len(items)
                ]
                if delete_updates:
                    from chat.services.platform.field_extractors.expense import clear_expense_blocked_add

                    clear_expense_blocked_add(memory)
                    defn = self.manager.ensure_definition("expense")
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=delete_updates,
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=["EXPENSE_RULES_DELETE"],
                        trace_id=trace_id,
                        understanding=understanding,
                    )

            if not is_vague_delete(message):
                parsed = parse_delete_request(message, items)
                if parsed and parsed.get("item_index") is not None:
                    defn = self.manager.ensure_definition("expense")
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=[
                            FieldUpdate(
                                field="items",
                                value={},
                                item_index=int(parsed["item_index"]),
                                action="delete",
                            )
                        ],
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=["EXPENSE_RULES_DELETE"],
                        trace_id=trace_id,
                        understanding=understanding,
                    )
                if parsed and parsed.get("needs_clarify"):
                    turn = {
                        "intent": "clarify_delete",
                        "item_patches": [],
                        "clarify": {
                            "kind": "which_delete",
                            "candidate_indices": list(parsed.get("candidate_indices") or []),
                            "category": parsed.get("category") or parsed.get("label"),
                        },
                    }
                    from chat.services.platform.field_extractors.expense import (
                        build_expense_pending_edit_from_turn,
                    )

                    pending_edit = build_expense_pending_edit_from_turn(turn, message=message, memory=memory)
                    if pending_edit:
                        state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
                    body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
                    return body, {
                        "outcome": "NEEDS_CLARIFICATION",
                        "rules_applied": ["EXPENSE_DELETE_CLARIFY"],
                    }
            turn = normalize_expense_delete_turn(
                dict((u.entities or {}).get("expense_turn") or {}) if u else {},
                message,
                memory,
            )
            if str(turn.get("intent") or "").lower() == "delete" and (
                turn.get("item_patches") or turn.get("delete_indices")
            ):
                from chat.services.platform.field_extractors.expense import (
                    expense_delete_field_updates,
                )

                fields = dict(draft.fields or {})
                retry_updates = expense_delete_field_updates(fields, turn)
                if retry_updates:
                    defn = self.manager.ensure_definition("expense")
                    return self._finish_expense_update_turn(
                        memory,
                        defn,
                        updates=retry_updates,
                        message=message,
                        lang=lang,
                        state=state,
                        was_submit=was_submit,
                        rules_applied=["EXPENSE_RULES_DELETE"],
                        trace_id=trace_id,
                        understanding=understanding,
                    )
            defn = self.manager.ensure_definition("expense")
            turn = {"intent": "clarify_delete", "item_patches": []}
            from chat.services.platform.field_extractors.expense import (
                build_expense_pending_edit_from_turn,
            )

            pending_edit = build_expense_pending_edit_from_turn(turn, message=message, memory=memory)
            if pending_edit:
                state.push("merge_last_entities", value={"expense_pending_edit": pending_edit})
            body = self.composer.expense_edit_clarify(memory, turn, lang=lang)
            return body, {
                "outcome": "NEEDS_CLARIFICATION",
                "rules_applied": ["EXPENSE_DELETE_CLARIFY"],
            }

        if u.action == UnderstandingAction.CLARIFICATION_NEEDED.value or is_vague_delete(message):
            pq = PendingQuestion(
                field="delete_which_item",
                prompt="Which entry number to delete?",
                workflow_id=draft.workflow_id,
                asked_at_turn=memory.turn_count,
            )
            state.push("set_pending_question", value=pq.to_dict())
            return self.composer.delete_pick(draft, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        applied = self.modifier.apply_understanding(memory, u, state=state)
        defn = self.manager.ensure_definition(draft.workflow_id)
        if applied:
            return self._continue_collection(
                memory, defn, lang=lang,
                prefix=self.composer.item_deleted(str(applied[0]), lang=lang),
                state=state,
            )
        return self.composer.delete_pick(draft, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

    def _apply_expense_message_to_state(
        self,
        message: str,
        *,
        memory: SessionMemory,
        state: StatePatchBuffer,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
    ) -> bool:
        """Re-parse expense narrative and apply patches (Phase A — switch / replay)."""
        from chat.services.platform.field_extractors.expense import expense_turn_to_field_updates

        if not (message or "").strip():
            return False
        fresh = self._is_new_expense_claim(
            message,
            active_workflow_id=(memory.active_workflow.id if memory.active_workflow else ""),
            memory=memory,
            trace_id=trace_id,
        )
        if fresh:
            self._prepare_fresh_expense_draft(state, memory)
        _, updates = expense_turn_to_field_updates(
            message,
            memory,
            trace_id=trace_id,
            conversation_history=conversation_history or [],
            fresh_claim=fresh,
        )
        if not updates:
            return False
        state.apply_field_updates(updates)
        self._push_default_expense_date(state, memory)
        return True

    @staticmethod
    def _synthesize_switch_pq_decision(
        understanding: UnderstandingResult | None,
    ) -> PendingQuestionDecision | None:
        """Active-route workflow switch — pending PQ is dropped but understanding has the target."""
        if understanding is None:
            return None
        target = (
            (understanding.interrupt_workflow or understanding.workflow or "")
            .strip()
            .lower()
        )
        if target not in ("leave", "expense"):
            return None
        return PendingQuestionDecision(
            kind=MessageIntentKind.NEW_WORKFLOW,
            confidence=float(understanding.confidence or 0.9),
            reasoning=understanding.reasoning or f"Switch to {target}",
            source=understanding.source or "active",
            blocks_new_workflow=False,
            target_workflow=target,
        )

    def _handle_switch(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision,
        understanding: UnderstandingResult,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]] | None:
        u = understanding
        from chat.services.platform.intent_rules import should_resume_suspended_expense

        target = (pq_decision.target_workflow or u.interrupt_workflow or u.workflow or "").strip().lower()
        if should_resume_suspended_expense(
            message=message,
            active_workflow_id=memory.active_workflow.id if memory.active_workflow else None,
            suspended_workflows=memory.suspended_workflows,
        ):
            target = "expense"
        if not target or not get_workflow_definition(target):
            return self.composer.which_workflow(lang=lang), {"outcome": "NEEDS_CLARIFICATION"}
        suspended_before = {sw.workflow_id for sw in memory.suspended_workflows}
        prev_active = memory.active_workflow.id if memory.active_workflow else None
        new_claim = target == "expense" and self._is_new_expense_claim(
            message,
            u,
            active_workflow_id=(memory.active_workflow.id if memory.active_workflow else ""),
            memory=memory,
            trace_id=trace_id,
        )
        if new_claim:
            self._prepare_fresh_expense_draft(state, memory)
        is_resume = target in suspended_before and not new_claim
        state.push("switch_to_workflow", value=target)
        state.flush()
        if new_claim:
            state.push("set_active_stage", value=WorkflowStage.COLLECTING.value)
            state.push("clear_pending_confirmation")
            state.flush()
        defn = self.manager.ensure_definition(target)
        draft = state.ensure_active_draft(target)
        if not draft:
            return self.composer.which_workflow(lang=lang), {"outcome": "NEEDS_CLARIFICATION"}
        applied = False
        if u.field_updates:
            state.apply_field_updates(u.field_updates)
            applied = True
        elif target == "expense":
            applied = self._apply_expense_message_to_state(
                message,
                memory=memory,
                state=state,
                conversation_history=conversation_history,
                trace_id=trace_id,
            )
        if is_resume and prev_active:
            prefix = self.composer.workflow_switch_resumed(
                memory,
                paused_workflow=prev_active,
                resumed_workflow=target,
                lang=lang,
            )
        else:
            prefix = self.composer.workflow_switched(target, lang=lang, memory=memory)
        entities = dict(u.entities or {})
        if pq_decision and pq_decision.extracted_entities:
            entities.update(pq_decision.extracted_entities)
        nav = str(entities.get("expense_navigation") or "").strip().lower()
        expense_intent = str(entities.get("expense_intent") or "").strip().lower()
        wants_summary_nav = (
            u.action == UnderstandingAction.REVIEW.value
            or nav == "summary"
            or expense_intent in ("show_summary", "show_list", "show_total")
        )
        has_expense_data = (
            applied
            or is_workflow_interrupt_expense(message, active_workflow=prev_active or "leave")
            or is_compound_expense_message(message)
        )
        if target == "expense" and wants_summary_nav:
            if has_expense_data or not is_pure_expense_navigation(message):
                return self._continue_collection(
                    memory, defn, lang=lang, prefix=prefix, state=state
                )
            review_msg, meta = self._show_review(
                memory,
                defn,
                lang=lang,
                state=state,
                message=message,
                trace_id=trace_id,
            )
            if prefix:
                review_msg = f"{prefix}\n\n{review_msg}".strip()
            return review_msg, meta
        if (
            target == "expense"
            and not applied
            and (is_expense_message(message) or is_compound_expense_message(message))
        ):
            from chat.services.llm_client import llm_rate_limit_active

            if (
                ((u.entities or {}).get("expense_llm_degraded") and not u.field_updates)
                or (
                    llm_rate_limit_active(trace_id)
                    and not (u.entities or {}).get("expense_wizard_fallback")
                )
            ):
                body = self.composer.expense_llm_unavailable(lang=lang, for_edit=True)
                if prefix:
                    body = f"{prefix}\n\n{body}".strip()
                return body, {
                    "outcome": "NEEDS_INPUT",
                    "rules_applied": ["EXPENSE_LLM_DEGRADED"],
                }
        if u.action == UnderstandingAction.REVIEW.value:
            review_msg, meta = self._show_review(
                memory,
                defn,
                lang=lang,
                state=state,
                message=message,
                trace_id=trace_id,
            )
            if prefix:
                review_msg = f"{prefix}\n\n{review_msg}".strip()
            return review_msg, meta
        return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

    def _continue_collection(
        self,
        memory: SessionMemory,
        defn,
        *,
        lang: str,
        prefix: str = "",
        state: StatePatchBuffer,
        extra_rules: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft:
            return prefix, {"outcome": "NEEDS_INPUT"}

        if defn.workflow_id == "leave" and not memory.active_workflow:
            state.push("start_workflow", value="leave")
            state.flush()
            draft = memory.active_draft()
            if not draft:
                return prefix, {"outcome": "NEEDS_INPUT"}

        if defn.workflow_id == "leave":
            from chat.services.platform.field_extractors.leave import apply_leave_derived_fields

            apply_leave_derived_fields(draft, message="")
            state.push("merge_last_entities", value={"leave_start_clarify": False})

        requested = (memory.last_entities or {}).get("requested_leave_type")
        if requested and defn.workflow_id == "leave" and not draft.fields.get("leave_type"):
            from chat.services.platform.field_extractors.leave import normalize_leave_type_value
            from chat.services.platform.schemas import FieldUpdate

            canonical = normalize_leave_type_value(requested)
            if canonical:
                state.apply_field_updates(
                    [FieldUpdate(field="leave_type", value=canonical)],
                    message="",
                )
                state.push(
                    "merge_last_entities",
                    value={"requested_leave_type": None},
                )
                draft = memory.active_draft() or draft
            else:
                type_note = self.composer.unrecognized_leave_type_prompt(
                    str(requested), lang=lang
                )
                if type_note and type_note not in prefix:
                    prefix = f"{prefix}\n\n{type_note}".strip() if prefix else type_note

        self._push_default_expense_date(state, memory)
        missing = self.fields.missing_fields(draft, defn)
        expense_collect = defn.workflow_id == "expense" and not memory.pending_confirmation
        errors = self.validator.validate(
            draft,
            defn,
            lang=lang,
            collect_mode=expense_collect,
        )

        if defn.workflow_id == "expense":
            from chat.services.platform.field_extractors.expense import (
                build_pending_queue,
                expense_focus_prompt,
                sync_expense_draft,
            )
            from chat.services.platform.summary import format_expense_collect_recap

            sync_expense_draft(draft)
            focus_q = None
            pq = self.fields.next_question(memory, draft, defn, lang=lang)
            if pq:
                state.push("set_pending_question", value=pq.to_dict())
                queue = build_pending_queue(list(draft.fields.get("items") or []))
                if queue:
                    focus_q = expense_focus_prompt(queue[0], lang=lang)
            else:
                state.push("set_pending_question", value=None)
            recap = format_expense_collect_recap(
                draft,
                lang=lang,
                include_focus_question=focus_q,
            )
            if recap:
                prefix = "\n\n".join(p for p in [prefix, recap] if p).strip()

        collect_rules = list(extra_rules or [])
        if "FIELD_COLLECTION" not in collect_rules:
            collect_rules.append("FIELD_COLLECTION")

        if not missing and not errors:
            if defn.workflow_id == "leave":
                blocked = self._leave_submitted_overlap_block(memory, lang, draft=draft)
                if blocked:
                    return blocked
            review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
            state.push("clear_pending_question")
            state.push("set_pending_confirmation", value="submit")
            state.push("set_active_stage", value=WorkflowStage.CONFIRM_SUBMIT.value)
            msg = self.composer.review_ready_message(prefix, review_text or "", lang=lang, memory=memory)
            review_rules = list(extra_rules or [])
            if "REVIEW_READY" not in review_rules:
                review_rules.append("REVIEW_READY")
            return msg, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": review_rules,
                "stage": WorkflowStage.REVIEW.value,
                "awaiting_confirmation": True,
            }

        pq = self.fields.next_question(memory, draft, defn, lang=lang)
        if defn.workflow_id != "expense":
            state.push("set_pending_question", value=pq.to_dict() if pq else None)
            parts = [p for p in [prefix, errors[0] if errors else ""] if p]
            if pq:
                parts.append(pq.prompt)
            msg = self.composer.collection_message(parts)
            return msg, {
                "outcome": "NEEDS_INPUT",
                "missing_fields": missing,
                "rules_applied": collect_rules,
            }

        msg = self.composer.collection_message([prefix] if prefix else [])
        return msg, {
            "outcome": "NEEDS_INPUT",
            "missing_fields": missing,
            "rules_applied": collect_rules,
        }

    def _request_submit(
        self, memory: SessionMemory, defn, *, lang: str, state: StatePatchBuffer
    ) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        if not draft:
            wf = defn.workflow_id
            return self.composer.no_open_draft(wf, lang=lang), {"outcome": "NEEDS_CLARIFICATION"}

        missing = self.fields.missing_fields(draft, defn)
        if missing:
            pq = self.fields.next_question(memory, draft, defn, lang=lang)
            state.push("set_pending_question", value=pq.to_dict() if pq else None)
            msg = self.composer.missing_for_submit(missing, lang=lang)
            if pq:
                msg += f"\n\n{pq.prompt}"
            return msg, {"outcome": "NEEDS_INPUT", "missing_fields": missing}

        errors = self.validator.validate(draft, defn, lang=lang)
        if errors:
            return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}

        if defn.workflow_id == "leave":
            blocked = self._leave_submitted_overlap_block(memory, lang, draft=draft)
            if blocked:
                return blocked

        review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
        state.push("set_pending_confirmation", value="submit")
        return self.composer.submit_confirm(review_text or "", lang=lang), {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
        }

    def _confirm_submit(
        self,
        memory: SessionMemory,
        defn,
        *,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        missing = self.fields.missing_fields(memory.active_draft(), defn) if memory.active_draft() else ["draft"]
        if missing:
            state.push("clear_pending_confirmation")
            return self.composer.missing_for_submit(missing, lang=lang), {"outcome": "NEEDS_INPUT"}

        if defn.workflow_id == "leave":
            blocked = self._leave_submitted_overlap_block(memory, lang)
            if blocked:
                state.push("clear_pending_confirmation")
                return blocked

        msg, meta = self.submission.confirm_and_submit(
            memory,
            defn,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            state=state,
        )
        outcome = "SUBMITTED" if meta.get("submitted") else "NEEDS_INPUT"
        return msg, {"outcome": outcome, **meta}

    def _resolve_submit_confirmation(
        self,
        message: str,
        memory: SessionMemory,
        u: UnderstandingResult,
        *,
        lang: str,
        company_id: str,
        employee_id: str,
        session_id: str,
        idempotency_key: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]] | None:
        if memory.pending_confirmation != "submit":
            return None
        defn = self._active_defn(memory)
        if not defn:
            return None
        if is_bare_rejection(message) or (
            u.action == UnderstandingAction.REVIEW.value
            and "declin" in (u.reasoning or "").lower()
        ):
            state.push("clear_pending_confirmation")
            state.push("set_active_stage", value=WorkflowStage.REVIEW.value)
            review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
            msg = self.composer.review_after_decline(review_text or "", lang=lang)
            return msg, {
                "outcome": "NEEDS_INPUT",
                "rules_applied": ["SUBMIT_DECLINED"],
            }
        if is_bare_confirmation(message) or parse_submit_workflow(
            message, active_workflow_id=defn.workflow_id
        ) or u.action in (
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SUBMIT.value,
        ):
            return self._confirm_submit(
                memory, defn, lang=lang,
                company_id=company_id, employee_id=employee_id,
                session_id=session_id, idempotency_key=idempotency_key,
                state=state,
            )
        if u.action in (
            UnderstandingAction.MODIFY.value,
            UnderstandingAction.DELETE.value,
        ) or (u.field_updates and u.action == UnderstandingAction.COLLECT.value):
            state.push("clear_pending_confirmation")
            state.push("set_active_stage", value=WorkflowStage.COLLECTING.value)
            return None
        from chat.services.platform.workflow_show import resolve_workflow_show_target

        show_target = str((u.entities or {}).get("show_workflow_target") or "").lower()
        if not show_target:
            show_target = resolve_workflow_show_target(
                message,
                memory,
                active_workflow_id=defn.workflow_id,
            ) or ""
        if show_target == "leave":
            return self.composer.leave_status_report(memory, lang=lang), {
                "outcome": "INFORMATIONAL",
                "rules_applied": ["LEAVE_SESSION_SUMMARY"],
            }
        review_text, _ = self.review.prepare_review(memory, defn, lang=lang)
        return self.composer.submit_confirm(review_text or "", lang=lang), {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
            "rules_applied": ["SUBMIT_CONFIRM_RETRY"],
        }

    def _block_submitted_leave_overlap(
        self,
        memory: SessionMemory,
        u: UnderstandingResult,
        lang: str,
        *,
        message: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        from chat.services.platform.field_extractors.date import parse_leave_dates
        from chat.services.platform.field_extractors.leave import find_submitted_leave_overlap

        wf = (u.workflow or "").strip().lower()
        if wf not in ("leave", "") and not (
            memory.active_workflow and memory.active_workflow.id == "leave"
        ):
            return None
        fields = {upd.field: upd.value for upd in (u.field_updates or [])}
        draft = memory.active_draft()
        start = fields.get("start_date") or (draft.fields.get("start_date") if draft else None)
        end = fields.get("end_date") or (draft.fields.get("end_date") if draft else None) or start
        if not start and message:
            parsed = parse_leave_dates(message)
            start = parsed.get("start_date")
            end = parsed.get("end_date") or start
        if not start:
            return None
        hit = find_submitted_leave_overlap(memory, str(start), str(end) if end else None)
        if not hit:
            return None
        msg = self.composer.submitted_leave_overlap(hit, lang=lang)
        return msg, {
            "outcome": "NEEDS_CLARIFICATION",
            "rules_applied": ["SUBMITTED_LEAVE_DATE_OVERLAP"],
        }

    def _leave_submitted_overlap_block(
        self,
        memory: SessionMemory,
        lang: str,
        *,
        draft=None,
    ) -> tuple[str, dict[str, Any]] | None:
        from chat.services.platform.field_extractors.leave import draft_overlaps_submitted_leave

        hit = draft_overlaps_submitted_leave(memory, draft)
        if not hit:
            return None
        msg = self.composer.submitted_leave_overlap(hit, lang=lang)
        return msg, {
            "outcome": "NEEDS_CLARIFICATION",
            "rules_applied": ["SUBMITTED_LEAVE_DATE_OVERLAP"],
        }

    def _handle_medical_document_skip(
        self, memory: SessionMemory, defn, *, lang: str, state: StatePatchBuffer
    ) -> tuple[str, dict[str, Any]]:
        state.apply_field_updates(
            [FieldUpdate(field="medical_document_skipped", value=True, action="set")],
            message="",
        )
        state.push("clear_pending_question")
        prefix = self.composer.medical_document_skipped(lang=lang)
        return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

    def _apply_pending_modify(
        self, memory: SessionMemory, lang: str, state: StatePatchBuffer
    ) -> tuple[str, dict[str, Any]]:
        raw = memory.pending_confirmation or ""
        parts = raw.split(":")
        state.push("clear_pending_confirmation")
        if len(parts) != 3:
            return self.composer.session_context(memory, lang=lang), {"outcome": "INFORMATIONAL"}
        idx = int(parts[1])
        new_amt = float(parts[2])
        draft = memory.active_draft()
        if not draft:
            return self.composer.no_draft(lang=lang), {"outcome": "NEEDS_CLARIFICATION"}
        items = list(draft.fields.get("items") or [])
        if 0 <= idx < len(items):
            state.apply_field_updates(
                [
                    FieldUpdate(
                        field="items",
                        value={"amount": new_amt},
                        item_index=idx,
                        action="update",
                    )
                ]
            )
        defn = self.manager.ensure_definition(draft.workflow_id)
        prefix = self.composer.item_updated_by_index(idx, new_amt, lang=lang)
        return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

    def _maybe_submit(self, memory, defn, u, *, lang, company_id, employee_id, session_id, idempotency_key, state):
        if u.action == UnderstandingAction.CONFIRM.value:
            return self._confirm_submit(
                memory, defn, lang=lang,
                company_id=company_id, employee_id=employee_id,
                session_id=session_id, idempotency_key=idempotency_key,
                state=state,
            )
        return self._request_submit(memory, defn, lang=lang, state=state)

    def _show_review(
        self,
        memory: SessionMemory,
        defn,
        *,
        lang: str,
        state: StatePatchBuffer | None = None,
        message: str = "",
        trace_id: str = "",
    ) -> tuple[str, dict[str, Any]]:
        from chat.services.platform.field_extractors.expense import memory_has_expense_draft
        from chat.services.platform.summary import resolve_expense_summary_scope
        from chat.services.platform.workflow_show import session_expense_draft

        draft = memory.active_draft()
        if defn.workflow_id == "expense":
            submitted = list((memory.conversation_facts or {}).get("submitted_expenses") or [])
            has_pending = memory_has_expense_draft(memory)
            if submitted or has_pending:
                draft = session_expense_draft(memory) or draft
                scope = resolve_expense_summary_scope(message, memory, trace_id=trace_id)
                return self.composer.workflow_summary(
                    defn,
                    draft,
                    lang=lang,
                    memory=memory,
                    expense_scope=scope,
                ), {"outcome": "INFORMATIONAL"}
        if defn.workflow_id == "leave":
            draft = memory.active_draft()
            if draft and draft.locked:
                return self.composer.leave_status_report(memory, lang=lang), {"outcome": "INFORMATIONAL"}
            if not draft and not memory.active_workflow:
                return self.composer.leave_status_report(memory, lang=lang), {"outcome": "INFORMATIONAL"}
            review_text, errors = self.review.prepare_review(memory, defn, lang=lang)
            if errors:
                return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}
            meta: dict[str, Any] = {"outcome": "NEEDS_INPUT", "rules_applied": ["LEAVE_REVIEW"]}
            if state is not None and draft and not draft.locked:
                missing = self.fields.missing_fields(draft, defn)
                val_errors = self.validator.validate(draft, defn, lang=lang)
                if not missing and not val_errors:
                    state.push("clear_pending_question")
                    state.push("set_pending_confirmation", value="submit")
                    state.push("set_active_stage", value=WorkflowStage.CONFIRM_SUBMIT.value)
                    state.push("merge_last_entities", value={"leave_start_clarify": False})
                    meta["awaiting_confirmation"] = True
                    meta["rules_applied"] = ["LEAVE_REVIEW", "SUBMIT_ARMED"]
            return review_text or "", meta
        review_text, errors = self.review.prepare_review(memory, defn, lang=lang)
        if errors:
            return errors[0], {"outcome": "NEEDS_INPUT", "errors": errors}
        return review_text or "", {"outcome": "NEEDS_INPUT"}

    def _locked_response(self, memory: SessionMemory, lang: str) -> tuple[str, dict[str, Any]]:
        draft = memory.active_draft()
        rid = draft.submitted_request_id if draft else ""
        if draft and draft.workflow_id == "expense" and (draft.locked or draft.status == "submitted"):
            msg = self.composer.expense_submitted_locked(rid, lang=lang)
        else:
            msg = self.composer.locked_with_reference(rid, lang=lang)
        return msg, {"outcome": "INFORMATIONAL", "rules_applied": ["POST_SUBMIT_LOCK"]}

    def _submitted_leave_overlap_response(
        self,
        message: str,
        memory: SessionMemory,
        *,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        submitted = list((memory.conversation_facts or {}).get("submitted_leave_ranges") or [])
        hit = find_submitted_leave_overlap_from_message(message, submitted)
        if not hit:
            return self.composer.clarification(
                UnderstandingResult(reasoning="Overlapping leave dates."),
                lang=lang,
            ), {"outcome": "NEEDS_CLARIFICATION"}
        msg = self.composer.submitted_leave_overlap(hit, lang=lang)
        return msg, {
            "outcome": "NEEDS_CLARIFICATION",
            "rules_applied": ["SUBMITTED_LEAVE_DATE_OVERLAP"],
        }

    def _reply_today_date(
        self,
        *,
        memory: SessionMemory,
        ctx: TurnContext,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        today_iso = ctx.today_iso or ""
        if not today_iso:
            from datetime import date

            today_iso = date.today().isoformat()
        msg = self.composer.today_date(today_iso=today_iso, lang=lang)
        msg = self.composer.with_continuation_hint(msg, memory)
        return msg, {
            "outcome": "INFORMATIONAL",
            "reason": "Today's calendar date.",
            "rules_applied": ["HR_TODAY_DATE_QUERY", "PLAN_REPLY_TODAY_DATE"],
        }

    def _reply_translation(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
    ) -> tuple[str, dict[str, Any]] | None:
        translate_to = (pq_decision.field_value if pq_decision else None) or is_translation_request(message)
        if not translate_to:
            return None
        prev = _assistant_text_for_translation(
            conversation_history,
            target_lang=translate_to,
        )
        if not prev:
            return None
        source = strip_policy_footer(prev)
        translated, ok = translate_text(
            source,
            target_lang=translate_to,
            trace_id=trace_id,
        )
        if ok:
            msg = translated.rstrip() + "\n\n" + self.composer.rules_footer(lang=translate_to)
            status = "success"
            rules = ["TRANSLATION_FOLLOWUP", "PLAN_REPLY_TRANSLATION"]
        else:
            msg = self.composer.translation_unavailable(prev, lang=translate_to)
            status = "degraded"
            rules = ["TRANSLATION_UNAVAILABLE", "PLAN_REPLY_TRANSLATION"]
        msg = self.composer.with_continuation_hint(msg, memory)
        return msg, {
            "outcome": "INFORMATIONAL",
            "reason": "Translated the previous assistant turn.",
            "rules_applied": rules,
            "response_status": status,
        }

    def _reply_policy(
        self,
        message: str,
        *,
        memory: SessionMemory,
        ctx: TurnContext,
        pq_decision: PendingQuestionDecision | None,
        conversation_history: list[str],
        company_id: str,
        trace_id: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        _ = state
        pq_log = pq_decision.to_log_dict() if pq_decision else None
        msg, resp_status, decision = self.composer.policy_turn(
            message,
            document_text=ctx.document_text,
            pq_decision_log=pq_log,
            conversation_history=conversation_history,
            company_id=company_id,
            trace_id=trace_id,
            pq_reasoning=pq_decision.reasoning if pq_decision else "",
        )
        msg = self.composer.with_continuation_hint(msg, memory)
        return msg, {
            **decision,
            "response_status": resp_status,
            "request_id": "",
        }

    def _reply_status(
        self,
        message: str,
        *,
        memory: SessionMemory,
        pq_decision: PendingQuestionDecision | None,
        company_id: str,
        employee_id: str,
        session_id: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        _ = state
        pq_log = pq_decision.to_log_dict() if pq_decision else None
        msg, resp_status, decision, request_id = self.composer.status_turn(
            message,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            pq_decision_log=pq_log,
        )
        msg = self.composer.with_continuation_hint(msg, memory)
        return msg, {
            **decision,
            "response_status": resp_status,
            "request_id": request_id,
        }

    def _reply_oos(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        lang: str,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        self._pause_for_oos(state)
        msg = self.composer.out_of_scope(
            message,
            lang=lang,
            context_lines=conversation_history,
            trace_id=trace_id,
        )
        msg = self.composer.with_continuation_hint(msg, memory)
        return msg, {
            "outcome": "INFORMATIONAL",
            "reason": "Out of scope.",
            "rules_applied": ["OUT_OF_SCOPE", "PLAN_REPLY_OOS"],
        }

    def _reply_greeting(
        self,
        message: str,
        *,
        conversation_history: list[str],
        trace_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        reply = self.composer.greeting(
            message,
            context_lines=conversation_history,
            trace_id=trace_id,
        )
        if not reply:
            return None
        return reply, {
            "outcome": "INFORMATIONAL",
            "reason": "Greeting/chitchat reply.",
            "rules_applied": ["CONVERSATIONAL_GREETING", "PLAN_REPLY_GREETING"],
        }

    def _reply_conversational(
        self,
        message: str,
        *,
        conversation_history: list[str],
        trace_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        reply = self.composer.conversational(
            message,
            context_lines=conversation_history,
            trace_id=trace_id,
        )
        if not reply:
            return None
        return reply, {
            "outcome": "INFORMATIONAL",
            "reason": "Conversational fallback.",
            "rules_applied": ["CONVERSATIONAL_FALLBACK", "PLAN_REPLY_CONVERSATIONAL"],
        }

    def _reply_platform_clarify(
        self,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        _ = memory
        wf_label = understanding.workflow or "workflow"
        clarify = self.composer.platform_continue_clarify(
            wf_label,
            reasoning=understanding.reasoning,
            lang=lang,
        )
        return clarify, {
            "outcome": "NEEDS_CLARIFICATION",
            "reason": clarify,
            "rules_applied": ["PLATFORM_ONLY", "PLAN_REPLY_PLATFORM_CLARIFY"],
        }

    def _reply_general_help(
        self,
        *,
        understanding: UnderstandingResult,
        lang: str,
    ) -> tuple[str, dict[str, Any]]:
        from chat.services.platform.turn_semantics import is_internal_reasoning_text

        reasoning = (understanding.reasoning or "").strip()
        if reasoning and not is_internal_reasoning_text(reasoning):
            fallback = reasoning
        else:
            fallback = self.composer.general_help(lang=lang)
        return fallback, {
            "outcome": "NEEDS_CLARIFICATION",
            "reason": fallback,
            "rules_applied": ["PLAN_REPLY_GENERAL_HELP"],
        }

    @staticmethod
    def _pause_active_workflow_for_interrupt(state: StatePatchBuffer) -> None:
        from chat.services._policy_interrupt import should_pause_workflow_for_informational

        if should_pause_workflow_for_informational(state._memory):
            state.push("suspend_active_workflow")

    @staticmethod
    def _pause_for_oos(state: StatePatchBuffer) -> None:
        WorkflowPipeline._pause_active_workflow_for_interrupt(state)

    def _maybe_workflow_switch_confirm(
        self,
        message: str,
        memory: SessionMemory,
        lang: str,
        *,
        understanding: UnderstandingResult | None = None,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]] | None:
        if memory.pending_confirmation and memory.pending_confirmation.startswith("switch:"):
            return None
        interrupt = self.manager.detect_interrupt(message, memory, understanding=understanding)
        if not interrupt:
            return None
        state.push(
            "arm_switch_confirm",
            from_workflow=interrupt.from_workflow,
            to_workflow=interrupt.to_workflow,
            pending_message=interrupt.pending_message,
        )
        msg = self.manager.switch_confirm_message(
            interrupt.from_workflow, interrupt.to_workflow, lang=lang
        )
        return msg, {
            "outcome": "NEEDS_INPUT",
            "awaiting_confirmation": True,
            "rules_applied": ["WORKFLOW_SWITCH_CONFIRM"],
        }

    def _resolve_workflow_switch(
        self,
        message: str,
        memory: SessionMemory,
        lang: str,
        *,
        state: StatePatchBuffer,
        conversation_history: list[str] | None = None,
        trace_id: str = "",
    ) -> tuple[str, dict[str, Any]]:
        raw = memory.pending_confirmation or ""
        parts = raw.split(":")
        if len(parts) != 3:
            state.push("clear_pending_confirmation")
            return self.composer.session_context(memory, lang=lang), {"outcome": "INFORMATIONAL"}

        from_wf, to_wf = parts[1], parts[2]
        choice = self.manager.parse_switch_reply(message, from_wf, to_wf)

        if choice == "continue":
            state.push("clear_switch_confirm")
            defn = self.manager.ensure_definition(from_wf)
            if not memory.active_workflow:
                state.push("switch_to_workflow", value=from_wf)
                state.flush()
            prefix = self.composer.continuing_workflow(from_wf, lang=lang)
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        if choice == "switch":
            pending_msg = str((memory.last_entities or {}).get("switch_pending_message") or "")
            prev_active = memory.active_workflow.id if memory.active_workflow else from_wf
            state.push("clear_switch_confirm")
            state.push("switch_to_workflow", value=to_wf)
            state.flush()
            defn = self.manager.ensure_definition(to_wf)
            if pending_msg:
                if to_wf == "expense":
                    self._apply_expense_message_to_state(
                        pending_msg,
                        memory=memory,
                        state=state,
                        conversation_history=conversation_history,
                        trace_id=trace_id,
                    )
                else:
                    fields = self.fields.extract_workflow_fields(to_wf, pending_msg, memory=memory)
                    for key, val in fields.items():
                        if key == "items" and isinstance(val, list):
                            for item in val:
                                state.apply_field_updates(
                                    [FieldUpdate(field="items", value=item, action="append")]
                                )
                        else:
                            state.apply_field_updates([FieldUpdate(field=key, value=val)])
                    self._push_default_expense_date(state, memory)
                prefix = self.composer.workflow_started(defn.name, lang=lang)
                return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)
            prefix = self.composer.workflow_switch_resumed(
                memory,
                paused_workflow=prev_active,
                resumed_workflow=to_wf,
                lang=lang,
            )
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        msg = self.manager.switch_retry_message(from_wf, to_wf, lang=lang)
        return msg, {"outcome": "NEEDS_INPUT", "awaiting_confirmation": True}

    def _reject_oos(self, lang: str) -> tuple[str, dict[str, Any]]:
        msg = self.composer.reject_oos(lang=lang)
        return msg, {"outcome": "INFORMATIONAL", "rules_applied": ["OUT_OF_SCOPE_REJECT"]}

    @staticmethod
    def _push_default_expense_date(state: StatePatchBuffer, memory: SessionMemory) -> None:
        from datetime import date

        draft = memory.active_draft()
        if draft and draft.workflow_id == "expense":
            if not draft.fields.get("incurred_date"):
                state.set_draft_field("incurred_date", date.today().isoformat())

    @staticmethod
    def _active_defn(memory: SessionMemory):
        if memory.active_workflow:
            return get_workflow_definition(memory.active_workflow.id)
        return None

    def _block_parallel_leave(
        self,
        message: str,
        memory: SessionMemory,
        u: UnderstandingResult,
        lang: str,
        *,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]] | None:
        draft = memory.active_draft()
        if not draft or not memory.active_workflow or memory.active_workflow.id != "leave":
            return None
        if not leave_draft_in_progress(draft):
            return None
        if not is_duplicate_leave_attempt(message, u, draft):
            return None
        msg = self.composer.active_leave_parallel_block(memory, lang=lang)
        return msg, {
            "outcome": "BLOCKED",
            "reason": "parallel leave blocked while draft in progress",
            "rules_applied": ["ACTIVE_LEAVE_PARALLEL_BLOCK"],
        }

    @staticmethod
    def parse_duplicate_leave_reply(message: str) -> str | None:
        """Return ``continue``, ``new``, or None."""
        low = (message or "").lower().strip()
        if re.search(r"\b(continue|chaliye|current|same|ager|jotokhon|thik\s*ache)\b", low):
            return "continue"
        if re.search(r"\b(cancel|new|notun|fresh|start over|yes|ha+h)\b", low):
            return "new"
        return None

    def _resolve_duplicate_leave(
        self,
        message: str,
        memory: SessionMemory,
        lang: str,
        *,
        understanding: UnderstandingResult | None = None,
        state: StatePatchBuffer,
    ) -> tuple[str, dict[str, Any]]:
        choice = self.parse_duplicate_leave_reply(message)
        entities = dict(memory.last_entities or {})
        pending_updates = entities.pop("duplicate_leave_updates", [])
        state.push("set_last_entities", value=entities)

        draft = memory.active_draft()
        defn = self.manager.ensure_definition("leave")

        if choice == "continue":
            state.push("clear_pending_confirmation")
            prefix = self.composer.duplicate_leave_continue(lang=lang)
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        if choice == "new":
            state.push("clear_pending_confirmation")
            if draft:
                draft_id = memory.active_workflow.draft_id if memory.active_workflow else "default"
                state.push("clear_draft_fields", draft_id=draft_id)
            state.push("clear_pending_question")
            u = understanding
            if u and u.field_updates:
                state.apply_field_updates(u.field_updates)
            elif pending_updates:
                updates = [
                    FieldUpdate(
                        field=pu["field"],
                        value=pu["value"],
                        action=pu.get("action") or "set",
                        item_index=pu.get("item_index"),
                    )
                    for pu in pending_updates
                    if pu.get("field")
                ]
                if updates:
                    state.apply_field_updates(updates)
            elif message.strip() and draft:
                fields = self.fields.extract_workflow_fields(draft.workflow_id, message, memory=memory)
                updates = [FieldUpdate(field=k, value=v) for k, v in fields.items()]
                if updates:
                    state.apply_field_updates(updates)
            prefix = self.composer.duplicate_leave_fresh(lang=lang)
            return self._continue_collection(memory, defn, lang=lang, prefix=prefix, state=state)

        msg = self.composer.duplicate_leave_prompt(draft, lang=lang) if draft else self.composer.duplicate_leave_retry(lang=lang)
        state.push("set_pending_confirmation", value="duplicate_leave")
        return msg, {"outcome": "NEEDS_INPUT", "awaiting_confirmation": True}
