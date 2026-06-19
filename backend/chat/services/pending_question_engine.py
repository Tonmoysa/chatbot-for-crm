"""
Pending Question Engine — runs BEFORE intent detection and workflow selection.

When ``pending_question`` is set, the user's message is preferentially interpreted
as an answer to that question unless stronger signals indicate otherwise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chat.services.llm_client import LLMClient
from chat.services.observability import log_step
from chat.services.policy_intent_helpers import (
    is_general_knowledge_out_of_scope,
    is_hr_assistant_in_scope,
    is_off_topic_for_hr_assistant,
    is_policy_kb_query,
    is_rules_query,
)
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult
from chat.services.session_memory import PendingQuestion, SessionMemory


class MessageIntentKind(str, Enum):
    ANSWER_PENDING = "answer_pending"
    MODIFY_DATA = "modify_data"
    DELETE_DATA = "delete_data"
    SWITCH_WORKFLOW = "switch_workflow"
    ASK_POLICY = "ask_policy"
    ASK_STATUS = "ask_status"
    OUT_OF_SCOPE = "out_of_scope"
    NEW_WORKFLOW = "new_workflow"
    CLARIFICATION_NEEDED = "clarification_needed"


# Kinds that must win over starting a new workflow when pending_question is active.
_INTERRUPT_KINDS = frozenset(
    {
        MessageIntentKind.MODIFY_DATA,
        MessageIntentKind.DELETE_DATA,
        MessageIntentKind.SWITCH_WORKFLOW,
        MessageIntentKind.ASK_POLICY,
        MessageIntentKind.ASK_STATUS,
        MessageIntentKind.OUT_OF_SCOPE,
    }
)

_STRONG_NEW_WORKFLOW_RE = re.compile(
    r"(?:"
    r"\b(apply|request|submit|start|open|new)\b.{0,30}\b(leave|expense|claim|wfh)\b|"
    r"\b(leave|expense|claim)\b.{0,30}\b(apply|request|submit|start|new|chah[iy]|lagbe|lage)\b|"
    r"(ছুটি|খরচ).{0,30}(চাই|লাগবে|apply|request|submit|নিতে)|"
    r"(?:i\s+)?(?:want|need)\s+(?:to\s+)?(?:apply|take|request)\s+(?:for\s+)?(?:a\s+)?leave\b|"
    r"(?:log|add|submit)\s+(?:an?\s+)?expense\b"
    r")",
    re.I | re.UNICODE,
)

_MODIFY_RE = re.compile(
    r"(?:"
    r"\b(change|update|modify|edit|correct|fix|instead|rather|use)\b|"
    r"\b(instead\s+of|not\s+\d+|make\s+it)\b|"
    r"\bamount\s*ta\b|"
    r"(?:lunch|bus|nasta|prothom|first).{0,25}(?:amount|taka|tk).{0,20}(?:kore|kor|koro|dao)|"
    r"prothom\s*ta\b|"
    r"(?:বদল|পরিবর্তন|ঠিক|instead|change\s*koro)"
    r")",
    re.I | re.UNICODE,
)

_NEW_LEAVE_DATE_RE = re.compile(
    r"(?:agami|next)?\s*\d{1,2}\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|march|july|august|"
    r"january|february|march|april|june|july|august|september|october|november|december)"
    r".{0,25}(?:leave|chuti|chhuti|ছুটি)\b",
    re.I | re.UNICODE,
)

_DELETE_RE = re.compile(
    r"(?:"
    r"\b(delete|remove|drop|cancel\s+that|undo)\b|"
    r"(?:মুছ|ডিলিট|remove\s*koro|bad\s*d[iy]ao)"
    r")",
    re.I | re.UNICODE,
)

_SWITCH_RE = re.compile(
    r"(?:"
    r"\b(switch|move\s+to|back\s+to|resume|continue)\b.{0,25}\b(leave|expense|claim|wfh)\b|"
    r"\b(leave|expense|claim)\b.{0,25}\b(instead|first|age|আগে)|"
    r"(?:ekhon|এখন).{0,20}(?:expense|leave|খরচ|ছুটি)"
    r")",
    re.I | re.UNICODE,
)

_STATUS_RE = re.compile(
    r"\b(ref|reference|request\s*id|rid|status|track)\b|"
    r"(রেফারেন্স|স্ট্যাটাস|ট্র্যাক)|"
    r"\b[A-Z]{2,}-\d{4,}\b",
    re.I | re.UNICODE,
)

_PENDING_ANSWER_BLOCKERS = re.compile(
    r"(?:"
    r"\?\s*$|"
    r"^\s*(?:what|when|where|why|how|who|which|can|could|is|are|do|does)\b|"
    r"\b(?:ki|kobe|keno|kothay|kemon|kon)\b|"
    r"(?:কী|কি|কেন|কোথায়|কখন|কিভাবে|কোন)"
    r")",
    re.I | re.UNICODE,
)

_LLM_SYSTEM = """You classify the user's latest message in an HR chatbot conversation.

Return ONLY valid JSON with this schema:
{
  "kind": one of [
    "answer_pending",
    "modify_data",
    "delete_data",
    "switch_workflow",
    "ask_policy",
    "ask_status",
    "out_of_scope",
    "new_workflow",
    "clarification_needed"
  ],
  "confidence": number between 0 and 1,
  "reasoning": "one or two sentences",
  "field_value": "extracted answer for pending field if kind is answer_pending, else null",
  "target_workflow": "leave|expense|wfh|policy|status|none or null"
}

CRITICAL RULES:
1. If pending_question is set, prefer "answer_pending" when the message plausibly answers that field — even if it also mentions another topic briefly.
2. Never choose "new_workflow" if "answer_pending" is reasonable (confidence >= 0.55).
3. "ask_policy" = company HR policy / handbook / rules (including reimbursement policy).
4. "ask_status" = tracking a submitted request by reference or status.
5. "out_of_scope" = general knowledge unrelated to HR (sports, politics, programming trivia, weather, national holiday dates).
6. "modify_data" = user wants to change already collected draft data ("use 200 instead of 150").
7. "delete_data" = user wants to remove collected data or a line item.
8. "switch_workflow" = user wants to pause current work and work on another HR workflow.
9. "new_workflow" = user clearly starts a fresh leave/expense/WFH request and pending_question does NOT explain the message.
10. Greetings (hi, hello, thanks) without HR content are NOT out_of_scope — use "new_workflow" with low confidence.

Be conservative: when unsure with an active pending question, choose answer_pending with moderate confidence — UNLESS the user explicitly starts a brand-new leave/expense request."""


@dataclass
class PendingQuestionDecision:
    kind: MessageIntentKind
    confidence: float
    reasoning: str
    source: str
    blocks_new_workflow: bool
    field_value: str | None = None
    target_workflow: str | None = None
    extracted_entities: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "source": self.source,
            "blocks_new_workflow": self.blocks_new_workflow,
            "field_value": self.field_value,
            "target_workflow": self.target_workflow,
            "extracted_entities": self.extracted_entities,
        }


class PendingQuestionEngine:
    """Classifies each turn before intent detection / workflow selection."""

    def classify(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        llm: LLMClient | None = None,
        understanding: UnderstandingResult | None = None,
    ) -> PendingQuestionDecision:
        raw = (message or "").strip()
        if not raw:
            decision = PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.2,
                reasoning="Empty message.",
                source="rules",
                blocks_new_workflow=False,
            )
            self._log(trace_id, raw, memory, decision, understanding=understanding)
            return decision

        from chat.services.platform.intent_rules import is_bare_confirmation

        if memory.pending_confirmation == "submit" and is_bare_confirmation(raw):
            aw = memory.active_workflow
            decision = PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=0.95,
                reasoning="Submit confirmation (yes/ha).",
                source="rules",
                blocks_new_workflow=True,
                target_workflow=aw.id if aw else None,
            )
            self._log(trace_id, raw, memory, decision, understanding=understanding)
            return decision

        if understanding is not None:
            u_decision = self._decide_from_understanding(message, memory, understanding)
            if u_decision is not None:
                decision = self._apply_confidence_guard(
                    self._apply_pending_guardrails(raw, memory, u_decision)
                )
                self._log(trace_id, raw, memory, decision, understanding=understanding)
                return decision
            if understanding.source == "llm":
                fallback = PendingQuestionDecision(
                    kind=MessageIntentKind.CLARIFICATION_NEEDED,
                    confidence=understanding.confidence,
                    reasoning=understanding.reasoning or "LLM understanding — needs clarification routing.",
                    source="llm",
                    blocks_new_workflow=bool(memory.pending_question),
                )
                decision = self._apply_confidence_guard(
                    self._apply_pending_guardrails(raw, memory, fallback)
                )
                self._log(trace_id, raw, memory, decision, understanding=understanding)
                return decision

        client = llm or LLMClient()
        if client.is_configured():
            llm_decision = self._classify_llm(
                raw,
                memory=memory,
                conversation_history=conversation_history,
                trace_id=trace_id,
                client=client,
            )
            if llm_decision is not None:
                decision = self._apply_confidence_guard(
                    self._apply_pending_guardrails(raw, memory, llm_decision)
                )
                self._log(trace_id, raw, memory, decision, understanding=understanding)
                return decision

        decision = self._apply_confidence_guard(
            self._apply_pending_guardrails(
                raw,
                memory,
                self._classify_rules(raw, memory=memory, understanding=understanding),
            )
        )
        self._log(trace_id, raw, memory, decision, understanding=understanding)
        return decision

    @staticmethod
    def _apply_confidence_guard(decision: PendingQuestionDecision) -> PendingQuestionDecision:
        """Block irreversible workflow routing when confidence is low."""
        if decision.kind in (
            MessageIntentKind.ASK_POLICY,
            MessageIntentKind.ASK_STATUS,
            MessageIntentKind.OUT_OF_SCOPE,
            MessageIntentKind.ANSWER_PENDING,
            MessageIntentKind.CLARIFICATION_NEEDED,
            MessageIntentKind.NEW_WORKFLOW,
        ):
            return decision
        if decision.confidence >= 0.70:
            return decision
        if decision.kind in (
            MessageIntentKind.MODIFY_DATA,
            MessageIntentKind.DELETE_DATA,
            MessageIntentKind.SWITCH_WORKFLOW,
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.CLARIFICATION_NEEDED,
                confidence=decision.confidence,
                reasoning=decision.reasoning or "Low confidence — clarification required.",
                source=decision.source,
                blocks_new_workflow=True,
                target_workflow=decision.target_workflow,
            )
        return decision

    def _log(
        self,
        trace_id: str,
        message: str,
        memory: SessionMemory,
        decision: PendingQuestionDecision,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> None:
        pq = memory.pending_question
        payload: dict[str, Any] = {
            "user_message": message,
            "decision": decision.to_log_dict(),
            "active_workflow": (
                memory.active_workflow.to_dict() if memory.active_workflow else None
            ),
            "pending_question": pq.to_dict() if pq else None,
            "has_draft": memory.active_draft() is not None,
        }
        if understanding is not None:
            payload["turn_understanding"] = understanding.to_dict()
        log_step(trace_id, "pending_question_engine", payload)

    @staticmethod
    def _decide_from_understanding(
        message: str,
        memory: SessionMemory,
        understanding: UnderstandingResult,
    ) -> PendingQuestionDecision | None:
        """Map AI Understanding Layer output to a pending-question decision."""
        pq = memory.pending_question
        aw = memory.active_workflow
        src = understanding.source or "understanding"
        conf = understanding.confidence

        if memory.pending_confirmation == "submit" and understanding.action in (
            UnderstandingAction.CONFIRM.value,
            UnderstandingAction.SUBMIT.value,
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(conf, 0.9),
                reasoning=understanding.reasoning or "Confirm leave/expense submit.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=aw.id if aw else understanding.workflow,
            )

        if understanding.is_greeting:
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.92,
                reasoning=understanding.reasoning or "Greeting — conversational reply.",
                source=src,
                blocks_new_workflow=False,
            )

        if understanding.is_out_of_scope:
            return PendingQuestionDecision(
                kind=MessageIntentKind.OUT_OF_SCOPE,
                confidence=max(conf, 0.88),
                reasoning=understanding.reasoning or "Out of scope (AI understanding).",
                source=src,
                blocks_new_workflow=bool(pq),
            )

        if understanding.action == UnderstandingAction.REVIEW.value:
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=max(conf, 0.86),
                reasoning=understanding.reasoning or "Summary/review (AI understanding).",
                source=src,
                blocks_new_workflow=False,
                target_workflow=understanding.workflow or (aw.id if aw else None),
            )

        if understanding.action == UnderstandingAction.QUERY.value or (
            understanding.workflow in ("none", "") and "policy" in (understanding.goal or "").lower()
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_POLICY,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Policy query (AI understanding).",
                source=src,
                blocks_new_workflow=bool(pq),
            )

        if understanding.action == UnderstandingAction.MODIFY.value:
            return PendingQuestionDecision(
                kind=MessageIntentKind.MODIFY_DATA,
                confidence=conf,
                reasoning=understanding.reasoning or "Modify (AI understanding).",
                source=src,
                blocks_new_workflow=True,
            )

        if understanding.action == UnderstandingAction.DELETE.value:
            return PendingQuestionDecision(
                kind=MessageIntentKind.DELETE_DATA,
                confidence=conf,
                reasoning=understanding.reasoning or "Delete (AI understanding).",
                source=src,
                blocks_new_workflow=True,
            )

        if (
            aw
            and understanding.interrupt_workflow
            and understanding.interrupt_workflow != aw.id
            and conf >= 0.65
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or f"Switch to {understanding.interrupt_workflow}.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=understanding.interrupt_workflow,
            )

        if (
            pq
            and aw
            and aw.id == "leave"
            and understanding.is_expense_intent()
            and conf >= 0.7
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Expense intent during leave (AI understanding).",
                source=src,
                blocks_new_workflow=True,
                target_workflow="expense",
            )

        if (
            pq
            and understanding.is_expense_intent()
            and understanding.interrupts_active_workflow(aw.id if aw else None)
            and conf >= 0.7
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=conf,
                reasoning=understanding.reasoning or "Cross-workflow expense intent.",
                source=src,
                blocks_new_workflow=True,
                target_workflow="expense",
            )

        if (
            aw
            and understanding.is_leave_intent()
            and aw.id == "expense"
            and conf >= 0.7
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=max(conf, 0.85),
                reasoning=understanding.reasoning or "Leave intent during expense.",
                source=src,
                blocks_new_workflow=True,
                target_workflow="leave",
            )

        if pq and understanding.action in (
            UnderstandingAction.COLLECT.value,
            UnderstandingAction.CONFIRM.value,
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(conf, 0.72),
                reasoning=understanding.reasoning or f"Answer pending field '{pq.field}'.",
                source=src,
                blocks_new_workflow=True,
                target_workflow=pq.workflow_id or (aw.id if aw else None),
            )

        if understanding.action in (
            UnderstandingAction.START.value,
            UnderstandingAction.COLLECT.value,
        ) and understanding.workflow in ("leave", "expense"):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=max(conf, 0.8),
                reasoning=understanding.reasoning or f"Start/collect {understanding.workflow}.",
                source=src,
                blocks_new_workflow=bool(pq),
                target_workflow=understanding.workflow,
            )

        return None

    def _apply_pending_guardrails(
        self,
        message: str,
        memory: SessionMemory,
        decision: PendingQuestionDecision,
    ) -> PendingQuestionDecision:
        pq = memory.pending_question

        if _STRONG_NEW_WORKFLOW_RE.search(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=max(decision.confidence, 0.84),
                reasoning=(
                    "Explicit new leave/expense/WFH phrasing — overrides pending slot interpretation."
                    if pq
                    else "Explicit new workflow phrasing detected."
                ),
                source=decision.source,
                blocks_new_workflow=False,
                target_workflow=self._infer_new_workflow(message),
            )

        if not pq:
            decision.blocks_new_workflow = decision.kind in _INTERRUPT_KINDS
            return decision

        if decision.kind == MessageIntentKind.NEW_WORKFLOW:
            decision.blocks_new_workflow = False
            return decision

        if decision.kind in _INTERRUPT_KINDS or decision.kind == MessageIntentKind.ANSWER_PENDING:
            decision.blocks_new_workflow = True
            return decision

        if decision.confidence < 0.85:
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=max(decision.confidence, 0.62),
                reasoning=(
                    f"Pending question on '{pq.field}' — message treated as slot answer "
                    f"instead of new workflow. Original: {decision.reasoning}"
                ),
                source=decision.source,
                blocks_new_workflow=True,
                field_value=decision.field_value or message.strip(),
                target_workflow=(
                    memory.active_workflow.id if memory.active_workflow else pq.workflow_id
                ),
            )

        decision.blocks_new_workflow = True
        return decision

    def _classify_llm(
        self,
        message: str,
        *,
        memory: SessionMemory,
        conversation_history: list[str],
        trace_id: str,
        client: LLMClient,
    ) -> PendingQuestionDecision | None:
        draft = memory.active_draft()
        context = {
            "active_workflow": (
                memory.active_workflow.to_dict() if memory.active_workflow else None
            ),
            "pending_question": (
                memory.pending_question.to_dict() if memory.pending_question else None
            ),
            "workflow_draft": draft.to_dict() if draft else None,
            "conversation_history": conversation_history[-8:],
        }
        user_prompt = (
            "Session context (JSON):\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
            f"User message:\n{message}\n"
        )
        parsed = client.chat_json(
            system_prompt=_LLM_SYSTEM,
            user_prompt=user_prompt,
            trace_id=trace_id,
        )
        if not isinstance(parsed, dict):
            return None

        kind_raw = str(parsed.get("kind") or "").strip().lower()
        try:
            kind = MessageIntentKind(kind_raw)
        except ValueError:
            return None

        confidence = float(parsed.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(parsed.get("reasoning") or "").strip() or "LLM classification."
        field_value = parsed.get("field_value")
        if field_value is not None:
            field_value = str(field_value).strip() or None

        target = parsed.get("target_workflow")
        target_workflow = str(target).strip() if target else None

        return PendingQuestionDecision(
            kind=kind,
            confidence=confidence,
            reasoning=reasoning,
            source="llm",
            blocks_new_workflow=False,
            field_value=field_value,
            target_workflow=target_workflow,
        )

    def _classify_rules(
        self,
        message: str,
        *,
        memory: SessionMemory,
        understanding: UnderstandingResult | None = None,
    ) -> PendingQuestionDecision:
        from chat.services.platform.intent_rules import is_greeting_or_chitchat

        if is_greeting_or_chitchat(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.92,
                reasoning="Greeting/chitchat — conversational reply.",
                source="rules",
                blocks_new_workflow=False,
            )

        pq = memory.pending_question
        draft = memory.active_draft()

        if is_general_knowledge_out_of_scope(message) or (
            is_off_topic_for_hr_assistant(message, wizard_active=bool(pq))
            and not is_hr_assistant_in_scope(message)
        ):
            return PendingQuestionDecision(
                kind=MessageIntentKind.OUT_OF_SCOPE,
                confidence=0.88,
                reasoning="General-knowledge / off-HR topic detected.",
                source="rules",
                blocks_new_workflow=bool(pq),
            )

        if _STATUS_RE.search(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_STATUS,
                confidence=0.9,
                reasoning="Request reference or status lookup phrasing.",
                source="rules",
                blocks_new_workflow=bool(pq),
            )

        if is_rules_query(message) or is_policy_kb_query(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ASK_POLICY,
                confidence=0.85,
                reasoning="Policy or rules query detected.",
                source="rules",
                blocks_new_workflow=bool(pq),
            )

        if _SWITCH_RE.search(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.SWITCH_WORKFLOW,
                confidence=0.82,
                reasoning="Explicit workflow switch/resume phrasing.",
                source="rules",
                blocks_new_workflow=True,
                target_workflow=self._infer_switch_target(message),
            )

        if draft and _DELETE_RE.search(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.DELETE_DATA,
                confidence=0.8,
                reasoning="Delete/remove phrasing with active draft.",
                source="rules",
                blocks_new_workflow=True,
            )

        if draft and _MODIFY_RE.search(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.MODIFY_DATA,
                confidence=0.78,
                reasoning="Modify/correct phrasing with active draft.",
                source="rules",
                blocks_new_workflow=True,
            )

        if pq and (self._is_summary_or_total(message)):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.86,
                reasoning="Summary/review request — not a slot answer.",
                source="rules",
                blocks_new_workflow=False,
                target_workflow=memory.active_workflow.id if memory.active_workflow else None,
            )

        if (
            pq
            and memory.active_workflow
            and memory.active_workflow.id == "leave"
            and understanding is None
        ):
            from chat.services.platform.intent_rules import is_workflow_interrupt_expense

            if is_workflow_interrupt_expense(message, active_workflow="leave"):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.SWITCH_WORKFLOW,
                    confidence=0.92,
                    reasoning="Strong expense signal while leave draft is open — workflow switch.",
                    source="rules",
                    blocks_new_workflow=True,
                    target_workflow="expense",
                )

        if pq and memory.active_workflow and memory.active_workflow.id == "leave":
            if _NEW_LEAVE_DATE_RE.search(message) or (
                _STRONG_NEW_WORKFLOW_RE.search(message) and re.search(r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|march|july|august)", message, re.I)
            ):
                return PendingQuestionDecision(
                    kind=MessageIntentKind.NEW_WORKFLOW,
                    confidence=0.86,
                    reasoning="New leave date while another leave draft is open.",
                    source="rules",
                    blocks_new_workflow=False,
                    target_workflow="leave",
                )

        if _STRONG_NEW_WORKFLOW_RE.search(message):
            return PendingQuestionDecision(
                kind=MessageIntentKind.NEW_WORKFLOW,
                confidence=0.84,
                reasoning="Explicit new leave/expense/WFH request phrasing.",
                source="rules",
                blocks_new_workflow=False,
                target_workflow=self._infer_new_workflow(message),
            )

        if pq and self._message_likely_answers_pending(message, pq, understanding=understanding):
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=0.72,
                reasoning=(
                    f"Active pending question on field '{pq.field}' — message plausibly "
                    "answers it (not a strong interrupt)."
                ),
                source="rules",
                blocks_new_workflow=True,
                field_value=message.strip(),
                target_workflow=pq.workflow_id or (
                    memory.active_workflow.id if memory.active_workflow else None
                ),
            )

        if pq:
            return PendingQuestionDecision(
                kind=MessageIntentKind.ANSWER_PENDING,
                confidence=0.58,
                reasoning=(
                    f"Pending question on '{pq.field}' — defaulting to slot answer "
                    "to avoid misrouting as new workflow."
                ),
                source="rules",
                blocks_new_workflow=True,
                field_value=message.strip(),
                target_workflow=pq.workflow_id,
            )

        return PendingQuestionDecision(
            kind=MessageIntentKind.NEW_WORKFLOW,
            confidence=0.45,
            reasoning="No pending question; defer to intent detection.",
            source="rules",
            blocks_new_workflow=False,
        )

    @staticmethod
    def _is_summary_or_total(message: str) -> bool:
        from chat.services.platform.intent_rules import is_summary_request, is_total_request

        return is_summary_request(message) or is_total_request(message)

    @staticmethod
    def _message_likely_answers_pending(
        message: str,
        pq: PendingQuestion,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> bool:
        if understanding and understanding.is_expense_intent():
            return False
        if _MODIFY_RE.search(message):
            return False
        if _NEW_LEAVE_DATE_RE.search(message):
            return False
        if _PENDING_ANSWER_BLOCKERS.search(message):
            return False
        if (
            _STATUS_RE.search(message)
            or _SWITCH_RE.search(message)
            or _DELETE_RE.search(message)
        ):
            return False
        if is_rules_query(message) or is_policy_kb_query(message):
            return False
        if is_general_knowledge_out_of_scope(message):
            return False
        from chat.services.platform.intent_rules import is_workflow_interrupt_expense

        if is_workflow_interrupt_expense(message, active_workflow="leave"):
            return False
        # Short/direct replies are usually slot answers.
        words = message.split()
        if len(words) <= 12:
            return True
        field = pq.field.lower()
        if field in ("reason", "notes", "description") and len(words) <= 40:
            return True
        return len(words) <= 20

    @staticmethod
    def _infer_switch_target(message: str) -> str | None:
        low = message.lower()
        if re.search(r"\b(expense|claim|খরচ)\b", low) or "খরচ" in message:
            return "expense"
        if re.search(r"\b(leave|wfh|ছুটি|chuti)\b", low) or "ছুটি" in message:
            return "leave"
        return None

    @staticmethod
    def _infer_new_workflow(message: str) -> str | None:
        low = message.lower()
        if re.search(r"\b(expense|claim|খরচ|reimburse)\b", low) or "খরচ" in message:
            return "expense"
        if re.search(r"\b(wfh|work\s+from\s+home)\b", low):
            return "wfh"
        if re.search(r"\b(leave|chuti|chhuti|ছুটি)\b", low) or "ছুটি" in message:
            return "leave"
        return None
