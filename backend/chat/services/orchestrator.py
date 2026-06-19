"""Central chat pipeline — pending question engine first, then policy / status / fallback."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from chat.constants import (
    INTENT_HR_POLICY,
    INTENT_REQUEST_STATUS,
    INTENT_UNKNOWN,
)
from chat.services.conversational import conversational_reply
from chat.services.crm.base import CRMError
from chat.services.crm.factory import get_crm_adapter
from chat.services.decision_engine import DecisionEngine
from chat.services.entity_extractor import EntityExtractor
from chat.services.intent_detector import IntentDetector
from chat.services.llm_client import clear_llm_trace_state
from chat.services.memory_store import ConversationMemoryStore
from chat.services.observability import log_step
from chat.services.pending_question_engine import (
    MessageIntentKind,
    PendingQuestionEngine,
)
from chat.services.platform.ai_understanding import AIUnderstandingLayer
from chat.services.platform.intent_rules import is_programming_question as is_programming_oos
from chat.services.pending_question_handlers import (
    build_pending_response,
    handle_pending_decision,
)
from chat.services.platform.pipeline import WorkflowPipeline
from chat.services.policy_intent_helpers import (
    build_out_of_scope_message,
    format_today_date_reply,
    is_general_knowledge_out_of_scope,
    is_hr_assistant_in_scope,
    is_hr_today_date_query,
    is_off_topic_for_hr_assistant,
    is_policy_kb_query,
    is_rules_query,
)
from chat.services.response_formatter import build_user_message
from chat.services.session_memory import load_session_memory, save_session_memory
from chat.services.translator import (
    align_policy_answer_language,
    detect_user_language,
    is_translation_request,
    resolve_reply_language,
    strip_policy_footer,
    translate_text,
)
from knowledge_base.services.rag_pipeline import (
    hr_policy_not_found_message,
    try_hr_policy_rag,
)

_RULES_FOOTER_EN = (
    "_(Answers come from your uploaded policies; ask using the policy title or topic.)_"
)
_RULES_FOOTER_BN = (
    "_(উত্তর আপনার আপলোড করা পলিসি থেকে আসে; পলিসির নাম বা বিষয় লিখে জিজ্ঞাসা করুন।)_"
)


def _rules_footer(*, lang: str) -> str:
    return _RULES_FOOTER_BN if lang in ("bn", "banglish") else _RULES_FOOTER_EN


def _workflow_continuation_hint(memory) -> str:
    if memory.suspended_workflows:
        sw = memory.suspended_workflows[-1]
        return (
            f"\n\n_(Your **{sw.workflow_id}** request is paused — "
            f"reply **{sw.workflow_id}** anytime to continue.)_"
        )
    pq = memory.pending_question
    wf = memory.active_workflow
    if not pq and not wf:
        return ""
    lang_note = ""
    if pq:
        return (
            f"\n\n_(Your **{pq.workflow_id or (wf.id if wf else 'workflow')}** draft is still open — "
            f"reply anytime to continue.)_"
        )
    if wf:
        return f"\n\n_(Your **{wf.id}** draft is still saved — you can continue when ready.)_"
    return ""


def new_trace_id() -> str:
    return str(uuid.uuid4())


class ChatOrchestrator:
    """User input → pending question engine → intent → entities → decision → CRM → response."""

    def __init__(self) -> None:
        self.memory = ConversationMemoryStore()
        self.pending_engine = PendingQuestionEngine()
        self.understanding_layer = AIUnderstandingLayer()
        self.workflow_pipeline = WorkflowPipeline()
        self.intents = IntentDetector()
        self.entities = EntityExtractor()
        self.engine = DecisionEngine()

    def run_chat(
        self,
        *,
        message: str,
        session_id: str | None,
        company_id: str,
        employee_id: str,
        trace_id: str,
        document_text: str | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        clear_llm_trace_state(trace_id)

        session = self.memory.get_or_create_session(
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id or "",
        )
        context_lines = self.memory.recent_context_lines(session)
        session_memory = load_session_memory(session)

        if is_hr_today_date_query(message):
            lang = detect_user_language(message)
            today_iso = date.today().isoformat()
            msg = format_today_date_reply(today_iso=today_iso, lang=lang)
            return self._finalize(session, message, msg, trace_id, {
                "intent": INTENT_HR_POLICY,
                "entities": {"calendar_date": today_iso},
                "decision": {
                    "outcome": "INFORMATIONAL",
                    "reason": "Today's calendar date.",
                    "rules_applied": ["HR_TODAY_DATE_QUERY"],
                },
                "response": {"message": msg, "status": "success", "request_id": ""},
                "status": "success",
            })

        translate_to = is_translation_request(message)
        if translate_to:
            prev = self._assistant_text_for_translation(context_lines, target_lang=translate_to)
            if prev:
                source = strip_policy_footer(prev)
                translated, ok = translate_text(
                    source,
                    target_lang=translate_to,
                    trace_id=trace_id,
                )
                if ok:
                    msg = translated.rstrip() + "\n\n" + _rules_footer(lang=translate_to)
                    status_str = "success"
                else:
                    msg = (
                        "Translation is briefly unavailable — re-posting the previous answer:\n\n"
                        + prev
                    )
                    status_str = "degraded"
                return self._finalize(session, message, msg, trace_id, {
                    "intent": INTENT_HR_POLICY,
                    "entities": {"translation_target_lang": translate_to},
                    "decision": {
                        "outcome": "INFORMATIONAL",
                        "reason": "Translated the previous assistant turn.",
                        "rules_applied": ["TRANSLATION_FOLLOWUP"],
                    },
                    "response": {"message": msg, "status": status_str, "request_id": ""},
                    "status": "success",
                })

        # ── Session Memory loaded ──
        # ── AI Understanding Layer (LLM primary) ──
        turn_understanding = self.understanding_layer.understand(
            message,
            memory=session_memory,
            conversation_history=context_lines,
            trace_id=trace_id,
        )
        session_memory.last_entities = {
            **dict(session_memory.last_entities or {}),
            "turn_understanding": turn_understanding.to_dict(),
        }

        # ── Pending Question Engine (uses AI understanding object) ──
        pq_decision = self.pending_engine.classify(
            message,
            memory=session_memory,
            conversation_history=context_lines,
            trace_id=trace_id,
            understanding=turn_understanding,
        )

        if (
            turn_understanding.is_greeting
            and not session_memory.active_workflow
            and not session_memory.pending_question
            and not str(session_memory.pending_confirmation or "").startswith("switch:")
        ):
            reply = conversational_reply(
                message=message,
                context_lines=context_lines,
                trace_id=trace_id,
            )
            if reply:
                save_session_memory(session, session_memory)
                return self._finalize(session, message, reply, trace_id, {
                    "intent": INTENT_UNKNOWN,
                    "entities": {},
                    "decision": {
                        "outcome": "INFORMATIONAL",
                        "reason": "Greeting/chitchat reply.",
                        "rules_applied": ["CONVERSATIONAL_GREETING"],
                        "pending_question_decision": pq_decision.to_log_dict(),
                    },
                    "response": {"message": reply, "status": "success", "request_id": ""},
                    "status": "success",
                })

        routed = self._route_pending_decision(
            message,
            pq_decision=pq_decision,
            understanding=turn_understanding,
            session=session,
            session_memory=session_memory,
            context_lines=context_lines,
            company_id=company_id,
            employee_id=employee_id,
            trace_id=trace_id,
            document_text=document_text,
            idempotency_key=idempotency_key,
        )
        if routed is not None:
            save_session_memory(session, session_memory)
            return routed

        # Active workflow: submit / confirm / review without PQ interrupt
        if session_memory.active_workflow:
            wf_result = self.workflow_pipeline.try_handle_active_workflow(
                message,
                memory=session_memory,
                understanding=turn_understanding,
                conversation_history=context_lines,
                trace_id=trace_id,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session.session_id,
                idempotency_key=idempotency_key,
            )
            if wf_result:
                msg, decision = wf_result
                save_session_memory(session, session_memory)
                return self._finalize(session, message, msg, trace_id, {
                    "intent": session_memory.active_workflow.id.upper() if session_memory.active_workflow else "",
                    "entities": session_memory.last_entities,
                    "decision": decision,
                    "response": {
                        "message": msg,
                        "status": "success" if decision.get("outcome") != "ERROR" else "error",
                        "request_id": decision.get("request_id", ""),
                    },
                    "status": "success",
                })

        # New message → AI understanding / workflow start
        if pq_decision.kind == MessageIntentKind.NEW_WORKFLOW and not pq_decision.blocks_new_workflow:
            wf_result = handle_pending_decision(
                pq_decision,
                message=message,
                memory=session_memory,
                understanding=turn_understanding,
                conversation_history=context_lines,
                trace_id=trace_id,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session.session_id,
                idempotency_key=idempotency_key,
            )
            if wf_result:
                msg, decision = wf_result
                save_session_memory(session, session_memory)
                return self._finalize(session, message, msg, trace_id, {
                    "intent": (session_memory.active_workflow.id.upper() if session_memory.active_workflow else INTENT_UNKNOWN),
                    "entities": session_memory.last_entities,
                    "decision": decision,
                    "response": {"message": msg, "status": "success", "request_id": decision.get("request_id", "")},
                    "status": "success",
                })

        save_session_memory(session, session_memory)

        log_step(trace_id, "intent_detection_start", {"user_message": message})
        intent_result = self.intents.detect(message, trace_id)
        intent = str(intent_result.get("intent") or INTENT_UNKNOWN)
        log_step(trace_id, "intent_detected", {"intent": intent, "source": intent_result.get("source")})

        entity_result = self.entities.extract(message, intent, context_lines, trace_id)
        entities = dict(entity_result.get("entities") or {})
        if document_text:
            entities["document_text"] = document_text
            entities["document_read"] = True

        crm = get_crm_adapter()
        crm_context: dict[str, Any] = {}
        request_id = ""

        if intent == INTENT_REQUEST_STATUS:
            rid = str(entities.get("request_id") or "").strip()
            if rid:
                crm_context = crm.get_request_status(
                    rid,
                    company_id=company_id,
                    employee_id=employee_id,
                    session_id=session.session_id,
                )
                request_id = rid

        decision = self.engine.evaluate(
            intent=intent,
            entities=entities,
            crm_context=crm_context,
        )
        decision["pending_question_decision"] = pq_decision.to_log_dict()

        crm_payload: dict[str, Any] = dict(crm_context)
        msg = ""
        resp_status = "success"

        if intent == INTENT_HR_POLICY or is_policy_kb_query(message) or is_rules_query(message):
            msg, resp_status, decision, crm_payload, request_id = self._run_policy_rag(
                message,
                intent=intent,
                entities=entities,
                decision=decision,
                crm_payload=crm_payload,
                context_lines=context_lines,
                company_id=company_id,
                trace_id=trace_id,
                request_id=request_id,
            )

        if not crm_payload.get("rules_answer"):
            if intent == INTENT_UNKNOWN or decision.get("outcome") == "NEEDS_CLARIFICATION":
                reply = conversational_reply(
                    message=message,
                    context_lines=context_lines,
                    trace_id=trace_id,
                )
                if reply:
                    msg = reply
                    decision = {
                        "outcome": "INFORMATIONAL",
                        "reason": "Conversational fallback.",
                        "rules_applied": ["CONVERSATIONAL_FALLBACK"],
                        "pending_question_decision": pq_decision.to_log_dict(),
                    }
                else:
                    msg, resp_status = build_user_message(
                        intent=intent,
                        entities=entities,
                        decision=decision,
                        crm_payload=crm_payload,
                    )
            else:
                if self._should_create_crm_request(intent, decision):
                    try:
                        created = crm.create_request(
                            company_id=company_id,
                            employee_id=employee_id,
                            session_id=session.session_id,
                            intent=intent,
                            entities=entities,
                            decision=decision,
                            idempotency_key=idempotency_key,
                        )
                        request_id = str(created.get("request_id") or "")
                        crm_payload.update(created.get("record") or {})
                        crm_payload["request_id"] = request_id
                    except CRMError as exc:
                        decision = {
                            "outcome": "ERROR",
                            "reason": str(exc),
                            "rules_applied": ["CRM_ERROR"],
                            "pending_question_decision": pq_decision.to_log_dict(),
                        }
                msg, resp_status = build_user_message(
                    intent=intent,
                    entities=entities,
                    decision=decision,
                    crm_payload=crm_payload,
                )
        elif not msg:
            msg, resp_status = build_user_message(
                intent=intent,
                entities=entities,
                decision=decision,
                crm_payload=crm_payload,
            )

        return self._finalize(session, message, msg, trace_id, {
            "intent": intent,
            "entities": entities,
            "decision": decision,
            "response": {
                "message": msg,
                "status": resp_status,
                "request_id": request_id,
            },
            "status": "success",
        })

    def _route_pending_decision(
        self,
        message: str,
        *,
        pq_decision,
        understanding,
        session,
        session_memory,
        context_lines: list[str],
        company_id: str,
        employee_id: str,
        trace_id: str,
        document_text: str | None,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        kind = pq_decision.kind

        if kind in (
            MessageIntentKind.ANSWER_PENDING,
            MessageIntentKind.MODIFY_DATA,
            MessageIntentKind.DELETE_DATA,
            MessageIntentKind.SWITCH_WORKFLOW,
            MessageIntentKind.CLARIFICATION_NEEDED,
        ):
            msg, decision = build_pending_response(
                pq_decision,
                memory=session_memory,
                user_message=message,
                conversation_history=context_lines,
                trace_id=trace_id,
                understanding=understanding,
                company_id=company_id,
                employee_id=employee_id,
                session_id=session.session_id,
                idempotency_key=idempotency_key,
            )
            if msg:
                return self._finalize(session, message, msg, trace_id, {
                    "intent": (
                        session_memory.active_workflow.id.upper()
                        if session_memory.active_workflow
                        else INTENT_UNKNOWN
                    ),
                    "entities": session_memory.last_entities,
                    "decision": decision,
                    "response": {"message": msg, "status": "success", "request_id": decision.get("request_id", "")},
                    "status": "success",
                })

        if kind == MessageIntentKind.OUT_OF_SCOPE:
            lang = detect_user_language(message)
            from chat.services.platform.workflow_manager import WorkflowManager

            if session_memory.active_workflow:
                WorkflowManager().suspend_active(session_memory)
            msg = build_out_of_scope_message(
                message, lang=lang, context_lines=context_lines, trace_id=trace_id
            )
            msg += _workflow_continuation_hint(session_memory)
            return self._finalize(session, message, msg, trace_id, {
                "intent": INTENT_UNKNOWN,
                "entities": {},
                "decision": {
                    "outcome": "INFORMATIONAL",
                    "reason": pq_decision.reasoning,
                    "rules_applied": ["OUT_OF_SCOPE", "PENDING_QUESTION_ENGINE"],
                    "pending_question_decision": pq_decision.to_log_dict(),
                },
                "response": {"message": msg, "status": "success", "request_id": ""},
                "status": "success",
            })

        if kind == MessageIntentKind.ASK_POLICY:
            entities: dict[str, Any] = {}
            if document_text:
                entities["document_text"] = document_text
                entities["document_read"] = True
            decision = {
                "outcome": "INFORMATIONAL",
                "reason": pq_decision.reasoning,
                "rules_applied": ["ASK_POLICY", "PENDING_QUESTION_ENGINE"],
                "pending_question_decision": pq_decision.to_log_dict(),
            }
            crm_payload: dict[str, Any] = {}
            msg, resp_status, decision, crm_payload, request_id = self._run_policy_rag(
                message,
                intent=INTENT_HR_POLICY,
                entities=entities,
                decision=decision,
                crm_payload=crm_payload,
                context_lines=context_lines,
                company_id=company_id,
                trace_id=trace_id,
                request_id="",
            )
            msg += _workflow_continuation_hint(session_memory)
            return self._finalize(session, message, msg, trace_id, {
                "intent": INTENT_HR_POLICY,
                "entities": entities,
                "decision": decision,
                "response": {"message": msg, "status": resp_status, "request_id": request_id},
                "status": "success",
            })

        if kind == MessageIntentKind.ASK_STATUS:
            entity_result = self.entities.extract(
                message, INTENT_REQUEST_STATUS, context_lines, trace_id
            )
            entities = dict(entity_result.get("entities") or {})
            crm = get_crm_adapter()
            crm_context: dict[str, Any] = {}
            request_id = ""
            rid = str(entities.get("request_id") or "").strip()
            if rid:
                crm_context = crm.get_request_status(
                    rid,
                    company_id=company_id,
                    employee_id=employee_id,
                    session_id=session.session_id,
                )
                request_id = rid
            decision = self.engine.evaluate(
                intent=INTENT_REQUEST_STATUS,
                entities=entities,
                crm_context=crm_context,
            )
            decision["pending_question_decision"] = pq_decision.to_log_dict()
            decision["rules_applied"] = list(decision.get("rules_applied") or []) + [
                "PENDING_QUESTION_ENGINE"
            ]
            msg, resp_status = build_user_message(
                intent=INTENT_REQUEST_STATUS,
                entities=entities,
                decision=decision,
                crm_payload=crm_context,
            )
            msg += _workflow_continuation_hint(session_memory)
            return self._finalize(session, message, msg, trace_id, {
                "intent": INTENT_REQUEST_STATUS,
                "entities": entities,
                "decision": decision,
                "response": {"message": msg, "status": resp_status, "request_id": request_id},
                "status": "success",
            })

        if kind == MessageIntentKind.NEW_WORKFLOW:
            if is_programming_oos(message):
                lang = detect_user_language(message)
                msg = (
                    "এটি HR assistant-এর scope-এর বাইরে।"
                    if lang == "bn"
                    else "That's outside my scope as an HR assistant."
                )
                return self._finalize(session, message, msg, trace_id, {
                    "intent": INTENT_UNKNOWN,
                    "entities": {},
                    "decision": {"outcome": "INFORMATIONAL", "reason": "Programming OOS", "rules_applied": ["OUT_OF_SCOPE"]},
                    "response": {"message": msg, "status": "success", "request_id": ""},
                    "status": "success",
                })
            if pq_decision.blocks_new_workflow:
                msg, decision = build_pending_response(
                    pq_decision,
                    memory=session_memory,
                    user_message=message,
                    conversation_history=context_lines,
                    trace_id=trace_id,
                    understanding=understanding,
                    company_id=company_id,
                    employee_id=employee_id,
                    session_id=session.session_id,
                    idempotency_key=idempotency_key,
                )
                if msg:
                    return self._finalize(session, message, msg, trace_id, {
                        "intent": INTENT_UNKNOWN,
                        "entities": {},
                        "decision": decision,
                        "response": {"message": msg, "status": "success", "request_id": ""},
                        "status": "success",
                    })
            if is_general_knowledge_out_of_scope(message) or (
                is_off_topic_for_hr_assistant(message) and not is_hr_assistant_in_scope(message)
            ):
                turn_u = (session_memory.last_entities or {}).get("turn_understanding") or {}
                if turn_u.get("action") not in ("review", "submit", "confirm"):
                    lang = detect_user_language(message)
                    msg = build_out_of_scope_message(
                        message, lang=lang, context_lines=context_lines, trace_id=trace_id
                    )
                    return self._finalize(session, message, msg, trace_id, {
                        "intent": INTENT_UNKNOWN,
                        "entities": {},
                        "decision": {
                            "outcome": "INFORMATIONAL",
                            "reason": "Out of scope.",
                            "rules_applied": ["OUT_OF_SCOPE"],
                            "pending_question_decision": pq_decision.to_log_dict(),
                        },
                        "response": {"message": msg, "status": "success", "request_id": ""},
                        "status": "success",
                    })
            return None

        return None

    def _run_policy_rag(
        self,
        message: str,
        *,
        intent: str,
        entities: dict[str, Any],
        decision: dict[str, Any],
        crm_payload: dict[str, Any],
        context_lines: list[str],
        company_id: str,
        trace_id: str,
        request_id: str,
    ) -> tuple[str, str, dict[str, Any], dict[str, Any], str]:
        msg = ""
        resp_status = "success"
        rag = try_hr_policy_rag(message, trace_id, company_id=company_id)
        answer_text = (rag or {}).get("text") or (rag or {}).get("answer")
        if rag and answer_text:
            reply_lang = resolve_reply_language(message, context_lines)
            answer = align_policy_answer_language(
                str(answer_text),
                user_message=message,
                target_lang=reply_lang,
                trace_id=trace_id,
            )
            crm_payload["rules_answer"] = answer.rstrip() + "\n\n" + _rules_footer(lang=reply_lang)
            decision = {
                **decision,
                "outcome": "INFORMATIONAL",
                "reason": "Policy answer from knowledge base.",
                "rules_applied": list(set(list(decision.get("rules_applied") or []) + ["HR_POLICY_RAG"])),
            }
        elif intent == INTENT_HR_POLICY or is_policy_kb_query(message) or is_rules_query(message):
            crm_payload["rules_answer"] = hr_policy_not_found_message()
            decision = {
                **decision,
                "outcome": "INFORMATIONAL",
                "reason": crm_payload["rules_answer"],
                "rules_applied": list(
                    set(list(decision.get("rules_applied") or []) + ["HR_POLICY_RAG_NOT_FOUND"])
                ),
            }
        msg, resp_status = build_user_message(
            intent=intent,
            entities=entities,
            decision=decision,
            crm_payload=crm_payload,
        )
        return msg, resp_status, decision, crm_payload, request_id

    def _finalize(
        self,
        session,
        user_message: str,
        assistant_message: str,
        trace_id: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        self.memory.append(session, "user", user_message)
        self.memory.append(session, "assistant", assistant_message)
        envelope["trace_id"] = trace_id
        envelope["_session_id"] = session.session_id
        log_step(trace_id, "chat_complete", {"intent": envelope.get("intent")})
        return envelope

    @staticmethod
    def _should_create_crm_request(intent: str, decision: dict[str, Any]) -> bool:
        if intent in (INTENT_HR_POLICY, INTENT_REQUEST_STATUS, INTENT_UNKNOWN):
            return False
        return decision.get("outcome") in ("PENDING_APPROVAL", "PENDING_REVIEW", "SUBMITTED")

    @staticmethod
    def _assistant_text_for_translation(
        context_lines: list[str],
        *,
        target_lang: str,
    ) -> str | None:
        for line in reversed(context_lines or []):
            if line.startswith("Assistant:"):
                return line[len("Assistant:") :].strip()
        return None
