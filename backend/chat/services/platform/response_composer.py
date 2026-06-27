"""Compose user-facing messages for the workflow platform (Phase 8).

Summary vs review — when to use which:
- ``format_expense_summary`` / ``format_leave_summary`` (``summary.py``):
  Read-only draft snapshots for **review/summary turns**, session context, and
  modify/delete pickers. Includes status, line items, totals (expense), and
  missing-field hints. Does **not** include the submit CTA footer.

- ``FieldEngine.build_review`` (``field_engine.py``):
  Formal **pre-submit review** block built from workflow field definitions.
  Always ends with "_Reply **yes** to submit, or tell me what to change._"
  Used by ``ReviewEngine.prepare_review`` immediately before confirmation.

- ``ResponseComposer`` methods below:
  Pipeline and orchestrator should call these instead of inline strings. Use
  ``workflow_summary()`` for informational summary turns;
  ``review_ready_message()`` / ``submit_confirm()`` when arming submit confirmation;
  ``general_help()`` / ``platform_continue_clarify()`` for orchestrator fallbacks;
  ``workflow_continuation_hint()`` when appending draft-continuation footers.

- ``informational_responses`` (Phase 8/11):
  Status lookup, policy RAG, ``policy_rules_footer()``, and ``build_user_message()``.
  Pipeline calls these **only** through ``ResponseComposer`` facade methods below.
"""

from __future__ import annotations

from typing import Any

from chat.services.informational_responses import (
    compose_policy_turn,
    policy_rules_footer,
    resolve_request_status_turn,
)
from chat.services.platform.schemas import UnderstandingAction, UnderstandingResult, WorkflowDefinition
from chat.services.platform.summary import (
    expense_total,
    format_expense_summary,
    format_session_context,
)
from chat.services.platform.field_extractors import format_iso_date_display
from chat.services.session_memory import PendingQuestion, SessionMemory, WorkflowDraft


def normalize_reply_lang(lang: str) -> str:
    """Normalize reply language to ``en``, ``bn``, or ``banglish``."""
    low = (lang or "en").strip().lower()
    if low in ("bn", "bang", "bangla", "bengali"):
        return "bn"
    if low == "banglish":
        return "banglish"
    return "en"


def localized(lang: str, *, en: str, bn: str, banglish: str | None = None) -> str:
    """Pick EN / BN script / Banglish copy."""
    bucket = normalize_reply_lang(lang)
    if bucket == "banglish":
        return banglish or bn
    if bucket == "bn":
        return bn
    return en


def _pick_copy(entry: dict[str, str], *, lang: str, fallback_en: str) -> str:
    bucket = normalize_reply_lang(lang)
    if bucket == "banglish":
        return entry.get("banglish") or entry.get("bn") or entry.get("en") or fallback_en
    if bucket == "bn":
        return entry.get("bn") or entry.get("banglish") or entry.get("en") or fallback_en
    return entry.get("en") or fallback_en


# Phase 5 — centralized leave user-facing copy (EN + BN + Banglish SSOT).
LEAVE_FIELD_PROMPTS: dict[str, dict[str, str]] = {
    "leave_type": {
        "en": "What type of leave would you like? Choose **annual**, **sick**, or **lwop**.",
        "bn": "কোন ধরনের ছুটি চান? **annual**, **sick**, বা **lwop** বলুন।",
        "banglish": "Kon dhoroner chuti chan? **annual**, **sick**, ba **lwop** bolen.",
    },
    "day_scope": {
        "en": "Is this a full day or half day?",
        "bn": "পুরো দিন নাকি অর্ধ দিন?",
        "banglish": "Puro din naki ordho din?",
    },
    "half_day_period": {
        "en": "Which half — **morning** or **afternoon**?",
        "bn": "কোন অর্ধ — **সকাল** নাকি **বিকেল**?",
        "banglish": "Kon ordho — **morning** naki **afternoon**?",
    },
    "start_date": {
        "en": "When does your leave start?",
        "bn": "ছুটি কখন থেকে শুরু?",
        "banglish": "Chuti kokhon theke shuru?",
    },
    "end_date": {
        "en": "When does your leave end? (optional for single-day leave)",
        "bn": "ছুটি কখন শেষ? (এক দিনের ছুটির জন্য ঐচ্ছিক)",
        "banglish": "Chuti kokhon shesh? (ek diner chutir jonno optional)",
    },
    "reason": {
        "en": "Would you like to share a reason? (optional — reply **skip** to continue)",
        "bn": "কারণ জানাতে চান? (ঐচ্ছিক — **skip** বললে এড়িয়ে যাব)",
        "banglish": "Karon janate chan? (optional — **skip** bollen continue korar jonno)",
    },
    "medical_document": {
        "en": "For sick leave of 3+ days, you may upload a medical document now (optional — reply **skip** or say you will provide it later).",
        "bn": "৩+ দিন sick leave-এ medical document দিতে পারেন (ঐচ্ছিক — **skip** বলুন বা পরে দেবেন বলুন)।",
        "banglish": "3+ din sick leave e medical document dite paren (optional — **skip** bolen ba pore diben bolen).",
    },
}

LEAVE_VALIDATION_MESSAGES: dict[str, dict[str, str]] = {
    "leave_type_required": {
        "en": "Leave type is required.",
        "bn": "ছুটির ধরন প্রয়োজন।",
        "banglish": "Chutir dhoron proyojon.",
    },
    "day_scope_required": {
        "en": "Please specify full day or half day.",
        "bn": "পুরো দিন নাকি অর্ধ দিন জানান।",
        "banglish": "Puro din naki ordho din janan.",
    },
    "half_day_period_conditional": {
        "en": "Half-day leave requires morning or afternoon.",
        "bn": "অর্ধ দিনের ছুটিতে সকাল/বিকেল জানান।",
        "banglish": "Ordho diner chutite morning/afternoon janan.",
    },
    "start_date_required": {
        "en": "Start date is required.",
        "bn": "শুরুর তারিখ প্রয়োজন।",
        "banglish": "Shurur tarikh proyojon.",
    },
    "end_date_gte_start": {
        "en": "End date must be on or after start date.",
        "bn": "শেষ তারিখ শুরুর তারিখের পরে হতে হবে।",
        "banglish": "Shesh tarikh shurur tarikher pore hote hobe.",
    },
    "medical_document_sick": {
        "en": "Sick leave of 3+ days requires a medical document.",
        "bn": "৩+ দিন sick leave-এ medical document প্রয়োজন।",
        "banglish": "3+ din sick leave e medical document proyojon.",
    },
}

LEAVE_FIELD_LABELS: dict[str, dict[str, str]] = {
    "leave_type": {"en": "leave type", "bn": "ছুটির ধরন", "banglish": "chutir dhoron"},
    "day_scope": {"en": "day scope", "bn": "দিনের ধরন", "banglish": "diner dhoron"},
    "half_day_period": {"en": "half day period", "bn": "অর্ধ দিনের সময়", "banglish": "ordho diner shomoy"},
    "start_date": {"en": "start date", "bn": "শুরুর তারিখ", "banglish": "shuru tarikh"},
    "end_date": {"en": "end date", "bn": "শেষ তারিখ", "banglish": "shesh tarikh"},
    "reason": {"en": "reason", "bn": "কারণ", "banglish": "karon"},
    "medical_document": {"en": "medical document", "bn": "medical document", "banglish": "medical document"},
}


def leave_field_prompt(field: str, *, lang: str = "en") -> str:
    entry = LEAVE_FIELD_PROMPTS.get(field, {})
    fallback = f"Please provide {field.replace('_', ' ')}."
    if normalize_reply_lang(lang) == "bn":
        fallback = f"অনুগ্রহ করে {field.replace('_', ' ')} দিন।"
    elif normalize_reply_lang(lang) == "banglish":
        fallback = f"Onugroho kore {field.replace('_', ' ')} din."
    return _pick_copy(entry, lang=lang, fallback_en=fallback)


def leave_validation_message(rule_id: str, *, lang: str = "en") -> str:
    entry = LEAVE_VALIDATION_MESSAGES.get(rule_id, {})
    return _pick_copy(
        entry,
        lang=lang,
        fallback_en="Please check the provided information.",
    ) if entry else localized(
        lang,
        en="Please check the provided information.",
        bn="অনুগ্রহ করে তথ্য যাচাই করুন।",
        banglish="Onugroho kore tothyo jachai korun.",
    )


def leave_field_label(field: str, *, lang: str = "en") -> str:
    entry = LEAVE_FIELD_LABELS.get(field, {})
    return _pick_copy(entry, lang=lang, fallback_en=field.replace("_", " "))


class ResponseComposer:
    def contextual_process_response(
        self,
        memory: SessionMemory,
        *,
        lang: str = "en",
    ) -> str:
        pq = memory.pending_question
        draft = memory.active_draft()
        field = pq.field if pq else ""
        if draft and memory.active_workflow and memory.active_workflow.id == "leave":
            summary = self.leave_summary(draft, lang=lang, include_status=True)
            screen = self.workflow_screen_context(memory, lang=lang)
            if lang == "bn":
                lead = (
                    "আপনার **বর্তমান leave draft** এখানে। "
                    "Submit করতে **yes** বলুন, বদলাতে field বলুন, বাতিল করতে **cancel**।"
                )
            elif normalize_reply_lang(lang) == "banglish":
                lead = (
                    "Apnar **current leave draft** ekhane. "
                    "Submit korte **yes** bolen, change korte field bolen, batil korte **cancel**."
                )
            else:
                lead = (
                    "Here is your **current leave draft**. "
                    "Say **yes** to submit, tell me what to change, or **cancel** to discard."
                )
            if screen:
                lead = f"{screen}\n\n{lead}"
            if field:
                label = leave_field_label(field, lang=lang) if field in LEAVE_FIELD_LABELS else field
                if lang == "bn":
                    lead = f"আমি **{label}** জিজ্ঞেস করেছিলাম। {lead}"
                else:
                    lead = f"I was asking for **{label}**. {lead}"
            return f"{lead}\n\n{summary}"
        return self.contextual_pending_clarify(memory, lang=lang)

    def contextual_pending_clarify(
        self,
        memory: SessionMemory,
        *,
        lang: str = "en",
    ) -> str:
        pq = memory.pending_question
        aw = memory.active_workflow
        field = pq.field if pq else "information"
        label = leave_field_label(field, lang=lang) if field in LEAVE_FIELD_LABELS else field.replace("_", " ")
        wf = aw.id if aw else "workflow"
        if aw and aw.id == "leave" and pq:
            from chat.services.platform.field_extractors.leave import _llm_client_configured

            hint = ""
            if not _llm_client_configured():
                hint = localized(
                    lang,
                    en="\n\n_Tip: short answers work — e.g. **annual**, **6 August**, **family program**._",
                    bn="\n\n_সংক্ষিপ্ত উত্তর দিন — যেমন **annual**, **6 August**, **family program**।_",
                    banglish="\n\n_Songkhipto uttor din — jemon **annual**, **6 August**, **family program**._",
                )
            if lang == "bn":
                return (
                    f"আপনি কি **{wf}** request-এর **summary** দেখতে চান, "
                    f"নাকি **{label}** বলছেন? Summary চাইলে **summary** বলুন।{hint}"
                )
            return (
                f"Do you want the **{wf}** **summary**, or are you answering **{label}**? "
                f"Say **summary** to see your current draft.{hint}"
            )
        if lang == "bn":
            return (
                f"আপনি কি **{wf}** request-এর **summary** দেখতে চান, "
                f"নাকি **{label}** বলছেন? Summary চাইলে **summary** বলুন।"
            )
        return (
            f"Do you want the **{wf}** **summary**, or are you answering **{label}**? "
            f"Say **summary** to see your current draft."
        )

    def contextual_meta_response(
        self,
        memory: SessionMemory,
        *,
        lang: str = "en",
    ) -> str:
        draft = memory.active_draft()
        aw = memory.active_workflow
        if aw and aw.id == "leave" and draft:
            summary = self.leave_summary(draft, lang=lang, include_status=True)
            if lang == "bn":
                lead = (
                    "বুঝতে পারছি — আপনি bot-এর আচরণ নিয়ে প্রশ্ন করছেন। "
                    "আপনার **বর্তমান leave draft** এখানে:"
                )
            else:
                lead = (
                    "I understand — you're asking about how I'm handling your request. "
                    "Here is your **current leave draft**:"
                )
            if lang == "bn":
                tail = "_যা বদলাতে চান (reason, date, type) স্পষ্ট করে বলুন।_"
            else:
                tail = "_Tell me clearly what to change (reason, date, type)._"
            return f"{lead}\n\n{summary}\n\n{tail}"
        if aw and aw.id == "expense" and draft:
            return self.expense_frustration_reply(memory, message="", lang=lang)
        return self.contextual_pending_clarify(memory, lang=lang)

    def expense_frustration_reply(
        self,
        memory: SessionMemory,
        *,
        message: str = "",
        lang: str = "en",
    ) -> str:
        """Phase 4 — empathetic expense reply instead of robotic slot/summary clarify."""
        from chat.services.platform.field_extractors.expense import (
            build_pending_queue,
            expense_focus_prompt,
            is_expense_anti_summary_request,
            sync_expense_draft,
        )
        from chat.services.platform.summary import format_expense_collect_recap

        draft = memory.active_draft()
        if not draft:
            return self.contextual_pending_clarify(memory, lang=lang)
        sync_expense_draft(draft)
        if is_expense_anti_summary_request(message):
            lead = localized(
                lang,
                en="Got it — I won't show the full summary unless you ask. Here's where we are:",
                bn="বুঝেছি — আপনি না বললে পুরো summary দেখাব না। এখন যা আছে:",
                banglish="Bujhechi — apni na bole pura summary dekhbo na. Ekhon ja ache:",
            )
        else:
            lead = localized(
                lang,
                en="Sorry for the confusion — here's your **current expense draft**:",
                bn="গোলমালের জন্য দুঃখিত — আপনার **বর্তমান expense draft**:",
                banglish="Golmal er jonno duhkhito — apnar **bortoman expense draft**:",
            )
        focus_q = None
        queue = build_pending_queue(list(draft.fields.get("items") or []))
        if queue:
            focus_q = expense_focus_prompt(queue[0], lang=lang)
        recap = format_expense_collect_recap(
            draft,
            lang=lang,
            include_focus_question=focus_q,
        )
        tail = localized(
            lang,
            en="_Tell me what to fix (route, category, remove a line) — I'll apply it._",
            bn="_কী ঠিক করবেন বলুন (route, category, কোনো লাইন বাদ) — আমি apply করব।_",
            banglish="_Ki thik korben bolen (route, category, kono line bad) — ami apply korbo._",
        )
        return "\n\n".join(p for p in [lead, recap, tail] if p).strip()

    def expense_wizard_fallback_notice(self, *, lang: str = "en", llm_degraded: bool = False) -> str:
        """Phase C — LLM limit hit but wizard seeded items from the message."""
        if llm_degraded:
            lead = localized(
                lang,
                en=(
                    "**AI limit reached** — I added what I could from your message. "
                    "I'll ask for any missing category or route next."
                ),
                bn=(
                    "**AI limit শেষ** — message থেকে যতটা সম্ভব যোগ করেছি। "
                    "বাকি category বা route লাগলে জিজ্ঞেস করব।"
                ),
                banglish=(
                    "**AI limit sesh** — message theke joto somvob jog korechi. "
                    "Baki category ba route lagle jigges korbo."
                ),
            )
        else:
            lead = localized(
                lang,
                en="I split your message into expense lines — I'll ask for anything still missing.",
                bn="আপনার message থেকে খরচের লাইন বানিয়েছি — যা বাকি আছে জিজ্ঞেস করব।",
                banglish="Apnar message theke khoroch er line baniyechi — ja baki ache jigges korbo.",
            )
        return lead

    def expense_llm_unavailable(self, *, lang: str = "en", for_edit: bool = False) -> str:
        """When expense LLM failed (rate limit / API busy) — honest user-facing message."""
        if for_edit:
            return localized(
                lang,
                en=(
                    "**AI couldn't process that edit right now** — the language model is "
                    "unavailable (often API rate limit on the free tier). "
                    "Wait 1–2 minutes and try again, or send a short line like "
                    "`lunch 120 taka` or `bike route mirpur to badda`."
                ),
                bn=(
                    "**AI এখন এই edit বুঝতে পারছে না** — language model কাজ করছে না "
                    "(free tier-এ API limit হলে হয়)। "
                    "১–২ মিনিট পর আবার চেষ্টা করুন, অথবা ছোট করে লিখুন — "
                    "যেমন `lunch 120 taka` বা `bike route mirpur to badda`।"
                ),
                banglish=(
                    "**AI ekhon ei edit bujhte parche na** — language model off "
                    "(free tier e API limit hole hoy). "
                    "1–2 minute wait kore abar try korun, ba choto kore likhun — "
                    "jemon `lunch 120 taka` ba `bike route mirpur to badda`."
                ),
            )
        return localized(
            lang,
            en=(
                "**AI is unavailable right now** — API limit reached or the model is busy. "
                "On the free tier, long messages often hit the limit. "
                "Wait 1–2 minutes and try again, or send one expense per line."
            ),
            bn=(
                "**AI এখন কাজ করছে না** — API limit শেষ বা model busy। "
                "Free tier-এ বড় message পাঠালে limit চলে যায়। "
                "১–২ মিনিট পর আবার চেষ্টা করুন, অথবা এক লাইনে একটা খরচ লিখুন।"
            ),
            banglish=(
                "**AI ekhon kaj korche na** — API limit sesh ba model busy. "
                "Free tier e boro message pathale limit chole jay. "
                "1–2 minute pore abar try korun, ba ek line e ekta khoroch likhun."
            ),
        )

    def expense_edit_clarify(
        self,
        memory: SessionMemory,
        turn: dict[str, Any],
        *,
        lang: str = "en",
    ) -> str:
        """Ask which draft line to edit — built from live draft, not static phrases."""
        from chat.services.platform.field_extractors.expense import (
            expense_item_label,
            sync_expense_draft,
        )

        draft = memory.active_draft()
        if not draft:
            return self.contextual_pending_clarify(memory, lang=lang)
        sync_expense_draft(draft)
        items = list(draft.fields.get("items") or [])
        intent = str(turn.get("intent") or "clarify_modify").lower()
        clarify = dict(turn.get("clarify") or {})
        indices = [
            int(i)
            for i in (clarify.get("candidate_indices") or [])
            if isinstance(i, (int, float)) or str(i).isdigit()
        ]
        indices = [i for i in indices if 0 <= i < len(items)]
        proposed = clarify.get("proposed_value")
        try:
            proposed_amt = float(proposed) if proposed is not None else None
        except (TypeError, ValueError):
            proposed_amt = None
        cat = str(clarify.get("category") or "").strip()

        if intent == "clarify_delete":
            lead = localized(
                lang,
                en="**Which expense should I delete?** Reply with the entry number (1, 2, 5…).",
                bn="**কোন expense delete করব?** entry নম্বর (1, 2, 5…) লিখুন।",
                banglish="**Kon expense delete korbo?** entry number (1, 2, 5…) likhun.",
            )
            raw_indices = list(clarify.get("candidate_indices") or turn.get("candidate_indices") or [])
            if not raw_indices and cat:
                from chat.services.platform.field_extractors.expense import normalize_expense_category

                norm_cat = normalize_expense_category(cat)
                raw_indices = [
                    idx
                    for idx, item in enumerate(items)
                    if normalize_expense_category(item.get("category")) == norm_cat
                ]
            indices = [
                int(i)
                for i in raw_indices
                if isinstance(i, (int, float)) or str(i).isdigit()
            ]
            indices = [i for i in indices if 0 <= i < len(items)]
            if not indices:
                indices = list(range(len(items)))
            numbered = [
                f"{pos + 1}. {expense_item_label(items[idx], index=idx)}"
                for pos, idx in enumerate(indices)
            ]
            return f"{lead}\n\n" + "\n".join(numbered)

        if indices:
            lines = [f"{pos + 1}. {expense_item_label(items[idx], index=idx)}" for pos, idx in enumerate(indices)]
            target_field = str(clarify.get("target_field") or "").strip().lower()
            match_amount = clarify.get("match_amount")
            try:
                match_amt = float(match_amount) if match_amount is not None else None
            except (TypeError, ValueError):
                match_amt = None
            operation = str(clarify.get("operation") or "").strip().lower()
            if intent == "clarify_delete" or operation == "delete":
                if cat:
                    lead = localized(
                        lang,
                        en=f"You have **{len(indices)} {cat}** expenses. Which one should I delete?",
                        bn=f"**{len(indices)} টা {cat}** expense আছে। কোনটা delete করব?",
                        banglish=f"**{len(indices)} ta {cat}** expense ache. Konta delete korbo?",
                    )
                else:
                    lead = localized(
                        lang,
                        en=f"**Which expense should I delete?** ({len(indices)} matches)",
                        bn=f"**কোন expense delete করব?** ({len(indices)} টা মিলেছে)",
                        banglish=f"**Kon expense delete korbo?** ({len(indices)} ta mileche)",
                    )
            elif target_field == "route":
                if cat:
                    lead = localized(
                        lang,
                        en=f"You have **{len(indices)} {cat}** expenses. Which route should I update?",
                        bn=f"আপনার draft-এ **{len(indices)} টা {cat}** expense আছে। কোনটা update করতে চান?",
                        banglish=f"Apnar draft-e **{len(indices)} ta {cat}** expense ache. Konta update korte chan?",
                    )
                else:
                    lead = localized(
                        lang,
                        en="**Which expense route should I update?** Reply with the entry number.",
                        bn="**কোন expense-এর route** update করব? entry নম্বর লিখুন।",
                        banglish="**Kon expense er route** update korbo? entry number likhun.",
                    )
            elif match_amt is not None and not cat:
                lead = localized(
                    lang,
                    en=f"You have **{len(indices)} expenses** at **{match_amt:.0f} taka**. Which one should I update?",
                    bn=f"**{match_amt:.0f} taka**-র **{len(indices)} টা** expense আছে। কোনটা update করতে চান?",
                    banglish=f"**{match_amt:.0f} taka** er **{len(indices)} ta** expense ache. Konta update korte chan?",
                )
            elif proposed_amt is not None and cat:
                lead = localized(
                    lang,
                    en=f"You have **{len(indices)} {cat}** expenses. Which one should be **{proposed_amt:.0f} taka**?",
                    bn=f"**{len(indices)} টা {cat}** expense আছে। কোনটা **{proposed_amt:.0f} taka** হবে?",
                    banglish=f"**{len(indices)} ta {cat}** expense ache. Konta **{proposed_amt:.0f} taka** hobe?",
                )
            elif proposed_amt is not None:
                lead = localized(
                    lang,
                    en=f"Which expense should be **{proposed_amt:.0f} taka**? Reply with the number.",
                    bn=f"কোন expense **{proposed_amt:.0f} taka** হবে? নম্বর লিখুন।",
                    banglish=f"Kon expense **{proposed_amt:.0f} taka** hobe? Number likhun.",
                )
            else:
                lead = localized(
                    lang,
                    en="**Which expense do you want to change?** Reply with the entry number.",
                    bn="**কোন expense বদলাবেন?** entry নম্বর লিখুন।",
                    banglish="**Kon expense badlaben?** entry number likhun.",
                )
            return f"{lead}\n\n" + "\n".join(lines)

        lead = localized(
            lang,
            en="Tell me **which expense** to change — use the entry number (1, 2, 5…) and what to update (amount, route, category).",
            bn="**কোন expense** বদলাবেন বলুন — entry নম্বর (1, 2, 5…) এবং কী বদলাবেন (amount, route, category)।",
            banglish="**Kon expense** badlaben bolen — entry number (1, 2, 5…) ar ki badlaben (amount, route, category).",
        )
        from chat.services.platform.summary import format_expense_collect_recap

        recap = format_expense_collect_recap(draft, lang=lang, include_focus_question=False)
        return f"{lead}\n\n{recap}".strip()

    def expense_repair_ack(
        self,
        memory: SessionMemory,
        *,
        message: str,
        lang: str = "en",
        repaired: bool = True,
    ) -> str:
        lead = localized(
            lang,
            en="Sorry about that — I removed the mistaken duplicate." if repaired else "I understand something went wrong.",
            bn="দুঃখিত — ভুল duplicate সরিয়ে দিয়েছি।" if repaired else "বুঝতে পারছি কিছু একটা ভুল হয়েছে।",
            banglish="Dukkhito — bhul duplicate soriye diyechi." if repaired else "Bujhte parchi kichu ekhta bhul hoyeche.",
        )
        body = self.expense_frustration_reply(memory, message=message, lang=lang)
        return f"{lead}\n\n{body}".strip() if repaired else body

    def expense_past_date_policy(self, *, lang: str = "en") -> str:
        return self.expense_date_policy_blocked(lang=lang)

    def expense_replay_blocked_ack(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en="Got it — adding those as **today's** expenses.",
            bn="ঠিক আছে — **আজকের** হিসেবে add করছি।",
            banglish="Thik ache — **ajker** hishebe add korchi.",
        )

    def expense_date_policy_blocked(
        self,
        *,
        lang: str = "en",
        requested_date: str | None = None,
    ) -> str:
        from datetime import date

        today = date.today().isoformat()
        is_future = bool(
            requested_date
            and requested_date > today
        )
        if is_future:
            return localized(
                lang,
                en=(
                    "I can only add **today's** expenses here. "
                    "Future dates aren't supported — please share today's items."
                ),
                bn=(
                    "আমি এখানে শুধু **আজকের** খরচ add করতে পারি। "
                    "ভবিষ্যত তারিখ গ্রহণযোগ্য নয় — আজকের item গুলো লিখুন।"
                ),
                banglish=(
                    "Ami ekhane shudhu **ajker** khoroch add korte pari. "
                    "Porer diner kharcha ekhane add kora jay na — ajker item gulo likhun."
                ),
            )
        return localized(
            lang,
            en=(
                "I can only add **today's** expenses here. "
                "Yesterday or older dates aren't supported — please share today's items."
            ),
            bn=(
                "আমি এখানে শুধু **আজকের** খরচ add করতে পারি। "
                "গতকাল বা আগের তারিখ গ্রহণযোগ্য নয় — আজকের item গুলো লিখুন।"
            ),
            banglish=(
                "Ami ekhane shudhu **ajker** khoroch add korte pari. "
                "Kalke ba ager diner kharcha ekhane add kora jay na — ajker item gulo likhun."
            ),
        )

    def expense_already_recorded(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en=(
                "These expenses are already on your draft — nothing new was added. "
                "Share **route** details or any **new items** if you have them."
            ),
            bn=(
                "এই খরচগুলো ইতিমধ্যে draft-এ আছে — আর কিছু add হয়নি। "
                "**Route** বা **নতুন item** থাকলে লিখুন।"
            ),
            banglish=(
                "Ei kharcha gulo already draft e ache — ar kichu add hoyni. "
                "**Route** ba **notun item** thakle likhun."
            ),
        )

    def contextual_review_modify_clarify(
        self,
        memory: SessionMemory,
        *,
        lang: str = "en",
        message: str = "",
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> str:
        if message.strip():
            natural = self.leave_review_natural_reply(
                message,
                memory,
                intent="unclear",
                lang=lang,
                trace_id=trace_id,
                conversation_history=conversation_history or [],
            )
            if natural:
                return natural
        draft = memory.active_draft()
        if not draft:
            return self.contextual_pending_clarify(memory, lang=lang)
        summary = self.leave_summary(draft, lang=lang, include_status=False)
        lead = localized(
            lang,
            en="You're on the **leave review** screen. Tell me what to change — **leave type**, **start date**, **end date**, or **reason** — or say **yes** to submit.",
            bn="আপনি **leave review** screen-এ আছেন। কী বদলাবেন বলুন — **leave type**, **start date**, **end date**, **reason** — অথবা submit করতে **ha** বলুন।",
            banglish="Apni **leave review** screen e achen. Ki badlaben bolen — **leave type**, **start date**, **end date**, **reason** — ba submit korte **ha** bolen.",
        )
        return f"{lead}\n\n{summary}"

    def leave_review_natural_reply(
        self,
        message: str,
        memory: SessionMemory,
        *,
        intent: str = "unclear",
        lang: str = "en",
        trace_id: str = "",
        conversation_history: list[str] | None = None,
    ) -> str:
        """LLM-natural reply for review questions / unclear modify — not static copy."""
        from chat.services.platform.field_extractors.leave import _llm_client_configured
        from chat.services.platform.summary import format_leave_summary

        draft = memory.active_draft()
        if not draft:
            return self.contextual_pending_clarify(memory, lang=lang)

        summary = format_leave_summary(draft, lang=lang, include_status=False)
        if not _llm_client_configured():
            if intent == "question":
                return localized(
                    lang,
                    en=(
                        "I hear you — here's what's in your leave draft right now. "
                        "Tell me clearly what to change (dates, reason, or leave type), or say **yes** to submit."
                    ),
                    bn=(
                        "বুঝতে পারছি — এখন আপনার leave draft-এ যা আছে তা নিচে। "
                        "কী বদলাতে চান (তারিখ, reason, leave type) স্পষ্ট করে বলুন, অথবা submit করতে **ha** বলুন।"
                    ),
                    banglish=(
                        "Bujhte parchi — ekhon apnar leave draft e ja ache ta niche. "
                        "Ki badlate chan (tarikh, reason, leave type) spostho kore bolen, ba submit korte **ha** bolen."
                    ),
                )
            return localized(
                lang,
                en="I'm not sure what you'd like to change. Could you say which field — **start date**, **end date**, **reason**, or **leave type**?",
                bn="ঠিক বুঝতে পারছি না কী বদলাতে চান। **start date**, **end date**, **reason**, নাকি **leave type**?",
                banglish="Thik bujhte parchi na ki badlate chan. **start date**, **end date**, **reason**, na **leave type**?",
            )

        hint = (
            "ACTIVE WORKFLOW: leave — user is on the REVIEW / submit-confirmation screen.\n"
            f"Detected intent: {intent}\n"
            f"Current draft summary:\n{summary}\n\n"
            "Reply naturally in the user's language (Bangla/Banglish/English).\n"
            "- If intent=question: explain what is currently in the draft; if they complain something is missing "
            "(e.g. end date or day count), point it out honestly and ask what to fix.\n"
            "- If intent=unclear: politely say you did not fully understand; ask which field to change.\n"
            "- Do NOT claim you updated any field unless a change was just applied.\n"
            "- Keep it short (2-4 sentences). No JSON. No internal reasoning."
        )
        reply = self.conversational(
            message,
            context_lines=conversation_history or [],
            trace_id=trace_id,
            workflow_hint=hint,
        )
        if (reply or "").strip():
            return reply.strip()
        return localized(
            lang,
            en="I'm not sure what you'd like to change — could you tell me which field to update?",
            bn="ঠিক বুঝতে পারছি না — কোন field বদলাতে চান বলবেন?",
            banglish="Thik bujhte parchi na — kon field badlate chan bolben?",
        )

    def family_care_sick_label_prompt(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en=(
                "Sick leave is for when **you** are unwell. Caring for a family member "
                "is usually **annual** or **lwop** — which one do you want?"
            ),
            bn=(
                "**Sick leave** শুধু নিজের অসুস্থতার জন্য। পরিবারের সদস্যের যত্নের জন্য সাধারণত "
                "**annual** বা **lwop** — কোনটা চান?"
            ),
            banglish=(
                "**Sick leave** shudhu nijer osusthotar jonno. Paribarer member er jonyo "
                "shadharon **annual** ba **lwop** — kon ta chan?"
            ),
        )

    def unrecognized_leave_type_prompt(self, mentioned: str, *, lang: str = "en") -> str:
        label = (mentioned or "that").replace("_", " ").strip()
        return localized(
            lang,
            en=(
                f"**{label}** isn't a leave type in our system. "
                "We support **annual**, **sick**, and **lwop** — which one do you want?"
            ),
            bn=(
                f"আমাদের system-এ **{label}** leave type নেই। "
                "**annual**, **sick**, **lwop** — কোনটা চান?"
            ),
            banglish=(
                f"Amader system e **{label}** leave type nei. "
                "**annual**, **sick**, **lwop** — kon ta chan?"
            ),
        )

    def clarification(
        self,
        understanding: UnderstandingResult,
        *,
        lang: str = "en",
        draft: WorkflowDraft | None = None,
        memory: SessionMemory | None = None,
    ) -> str:
        if memory and (understanding.entities or {}).get("meta_complaint"):
            if memory.active_workflow and memory.active_workflow.id == "expense":
                return self.expense_frustration_reply(
                    memory,
                    message="",
                    lang=lang,
                )
            return self.contextual_meta_response(memory, lang=lang)
        expense_intent = str((understanding.entities or {}).get("expense_intent") or "").lower()
        if (
            memory
            and memory.active_workflow
            and memory.active_workflow.id == "expense"
            and expense_intent in ("clarify_delete", "clarify_modify")
        ):
            turn = dict((understanding.entities or {}).get("expense_turn") or {})
            turn.setdefault("intent", expense_intent)
            return self.expense_edit_clarify(memory, turn, lang=lang)
        if memory and (understanding.entities or {}).get("process_question"):
            return self.contextual_process_response(memory, lang=lang)

        from chat.services.platform.field_extractors.leave import is_leave_review_mode

        reason = understanding.reasoning or ""
        if memory and is_leave_review_mode(memory):
            if understanding.action == UnderstandingAction.CLARIFICATION_NEEDED.value:
                if "leave field" in reason.lower():
                    return self.contextual_review_modify_clarify(memory, lang=lang)
                return self.contextual_process_response(memory, lang=lang)
            if understanding.workflow == "leave":
                return self.contextual_process_response(memory, lang=lang)

        if (
            memory
            and memory.pending_question
            and memory.active_workflow
            and memory.active_workflow.id == "expense"
            and understanding.answers_pending_field is False
            and understanding.action == UnderstandingAction.CLARIFICATION_NEEDED.value
            and expense_intent
            not in (
                "clarify_delete",
                "clarify_modify",
                "date_correction",
                "replay_blocked_add",
            )
        ):
            return self.expense_frustration_reply(memory, message="", lang=lang)
        if (
            memory
            and memory.pending_question
            and memory.active_workflow
            and understanding.answers_pending_field is False
            and understanding.action == UnderstandingAction.CLARIFICATION_NEEDED.value
        ):
            return self.contextual_pending_clarify(memory, lang=lang)
        if (
            memory
            and memory.active_workflow
            and understanding.action == UnderstandingAction.CLARIFICATION_NEEDED.value
            and not (understanding.entities or {}).get("leave_start_clarify")
        ):
            return self.contextual_pending_clarify(memory, lang=lang)

        if "open leave" in reason.lower() or "already exists" in reason.lower() or "already in progress" in reason.lower():
            if lang == "bn":
                return (
                    "আপনার একটি **leave request** ইতিমধ্যে open আছে।\n\n"
                    "আগে এটি **submit** করুন বা **cancel** করুন, তারপর নতুন leave নিন।"
                )
            return (
                "You already have an **open leave request**.\n\n"
                "Please **submit** or **cancel** it first, then start a new leave request."
            )
        if "which item" in reason.lower() or "which entry" in reason.lower() or "multiple amounts" in reason.lower():
            if draft and draft.workflow_id == "expense":
                summary = format_expense_summary(draft, lang=lang)
                if lang == "bn":
                    return f"{summary}\n\n**কোন entry delete/modify করব?** নম্বর বা বিবরণ লিখুন (যেমন: lunch, 1, prothom ta)।"
                return f"{summary}\n\n**Which entry should I update/delete?** Tell me the number or description (e.g. lunch, 1, first one)."
            if lang == "bn":
                return "কোন amount বদলাব? lunch, bus, নম্বর (1/2) বা prothom ta লিখুন।"
            return "Which amount should I change? Say **lunch**, **bus**, item **number (1/2)**, or **first one**."

        wf = understanding.workflow
        if wf == "leave":
            if memory and memory.active_workflow and memory.active_workflow.id == "leave":
                from chat.services.platform.field_engine import leave_draft_in_progress

                draft = memory.active_draft()
                if leave_draft_in_progress(draft):
                    return self.contextual_pending_clarify(memory, lang=lang)
            return localized(
                lang,
                en="I understand you may need time away from work. Would you like to submit a **leave request**?",
                bn="মনে হচ্ছে আপনি ছুটি নিতে চান। **Leave request** শুরু করব?",
                banglish="Mone hocche apni chuti nite chan. **Leave request** shuru korbo?",
            )
        if wf == "expense":
            if memory and memory.active_workflow and memory.active_workflow.id == "expense":
                draft = memory.active_draft()
                if draft and (
                    (draft.fields or {}).get("items")
                    or memory.pending_question
                ):
                    return self.contextual_pending_clarify(memory, lang=lang)
            return localized(
                lang,
                en="I think this may be an expense entry. Would you like to create an **expense claim**?",
                bn="মনে হচ্ছে এটি expense entry। **Expense claim** তৈরি করব?",
                banglish="Mone hocche eta expense entry. **Expense claim** toiri korbo?",
            )
        if understanding.goal == "Greeting" or "greeting" in reason.lower():
            return localized(
                lang,
                en="Hello! How can I help you today? I can assist with **leave**, **expense**, or **company policies**.",
                bn="হ্যালো! আমি আপনাকে কীভাবে সাহায্য করতে পারি? Leave, expense বা company policy নিয়ে জিজ্ঞাসা করতে পারেন।",
                banglish="Hello! Ami apnake kivabe sahajjo korte pari? Leave, expense ba company policy niye jigges korte paren.",
            )
        if reason and (
            "no workflow" in reason.lower()
            or "could not determine" in reason.lower()
            or "clarification" in reason.lower()
        ):
            return self.general_help(lang=lang)
        return reason or localized(lang, en="Could you clarify?", bn="আরও স্পষ্ট করে বলবেন?", banglish="Aro spostho kore bolben?")

    def field_saved(self, field: str, *, lang: str = "en", workflow_id: str = "") -> str:
        label = leave_field_label(field, lang=lang) if workflow_id == "leave" else field.replace("_", " ")
        return localized(
            lang,
            en=f"Saved **{label}**.",
            bn=f"**{label}** সংরক্ষণ করা হয়েছে।",
            banglish=f"**{label}** save kora hoyeche.",
        )

    def leave_reason_skipped(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en="Skipped **reason**.",
            bn="**কারণ** এড়িয়ে গেছে।",
            banglish="**Karon** eriye geche.",
        )

    def leave_review_title(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en="**Leave Request — Review**",
            bn="**ছুটি আবেদন — পর্যালোচনা**",
            banglish="**Chuti abedon — porjalochona**",
        )

    def leave_review_submit_cta(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en="_Reply **yes** to submit, or tell me what to change._",
            bn="_Submit করতে **ha** বলুন, অথবা যা বদলাতে চান তা বলুন।_",
            banglish="_Submit korte **ha** bolen, ba ja bodlate chan ta bolen._",
        )

    def leave_review(self, draft: WorkflowDraft, definition: WorkflowDefinition, *, lang: str = "en") -> str:
        from chat.services.platform.field_extractors.leave import LEAVE_INTERNAL_DRAFT_FIELDS
        from chat.services.platform.field_engine import FieldEngine

        lines = [self.leave_review_title(lang=lang), ""]

        def _fmt(name: str, val: Any) -> str:
            if name in ("start_date", "end_date") and val:
                return format_iso_date_display(str(val))
            if name == "day_scope" and val:
                return str(val).replace("_", " ")
            if name == "reason" and val:
                text = str(val).strip()
                return text if len(text) <= 120 else text[:117] + "..."
            return str(val)

        engine = FieldEngine()
        for f in definition.fields:
            if f.name in LEAVE_INTERNAL_DRAFT_FIELDS:
                continue
            if not engine.field_is_active(f, draft):
                continue
            val = draft.fields.get(f.name)
            if val not in (None, ""):
                lines.append(f"- **{leave_field_label(f.name, lang=lang)}**: {_fmt(f.name, val)}")
        lines.append("")
        lines.append(self.leave_review_submit_cta(lang=lang))
        return "\n".join(lines)

    def expense_review(self, draft: WorkflowDraft, definition: WorkflowDefinition, *, lang: str = "en") -> str:
        from chat.services.platform.field_extractors.expense import (
            category_display_name,
            is_travel_category,
            is_valid_expense_route,
            normalize_expense_category,
        )

        lines = [
            localized(
                lang,
                en="Please review your expenses before submission.",
                bn="Submit করার আগে আপনার expenses review করুন।",
                banglish="Submit korar age apnar expenses review korun.",
            ),
            "",
        ]
        items = list(draft.fields.get("items") or [])
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            cat = category_display_name(normalize_expense_category(item.get("category")) or "?")
            amt = item.get("amount", "?")
            line = f"- **{cat}** — {amt} taka"
            frm, to = item.get("from_location"), item.get("to_location")
            if is_travel_category(item.get("category")) and is_valid_expense_route(frm, to):
                line += f" ({frm} → {to})"
            lines.append(line)
        lines.append("")
        lines.append(
            localized(
                lang,
                en="Is everything correct? Would you like to submit?",
                bn="সব ঠিক আছে? Submit করতে চান?",
                banglish="Shob thik ache? Submit korte chan?",
            )
        )
        lines.append("")
        lines.append(self.leave_review_submit_cta(lang=lang))
        return "\n".join(lines)

    def unsupported_expense_category(self, category: str, *, lang: str = "en") -> str:
        return localized(
            lang,
            en=(
                f"Sorry, **{category}** is not a supported expense type. "
                "I can only accept: Lunch, Snack, Bus, Train, Bike, Metro Rail, Metro, Rickshaw."
            ),
            bn=(
                f"দুঃখিত, **{category}** supported expense type নয়। "
                "আমি শুধু Lunch, Snack, Bus, Train, Bike, Metro Rail, Metro, Rickshaw নিতে পারি।"
            ),
            banglish=(
                f"Dukkhito, **{category}** supported expense type noy. "
                "Ami shudhu Lunch, Snack, Bus, Train, Bike, Metro Rail, Metro, Rickshaw nite pari."
            ),
        )

    def leave_summary(self, draft: WorkflowDraft, *, lang: str = "en", include_status: bool = True) -> str:
        if normalize_reply_lang(lang) == "en":
            lines = ["**Leave Summary**", ""]
            status = "Submitted" if draft.locked else "Pending (not submitted)"
        else:
            lines = [localized(lang, en="**Leave Summary**", bn="**ছুটি সারাংশ**", banglish="**Chuti saransho**"), ""]
            status = localized(
                lang,
                en="Submitted" if draft.locked else "Pending (not submitted)",
                bn="জমা হয়েছে" if draft.locked else "অপেক্ষমাণ (জমা হয়নি)",
                banglish="Joma hoyeche" if draft.locked else "Opekkhaman (joma hoyni)",
            )
        if include_status:
            lines.append(f"Status: **{status}**")
            if draft.submitted_request_id:
                ref = localized(lang, en="Reference", bn="রেফারেন্স", banglish="Reference")
                lines.append(f"{ref}: **`{draft.submitted_request_id}`**")
            lines.append("")

        for key in ("leave_type", "start_date", "end_date", "day_scope", "half_day_period", "reason"):
            val = draft.fields.get(key)
            if val in (None, ""):
                continue
            if key in ("start_date", "end_date"):
                display = format_iso_date_display(str(val))
            elif key == "day_scope":
                display = str(val).replace("_", " ")
            elif key == "reason":
                text = str(val).strip()
                display = text if len(text) <= 120 else text[:117] + "..."
            else:
                display = val
            lines.append(f"- **{leave_field_label(key, lang=lang)}**: {display}")

        missing = []
        for key in ("leave_type", "start_date", "day_scope"):
            if not draft.fields.get(key):
                missing.append(leave_field_label(key, lang=lang))
        if missing and not draft.locked:
            lines.append("")
            lines.append(
                localized(
                    lang,
                    en=f"_Still needed: {', '.join(missing)}_",
                    bn=f"_এখনও দরকার: {', '.join(missing)}_",
                    banglish=f"_Ekhono dorkar: {', '.join(missing)}_",
                )
            )
        return "\n".join(lines)

    def _submitted_range_summary(self, entry: dict, *, lang: str = "en") -> str:
        from chat.services.platform.field_extractors.date import format_iso_date_display

        start = format_iso_date_display(str(entry.get("start_date") or ""))
        end = format_iso_date_display(
            str(entry.get("end_date") or entry.get("start_date") or "")
        )
        rid = str(entry.get("request_id") or "")
        if lang == "bn":
            lines = ["**জমা দেওয়া ছুটি**", "", f"- **তারিখ**: {start} থেকে {end}"]
            if rid:
                lines.append(f"- **রেফারেন্স**: `{rid}`")
        else:
            lines = ["**Submitted Leave**", "", f"- **Dates**: {start} to {end}"]
            if rid:
                lines.append(f"- **Reference**: `{rid}`")
        return "\n".join(lines)

    def leave_status_report(self, memory: SessionMemory, *, lang: str = "en") -> str:
        """Pending draft plus any submitted leave ranges recorded this session."""
        from chat.services.platform.field_engine import leave_draft_in_progress
        from chat.services.platform.workflow_show import session_leave_draft

        parts: list[str] = []
        draft = session_leave_draft(memory)
        if draft and (
            draft.locked
            or leave_draft_in_progress(draft)
            or (draft.fields or {}).get("reason")
        ):
            parts.append(self.leave_summary(draft, lang=lang))

        current_ref = str(draft.submitted_request_id or "") if draft and draft.locked else ""
        seen_ranges: set[tuple[str, str]] = set()
        for entry in (memory.conversation_facts or {}).get("submitted_leave_ranges") or []:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get("request_id") or "")
            if rid and rid == current_ref:
                continue
            start = str(entry.get("start_date") or "")[:10]
            end = str(entry.get("end_date") or entry.get("start_date") or "")[:10]
            key = (start, end)
            if not start or key in seen_ranges:
                continue
            seen_ranges.add(key)
            parts.append(self._submitted_range_summary(entry, lang=lang))

        if not parts:
            submitted = list((memory.conversation_facts or {}).get("submitted_leave_ranges") or [])
            if submitted:
                seen_ranges: set[tuple[str, str]] = set()
                for entry in submitted:
                    if not isinstance(entry, dict):
                        continue
                    start = str(entry.get("start_date") or "")[:10]
                    end = str(entry.get("end_date") or entry.get("start_date") or "")[:10]
                    key = (start, end)
                    if not start or key in seen_ranges:
                        continue
                    seen_ranges.add(key)
                    parts.append(self._submitted_range_summary(entry, lang=lang))
                if parts:
                    return "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0]
            if not memory.active_workflow:
                return localized(
                    lang,
                    en=(
                        "I couldn't find a leave summary in this session. "
                        "Start a new leave or share a submitted leave reference."
                    ),
                    bn=(
                        "এই সেশনে leave সারাংশ খুঁজে পাচ্ছি না। "
                        "নতুন leave শুরু করুন, অথবা submit করা leave এর reference দিন।"
                    ),
                    banglish=(
                        "Apnar leave summery khuje pacchi nah ei session e. "
                        "Notun leave shuru korte bolun, ba submit kora leave er reference din."
                    ),
                )
            return self.no_open_draft("leave", lang=lang)
        if len(parts) == 1:
            return parts[0]
        divider = "\n\n---\n\n"
        return divider.join(parts)

    def item_added(self, item: dict, *, lang: str = "en") -> str:
        cat = item.get("category", "?")
        amt = item.get("amount", "?")
        if lang == "bn":
            return f"**{cat}** expense {amt} taka যোগ হয়েছে।"
        return f"Added **{cat}** expense: **{amt} taka**."

    def slot_still_needed(self, field: str, prompt: str, *, lang: str = "en") -> str:
        label = leave_field_label(field, lang=lang) if field in LEAVE_FIELD_LABELS else field.replace("_", " ")
        body = prompt or localized(
            lang,
            en="Please provide the requested information.",
            bn="অনুগ্রহ করে তথ্য দিন।",
            banglish="Onugroho kore tothyo din.",
        )
        return localized(
            lang,
            en=f"I still need your **{label}**.\n\n{body}",
            bn=f"এখনও **{label}** দরকার।\n\n{body}",
            banglish=f"Ekhono **{label}** dorkar.\n\n{body}",
        )

    def item_removed_by_index(self, index: int, *, lang: str = "en") -> str:
        n = index + 1
        if lang == "bn":
            return f"Entry {n} মুছে ফেলা হয়েছে।"
        return f"Removed item {n}."

    def item_updated(self, label: str, new_amt: float, *, lang: str = "en") -> str:
        if lang == "bn":
            return f"**{label}** **{new_amt:.0f} taka** করা হয়েছে।"
        return f"Updated **{label}** to **{new_amt:.0f} taka**."

    def item_updated_by_index(self, index: int, new_amt: float, *, lang: str = "en") -> str:
        n = index + 1
        if lang == "bn":
            return f"Item {n} **{new_amt:.0f} taka** করা হয়েছে।"
        return f"Updated item {n} to **{new_amt:.0f} taka**."

    def item_deleted(self, label: str, *, lang: str = "en") -> str:
        if lang == "bn":
            return f"**{label}** মুছে ফেলা হয়েছে।"
        return f"Removed **{label}**."

    def no_draft_to_delete(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "Delete করার মতো কোনো draft নেই।"
        return "No draft to delete from."

    def which_workflow(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "কোন workflow?"
        return "Which workflow?"

    def workflow_switched(
        self,
        target: str,
        *,
        lang: str = "en",
        memory: SessionMemory | None = None,
    ) -> str:
        label = target.replace("_", " ")
        base = localized(
            lang,
            en=f"Switched to **{label}**.",
            bn=f"**{label}** workflow-এ গেলাম।",
            banglish=f"**{label}** workflow e gelam.",
        )
        if memory and memory.suspended_workflows:
            hint = self._paused_workflow_resume_hint(memory.suspended_workflows[-1].workflow_id, lang)
            if hint:
                return f"{base}\n\n{hint}"
        return base

    def workflow_switch_resumed(
        self,
        memory: SessionMemory,
        *,
        paused_workflow: str,
        resumed_workflow: str,
        lang: str = "en",
    ) -> str:
        """Rich copy when resuming a suspended workflow (Fix 6)."""
        paused = paused_workflow.replace("_", " ")
        resumed = resumed_workflow.replace("_", " ")
        lines = [
            localized(
                lang,
                en=f"Your **{paused}** request is now paused.",
                bn=f"আপনার **{paused}** request সাময়িক বিরতিতে রাখা হলো।",
                banglish=f"Apnar **{paused}** request temporarily pause kora holo.",
            ),
            localized(
                lang,
                en=f"Resuming your **{resumed}** request.",
                bn=f"আপনার **{resumed}** request আবার খুলছি।",
                banglish=f"Apnar **{resumed}** request abar khulchi.",
            ),
        ]
        resume_hint = self._paused_workflow_resume_hint(paused_workflow, lang)
        if resume_hint:
            lines.append(resume_hint)
        footer = self.workflow_turn_footer(memory, lang=lang, compact=True, include_suspended_hint=False)
        if footer:
            lines.append(footer)
        return "\n\n".join(line for line in lines if line).strip()

    def _paused_workflow_resume_hint(self, workflow_id: str, lang: str) -> str:
        if workflow_id == "expense":
            return localized(
                lang,
                en="To return to expense later, say **expense continue** or **expense summary**.",
                bn="Expense-এ ফিরতে **expense continue** বা **expense summary** বলুন।",
                banglish="Expense e fire jete **expense continue** ba **expense summary** bolen.",
            )
        if workflow_id == "leave":
            return localized(
                lang,
                en="To return to leave later, say **leave continue** or **leave summary**.",
                bn="Leave-এ ফিরতে **leave continue** বা **leave summary** বলুন।",
                banglish="Leave e fire jete **leave continue** ba **leave summary** bolen.",
            )
        return ""

    def workflow_screen_context(self, memory: SessionMemory, *, lang: str = "en") -> str:
        wf = memory.active_workflow
        if not wf:
            return ""
        wf_id = wf.id.replace("_", " ")
        pending = memory.pending_confirmation or ""
        if pending == "submit":
            return localized(
                lang,
                en=f"You are on the **{wf_id} review** screen — ready to submit.",
                bn=f"আপনি **{wf_id} review** screen-এ আছেন — submit করার জন্য প্রস্তুত।",
                banglish=f"Apni **{wf_id} review** screen e achen — submit korar jonno proshuto.",
            )
        if pending.startswith("switch:"):
            parts = pending.split(":")
            if len(parts) == 3:
                from_wf, to_wf = parts[1], parts[2]
                return localized(
                    lang,
                    en=f"Confirm: continue **{from_wf}** or switch to **{to_wf}**?",
                    bn=f"নিশ্চিত করুন: **{from_wf}** চালিয়ে যাবেন নাকি **{to_wf}**-এ যাবেন?",
                    banglish=f"Confirm korun: **{from_wf}** continue korben naki **{to_wf}** e jaben?",
                )
        pq = memory.pending_question
        if pq and pq.field:
            field = leave_field_label(pq.field, lang=lang) if pq.field in LEAVE_FIELD_LABELS else pq.field.replace("_", " ")
            return localized(
                lang,
                en=f"Collecting **{wf_id}** — waiting for **{field}**.",
                bn=f"**{wf_id}** collect করছি — **{field}** এর জন্য অপেক্ষা করছি।",
                banglish=f"**{wf_id}** collect korchi — **{field}** er jonno wait korchi.",
            )
        return localized(
            lang,
            en=f"Working on your **{wf_id}** request.",
            bn=f"আপনার **{wf_id}** request নিয়ে কাজ করছি।",
            banglish=f"Apnar **{wf_id}** request niye kaj korchi.",
        )

    def workflow_turn_footer(
        self,
        memory: SessionMemory,
        *,
        lang: str = "en",
        compact: bool = False,
        include_suspended_hint: bool = True,
    ) -> str:
        """Where-am-I + what-you-can-do hints (Fix 7)."""
        parts: list[str] = []
        screen = self.workflow_screen_context(memory, lang=lang)
        if screen and not compact:
            parts.append(screen)
        next_steps = self._workflow_next_steps(memory, lang=lang)
        if next_steps:
            parts.append(next_steps)
        if include_suspended_hint and memory.suspended_workflows:
            sw = memory.suspended_workflows[-1]
            hint = self._paused_workflow_resume_hint(sw.workflow_id, lang)
            if hint:
                parts.append(hint)
        return "\n\n".join(p for p in parts if p).strip()

    def _workflow_next_steps(self, memory: SessionMemory, *, lang: str) -> str:
        wf = memory.active_workflow
        if not wf:
            return ""
        if memory.pending_confirmation == "submit":
            return localized(
                lang,
                en="_You can **submit** (yes), **modify**, or **cancel**._",
                bn="_**submit** (ha), **modify**, ba **cancel** করতে পারেন।_",
                banglish="_Apni **submit** (ha), **modify**, ba **cancel** korte paren._",
            )
        if memory.pending_question:
            return localized(
                lang,
                en="_Say **summary** / **review** for your draft, or **cancel** to discard._",
                bn="_Draft দেখতে **summary** / **review** বলুন, বাতিল করতে **cancel**।_",
                banglish="_Draft dekhte **summary** / **review** bolen, batil korte **cancel**._",
            )
        return ""

    def continuing_workflow(self, workflow_id: str, *, lang: str = "en") -> str:
        return localized(
            lang,
            en=f"Continuing your **{workflow_id}** request.",
            bn=f"**{workflow_id}** request চালিয়ে যাচ্ছি।",
            banglish=f"**{workflow_id}** request chaliye jacchi.",
        )

    def no_open_draft(self, workflow_id: str, *, lang: str = "en") -> str:
        return localized(
            lang,
            en=f"No open **{workflow_id}** draft to submit.",
            bn=f"কোনো open **{workflow_id}** draft নেই।",
            banglish=f"Kono open **{workflow_id}** draft nei.",
        )

    def no_draft(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "কোনো draft নেই।"
        return "No draft."

    def reject_oos(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "এটি company HR assistant-এর scope-এর বাইরে। আমি leave, expense ও company policy নিয়ে সাহায্য করি।"
        return "That's outside my scope as an HR assistant. I help with leave, expense, and company policies."

    def modify_confirm(self, *, label: str, old: float, new: float, draft: WorkflowDraft, lang: str = "en") -> str:
        summary = format_expense_summary(draft, lang=lang)
        if lang == "bn":
            return (
                f"{summary}\n\n**{label}** এর amount **{old:.0f}** থেকে **{new:.0f} taka** করব?\n"
                "Confirm করতে **ha** বলুন।"
            )
        return (
            f"{summary}\n\nChange **{label}** amount from **{old:.0f}** to **{new:.0f} taka**?\n"
            "Reply **yes** to confirm."
        )

    def delete_pick(self, draft: WorkflowDraft, *, lang: str = "en") -> str:
        summary = format_expense_summary(draft, lang=lang)
        if lang == "bn":
            return f"{summary}\n\n**কোন entry delete করব?** নম্বর (1, 2...) বা বিবরণ লিখুন।"
        return f"{summary}\n\n**Which entry should I delete?** Reply with the number (1, 2...) or description."

    def submit_confirm(self, review_text: str, *, lang: str = "en") -> str:
        extra = localized(lang, en="Reply **yes** to submit.", bn="Submit করতে **ha** বলুন।", banglish="Submit korte **ha** bolen.")
        return f"{review_text}\n\n{extra}"

    def review_after_decline(self, review_text: str, *, lang: str = "en") -> str:
        extra = localized(
            lang,
            en="\n\nTell me if you want to **submit**, **modify**, or **cancel**.",
            bn="\n\nSubmit, modify, বা cancel করতে বলুন।",
            banglish="\n\nSubmit, modify, ba cancel korte bolen.",
        )
        return f"{review_text}{extra}".strip()

    def review_ready_message(
        self,
        prefix: str,
        review_text: str,
        *,
        lang: str = "en",
        memory: SessionMemory | None = None,
    ) -> str:
        """Prefix + formal pre-submit review (from ``build_review``)."""
        body = f"{prefix}\n\n{review_text or ''}".strip() if prefix else (review_text or "").strip()
        if memory:
            footer = self.workflow_turn_footer(memory, lang=lang, compact=True)
            if footer:
                body = f"{body}\n\n{footer}".strip()
        return body

    def missing_for_submit(self, missing: list[str], *, lang: str = "en") -> str:
        joined = ", ".join(m.replace("_", " ") for m in missing)
        if lang == "bn":
            return f"Submit করার আগে এগুলো দরকার: **{joined}**"
        return f"Before submitting, I still need: **{joined}**"

    def locked_message(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "এই request submit হয়ে গেছে — আর modify/delete/cancel করা যাবে না।"
        return "This request is already submitted — modify, delete, and cancel are no longer allowed."

    def locked_with_reference(self, request_id: str, *, lang: str = "en") -> str:
        msg = self.locked_message(lang=lang)
        if request_id:
            msg += f"\n\nReference: **`{request_id}`**"
        return msg

    def workflow_started(self, workflow_name: str, *, lang: str = "en") -> str:
        return localized(
            lang,
            en=f"**{workflow_name}** started.",
            bn=f"**{workflow_name}** শুরু হলো।",
            banglish=f"**{workflow_name}** shuru holo.",
        )

    def category_clarify(self, amount: float, *, lang: str = "en") -> str:
        if lang == "bn":
            return (
                f"**{amount:.0f} taka** খরচ নোট করেছি। "
                "খরচের ধরন কী? (Lunch, Snack, Bus, Train, Metro, Rickshaw, Bike)"
            )
        return (
            f"Noted **{amount:.0f} taka**. "
            "What type of expense was it? (Lunch, Snack, Bus, Train, Metro, Rickshaw, Bike)"
        )

    def medical_document_skipped(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en="Noted — you can provide the medical document later. Continuing with your **sick leave** request.",
            bn="বুঝেছি — medical document পরে দিতে পারবেন। আপনার **sick leave** request চালিয়ে যাচ্ছি।",
            banglish="Bujhechi — medical document pore dite paren. Apnar **sick leave** request chaliye jacchi.",
        )

    def medical_document_unavailable(self, *, lang: str = "en") -> str:
        return self.medical_document_skipped(lang=lang)

    def submitted_leave_overlap(self, entry: dict, *, lang: str = "en") -> str:
        from chat.services.platform.field_extractors.date import format_iso_date_display

        start = format_iso_date_display(str(entry.get("start_date") or ""))
        end = format_iso_date_display(str(entry.get("end_date") or entry.get("start_date") or ""))
        rid = entry.get("request_id") or ""
        if lang == "bn":
            msg = f"**{start}** থেকে **{end}** পর্যন্ত leave ইতিমধ্যে submit করা আছে"
            if rid:
                msg += f" (ref: `{rid}`)"
            msg += "।\n\nঅন্য তারিখে leave নিতে চাইলে নতুন তারিখ বলুন।"
            return msg
        msg = f"You already submitted leave for **{start}** to **{end}**"
        if rid:
            msg += f" (ref: `{rid}`)"
        msg += ".\n\nUse different dates if you need another leave request."
        return msg

    def duplicate_leave_prompt(self, draft: WorkflowDraft, *, lang: str = "en") -> str:
        summary = self.leave_summary(draft, lang=lang)
        if lang == "bn":
            return (
                f"{summary}\n\n"
                "আপনার একটি **leave request** ইতিমধ্যে চলছে।\n\n"
                "**continue** বললে এটাই চালিয়ে যাব, **new** বা **cancel** বললে নতুন leave শুরু করব।"
            )
        return (
            f"{summary}\n\n"
            "You already have a **leave request** in progress.\n\n"
            "Reply **continue** to keep it, or **new** / **cancel** to start a fresh leave request."
        )

    def duplicate_leave_continue(self, *, lang: str = "en") -> str:
        return self.continuing_workflow("leave", lang=lang)

    def duplicate_leave_fresh(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "নতুন **leave request** শুরু হলো।"
        return "Started a fresh **leave request**."

    def duplicate_leave_retry(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "**continue** বা **new** বলুন।"
        return "Reply **continue** or **new**."

    def active_leave_parallel_block(
        self,
        memory: SessionMemory | None = None,
        *,
        lang: str = "en",
    ) -> str:
        if lang == "bn":
            block = (
                "আপনার একটি **active leave request** ইতিমধ্যে চলছে। "
                "নতুন leave শুরু করার আগে এটি **complete** করুন বা **cancel** করুন।"
            )
            hints = (
                "_**summary** / **review** বললে বর্তমান draft দেখাব। "
                "**cancel** বললে বাতিল হবে।_"
            )
        else:
            block = (
                "You already have an active **leave request**. "
                "Please **complete** or **cancel** it before starting a new one."
            )
            hints = (
                "_Say **summary** or **review** to see your current draft. "
                "Say **cancel** to discard it._"
            )

        if not memory or not memory.active_workflow or memory.active_workflow.id != "leave":
            return f"{block}\n\n{hints}"

        draft = memory.active_draft()
        if not draft:
            return f"{block}\n\n{hints}"

        summary = self.leave_summary(draft, lang=lang, include_status=True)
        return f"{block}\n\n{summary}\n\n{hints}"

    def workflow_cancelled(self, workflow_id: str, *, lang: str = "en") -> str:
        label = workflow_id.replace("_", " ")
        return localized(
            lang,
            en=f"Your **{label}** request has been cancelled.",
            bn=f"**{label}** request বাতিল করা হয়েছে।",
            banglish=f"**{label}** request batil kora hoyeche.",
        )

    def session_context(self, memory: SessionMemory, *, lang: str = "en") -> str:
        return format_session_context(memory, lang=lang)

    def workflow_summary(
        self,
        definition: WorkflowDefinition,
        draft: WorkflowDraft,
        *,
        lang: str = "en",
        emphasize_total: bool = False,
        memory: SessionMemory | None = None,
    ) -> str:
        """Informational summary turn — uses ``format_*_summary``, not ``build_review``."""
        if definition.workflow_id == "expense":
            from chat.services.platform.summary import format_expense_status_report

            if memory is not None:
                msg = format_expense_status_report(memory, lang=lang, focus_draft=draft)
            else:
                msg = format_expense_summary(draft, lang=lang)
            if emphasize_total:
                msg += f"\n\n**Total: {expense_total(draft):.0f} taka**"
            return msg
        if definition.workflow_id == "leave":
            return self.leave_summary(draft, lang=lang)
        return ""

    def collection_message(self, parts: list[str]) -> str:
        return "\n\n".join(p for p in parts if p).strip()

    def general_help(self, *, lang: str = "en") -> str:
        return localized(
            lang,
            en="How can I help? You can ask about **leave**, **expense**, or **company policies**.",
            bn="আমি কীভাবে সাহায্য করতে পারি? Leave, expense বা company policy নিয়ে জিজ্ঞাসা করুন।",
            banglish="Ami kivabe sahajjo korte pari? Leave, expense ba company policy niye jigges korun.",
        )

    def platform_continue_clarify(
        self,
        workflow_label: str,
        *,
        reasoning: str | None = None,
        lang: str = "en",
    ) -> str:
        from chat.services.platform.turn_semantics import is_internal_reasoning_text

        if reasoning and not is_internal_reasoning_text(reasoning):
            return reasoning
        if lang == "bn":
            return f"আরও একটু তথ্য দরকার {workflow_label} request চালিয়ে যেতে।"
        return f"I need a bit more information to continue your {workflow_label} request."

    def translation_unavailable(self, previous_answer: str, *, lang: str = "en") -> str:
        if lang == "bn":
            prefix = "Translation এখন unavailable — আগের উত্তর আবার দিচ্ছি:\n\n"
        else:
            prefix = "Translation is briefly unavailable — re-posting the previous answer:\n\n"
        return prefix + previous_answer

    def workflow_continuation_hint(self, memory: SessionMemory) -> str:
        lang = normalize_reply_lang(str((memory.last_entities or {}).get("reply_language") or "en"))
        if memory.suspended_workflows:
            sw = memory.suspended_workflows[-1]
            return localized(
                lang,
                en=(
                    f"\n\n_(Your **{sw.workflow_id}** request is paused — "
                    f"reply **{sw.workflow_id}** anytime to continue.)_"
                ),
                bn=(
                    f"\n\n_(আপনার **{sw.workflow_id}** request pause আছে — "
                    f"চালিয়ে যেতে **{sw.workflow_id}** বলুন।)_"
                ),
                banglish=(
                    f"\n\n_(Apnar **{sw.workflow_id}** request pause ache — "
                    f"chaliye jete **{sw.workflow_id}** bolen.)_"
                ),
            )
        pq = memory.pending_question
        wf = memory.active_workflow
        if not pq and not wf:
            return ""
        if pq:
            wf_label = pq.workflow_id or (wf.id if wf else "workflow")
            return localized(
                lang,
                en=(
                    f"\n\n_(Your **{wf_label}** draft is still open — "
                    f"reply anytime to continue.)_"
                ),
                bn=(
                    f"\n\n_(আপনার **{wf_label}** draft এখনও open — "
                    f"চালিয়ে যেতে যেকোনো সময় reply করুন।)_"
                ),
                banglish=(
                    f"\n\n_(Apnar **{wf_label}** draft ekhono open — "
                    f"chaliye jete jekono somoy reply korun.)_"
                ),
            )
        if wf:
            return localized(
                lang,
                en=f"\n\n_(Your **{wf.id}** draft is still saved — you can continue when ready.)_",
                bn=f"\n\n_(আপনার **{wf.id}** draft এখনও saved — প্রস্তুত হলে চালিয়ে যান।)_",
                banglish=f"\n\n_(Apnar **{wf.id}** draft ekhono saved — prostut hole chaliye jan.)_",
            )
        return ""

    def item_prefix_from_updates(self, updates, *, lang: str = "en") -> str:
        lines: list[str] = []
        for upd in updates or []:
            if upd.field == "items" and isinstance(upd.value, dict) and upd.action == "append":
                return self.item_added(upd.value, lang=lang)
            if upd.field in LEAVE_FIELD_LABELS and upd.value not in (None, ""):
                label = leave_field_label(str(upd.field), lang=lang)
                val = upd.value
                if upd.field in ("start_date", "end_date"):
                    val = format_iso_date_display(str(val))
                elif upd.field == "day_scope":
                    val = str(val).replace("_", " ")
                lines.append(
                    localized(
                        lang,
                        en=f"Updated **{label}** to **{val}**.",
                        bn=f"**{label}** **{val}**-এ আপডেট করা হয়েছে।",
                        banglish=f"**{label}** **{val}** e update kora hoyeche.",
                    )
                )
            elif upd.field == "reason_skipped":
                lines.append(self.leave_reason_skipped(lang=lang))
        return "\n".join(lines)

    # --- Phase 8 facade — pipeline must call these instead of stray copy imports ---

    def rules_footer(self, *, lang: str) -> str:
        return policy_rules_footer(lang=lang)

    def with_continuation_hint(self, message: str, memory: SessionMemory) -> str:
        return message + self.workflow_continuation_hint(memory)

    def conversational(
        self,
        message: str,
        *,
        context_lines: list[str],
        trace_id: str,
    ) -> str | None:
        from chat.services.conversational import conversational_reply

        return conversational_reply(
            message=message,
            context_lines=context_lines,
            trace_id=trace_id,
        )

    def policy_turn(
        self,
        message: str,
        *,
        document_text: str | None,
        pq_decision_log: dict[str, Any] | None,
        conversation_history: list[str],
        company_id: str,
        trace_id: str,
        pq_reasoning: str = "",
    ) -> tuple[str, str, dict[str, Any]]:
        return compose_policy_turn(
            message,
            document_text=document_text,
            pq_decision_log=pq_decision_log,
            conversation_history=conversation_history,
            company_id=company_id,
            trace_id=trace_id,
            pq_reasoning=pq_reasoning,
        )

    def status_turn(
        self,
        message: str,
        *,
        company_id: str,
        employee_id: str,
        session_id: str,
        pq_decision_log: dict[str, Any] | None = None,
        rules_tag: str = "PLAN_REPLY_STATUS",
    ) -> tuple[str, str, dict[str, Any], str]:
        msg, resp_status, decision, _entities, request_id = resolve_request_status_turn(
            message,
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            pq_decision_log=pq_decision_log,
            rules_tag=rules_tag,
        )
        return msg, resp_status, decision, request_id

    def out_of_scope(
        self,
        message: str,
        *,
        lang: str,
        context_lines: list[str],
        trace_id: str,
    ) -> str:
        from chat.services.policy_intent_helpers import build_out_of_scope_message

        return build_out_of_scope_message(
            message,
            lang=lang,
            context_lines=context_lines,
            trace_id=trace_id,
        )

    def today_date(self, *, today_iso: str, lang: str) -> str:
        from chat.services.policy_intent_helpers import format_today_date_reply

        return format_today_date_reply(today_iso=today_iso, lang=lang)

    def greeting(self, message: str, *, context_lines: list[str], trace_id: str) -> str | None:
        return self.conversational(message=message, context_lines=context_lines, trace_id=trace_id)