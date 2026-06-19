"""Compose user-facing messages for workflow platform."""

from __future__ import annotations

from chat.services.platform.schemas import UnderstandingResult
from chat.services.platform.summary import format_expense_summary, format_leave_summary
from chat.services.session_memory import PendingQuestion, SessionMemory, WorkflowDraft


class ResponseComposer:
    def clarification(self, understanding: UnderstandingResult, *, lang: str = "en", draft: WorkflowDraft | None = None) -> str:
        reason = understanding.reasoning or ""
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
            if lang == "bn":
                return "মনে হচ্ছে আপনি ছুটি নিতে চান। **Leave request** শুরু করব?"
            return "I understand you may need time away from work. Would you like to submit a **leave request**?"
        if wf == "expense":
            if lang == "bn":
                return "মনে হচ্ছে এটি expense entry। **Expense claim** তৈরি করব?"
            return "I think this may be an expense entry. Would you like to create an **expense claim**?"
        if understanding.goal == "Greeting" or "greeting" in reason.lower():
            if lang == "bn":
                return "হ্যালো! আমি আপনাকে কীভাবে সাহায্য করতে পারি? Leave, expense বা company policy নিয়ে জিজ্ঞাসা করতে পারেন।"
            return "Hello! How can I help you today? I can assist with **leave**, **expense**, or **company policies**."
        if reason and (
            "no workflow" in reason.lower()
            or "could not determine" in reason.lower()
            or "clarification" in reason.lower()
        ):
            if lang == "bn":
                return "আমি কীভাবে সাহায্য করতে পারি? Leave, expense বা company policy নিয়ে বলুন।"
            return "How can I help? You can ask about **leave**, **expense**, or **company policies**."
        return reason or ("আরও স্পষ্ট করে বলবেন?" if lang == "bn" else "Could you clarify?")

    def field_saved(self, field: str, *, lang: str = "en") -> str:
        label = field.replace("_", " ")
        if lang == "bn":
            return f"**{label}** সংরক্ষণ করা হয়েছে।"
        return f"Saved **{label}**."

    def item_added(self, item: dict, *, lang: str = "en") -> str:
        cat = item.get("category", "?")
        amt = item.get("amount", "?")
        if lang == "bn":
            return f"**{cat}** expense {amt} taka যোগ হয়েছে।"
        return f"Added **{cat}** expense: **{amt} taka**."

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
        if lang == "bn":
            return f"{review_text}\n\nSubmit করতে **ha** বলুন।"
        return f"{review_text}\n\nReply **yes** to submit."

    def missing_for_submit(self, missing: list[str], *, lang: str = "en") -> str:
        joined = ", ".join(m.replace("_", " ") for m in missing)
        if lang == "bn":
            return f"Submit করার আগে এগুলো দরকার: **{joined}**"
        return f"Before submitting, I still need: **{joined}**"

    def locked_message(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return "এই request submit হয়ে গেছে — আর modify/delete/cancel করা যাবে না।"
        return "This request is already submitted — modify, delete, and cancel are no longer allowed."

    def workflow_started(self, workflow_name: str, *, lang: str = "en") -> str:
        if lang == "bn":
            return f"**{workflow_name}** শুরু হলো।"
        return f"**{workflow_name}** started."

    def category_clarify(self, amount: float, *, lang: str = "en") -> str:
        if lang == "bn":
            return f"**{amount:.0f} taka** খরচ নোট করেছি। Category কী (travel/bus, lunch/meals, snack)?"
        return f"Noted **{amount:.0f} taka**. What category is this (travel/bus, lunch/meals, snack)?"

    def medical_document_unavailable(self, *, lang: str = "en") -> str:
        if lang == "bn":
            return (
                "৩+ দিন **sick leave**-এ medical document লাগে। আপনার কাছে document নেই বলে "
                "sick leave apply করা যাবে না।\n\n"
                "**annual**, **sick**, বা **lwop** leave — কোনটা নেবেন?"
            )
        return (
            "Medical documentation is required for **sick leave** of 3+ days, and you don't have one.\n\n"
            "Please choose **annual**, **sick**, or **lwop** leave instead."
        )

    def submitted_leave_overlap(self, entry: dict, *, lang: str = "en") -> str:
        from chat.services.platform.field_extractors.date import format_iso_date_display

        start = format_iso_date_display(str(entry.get("start_date") or ""))
        end = format_iso_date_display(str(entry.get("end_date") or entry.get("start_date") or ""))
        rid = entry.get("request_id") or ""
        if lang == "bn":
            msg = (
                f"**{start}** থেকে **{end}** পর্যন্ত leave ইতিমধ্যে submit করা আছে"
            )
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
        from chat.services.platform.summary import format_leave_summary

        summary = format_leave_summary(draft, lang=lang)
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
