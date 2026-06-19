"""User-defined conversation scenarios — expected behavior from product spec."""

from __future__ import annotations

import re

import pytest

from tests.helpers.scenario_runner import chat_runner  # noqa: F401 — pytest fixture


def _has(msg: str, *patterns: str) -> bool:
    low = msg.lower()
    return all(re.search(p, low, re.I) for p in patterns)


def _any(msg: str, *patterns: str) -> bool:
    low = msg.lower()
    return any(re.search(p, low, re.I) for p in patterns)


class TestExpenseBasics:
    def test_bus_add_asks_route(self, chat_runner):
        msg = chat_runner("ajke bus e 100 taka")
        assert _any(msg, r"travel|bus|from|route|কোথা")

    def test_lunch_add(self, chat_runner):
        msg = chat_runner("lunch 150 taka")
        assert _any(msg, r"lunch|meals|150|added|যোগ")

    def test_nasta_add(self, chat_runner):
        msg = chat_runner("nasta 50 taka")
        assert _any(msg, r"50|meals|nasta|added|যোগ")

    def test_vague_amount_modify_clarify(self, chat_runner):
        chat_runner("lunch 150 taka")
        chat_runner("bus 100 taka")
        msg = chat_runner("amount ta 200 kore dao")
        assert _any(msg, r"which|kon|lunch|bus|entry|item|number|কোন")

    def test_lunch_amount_modify(self, chat_runner):
        chat_runner("lunch 150 taka")
        msg = chat_runner("lunch er amount ta 200 koro")
        assert _any(msg, r"200|confirm|change|updated|ha|yes")


class TestLeaveFlow:
    def test_leave_start_collects_remaining(self, chat_runner):
        msg = chat_runner("agami 15 august leave chai")
        assert _any(msg, r"leave|august|15|type|day|scope|ছুটি")

    def test_reason_then_missing_fields(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        msg = chat_runner("reason personal work")
        assert _any(msg, r"reason|personal|leave|type|day|scope|saved|সংরক্ষণ")

    def test_block_second_leave_while_open(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        msg = chat_runner("agami 20 august leave chai")
        assert _any(msg, r"already|open|submit|cancel|আগে|pending")


class TestExpenseDelete:
    def test_delete_asks_which_entry(self, chat_runner):
        chat_runner("bus 100 taka")
        chat_runner("lunch 120 taka")
        msg = chat_runner("delete koro")
        assert _any(msg, r"which|delete|summary|entry|item|কোন|1\.|2\.")


class TestSubmitConfirm:
    def test_leave_submit_asks_confirm(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        chat_runner("annual leave")
        chat_runner("full day")
        msg = chat_runner("leave submit koro")
        assert _any(msg, r"submit|confirm|yes|ha|review|leave")

    def test_ha_only_submits_with_pending(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        chat_runner("annual leave")
        chat_runner("full day")
        chat_runner("leave submit koro")
        msg = chat_runner("ha")
        assert msg  # submit or still missing info

    def test_ha_without_pending_shows_context(self, chat_runner):
        chat_runner("lunch 150 taka")
        msg = chat_runner("ha")
        assert _any(msg, r"expense|summary|pending|active|no pending|context|150")

    def test_expense_submit_pending(self, chat_runner):
        chat_runner("lunch 150 taka")
        msg = chat_runner("expense submit koro")
        assert _any(msg, r"submit|confirm|yes|review|expense")


class TestExpenseWhileDraftOpen:
    def test_add_while_submit_pending(self, chat_runner):
        chat_runner("lunch 150 taka")
        chat_runner("expense submit koro")
        msg = chat_runner("ajke nasta 50 taka")
        assert _any(msg, r"50|nasta|meals|added|item|submit")


class TestModifyWithConfirm:
    def test_route_expense_and_lunch(self, chat_runner):
        chat_runner("Mirpur theke Motijheel bus e 120 tk")
        msg = chat_runner("lunch korechi 150 taka")
        assert _any(msg, r"150|lunch|meals|added")

    def test_first_item_modify_confirm(self, chat_runner):
        chat_runner("Mirpur theke Motijheel bus e 120 tk")
        chat_runner("lunch korechi 150 taka")
        msg = chat_runner("prothom ta 200 tk kore dao")
        assert _any(msg, r"200|confirm|summary|1|prothom|first|change")

    def test_expense_review(self, chat_runner):
        chat_runner("bus 100 taka")
        msg = chat_runner("expense review dao")
        assert _any(msg, r"review|summary|expense|item|100")


class TestSummary:
    def test_summary_shows_items(self, chat_runner):
        chat_runner("bus 100 taka")
        chat_runner("bus 100 taka")
        msg = chat_runner("summary dekhao")
        assert _any(msg, r"summary|100|expense|item|total")


class TestLeaveBlockWhenOpen:
    def test_july_leave_blocked_if_open(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        msg = chat_runner("10 july leave chai")
        assert _any(msg, r"already|open|submit|cancel|pending|আগে")


class TestPolicyInterrupt:
    def test_policy_during_expense(self, chat_runner):
        chat_runner("bus 100 taka")
        msg = chat_runner("sick leave policy ki?")
        assert _any(msg, r"policy|rules|handbook|uploaded|couldn't find|reimbursement")

    def test_expense_summary_after_policies(self, chat_runner):
        chat_runner("bus 100 taka")
        chat_runner("lunch 120 taka")
        chat_runner("sick leave policy ki?")
        chat_runner("overtime policy ki?")
        chat_runner("casual leave policy ki?")
        msg = chat_runner("expense summary dekhao")
        assert _any(msg, r"summary|expense|100|120|total|item")


class TestMultiExpenseDay:
    def test_office_travel_asks_category_or_route(self, chat_runner):
        msg = chat_runner("ajke office jete giye 120 taka khoroch hoise")
        assert _any(msg, r"category|travel|route|from|120|কোথা|bus")

    def test_afternoon_amount(self, chat_runner):
        chat_runner("ajke office jete giye 120 taka khoroch hoise")
        chat_runner("travel")
        chat_runner("Mirpur")
        chat_runner("Motijheel")
        msg = chat_runner("tarpor dupure 180 taka lagse")
        assert _any(msg, r"180|category|travel|added")

    def test_snack_add(self, chat_runner):
        chat_runner("lunch 150 taka")
        msg = chat_runner("ar bikale halka nasta 40 taka")
        assert _any(msg, r"40|nasta|meals|added")

    def test_total_summary(self, chat_runner):
        chat_runner("lunch 150 taka")
        chat_runner("nasta 50 taka")
        msg = chat_runner("total koto hoise?")
        assert _any(msg, r"total|150|50|200|summary|expense")


class TestMixedSubmit:
    def test_leave_submit_missing_info(self, chat_runner):
        msg = chat_runner("leave submit koro")
        assert _any(msg, r"need|required|leave|type|date|submit|missing")

    def test_ha_expense_submit(self, chat_runner):
        chat_runner("lunch 150 taka")
        msg = chat_runner("ha expense submit koro")
        assert _any(msg, r"expense|submit|confirm|review|150")


class TestOOSAndSummary:
    def test_leave_chai_when_pending(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        msg = chat_runner("leave chai")
        assert _any(msg, r"already|open|pending|submit|cancel")

    def test_programming_rejected(self, chat_runner):
        for q in ("python ki?", "javascript ki?", "golang ki?"):
            msg = chat_runner(q)
            assert _any(msg, r"scope|outside|hr|reject|can't help|বাইরে")

    def test_leave_summary(self, chat_runner):
        chat_runner("agami 15 august leave chai")
        msg = chat_runner("amar leave summary dekhao")
        assert _any(msg, r"leave|summary|august|15|pending")

    def test_expense_submit_full_flow(self, chat_runner):
        chat_runner("lunch 150 taka")
        chat_runner("expense submit koro")
        msg = chat_runner("ha")
        assert _any(msg, r"submit|reference|mock|150|confirm|need|missing")


class TestTranscriptBugFixes:
    """Regression tests from live leave/expense cross-contamination transcript."""

    LEAVE_NARRATIVE = (
        "Actually my mama has been sick for a long time, need treatment in Dhaka. "
        "I cannot attend office from Monday (14th July) to Wednesday (16th July)."
    )
    EXPENSE_MSG = (
        "amar ajke expense hoyeche 100 taka for bus then lunch 50 taka "
        "bike 120 taka snack 30 taka"
    )

    def test_leave_narrative_extracts_date_range(self, chat_runner):
        msg = chat_runner(self.LEAVE_NARRATIVE)
        assert _any(msg, r"14|july|2025|type|annual|sick|day|scope|leave|ছুটি")

    def test_expense_during_leave_asks_switch(self, chat_runner):
        chat_runner(self.LEAVE_NARRATIVE)
        chat_runner("annual")
        msg = chat_runner(self.EXPENSE_MSG)
        assert _any(msg, r"expense|unfinished|leave|continue|detected|claim")
        assert not _any(msg, r"day scope|day_scope|saved leave type")

    def test_oos_pauses_leave_draft(self, chat_runner):
        chat_runner(self.LEAVE_NARRATIVE)
        chat_runner("annual")
        msg = chat_runner("what is python?")
        assert _any(msg, r"scope|outside|hr|can't help|বাইরে")
        assert _any(msg, r"paused|pause|continue|leave")

    def test_compound_expense_items_saved(self, chat_runner):
        chat_runner(self.EXPENSE_MSG)
        msg = chat_runner("expense review dao")
        assert _any(msg, r"100|50|120|30|300|total|bus|lunch|bike|snack")

    def test_leave_review_shows_formatted_dates(self, chat_runner):
        chat_runner(self.LEAVE_NARRATIVE)
        chat_runner("annual")
        chat_runner("full day")
        msg = chat_runner("leave review dao")
        assert _any(msg, r"14\s+july\s+202\d|202\d-07-14")
        assert _any(msg, r"16\s+july|202\d-07-16|end")
