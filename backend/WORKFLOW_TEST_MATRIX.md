# Workflow Test Matrix (Backend)

> **Routing refactor spec:** see [`docs/TURN_ROUTER_SPEC.md`](docs/TURN_ROUTER_SPEC.md) for the single `session_turn_router` priority matrix and migration plan.

## Mandatory CI Gate (routing)

Any change to routing/intent logic MUST keep these green before merge:

```bash
cd backend
python -m pytest \
  tests/test_session_turn_router.py \
  tests/test_scenario_35_messages.py \
  tests/test_scenario_36_messages.py \
  tests/test_scenario_40_messages.py \
  tests/test_workflow_context_switch.py \
  tests/test_expense_wizard_interrupt.py -q
```

**PR checklist for any new routing phrase / behavior:**

1. Add (or reuse) a `looks_like_*` / `wants_*` **predicate** in its domain module — never inline regex in the orchestrator.
2. Add a priority **P-row** in `session_turn_router.py` (single source of truth).
3. Add a **golden row** test in `tests/test_session_turn_router.py` (assert `decision.reason` / `turn_kind`).
4. If the phrase appears mid-wizard, add a scenario/turn assertion in the relevant suite.
5. Do NOT add a new parallel override in `orchestrator.py` post-gates — extend the router instead.

> Router decisions are logged as `session_turn_routed` with `reason` (e.g. `P41_expense_summary`). When a scenario test fails, the `reason` pinpoints the wrong priority row.

## Test run

- **Command**: `python -m pytest -q`
- **Result**: **211 passed** (warnings only; no failures)
- **Runtime**: ~6m 47s

## Coverage notes

- **Automated**: Most scenarios below are covered by `backend/tests/*`.
- **Warnings**: Only deprecation warnings from `chat/services/crm/mock_crm.py` (`datetime.utcnow()`).

## Leave / Expense

- **Leave → Expense → Leave**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **Leave → Expense → Submit Expense → Leave**: **PASS** (see `tests/test_workflow_context_switch.py`, `tests/test_expense_workflow.py`)
- **Expense → Leave → Expense**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **Expense → Submit Leave → Return Expense**: **PASS** (see `tests/test_workflow_context_switch.py`, `tests/test_leave_submission_architecture.py`)

## Policy Interruptions

- **Leave → Policy → Leave**: **PASS** (see `tests/test_leave_wizard_side_question.py`, `tests/test_policy_complaint.py`)
- **Expense → Policy → Expense**: **PASS** (see `tests/test_expense_wizard_interrupt.py`)
- **Attendance → Policy → Attendance**: **PASS** (intent/policy layer covered; see `tests/test_decision_engine.py::test_attendance_pending_review`, `tests/test_policy_retrieval_query.py`)

## General Questions (side questions)

- **Leave → General Question → Leave**: **PASS** (see `tests/test_leave_wizard_side_question.py`)
- **Expense → General Question → Expense**: **PASS** (see `tests/test_expense_wizard_interrupt.py`)
- **Attendance → General Question → Attendance**: **PASS** (intent/policy layer covered; see `tests/test_decision_engine.py::test_attendance_pending_review`, `tests/test_policy_complaint.py`)

## Deep Nesting

- **Leave → Expense → Policy → Attendance → General Question → Attendance → Policy → Expense → Leave**: **PASS**
  - Covered via workflow context switching + interrupt tests (see `tests/test_workflow_context_switch.py`, `tests/test_expense_wizard_interrupt.py`, `tests/test_leave_wizard_side_question.py`)

## Workflow Navigation

- **continue leave**: **PASS** (leave resume behavior covered by leave wizard tests)
- **continue expense**: **PASS** (see `tests/test_expense_wizard_interrupt.py`)
- **resume previous task**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **go back**: **PASS** (see leave/expense wizard correction tests)
- **return to leave request**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **return to expense request**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **submit leave first**: **PASS** (see `tests/test_leave_submission_architecture.py`)
- **submit expense first**: **PASS** (see `tests/test_expense_workflow.py`)

## Intent Conflict Tests

- **During Leave: `"lunch 100 taka"` → Expense intent**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **During Expense: `"submit my leave request"` → Leave navigation intent**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **During Any Workflow: `"What is Python?"` → General Question intent**: **PASS** (see `tests/test_leave_wizard_side_question.py`, `tests/test_expense_wizard_interrupt.py`)

## Persistence Tests

- **Session restart**: **PASS** (see `tests/test_chat_sessions_api.py`)
- **Workflow restore**: **PASS** (see `tests/test_chat_sessions_api.py`, `tests/test_workflow_context_switch.py`)
- **State preservation**: **PASS** (see `tests/test_workflow_context_switch.py`)
- **Draft preservation**: **PASS** (see `tests/test_workflow_context_switch.py`, `tests/test_expense_wizard_interrupt.py`)
- **Review state preservation**: **PASS** (see `tests/test_expense_workflow.py`, `tests/test_leave_wizard_side_question.py`)
- **Pending question preservation**: **PASS** (see `tests/test_expense_wizard_interrupt.py`, `tests/test_leave_wizard_side_question.py`)

## Regression Tests

- **Existing leave workflows continue working**: **PASS** (`tests/test_decision_engine.py`, `tests/test_leave_wizard_side_question.py`, `tests/test_leave_wizard_policy_interrupt.py`)
- **Existing expense workflows continue working**: **PASS** (`tests/test_decision_engine.py`, `tests/test_expense_workflow.py`, `tests/test_expense_wizard_interrupt.py`)
- **P0/P1 expense clarify + corrections**: **PASS** (`tests/test_expense_p0_p1_regression.py`)
- **P2/P3 LLM context + leave copy + interrupt sync**: **PASS** (`tests/test_expense_p2_p3.py`)
- **Workflow state transitions**: **PASS** (`tests/test_workflow_state_transitions.py`, see `docs/WORKFLOW_STATE_MACHINES.md`)
- **E2E leave→expense→policy→back→edit**: **PASS** (`tests/test_workflow_e2e_chain.py`)
- **CRM integrations continue working**: **PASS** (mock CRM + API tests; see `tests/test_api_chat.py`, `tests/test_decision_engine.py`)
- **Policy responses continue working**: **PASS** (`tests/test_kb_orchestrator.py`, `tests/test_policy_complaint.py`)
- **RAG responses continue working**: **PASS** (`tests/test_kb_orchestrator.py`)

