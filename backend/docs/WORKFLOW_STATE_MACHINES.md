# Workflow State Machines (P3)

Deterministic stage transitions for leave and expense wizards. Tests live in
`tests/test_workflow_state_transitions.py`.

## Expense wizard

```mermaid
stateDiagram-v2
    [*] --> collecting: first expense line
    collecting --> collecting: add line / pending slot
    collecting --> review: done / advance (no pending)
    review --> submit_confirm: yes (valid)
    review --> collecting: validation blocked
    submit_confirm --> [*]: yes submit
    submit_confirm --> review: no
    collecting --> [*]: cancel / deactivate
```

| Stage | User signal | Next stage |
|-------|-------------|------------|
| `collecting` | new lines / slot answers | `collecting` |
| `collecting` | done / finish (valid draft) | `review` |
| `review` | yes (valid) | `submit_confirm` |
| `review` | edit / correction | `review` |
| `submit_confirm` | yes | submitted (inactive) |
| `submit_confirm` | no | `review` |

Pending sub-steps during `collecting`: `category`, `from_to`, `clarify`.

## Leave wizard

```mermaid
stateDiagram-v2
    [*] --> collecting: leave intent
    collecting --> collecting: slot answers
    collecting --> review_pending: all slots filled
    review_pending --> submitted: yes
    review_pending --> edit_menu: edit
    edit_menu --> edit_slot: pick field
    edit_slot --> review_pending: new value
```

| State | User signal | Next |
|-------|-------------|------|
| collecting | slot answer | collecting / review_pending |
| review_pending | yes | submitted |
| review_pending | edit | edit_menu |
| edit_menu | field name | edit_slot |
| edit_slot | new value | review_pending |

## Cross-workflow suspend

- Active leave + expense claim → `suspended_leave` snapshot, expense active.
- Active expense + leave claim → `suspended_expense` snapshot, leave active.
- Resume commands restore the parked snapshot.

See `workflow_suspend.py` and `tests/test_workflow_context_switch.py`.
