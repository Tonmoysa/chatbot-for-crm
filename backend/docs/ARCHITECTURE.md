# Chat architecture (Phases 1–10)

Production entry: `POST /api/chat/` → `ChatOrchestrator.run_chat()`.

## Turn pipeline

```
User Message
  → SessionStore.open()              TurnContext + workflow memory + transcript
  → AIUnderstandingLayer.understand()  rules + optional LLM
  → PendingQuestionEngine            Decision Core (classify + plan execute)
  → PlanBuilder.build()              ExecutionPlan
  → WorkflowPipeline.execute_*       Python executor (not YAML-driven runtime)
  → StatePatchBuffer / apply_state_patches   sole workflow state writer
  → SessionStore.commit_turn()       persist workflow + transcript
  → ResponseComposer                 user-facing copy facade
```

## Leave workflow (semantic)

Leave uses **context-driven field patches**, not regex command routing.

### Stages

| Stage | Session signals | User can |
|-------|-----------------|----------|
| Collect | `pending_question` set | Answer one slot (type, dates, reason, …) |
| Review | `pending_confirmation=submit` or `stage=confirm_submit` | Edit fields naturally, `summary`, `yes`, `cancel` |
| Submitted | `draft.locked` | Start a new request (new dates/narrative) |

### Field intelligence (`field_extractors/leave.py`)

| Mode | Entry | LLM prompt |
|------|-------|------------|
| Collect slot | `collect_slot_field_updates()` | `LEAVE_COLLECT_SLOT_SYSTEM` |
| Review edit | `review_field_updates_from_message()` | `LEAVE_REVIEW_EDIT_SYSTEM` |
| Start / narrative | `ground_leave_understanding()` | `UNDERSTAND_SYSTEM` |

**Policies**

- Canonical `leave_type`: `annual`, `sick`, `lwop` only.
- Non-canonical labels (e.g. casual) → ask, do not infer.
- Review edits are **partial patches** (change start only, end unchanged).
- Review fallback (`_review_edit_fallback`) — keyword hints + date parser when LLM unavailable.
- Long narratives / reasons — **LLM only** via `ground_leave_understanding()`; rules extract dates + explicit types.

### Grounding (`field_engine.ground_leave_understanding`)

1. Review → sanitize LLM/review interpreter patches.
2. Collect with pending slot → keep only pending field (+ `half_day_period` when scope asked).
3. Start / narrative → merge deterministic dates + LLM gaps; strip unknown leave types.

### Pipeline apply path

`WorkflowPipeline._finish_leave_update_turn()` — single path for collect apply, modify, and review re-show.

## Module map

| Layer | Module |
|-------|--------|
| API | `chat/views.py` |
| Orchestrator | `chat/services/orchestrator.py` |
| Turn context | `chat/services/session_memory.py` (`build_turn_context`) |
| Understanding | `chat/services/platform/ai_understanding.py` |
| Decision Core | `chat/services/pending_question_engine.py` |
| Plan + executor | `chat/services/platform/pipeline.py` |
| Workflow schema (YAML) | `chat/services/platform/workflow_definitions/*.yaml` |
| State reducer | `chat/services/session_memory.py` (`reduce_*`, `StatePatchBuffer`) |
| Memory Store | `chat/services/session_store.py` |
| Response | `chat/services/platform/response_composer.py` |
| Informational CRM/policy | `chat/services/informational_responses.py` |
| Leave semantics | `chat/services/platform/field_extractors/leave.py` |
| Turn slot vs meta | `chat/services/platform/turn_semantics.py` |
| LLM prompts | `chat/services/platform/llm_prompts.py` |

## Observability

`classify_leave_field_apply_mode()` labels field writes:

- `semantic_review` — LLM review edit
- `legacy_review_fallback` — LLM off, `_review_edit_fallback` deterministic path
- `collect_slot` — pending question, LLM slot fill
- `collect_deterministic` — pending question, rules-only parse
- `submit_review` — armed for submit confirm

## Tests

- Architecture guards: `tests/test_architecture_guards.py`
- YAML scenarios: `tests/scenarios/*.yaml` + `tests/test_yaml_scenarios.py`
- Python scenarios: `tests/test_user_scenarios.py` (legacy; migrate to YAML over time)
- Review fallback shims: `leave_modify_updates_as_dict()` / `parse_leave_modify_command()` — tests only

## Deprecated (debug / shims only)

- `POST /api/intent/`, `/extract/`, `/decision/` — gated by `ENABLE_LEGACY_DEBUG_ENDPOINTS`
- `intent_detector`, `entity_extractor`, `decision_engine`, `response_formatter` — shims with `DeprecationWarning`
- `WorkflowPipeline.handle()` / `execute_turn()` — use `execute_workflow_turn()`
- `parse_leave_modify_command()` — test shim delegating to `_review_edit_fallback`

## Not in scope of architecture phases

- Live transcript bug fixes (switch regex, suspended summary, etc.) — targeted PRs on top of this stack
- Runtime YAML workflow engine — executor remains Python; YAML defines field schemas + test scenarios only
