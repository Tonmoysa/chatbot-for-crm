import datetime as dt
import pytest
from tests.test_scenario_36_messages import (
    SCENARIO_STEPS,
    _assert_step,
    _patch_dates,
    _patch_no_llm,
    _patch_policy_rag_miss,
    _patch_polish_passthrough,
    COMPANY_ID,
    EMP,
)
from chat.services.orchestrator import ChatOrchestrator

class S:
    KB_RAG_ENABLED = True

def main():
    import chat.services.orchestrator as orch_mod
    monkeypatch = pytest.MonkeyPatch()
    _patch_dates(monkeypatch)
    _patch_no_llm(monkeypatch)
    _patch_policy_rag_miss(monkeypatch)
    _patch_polish_passthrough(monkeypatch)
    monkeypatch.setattr(
        "chat.services.orchestrator.conversational_reply",
        lambda **_k: "আমি শুধু HR বিষয়ে সাহায্য করি।",
    )
    orch = ChatOrchestrator()
    sid = None
    for step in SCENARIO_STEPS:
        result = orch.run_chat(
            company_id=COMPANY_ID,
            message=step.message,
            session_id=sid,
            employee_id=EMP,
            trace_id=f"dbg-{step.id:02d}",
        )
        sid = result["_session_id"]
        session = orch.memory.get_or_create_session(
            company_id=COMPANY_ID, employee_id=EMP, session_id=sid
        )
        wf = dict(session.workflow_state or {})
        failures = _assert_step(step, result, wf)
        if failures:
            body = (result.get("response") or {}).get("message") or ""
            print(f"\n=== STEP {step.id} FAIL ===")
            print("msg:", step.message[:60])
            print("failures:", failures)
            print("reply:", body[:400])
            print("intent:", result.get("intent"), "outcome:", (result.get("decision") or {}).get("outcome"))
    monkeypatch.undo()

if __name__ == "__main__":
    main()
