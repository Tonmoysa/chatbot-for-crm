import uuid
from datetime import datetime
from typing import Any

from chat.services.crm.base import CRMAdapter


_MOCK_SINGLETON: "MockCRMAdapter | None" = None


def get_mock_singleton() -> "MockCRMAdapter":
    global _MOCK_SINGLETON
    if _MOCK_SINGLETON is None:
        _MOCK_SINGLETON = MockCRMAdapter()
    return _MOCK_SINGLETON


class MockCRMAdapter(CRMAdapter):
    """In-memory CRM for local testing."""

    def __init__(self) -> None:
        self._requests: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[tuple[str, str], str] = {}

    def health(self) -> dict[str, Any]:
        return {"crm": "mock", "ok": True}

    def _identity(self, company_id: str, employee_id: str, session_id: str) -> tuple[str, str]:
        company = (company_id or "").strip()
        emp = (employee_id or "").strip()
        sid = (session_id or "").strip()
        if not company or not emp or not sid:
            raise ValueError("company_id, employee_id, and session_id are required.")
        return company, emp

    def create_request(
        self,
        *,
        company_id: str,
        employee_id: str,
        session_id: str,
        intent: str,
        entities: dict[str, Any],
        decision: dict[str, Any],
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        company, emp = self._identity(company_id, employee_id, session_id)
        idem = (idempotency_key or "").strip()
        if idem and (company, idem) in self._idempotency:
            rid = self._idempotency[(company, idem)]
            return {"request_id": rid, "record": self._requests[rid], "_idempotent_replay": True}

        rid = f"MOCK-{uuid.uuid4().hex[:10].upper()}"
        crm_status = self._initial_status(decision)
        self._requests[rid] = {
            "request_id": rid,
            "company_id": company,
            "employee_id": emp,
            "session_id": session_id,
            "idempotency_key": idem,
            "intent": intent,
            "entities": entities,
            "decision": decision,
            "status": crm_status,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        if idem:
            self._idempotency[(company, idem)] = rid
        return {"request_id": rid, "record": self._requests[rid]}

    def _initial_status(self, decision: dict[str, Any]) -> str:
        outcome = (decision or {}).get("outcome")
        mapping = {
            "AUTO_APPROVED": "APPROVED",
            "APPROVED": "APPROVED",
            "SUBMITTED": "PENDING",
            "REJECTED": "REJECTED",
            "PENDING_APPROVAL": "PENDING",
            "PENDING_REVIEW": "PENDING_REVIEW",
            "INFORMATIONAL": "COMPLETED",
            "NEEDS_CLARIFICATION": "DRAFT",
        }
        return mapping.get(str(outcome), "PENDING")

    def get_request_status(
        self,
        request_id: str,
        *,
        company_id: str,
        employee_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        company, emp = self._identity(company_id, employee_id, session_id)
        rec = self._requests.get(request_id)
        if not rec or rec.get("company_id") != company or rec.get("employee_id") != emp:
            return {"request_id": request_id, "status": "NOT_FOUND", "detail": "Unknown request"}
        return {
            "request_id": request_id,
            "status": rec["status"],
            "intent": rec.get("intent"),
            "updated_at": rec.get("created_at"),
        }
