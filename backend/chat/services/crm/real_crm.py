import logging
import time
from typing import Any

import httpx
from django.conf import settings

from chat.services.crm.base import CRMAdapter, CRMError

logger = logging.getLogger("hr_chatbot")


class RealCRMAdapter(CRMAdapter):
    """PHP HR/CRM integration via REST only (/api/v1/)."""

    def __init__(self) -> None:
        base = settings.PHP_CRM_BASE_URL
        if not base:
            raise CRMError("PHP_CRM_BASE_URL is not configured.")
        self.base = base.rstrip("/") + "/api/v1"
        self.api_key = settings.PHP_CRM_API_KEY or ""
        self.timeout = settings.CRM_HTTP_TIMEOUT_SECONDS
        self.max_retries = settings.CRM_HTTP_MAX_RETRIES

    def _headers(
        self,
        *,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
    ) -> dict[str, str]:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if company_id:
            h["X-Company-Id"] = company_id
        if employee_id:
            h["X-Employee-Id"] = employee_id
        if session_id:
            h["X-Session-Id"] = session_id
        if idempotency_key:
            h["Idempotency-Key"] = idempotency_key
        return h

    def _request(
        self,
        method: str,
        path: str,
        *,
        company_id: str = "",
        employee_id: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        url = f"{self.base}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.request(
                        method,
                        url,
                        headers=self._headers(
                            company_id=company_id,
                            employee_id=employee_id,
                            session_id=session_id,
                            idempotency_key=idempotency_key,
                        ),
                        **kwargs,
                    )
                    if r.status_code >= 500 and attempt < self.max_retries:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    if r.content:
                        return r.json()
                    return {}
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                logger.warning("crm_http_failed url=%s err=%s", url, type(exc).__name__)
                raise CRMError("CRM request failed.", transient=True) from exc
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise CRMError("CRM request failed.", transient=True) from exc
        raise CRMError(str(last_exc or "CRM error"))

    def health(self) -> dict[str, Any]:
        try:
            self._request("GET", "/health/")
            return {"crm": "real", "ok": True}
        except Exception:
            return {"crm": "real", "ok": False}

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
        payload = {
            "company_id": company_id,
            "employee_id": employee_id,
            "session_id": session_id,
            "intent": intent,
            "entities": entities,
            "decision": decision,
        }
        return self._request(
            "POST",
            "/hr-requests/",
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            json=payload,
        )

    def get_request_status(
        self,
        request_id: str,
        *,
        company_id: str,
        employee_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/hr-requests/{request_id}/status/",
            company_id=company_id,
            employee_id=employee_id,
            session_id=session_id,
        )
