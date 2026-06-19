from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rest_framework import serializers
from rest_framework.exceptions import ValidationError


@dataclass(frozen=True)
class RequestIdentity:
    company_id: str
    employee_id: str
    session_id: str
    idempotency_key: str = ""


class TenantIdentitySerializerMixin(serializers.Serializer):
    company_id = serializers.CharField(max_length=64)
    employee_id = serializers.CharField(max_length=64)
    session_id = serializers.CharField(max_length=64)
    idempotency_key = serializers.CharField(
        max_length=128,
        required=False,
        allow_blank=True,
        default="",
    )


def _value_from_request(request, key: str) -> str:
    header = "HTTP_X_" + key.upper().replace("_", "_")
    candidates = (
        request.data.get(key) if hasattr(request, "data") else None,
        request.query_params.get(key) if hasattr(request, "query_params") else None,
        request.META.get(header),
    )
    if key == "idempotency_key":
        candidates = (*candidates, request.META.get("HTTP_IDEMPOTENCY_KEY"))
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def identity_from_validated_data(data: dict[str, Any]) -> RequestIdentity:
    company_id = str(data.get("company_id") or "").strip()
    employee_id = str(data.get("employee_id") or "").strip()
    session_id = str(data.get("session_id") or "").strip()
    if not company_id or not employee_id or not session_id:
        raise ValidationError(
            {
                "detail": (
                    "company_id, employee_id, and session_id are required for "
                    "tenant-scoped requests."
                )
            }
        )
    return RequestIdentity(
        company_id=company_id,
        employee_id=employee_id,
        session_id=session_id,
        idempotency_key=str(data.get("idempotency_key") or "").strip(),
    )


def identity_from_request(request) -> RequestIdentity:
    data = {
        "company_id": _value_from_request(request, "company_id"),
        "employee_id": _value_from_request(request, "employee_id"),
        "session_id": _value_from_request(request, "session_id"),
        "idempotency_key": _value_from_request(request, "idempotency_key"),
    }
    return identity_from_validated_data(data)
