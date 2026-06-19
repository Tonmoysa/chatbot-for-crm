from dataclasses import dataclass

from django.conf import settings
from rest_framework import authentication, exceptions


@dataclass
class ServicePrincipal:
    """Non-Django-user principal authenticated via API key."""

    key_id: str = "apikey"

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def pk(self) -> int:
        return 0

    def __str__(self) -> str:
        return "ServicePrincipal"


class ApiKeyAuthentication(authentication.BaseAuthentication):
    """
    API key via X-API-Key header (preferred) or Authorization: ApiKey <key>.
    """

    keyword = b"ApiKey"

    def authenticate(self, request):
        expected = getattr(settings, "HR_SERVICE_API_KEY", "") or ""
        if not expected:
            raise exceptions.AuthenticationFailed(
                "Server misconfiguration: HR_SERVICE_API_KEY is not set."
            )

        key = request.META.get("HTTP_X_API_KEY")
        if not key:
            auth = request.META.get("HTTP_AUTHORIZATION", "")
            if auth.startswith("ApiKey "):
                key = auth.split(" ", 1)[1].strip()

        if key and key != expected:
            raise exceptions.AuthenticationFailed("Invalid API key.")

        if not key:
            return None

        return (ServicePrincipal(), None)
