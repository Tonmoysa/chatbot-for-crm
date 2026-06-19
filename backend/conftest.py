import os

import pytest
from rest_framework.test import APIClient

from chat.authentication import ServicePrincipal

os.environ.setdefault("KB_RAG_ENABLED", "false")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("HR_SERVICE_API_KEY", "test-key")


@pytest.fixture
def api_client(settings):
    settings.HR_SERVICE_API_KEY = "test-key"
    c = APIClient()
    c.force_authenticate(user=ServicePrincipal())
    c.defaults["HTTP_X_API_KEY"] = "test-key"
    return c
