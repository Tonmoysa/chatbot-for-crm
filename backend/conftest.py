import os

import pytest
from rest_framework.test import APIClient

from chat.authentication import ServicePrincipal

os.environ.setdefault("KB_RAG_ENABLED", "false")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("HR_SERVICE_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _pipeline_handle_auto_understand(monkeypatch):
    """Unit tests that call WorkflowPipeline.handle directly must supply understanding."""
    from chat.services.platform.ai_understanding import AIUnderstandingLayer
    from chat.services.platform.pipeline import WorkflowPipeline

    original_handle = WorkflowPipeline.handle
    layer = AIUnderstandingLayer()

    def handle_with_understanding(self, message, *, memory, pq_decision, understanding=None, **kwargs):
        if understanding is None:
            understanding = layer.understand(
                message,
                memory=memory,
                conversation_history=kwargs.get("conversation_history") or [],
                trace_id=kwargs.get("trace_id") or "",
            )
        return original_handle(
            self,
            message,
            memory=memory,
            pq_decision=pq_decision,
            understanding=understanding,
            **kwargs,
        )

    monkeypatch.setattr(WorkflowPipeline, "handle", handle_with_understanding)


@pytest.fixture
def api_client(settings):
    settings.HR_SERVICE_API_KEY = "test-key"
    c = APIClient()
    c.force_authenticate(user=ServicePrincipal())
    c.defaults["HTTP_X_API_KEY"] = "test-key"
    return c
