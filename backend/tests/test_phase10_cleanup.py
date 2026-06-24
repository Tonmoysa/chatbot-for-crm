"""Phase 10 — architecture cleanup guards."""

from __future__ import annotations

from pathlib import Path


def test_architecture_doc_exists():
    root = Path(__file__).resolve().parents[1]
    doc = root / "docs" / "ARCHITECTURE.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "SessionStore" in text
    assert "ResponseComposer" in text
    assert "execute_workflow_turn" in text


def test_orchestrator_has_no_dead_routed_turn_helper():
    root = Path(__file__).resolve().parents[1]
    source = (root / "chat" / "services" / "orchestrator.py").read_text(encoding="utf-8")
    assert "_complete_routed_turn" not in source
    assert "ResponseComposer" not in source


def test_production_services_avoid_legacy_shims():
    """Deprecated modules are only used from debug views and deprecation tests."""
    root = Path(__file__).resolve().parents[1]
    production = [
        root / "chat" / "services" / "orchestrator.py",
        root / "chat" / "services" / "pending_question_engine.py",
        root / "chat" / "services" / "platform" / "pipeline.py",
    ]
    forbidden = [
        "from chat.services.intent_detector",
        "from chat.services.entity_extractor",
        "from chat.services.decision_engine",
        "from chat.services.response_formatter",
    ]
    for path in production:
        text = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in text]
        assert hits == [], f"{path.name} must not import legacy shims: {hits}"


def test_legacy_path_disabled_in_settings():
    from django.conf import settings

    assert settings.ENABLE_LEGACY_PATH is False
