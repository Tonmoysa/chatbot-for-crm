"""Embedding helpers; use ``LLMClient.embed_texts`` (``EMBEDDING_BACKEND=openai`` or ``local``)."""

from __future__ import annotations

from chat.services.llm_client import LLMClient


def embed_strings(texts: list[str], trace_id: str) -> list[list[float]] | None:
    """Batch-embed texts using the shared LLM client."""
    return LLMClient().embed_texts(texts, trace_id)
