"""Local dense embeddings via sentence-transformers (no OpenAI /embeddings HTTP)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from django.conf import settings

logger = logging.getLogger("hr_chatbot")

_model: Any = None
_model_lock = threading.Lock()


def _get_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            name = getattr(
                settings,
                "LOCAL_EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            device = getattr(settings, "LOCAL_EMBED_DEVICE", "").strip() or None
            t0 = time.perf_counter()
            logger.info(
                "local_embed_model_loading trace_id=startup name=%s device=%s",
                name,
                device or "auto",
            )
            m = SentenceTransformer(name, device=device)
            ms = int((time.perf_counter() - t0) * 1000)
            dim = int(m.get_sentence_embedding_dimension())
            expected = int(getattr(settings, "QDRANT_VECTOR_SIZE", dim))
            if dim != expected:
                logger.error(
                    "local_embed_dimension_mismatch model_dim=%s QDRANT_VECTOR_SIZE=%s "
                    "Set QDRANT_VECTOR_SIZE=%s or use a new QDRANT_COLLECTION.",
                    dim,
                    expected,
                    dim,
                )
                raise RuntimeError(
                    f"Embedding dimension {dim} does not match QDRANT_VECTOR_SIZE={expected}."
                )
            logger.info(
                "local_embed_model_loaded trace_id=startup name=%s ms=%s dim=%s",
                name,
                ms,
                dim,
            )
            _model = m
        return _model


def encode_texts_local(
    texts: list[str],
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    """Return one float vector per input string (L2-normalized for cosine search)."""
    if not texts:
        return []
    model = _get_model()
    bs = max(1, min(int(batch_size), 128))
    arr = model.encode(
        list(texts),
        batch_size=bs,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    if arr.ndim == 1:
        return [arr.astype(np.float64).tolist()]
    return [np.asarray(row, dtype=np.float64).tolist() for row in arr]
