import hashlib
import json
import logging
import re
import time
from typing import Any

import httpx
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger("hr_chatbot")

_JSON_HTTP_FAILURES: dict[str, int] = {}
_JSON_CIRCUIT_THRESHOLD = 3

# Once a provider rate-limit (HTTP 429) is hit for a trace, every further LLM
# call in that same chat turn would also fail — so we trip a hard circuit and
# fall back to rules immediately instead of burning latency on doomed retries.
_RATE_LIMIT_TRIPPED: set[str] = set()


def clear_llm_trace_state(trace_id: str = "") -> None:
    """Reset per-trace LLM failure counters / circuits (call once per chat turn)."""
    if not trace_id:
        _JSON_HTTP_FAILURES.clear()
        _RATE_LIMIT_TRIPPED.clear()
        return
    key = trace_id.strip()
    _JSON_HTTP_FAILURES.pop(key, None)
    _RATE_LIMIT_TRIPPED.discard(key)


def _rate_limit_tripped(trace_id: str) -> bool:
    return bool(trace_id) and trace_id in _RATE_LIMIT_TRIPPED


def _maybe_trip_rate_limit(trace_id: str, exc: Exception) -> None:
    if not trace_id:
        return
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        if trace_id not in _RATE_LIMIT_TRIPPED:
            logger.warning("llm_rate_limit_circuit_open trace_id=%s", trace_id)
        _RATE_LIMIT_TRIPPED.add(trace_id)


def _http_error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body_txt = ""
        try:
            body_txt = (exc.response.text or "")[:400]
        except Exception:
            pass
        return f"HTTP {exc.response.status_code} {body_txt}".strip()
    return type(exc).__name__


def _json_circuit_open(trace_id: str) -> bool:
    return _JSON_HTTP_FAILURES.get(trace_id, 0) >= _JSON_CIRCUIT_THRESHOLD


def _record_json_http_failure(trace_id: str) -> None:
    _JSON_HTTP_FAILURES[trace_id] = _JSON_HTTP_FAILURES.get(trace_id, 0) + 1


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


class LLMClient:
    """OpenAI-compatible chat completions; strict JSON-only responses."""

    def __init__(self) -> None:
        self.base = settings.LLM_API_BASE_URL
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.embedding_backend = getattr(
            settings, "EMBEDDING_BACKEND", "openai"
        ).strip().lower()
        self.local_embed_model = getattr(
            settings,
            "LOCAL_EMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.embed_base = getattr(settings, "EMBED_API_BASE_URL", self.base).rstrip("/")
        self.embed_api_key = getattr(settings, "EMBED_API_KEY", self.api_key)
        self.embed_model = getattr(
            settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small"
        )
        self.embed_timeout = float(
            getattr(settings, "EMBED_TIMEOUT_SECONDS", self.timeout)
        )
        self.embed_batch_size = int(getattr(settings, "EMBED_BATCH_SIZE", 64))
        self.embed_cache_ttl = int(getattr(settings, "EMBED_CACHE_TTL_SECONDS", 86_400))

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def is_embedding_configured(self) -> bool:
        if self.embedding_backend == "local":
            return True
        return bool((self.embed_api_key or "").strip())

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace_id: str,
    ) -> dict[str, Any] | None:
        if not self.is_configured():
            return None
        if _rate_limit_tripped(trace_id):
            return None
        if _json_circuit_open(trace_id):
            logger.warning("llm_json_circuit_open trace_id=%s", trace_id)
            return None
        for attempt in range(2):
            use_json_format = attempt == 0
            raw = self._complete(
                system_prompt,
                user_prompt,
                trace_id,
                attempt,
                json_format=use_json_format,
            )
            parsed = self._parse_json_object(raw)
            if parsed is not None:
                return parsed
            logger.warning(
                "llm_invalid_json trace_id=%s attempt=%s json_format=%s",
                trace_id,
                attempt + 1,
                use_json_format,
            )
        return None

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace_id: str,
        max_tokens: int | None = None,
    ) -> str | None:
        """Plain-text completion (no JSON shaping). Used for translation, etc.

        ``max_tokens`` is forwarded to the provider when set — important for
        long translations where the default cap would truncate the answer.
        """
        if not self.is_configured():
            return None
        if _rate_limit_tripped(trace_id):
            return None
        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = int(max_tokens)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            ) or None
        except Exception as exc:
            _maybe_trip_rate_limit(trace_id, exc)
            logger.warning(
                "llm_http_error trace_id=%s err=%s mode=text",
                trace_id,
                _http_error_detail(exc),
            )
            return None

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        trace_id: str,
        attempt: int,
        *,
        json_format: bool = True,
    ) -> str | None:
        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        extra = ""
        if attempt == 1 or not json_format:
            extra = (
                "\nReturn a single JSON object only. No markdown. No explanation."
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt + extra},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        if json_format:
            body["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as exc:
            _maybe_trip_rate_limit(trace_id, exc)
            if json_format:
                _record_json_http_failure(trace_id)
            logger.warning(
                "llm_http_error trace_id=%s err=%s json_format=%s",
                trace_id,
                _http_error_detail(exc),
                json_format,
            )
            return None

    def embed_texts(
        self,
        texts: list[str],
        trace_id: str,
        *,
        model: str | None = None,
    ) -> list[list[float]] | None:
        """
        Embeddings: either local ``sentence-transformers`` or OpenAI-compatible HTTP
        ``/v1/embeddings`` (see ``EMBEDDING_BACKEND`` in settings).
        """
        if not texts:
            return None
        if self.embedding_backend == "local":
            return self._embed_texts_local(texts, trace_id)

        key = (self.embed_api_key or "").strip()
        if not key:
            logger.warning("embed_skipped trace_id=%s reason=no_embed_api_key", trace_id)
            return None

        mdl = model or self.embed_model
        url = f"{self.embed_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if "groq.com" in (self.embed_base or "").lower():
            logger.error(
                "embed_skipped trace_id=%s reason=groq_no_embeddings "
                "Use EMBEDDING_BACKEND=local or set EMBED_API_BASE_URL to an OpenAI-compatible host.",
                trace_id,
            )
            return None
        if "api.openai.com" in (self.embed_base or "").lower() and key.startswith("gsk_"):
            logger.error(
                "embed_skipped trace_id=%s reason=groq_key_on_openai_host "
                "Set EMBED_API_KEY or OPENAI_API_KEY to an OpenAI sk- key.",
                trace_id,
            )
            return None

        out_vectors: list[list[float] | None] = [None] * len(texts)
        batch_size = max(1, self.embed_batch_size)
        ttl = self.embed_cache_ttl
        expected = int(getattr(settings, "QDRANT_VECTOR_SIZE", 1536))

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            to_fetch_texts: list[str] = []
            to_fetch_local_idx: list[int] = []
            for i, t in enumerate(batch):
                global_i = start + i
                ck = f"emb:{mdl}:{hashlib.sha256((t or '').encode('utf-8', errors='ignore')).hexdigest()}"
                if ttl > 0:
                    cached = cache.get(ck)
                    if isinstance(cached, list) and cached:
                        out_vectors[global_i] = cached
                        continue
                to_fetch_local_idx.append(global_i)
                to_fetch_texts.append(t)

            if not to_fetch_texts:
                continue

            body = {"model": mdl, "input": to_fetch_texts}
            vec_result: list[list[float]] | None = None
            t0 = time.perf_counter()
            for attempt in range(2):
                try:
                    with httpx.Client(timeout=self.embed_timeout) as client:
                        r = client.post(url, headers=headers, json=body)
                        r.raise_for_status()
                        data = r.json()
                    rows = sorted(
                        data.get("data") or [],
                        key=lambda x: int(x.get("index", 0)),
                    )
                    parsed: list[list[float]] = []
                    for row in rows:
                        emb = row.get("embedding")
                        if not isinstance(emb, list) or not emb:
                            logger.warning(
                                "embed_invalid_row trace_id=%s attempt=%s",
                                trace_id,
                                attempt + 1,
                            )
                            parsed = []
                            break
                        parsed.append([float(x) for x in emb])
                    if len(parsed) == len(to_fetch_texts):
                        vec_result = parsed
                        break
                except Exception as exc:
                    detail = type(exc).__name__
                    if isinstance(exc, httpx.HTTPStatusError):
                        body_txt = ""
                        try:
                            body_txt = (exc.response.text or "")[:400]
                        except Exception:
                            pass
                        detail = f"HTTP {exc.response.status_code} {body_txt}"
                    logger.warning(
                        "embed_http_error trace_id=%s attempt=%s err=%s",
                        trace_id,
                        attempt + 1,
                        detail,
                    )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "embedding_batch trace_id=%s model=%s batch=%s ms=%s ok=%s",
                trace_id,
                mdl,
                len(to_fetch_texts),
                elapsed_ms,
                bool(vec_result),
            )
            if not vec_result:
                return None
            for j, vec in enumerate(vec_result):
                if len(vec) != expected:
                    logger.warning(
                        "embed_dimension_mismatch trace_id=%s expected=%s got=%s",
                        trace_id,
                        expected,
                        len(vec),
                    )
                    return None
                global_i = to_fetch_local_idx[j]
                out_vectors[global_i] = vec
                if ttl > 0:
                    t = texts[global_i]
                    ck = f"emb:{mdl}:{hashlib.sha256((t or '').encode('utf-8', errors='ignore')).hexdigest()}"
                    cache.set(ck, vec, ttl)

        if any(v is None for v in out_vectors):
            return None
        return [v for v in out_vectors if v is not None]

    def _embed_texts_local(
        self,
        texts: list[str],
        trace_id: str,
    ) -> list[list[float]] | None:
        from knowledge_base.services.local_embeddings import encode_texts_local

        mname = self.local_embed_model
        out_vectors: list[list[float] | None] = [None] * len(texts)
        batch_size = max(1, min(self.embed_batch_size, 64))
        ttl = self.embed_cache_ttl
        expected = int(getattr(settings, "QDRANT_VECTOR_SIZE", 384))

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            to_fetch_texts: list[str] = []
            to_fetch_local_idx: list[int] = []
            for i, t in enumerate(batch):
                global_i = start + i
                ck = f"emb:local:{mname}:{hashlib.sha256((t or '').encode('utf-8', errors='ignore')).hexdigest()}"
                if ttl > 0:
                    cached = cache.get(ck)
                    if isinstance(cached, list) and cached:
                        out_vectors[global_i] = cached
                        continue
                to_fetch_local_idx.append(global_i)
                to_fetch_texts.append(t)

            if not to_fetch_texts:
                continue

            t0 = time.perf_counter()
            try:
                parsed = encode_texts_local(
                    to_fetch_texts,
                    batch_size=min(32, batch_size),
                )
            except Exception as exc:
                logger.warning(
                    "embed_local_error trace_id=%s err=%s",
                    trace_id,
                    type(exc).__name__,
                )
                return None
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "embedding_batch_local trace_id=%s model=%s batch=%s ms=%s ok=%s",
                trace_id,
                mname,
                len(to_fetch_texts),
                elapsed_ms,
                len(parsed) == len(to_fetch_texts),
            )
            if len(parsed) != len(to_fetch_texts):
                return None
            for j, vec in enumerate(parsed):
                if len(vec) != expected:
                    logger.warning(
                        "embed_dimension_mismatch trace_id=%s expected=%s got=%s",
                        trace_id,
                        expected,
                        len(vec),
                    )
                    return None
                global_i = to_fetch_local_idx[j]
                out_vectors[global_i] = vec
                if ttl > 0:
                    t = texts[global_i]
                    ck = f"emb:local:{mname}:{hashlib.sha256((t or '').encode('utf-8', errors='ignore')).hexdigest()}"
                    cache.set(ck, vec, ttl)

        if any(v is None for v in out_vectors):
            return None
        return [v for v in out_vectors if v is not None]

    def _parse_json_object(self, content: str | None) -> dict[str, Any] | None:
        if not content:
            return None
        try:
            parsed = json.loads(_strip_json_fence(content))
            if isinstance(parsed, dict):
                return parsed
            # Some providers occasionally return a single-item JSON array even when
            # asked for an object. Accept it to avoid breaking the pipeline.
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
            return None
        except json.JSONDecodeError:
            return None
