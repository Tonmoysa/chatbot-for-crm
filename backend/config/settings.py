"""
Django settings for HR AI chatbot microservice.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-dev-only")
DEBUG = os.environ.get("DEBUG", "true").lower() in ("1", "true", "yes")

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "drf_spectacular",
    "chat",
    "knowledge_base",
    "voice",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "chat.middleware.TraceIdMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- HR microservice configuration ---
HR_SERVICE_API_KEY = os.environ.get("HR_SERVICE_API_KEY", "")
USE_MOCK_CRM = os.environ.get("USE_MOCK", "true").lower() in ("1", "true", "yes")
USE_TURN_CONTEXT = os.environ.get("USE_TURN_CONTEXT", "true").lower() in ("1", "true", "yes")
USE_DECISION_CORE = True
USE_LEAVE_PLAN = os.environ.get("USE_LEAVE_PLAN", "true").lower() in ("1", "true", "yes")
# Phase 10 — architecture cleanup complete; legacy orchestrator path removed (no effect).
ENABLE_LEGACY_PATH = False
# Deprecated debug endpoints (intent/extract/decision); production uses POST /chat/ only.
ENABLE_LEGACY_DEBUG_ENDPOINTS = os.environ.get("ENABLE_LEGACY_DEBUG_ENDPOINTS", "false").lower() in (
    "1",
    "true",
    "yes",
)
# Phase 8 — expense plan executor + platform-only routing for expense turns.
EXPENSE_NEW_ARCH = os.environ.get("EXPENSE_NEW_ARCH", "true").lower() in ("1", "true", "yes")
# Phase 4 — informational turns via PlanBuilder (policy, status, OOS, greeting).
USE_INFORMATIONAL_PLAN = os.environ.get("USE_INFORMATIONAL_PLAN", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Phase 10 — consolidated turn_complete log (context → understanding → decision → plan → state → response).
FULL_TURN_OBSERVABILITY = os.environ.get("FULL_TURN_OBSERVABILITY", "true").lower() in (
    "1",
    "true",
    "yes",
)

PHP_CRM_BASE_URL = os.environ.get("PHP_CRM_BASE_URL", "").rstrip("/")
PHP_CRM_API_KEY = os.environ.get("PHP_CRM_API_KEY", "")

CRM_HTTP_TIMEOUT_SECONDS = float(os.environ.get("CRM_HTTP_TIMEOUT_SECONDS", "5"))
CRM_HTTP_MAX_RETRIES = int(os.environ.get("CRM_HTTP_MAX_RETRIES", "2"))

LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
# Optional dedicated model for expense draft interpreter (smaller / faster on Groq).
LLM_EXPENSE_MODEL = os.environ.get("LLM_EXPENSE_MODEL", "").strip() or LLM_MODEL
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "25"))
LLM_MESSAGE_POLISH = os.environ.get("LLM_MESSAGE_POLISH", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Voice / STT (Phase 2 — OpenAI Whisper; unused when frontend uses Web Speech API)
OPENAI_WHISPER_API_KEY = os.environ.get(
    "OPENAI_WHISPER_API_KEY",
    os.environ.get("OPENAI_API_KEY", ""),
).strip()
OPENAI_WHISPER_API_BASE_URL = os.environ.get(
    "OPENAI_WHISPER_API_BASE_URL", "https://api.openai.com/v1"
).rstrip("/")
OPENAI_WHISPER_MODEL = os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1")
OPENAI_WHISPER_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_WHISPER_TIMEOUT_SECONDS", "60"))
VOICE_STT_PROVIDER = os.environ.get("VOICE_STT_PROVIDER", "openai_whisper").strip().lower()
VOICE_MAX_UPLOAD_BYTES = int(os.environ.get("VOICE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
VOICE_MAX_DURATION_SECONDS = float(os.environ.get("VOICE_MAX_DURATION_SECONDS", "120"))

# HTTP embeddings (when EMBEDDING_BACKEND=openai). If host differs from LLM, do not reuse chat key.
_embed_base_raw = os.environ.get("EMBED_API_BASE_URL", "").strip().rstrip("/")
EMBED_API_BASE_URL = _embed_base_raw or LLM_API_BASE_URL
if EMBED_API_BASE_URL.rstrip("/") == LLM_API_BASE_URL.rstrip("/"):
    EMBED_API_KEY = (
        os.environ.get("EMBED_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or LLM_API_KEY
    )
else:
    EMBED_API_KEY = (
        os.environ.get("EMBED_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )

# openai = remote /v1/embeddings; local = sentence-transformers on this machine
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "openai").strip().lower()
LOCAL_EMBED_MODEL = os.environ.get(
    "LOCAL_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LOCAL_EMBED_DEVICE = os.environ.get("LOCAL_EMBED_DEVICE", "").strip()

# --- Embeddings + Qdrant (RAG) ---
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBEDDING_VERSION = os.environ.get("EMBEDDING_VERSION", "").strip()
EMBED_TIMEOUT_SECONDS = float(os.environ.get("EMBED_TIMEOUT_SECONDS", "30"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
EMBED_CACHE_TTL_SECONDS = int(os.environ.get("EMBED_CACHE_TTL_SECONDS", "86400"))

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
# Default aligns with local dev + sentence-transformers; override if you use another name.
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "hr_policies_local").strip()
_qdrant_vs_env = os.environ.get("QDRANT_VECTOR_SIZE", "").strip()
if EMBEDDING_BACKEND == "local" and not _qdrant_vs_env:
    QDRANT_VECTOR_SIZE = 384
else:
    QDRANT_VECTOR_SIZE = int(_qdrant_vs_env or "1536")
QDRANT_TIMEOUT_SECONDS = float(os.environ.get("QDRANT_TIMEOUT_SECONDS", "180"))
QDRANT_UPSERT_BATCH_SIZE = int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "128"))
QDRANT_UPSERT_WAIT = os.environ.get("QDRANT_UPSERT_WAIT", "false").lower() in (
    "1",
    "true",
    "yes",
)

KB_RAG_ENABLED = os.environ.get("KB_RAG_ENABLED", "true").lower() in ("1", "true", "yes")
KB_RAG_NOT_FOUND_MESSAGE = os.environ.get("KB_RAG_NOT_FOUND_MESSAGE", "").strip() or None
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "12"))
# Qdrant cosine first pass (server-side). If no hits, retriever retries without threshold
# and keeps chunks with score >= RAG_MIN_SIMILARITY (soft ranking may still return low scores).
# Conservative defaults biased toward recall; grounded-generation still filters weak evidence.
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.38"))
RAG_MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", "0.24"))
# Retrieve more candidates on the relaxed pass, then clamp to TOP_K after min-sim filtering.
RAG_RELAXED_CANDIDATE_MULTIPLIER = int(os.environ.get("RAG_RELAXED_CANDIDATE_MULTIPLIER", "3"))
RAG_MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "10000"))
RAG_QUERY_DEBUG = os.environ.get("RAG_QUERY_DEBUG", "false").lower() in ("1", "true", "yes")

KB_MAX_UPLOAD_BYTES = int(os.environ.get("KB_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
KB_MAX_EXTRACT_CHARS = int(os.environ.get("KB_MAX_EXTRACT_CHARS", "200000"))
KB_CHUNK_TARGET_TOKENS = int(os.environ.get("KB_CHUNK_TARGET_TOKENS", "500"))
KB_CHUNK_OVERLAP_TOKENS = int(os.environ.get("KB_CHUNK_OVERLAP_TOKENS", "140"))

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "hr-chatbot-local",
    }
}

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["chat.authentication.ApiKeyAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "user": "180/minute",
    },
    "EXCEPTION_HANDLER": "chat.exceptions.hr_exception_handler",
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

ENABLE_API_DOCS = os.environ.get("ENABLE_API_DOCS", "true").lower() in (
    "1",
    "true",
    "yes",
)

SPECTACULAR_SETTINGS = {
    "TITLE": "HR AI Chatbot API",
    "DESCRIPTION": "HR chatbot microservice. Send `X-API-Key` on protected routes. "
    "Chat responses include `X-Session-Id` for multi-turn `session_id`.",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "APPEND_COMPONENTS": {
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Same value as server env `HR_SERVICE_API_KEY`.",
            }
        }
    },
    "SECURITY": [{"ApiKeyAuth": []}],
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "jsonish": {
            "format": "%(levelname)s %(name)s trace_id=%(trace_id)s %(message)s",
        },
        "simple": {"format": "%(levelname)s %(name)s %(message)s"},
    },
    "filters": {
        "require_debug_false": {"()": "django.utils.log.RequireDebugFalse"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "loggers": {
        "hr_chatbot": {"handlers": ["console"], "level": "INFO"},
        "django.request": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}
