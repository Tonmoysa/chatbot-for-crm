# HR AI Chatbot (Django REST Framework microservice)

Production-oriented HR assistant service: **LLM is used only for intent detection and entity extraction**. All approvals and business outcomes are decided by a **rule-based decision engine**. CRM integration is **REST-only** (no shared database with PHP).

## Security

- **Never commit API keys or paste them into chat.** Configure secrets only via environment variables or a local `.env` file (ignored by git).
- If a key was exposed, **revoke and rotate it** in the provider dashboard before continuing.

## Requirements

- Python 3.11+
- Optional: Redis (`REDIS_URL`) for shared caching across processes (otherwise LocMem cache is used for rate limiting and non-session data).

## Quick start (local, Mock CRM)

```powershell
cd f:\chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

- Set `HR_SERVICE_API_KEY` to a strong random string (clients send it as `X-API-Key`).
- For LLM calls, set `LLM_API_KEY` (and optionally `LLM_API_BASE_URL` / `LLM_MODEL`).  
  - OpenAI: default base URL works.  
  - Groq (OpenAI-compatible): set `LLM_API_BASE_URL=https://api.groq.com/openai/v1` and a Groq model name.

```powershell
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

### OpenAPI / Swagger

After `runserver`, open:

- **Swagger UI:** `http://127.0.0.1:8000/api/docs/`
- **ReDoc:** `http://127.0.0.1:8000/api/redoc/`
- **OpenAPI schema:** `http://127.0.0.1:8000/api/schema/`

The UI is public by default; use **Authorize** and send **`X-API-Key`** (same value as `HR_SERVICE_API_KEY`) for “Try it out” on protected routes. Set **`ENABLE_API_DOCS=false`** in `.env` to hide these URLs in production if you prefer.

### Example requests

Health (no API key):

```bash
curl -s http://127.0.0.1:8000/api/health/
```

Chat (API key required):

```bash
curl -s http://127.0.0.1:8000/api/chat/ ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: YOUR_KEY" ^
  -H "X-Trace-Id: optional-trace" ^
  -d "{\"message\":\"What is my PTO balance?\",\"employee_id\":\"E001\"}"
```

The response includes `X-Session-Id` for multi-turn conversations; send it back as `session_id` in the JSON body on the next call.

Mock CRM helpers:

```bash
curl -s "http://127.0.0.1:8000/api/mock/leave-balance/?employee_id=E001" -H "X-API-Key: YOUR_KEY"
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `HR_SERVICE_API_KEY` | **Required.** Authenticates callers (`X-API-Key`). |
| `USE_MOCK` | `true` (default) for `MockCRM`; `false` for PHP REST adapter. |
| `PHP_CRM_BASE_URL` | Base site URL (adapter calls `/api/v1/...`). |
| `PHP_CRM_API_KEY` | Optional bearer token for PHP API. |
| `CRM_HTTP_TIMEOUT_SECONDS` | Default `5`. |
| `CRM_HTTP_MAX_RETRIES` | Default `2`. |
| `LLM_API_KEY` / `OPENAI_API_KEY` | LLM provider secret. |
| `LLM_API_BASE_URL` | OpenAI-compatible base URL. |
| `LLM_MODEL` | Model id. |
| `REDIS_URL` | Optional `redis://...:6379/0`. |

## Architecture (layers)

1. **API (DRF)** — `chat/views.py`, `chat/serializers.py`
2. **Orchestrator** — `chat/services/orchestrator.py`
3. **Intent** — `chat/services/intent_detector.py` (+ rules)
4. **Entity extraction** — `chat/services/entity_extractor.py` (+ rules)
5. **Decision engine** — `chat/services/decision_engine.py` (rules only)
6. **CRM adapters** — `chat/services/crm/` (`MockCRMAdapter`, `RealCRMAdapter`)
7. **Conversation memory** — `chat/services/memory_store.py` + `chat/models.py`
8. **Response formatting** — `chat/services/response_formatter.py`
9. **Observability** — `TraceIdMiddleware`, structured logs under logger `hr_chatbot`

## Endpoints

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/health/` | No |
| POST | `/api/chat/` | API key |
| POST | `/api/intent/` | API key |
| POST | `/api/extract/` | API key |
| POST | `/api/decision/` | API key |
| GET | `/api/status/<id>/` | API key |
| POST | `/api/mock/request-create/` | API key |
| GET | `/api/mock/request-status/?request_id=` | API key |
| GET | `/api/mock/leave-balance/?employee_id=` | API key |

## Response contract

Successful and error-shaped API bodies follow:

```json
{
  "trace_id": "",
  "intent": "",
  "entities": {},
  "decision": {},
  "response": {
    "message": "",
    "status": "",
    "request_id": ""
  },
  "status": "success | failed"
}
```

## Tests

```powershell
pytest -q
```

## PHP integration (later)

Set `USE_MOCK=false` and `PHP_CRM_BASE_URL` to your CRM origin. Implement or align PHP routes under `/api/v1/` as used in `chat/services/crm/real_crm.py` (leave balance, create HR request, request status). This service never connects to the PHP database directly.
