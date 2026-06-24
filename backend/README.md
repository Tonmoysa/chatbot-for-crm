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

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the Phase 1–10 pipeline.

Production path: `POST /api/chat/` only.

| Layer | Module |
|-------|--------|
| API | `chat/views.py` |
| Orchestrator | `chat/services/orchestrator.py` |
| Understanding | `chat/services/platform/ai_understanding.py` |
| Decision Core | `chat/services/pending_question_engine.py` |
| Plan + executor | `chat/services/platform/pipeline.py` |
| State + memory | `chat/services/session_memory.py`, `session_store.py` |
| Response | `chat/services/platform/response_composer.py` |
| CRM | `chat/services/crm/` |

Deprecated debug routes (`/api/intent/`, `/extract/`, `/decision/`) default off — use `ENABLE_LEGACY_DEBUG_ENDPOINTS=true` only for local debugging.

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
