import axios from "axios";
import { getClientIdentity } from "../utils/session.js";
import { generateTraceId } from "../utils/trace.js";

const resolvedBaseURL = () => {
  const fromEnv = import.meta.env.VITE_API_BASE_URL?.trim();
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  if (import.meta.env.DEV) return "";
  return "http://127.0.0.1:8000";
};

const client = axios.create({
  baseURL: resolvedBaseURL(),
  headers: { "Content-Type": "application/json" },
  timeout: 120_000,
});

client.interceptors.request.use((config) => {
  const key = import.meta.env.VITE_HR_API_KEY?.trim();
  if (key) {
    config.headers["X-API-Key"] = key;
  }
  if (!config.headers["X-Trace-Id"] && !config.headers["x-trace-id"]) {
    config.headers["X-Trace-Id"] = generateTraceId();
  }
  return config;
});

/**
 * POST /api/chat/
 * @returns {{ data: object, sessionIdHeader: string | null }}
 */
function resolveIdentity(identity) {
  return identity || getClientIdentity();
}

function appendIdentity(form, identity) {
  form.append("company_id", identity.company_id);
  form.append("employee_id", identity.employee_id);
  form.append("session_id", identity.session_id);
  if (identity.idempotency_key) {
    form.append("idempotency_key", identity.idempotency_key);
  }
}

/**
 * GET /api/chat/sessions/
 */
export async function fetchChatSessions({ identity, limit = 30 } = {}) {
  const requestIdentity = resolveIdentity(identity);
  const res = await client.get("/api/chat/sessions/", {
    params: {
      company_id: requestIdentity.company_id,
      employee_id: requestIdentity.employee_id,
      limit,
    },
  });
  return {
    sessions: Array.isArray(res.data?.sessions) ? res.data.sessions : [],
  };
}

/**
 * GET /api/chat/sessions/:sessionId/
 */
export async function fetchChatSession({ sessionId, identity }) {
  const requestIdentity = resolveIdentity(identity);
  const sid = (sessionId || requestIdentity.session_id || "").trim();
  const res = await client.get(`/api/chat/sessions/${encodeURIComponent(sid)}/`, {
    params: {
      company_id: requestIdentity.company_id,
      employee_id: requestIdentity.employee_id,
    },
  });
  return {
    sessionId: res.data?.session_id || sid,
    messages: Array.isArray(res.data?.messages) ? res.data.messages : [],
  };
}

export async function postChat({ message, sessionId, documentText, identity }) {
  const requestIdentity = resolveIdentity(identity);
  const res = await client.post("/api/chat/", {
    company_id: requestIdentity.company_id,
    employee_id: requestIdentity.employee_id,
    session_id: sessionId || requestIdentity.session_id,
    message,
    document_text: documentText || "",
    idempotency_key: requestIdentity.idempotency_key || "",
  });
  const sessionIdHeader =
    res.headers["x-session-id"] ?? res.headers["X-Session-Id"] ?? null;
  return {
    data: res.data,
    sessionIdHeader: typeof sessionIdHeader === "string" ? sessionIdHeader : null,
  };
}

/**
 * POST /api/document/extract/ (multipart)
 * @returns {{ data: object, documentText: string }}
 */
export async function postDocumentExtract({ file, identity }) {
  const requestIdentity = resolveIdentity(identity);
  const form = new FormData();
  form.append("file", file);
  appendIdentity(form, requestIdentity);
  const res = await client.post("/api/document/extract/", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return {
    data: res.data,
    documentText: typeof res.data?.document_text === "string" ? res.data.document_text : "",
  };
}

/**
 * POST /api/voice/transcribe/ (Phase 2 — OpenAI Whisper via backend).
 * @returns {Promise<{ data: object, transcript: string, traceId: string | null }>}
 */
export async function postVoiceTranscribe({ blob, mimeType, language, traceId, identity }) {
  const requestIdentity = resolveIdentity(identity);
  const form = new FormData();
  const ext = mimeType?.includes("mp4") ? "m4a" : "webm";
  form.append("file", blob, `recording.${ext}`);
  appendIdentity(form, requestIdentity);
  if (language) {
    form.append("language", language);
  }
  const headers = { "Content-Type": "multipart/form-data" };
  if (traceId) {
    headers["X-Trace-Id"] = traceId;
  }
  const res = await client.post("/api/voice/transcribe/", form, { headers });
  const traceIdHeader =
    res.headers["x-trace-id"] ?? res.headers["X-Trace-Id"] ?? traceId ?? null;
  const transcript =
    typeof res.data?.transcript === "string"
      ? res.data.transcript
      : typeof res.data?.response?.message === "string"
        ? res.data.response.message
        : "";
  return {
    data: res.data,
    transcript,
    traceId: typeof traceIdHeader === "string" ? traceIdHeader : null,
  };
}
