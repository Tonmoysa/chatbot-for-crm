const STORAGE_PREFIX = "hr-chatbot-session-id";

export function newSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function runtimeIdentity() {
  if (typeof window === "undefined") return {};
  return window.__HR_CHATBOT_IDENTITY__ || {};
}

function envIdentity() {
  return {
    company_id: import.meta.env.VITE_COMPANY_ID?.trim() || "",
    employee_id: import.meta.env.VITE_EMPLOYEE_ID?.trim() || "",
  };
}

function scopedStorageKey(companyId, employeeId) {
  return `${STORAGE_PREFIX}:${companyId}:${employeeId}`;
}

export function getSessionId(identity = {}) {
  const companyId = (identity.company_id || identity.companyId || "").trim();
  const employeeId = (identity.employee_id || identity.employeeId || "").trim();
  const key =
    companyId && employeeId
      ? scopedStorageKey(companyId, employeeId)
      : `${STORAGE_PREFIX}:unscoped`;
  try {
    const existing = localStorage.getItem(key);
    if (existing && existing.trim()) {
      return existing.trim();
    }
  } catch {
    /* ignore */
  }
  const id = newSessionId();
  setSessionId(id, identity);
  return id;
}

export function setSessionId(sessionId, identity = {}) {
  const v = (sessionId || "").trim();
  if (!v) return;
  const companyId = (identity.company_id || identity.companyId || "").trim();
  const employeeId = (identity.employee_id || identity.employeeId || "").trim();
  const key =
    companyId && employeeId
      ? scopedStorageKey(companyId, employeeId)
      : `${STORAGE_PREFIX}:unscoped`;
  try {
    localStorage.setItem(key, v);
  } catch {
    /* ignore */
  }
}

/** Start a fresh chat thread (new session id in local storage). */
export function rotateSessionId(identity) {
  const id = newSessionId();
  setSessionId(id, identity);
  return id;
}

export function getClientIdentity() {
  const runtime = runtimeIdentity();
  const env = envIdentity();
  const companyId = (runtime.company_id || runtime.companyId || env.company_id || "").trim();
  const employeeId = (runtime.employee_id || runtime.employeeId || env.employee_id || "").trim();
  if (!companyId || !employeeId) {
    throw new Error("Missing company_id or employee_id from CRM identity bootstrap.");
  }
  const sessionId = (
    runtime.session_id ||
    runtime.sessionId ||
    getSessionId({ company_id: companyId, employee_id: employeeId })
  ).trim();
  return {
    company_id: companyId,
    employee_id: employeeId,
    session_id: sessionId,
  };
}
