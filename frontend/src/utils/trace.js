/**
 * Client-side trace IDs for speech observability (aligns with backend X-Trace-Id).
 */

export function generateTraceId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `speech-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

const SPEECH_LOG_PREFIX = "[speech]";

/**
 * @param {string} event
 * @param {Record<string, unknown>} [extra]
 */
export function logSpeechEvent(event, extra = {}) {
  if (import.meta.env.DEV) {
    console.info(SPEECH_LOG_PREFIX, event, extra);
  }
}
