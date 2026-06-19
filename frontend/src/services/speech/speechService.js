import { ACTIVE_SPEECH_PROVIDER, SPEECH_PROVIDERS } from "./speechConfig.js";
import { canUseSpeechProvider } from "./speechUtils.js";
import { logSpeechEvent, generateTraceId } from "../../utils/trace.js";
import { WebSpeechProvider } from "./providers/webSpeechProvider.js";
import { WhisperProvider } from "./providers/whisperProvider.js";

/** @type {import('./providers/baseProvider.js').BaseSpeechProvider | null} */
let activeInstance = null;

/** @type {import('./providers/baseProvider.js').SpeechRecognitionCallbacks} */
let registeredCallbacks = {};

/**
 * @param {string} [providerKey]
 */
export function createSpeechProvider(providerKey = ACTIVE_SPEECH_PROVIDER) {
  if (providerKey === SPEECH_PROVIDERS.WHISPER) {
    return new WhisperProvider();
  }
  return new WebSpeechProvider();
}

/**
 * @returns {string}
 */
export function getActiveProviderName() {
  return ACTIVE_SPEECH_PROVIDER;
}

/**
 * @returns {{ supported: boolean, provider: string, webSpeech: boolean, whisper: boolean }}
 */
export function getSpeechCapabilities() {
  return {
    provider: ACTIVE_SPEECH_PROVIDER,
    supported: canUseSpeechProvider(ACTIVE_SPEECH_PROVIDER),
    webSpeech: canUseSpeechProvider(SPEECH_PROVIDERS.WEB_SPEECH),
    whisper: canUseSpeechProvider(SPEECH_PROVIDERS.WHISPER),
  };
}

/**
 * @param {import('./providers/baseProvider.js').SpeechRecognitionCallbacks} callbacks
 */
export function setSpeechCallbacks(callbacks) {
  registeredCallbacks = { ...registeredCallbacks, ...callbacks };
  activeInstance?.setCallbacks(registeredCallbacks);
}

/**
 * @param {{ language?: string, traceId?: string }} [options]
 */
export async function startRecognition(options = {}) {
  const traceId = options.traceId || generateTraceId();
  const providerKey = ACTIVE_SPEECH_PROVIDER;

  if (!canUseSpeechProvider(providerKey)) {
    const fallback =
      providerKey === SPEECH_PROVIDERS.WHISPER
        ? "Whisper transcription requires audio recording and the voice API."
        : "Speech recognition is not supported in this browser. Use Chrome or Edge.";
    throw new Error(fallback);
  }

  if (activeInstance) {
    await activeInstance.abort().catch(() => {});
  }

  activeInstance = createSpeechProvider(providerKey);
  activeInstance.setCallbacks(registeredCallbacks);
  logSpeechEvent("speech_service_start", { provider: providerKey, traceId });

  await activeInstance.start({ ...options, traceId });
  return traceId;
}

export async function stopRecognition() {
  if (!activeInstance) return;
  logSpeechEvent("speech_service_stop", { provider: activeInstance.name });
  await activeInstance.stop();
}

export async function abortRecognition() {
  if (!activeInstance) return;
  await activeInstance.abort();
  activeInstance = null;
}

/**
 * Switch Web Speech language/mode while dictating (no-op if provider lacks support).
 * @param {{ language: string, mode?: string, languageChain?: string[] }} options
 */
export async function updateRecognitionLanguage(options) {
  if (!activeInstance || typeof activeInstance.updateLanguage !== "function") return;
  await activeInstance.updateLanguage(options);
}

export function isListening() {
  return Boolean(
    activeInstance &&
      activeInstance.name === "webSpeech" &&
      "listening" in activeInstance &&
      activeInstance.listening
  );
}
