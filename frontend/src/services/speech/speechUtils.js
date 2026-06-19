import {
  SPEECH_AUTO_DETECT_LANGUAGE,
  SPEECH_DEFAULT_START_MODE,
  SPEECH_EN_SWITCH_MIN_CHARS,
  SPEECH_FORCE_PRIMARY_LANGUAGE,
  SPEECH_LANG_SWITCH_MIN_CHARS,
  SPEECH_LANGUAGE_PREFERENCE,
} from "./speechConfig.js";
import { postProcessBanglaTranscript } from "./banglaTranscript.js";
import {
  SPEECH_LANG_MODES,
  detectSpeechLanguageMode,
  isBengaliPhoneticEnglish,
  readStoredSpeechMode,
  repairBengaliPhoneticEnglish,
  resolveTranscriptProcessingMode,
  scoreTranscriptForMode,
  speechLangChainForMode,
  speechLangTagForMode,
} from "./speechLanguageDetect.js";

/**
 * @returns {typeof SpeechRecognition | null}
 */
export function getSpeechRecognitionConstructor() {
  if (typeof window === "undefined") return null;
  const w = window;
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

export function isWebSpeechSupported() {
  return Boolean(getSpeechRecognitionConstructor());
}

export function isMediaRecorderSupported() {
  return typeof window !== "undefined" && typeof MediaRecorder !== "undefined";
}

/**
 * Pick the first BCP-47 tag for Web Speech `lang`.
 * For Bangla demo: force bn-BD even when the browser UI is English.
 * @param {string[]} [preference]
 * @param {{ forcePrimary?: boolean }} [options]
 */
export function resolveSpeechLanguage(
  preference = SPEECH_LANGUAGE_PREFERENCE,
  options = {}
) {
  const resolved = resolveSpeechSession(preference, options);
  return resolved.lang;
}

/**
 * Resolve STT language + mode for a dictation session.
 * @param {string[]} [preference]
 * @param {{ forcePrimary?: boolean, autoDetect?: boolean, draftText?: string, storedMode?: string | null, mode?: string }} [options]
 * @returns {{ lang: string, mode: string, languageChain: string[] }}
 */
export function resolveSpeechSession(
  preference = SPEECH_LANGUAGE_PREFERENCE,
  options = {}
) {
  const autoDetect = options.autoDetect ?? SPEECH_AUTO_DETECT_LANGUAGE;
  const forcePrimary = options.forcePrimary ?? SPEECH_FORCE_PRIMARY_LANGUAGE;

  if (autoDetect) {
    const draft = (options.draftText || "").trim();
    if (draft) {
      const det = detectSpeechLanguageMode(draft);
      if (det.confidence >= 0.45) {
        const mode = det.mode;
        const lang = speechLangTagForMode(mode, preference);
        return {
          lang,
          mode,
          languageChain: speechLangChainForMode(mode, preference),
          modeLocked: true,
        };
      }
    }

    const stored = options.storedMode ?? readStoredSpeechMode();
    if (stored) {
      return {
        lang: speechLangTagForMode(stored, preference),
        mode: stored,
        languageChain: speechLangChainForMode(stored, preference),
        modeLocked: false,
      };
    }

    const startMode =
      SPEECH_DEFAULT_START_MODE === SPEECH_LANG_MODES.EN
        ? SPEECH_LANG_MODES.EN
        : SPEECH_DEFAULT_START_MODE === SPEECH_LANG_MODES.BANGLISH
          ? SPEECH_LANG_MODES.BANGLISH
          : SPEECH_LANG_MODES.BN;
    return {
      lang: speechLangTagForMode(startMode, preference),
      mode: startMode,
      languageChain: speechLangChainForMode(startMode, preference),
      modeLocked: false,
    };
  }

  if (forcePrimary && preference[0]) {
    const lang = preference[0];
    return {
      lang,
      mode: /^bn/i.test(lang) ? SPEECH_LANG_MODES.BN : SPEECH_LANG_MODES.EN,
      languageChain: speechLanguageFallbackChain(lang, preference),
    };
  }

  if (typeof navigator === "undefined") {
    const lang = preference[0] || "bn-BD";
    return {
      lang,
      mode: SPEECH_LANG_MODES.BN,
      languageChain: speechLanguageFallbackChain(lang, preference),
    };
  }
  const available = new Set(
    (navigator.languages || [navigator.language || "en-US"]).map((l) =>
      l.toLowerCase()
    )
  );
  for (const tag of preference) {
    const lower = tag.toLowerCase();
    if (available.has(lower)) {
      return {
        lang: tag,
        mode: lower.startsWith("bn") ? SPEECH_LANG_MODES.BN : SPEECH_LANG_MODES.EN,
        languageChain: speechLanguageFallbackChain(tag, preference),
      };
    }
    const base = lower.split("-")[0];
    if ([...available].some((a) => a === base || a.startsWith(`${base}-`))) {
      return {
        lang: tag,
        mode: base === "bn" ? SPEECH_LANG_MODES.BN : SPEECH_LANG_MODES.EN,
        languageChain: speechLanguageFallbackChain(tag, preference),
      };
    }
  }
  const lang = preference[0] || "bn-BD";
  return {
    lang,
    mode: SPEECH_LANG_MODES.BN,
    languageChain: speechLanguageFallbackChain(lang, preference),
  };
}

const BENGALI_SCRIPT_RE = /[\u0980-\u09FF]/;

/**
 * Re-detect mode from accumulated transcript; switch STT language when confident.
 * @param {string} text
 * @param {{ mode: string, lang: string, modeLocked?: boolean }} current
 */
export function refineSpeechSessionFromTranscript(text, current) {
  const raw = (text || "").trim();
  if (!raw) {
    return current;
  }

  if (BENGALI_SCRIPT_RE.test(raw)) {
    if (isBengaliPhoneticEnglish(raw)) {
      const lang = speechLangTagForMode(SPEECH_LANG_MODES.EN, SPEECH_LANGUAGE_PREFERENCE);
      return {
        mode: SPEECH_LANG_MODES.EN,
        lang,
        languageChain: speechLangChainForMode(SPEECH_LANG_MODES.EN, SPEECH_LANGUAGE_PREFERENCE),
        confidence: 0.92,
        modeLocked: true,
      };
    }
    if (current.mode === SPEECH_LANG_MODES.BN) return current;
    const lang = speechLangTagForMode(SPEECH_LANG_MODES.BN, SPEECH_LANGUAGE_PREFERENCE);
    return {
      mode: SPEECH_LANG_MODES.BN,
      lang,
      languageChain: speechLangChainForMode(SPEECH_LANG_MODES.BN, SPEECH_LANGUAGE_PREFERENCE),
      confidence: 0.99,
      modeLocked: true,
    };
  }

  const minCharsForSwitch =
    current.mode === SPEECH_LANG_MODES.BN && !current.modeLocked
      ? SPEECH_EN_SWITCH_MIN_CHARS
      : SPEECH_LANG_SWITCH_MIN_CHARS;
  if (raw.length < minCharsForSwitch) {
    return current;
  }

  const det = detectSpeechLanguageMode(raw, {
    minChars: Math.min(8, minCharsForSwitch),
  });
  if (det.mode === SPEECH_LANG_MODES.UNKNOWN) {
    return current;
  }
  if (det.mode === current.mode) {
    return det.confidence >= 0.72
      ? { ...current, modeLocked: true, confidence: det.confidence }
      : current;
  }

  const leaveBnThreshold =
    current.mode === SPEECH_LANG_MODES.BN && !current.modeLocked ? 0.52 : 0.82;
  const requiredConfidence =
    current.mode === SPEECH_LANG_MODES.BN && det.mode === SPEECH_LANG_MODES.EN
      ? leaveBnThreshold
      : current.mode === SPEECH_LANG_MODES.BN
        ? 0.74
        : 0.68;

  if (det.confidence < requiredConfidence) {
    return current;
  }

  const lang = speechLangTagForMode(det.mode, SPEECH_LANGUAGE_PREFERENCE);
  if (lang === current.lang) {
    return { ...current, mode: det.mode, modeLocked: true, confidence: det.confidence };
  }

  return {
    mode: det.mode,
    lang,
    languageChain: speechLangChainForMode(det.mode, SPEECH_LANGUAGE_PREFERENCE),
    confidence: det.confidence,
    modeLocked: true,
  };
}

/**
 * Fallback language chain when the browser rejects the primary tag.
 * @param {string} primary
 * @param {string[]} [preference]
 */
export function speechLanguageFallbackChain(
  primary,
  preference = SPEECH_LANGUAGE_PREFERENCE
) {
  const chain = [primary, ...preference].filter(Boolean);
  return [...new Set(chain)];
}

/**
 * Pick best alternative — confidence + language-mode fit when auto-detect is on.
 * @param {SpeechRecognitionResult} result
 * @param {{ mode?: string }} [options]
 */
export function pickBestTranscript(result, options = {}) {
  if (!result || result.length === 0) return "";
  const mode = options.mode;
  const modeLocked = Boolean(options.modeLocked);

  let best = result[0];
  let bestScore = -1;
  for (let i = 0; i < result.length; i += 1) {
    const alt = result[i];
    const conf = typeof alt.confidence === "number" ? alt.confidence : 0.5;
    let score = conf;
    if (modeLocked && mode && mode !== SPEECH_LANG_MODES.UNKNOWN) {
      const fit = scoreTranscriptForMode(alt.transcript || "", mode);
      score = conf * 0.82 + fit * 0.18;
    }
    if (score > bestScore) {
      bestScore = score;
      best = alt;
    }
  }
  return best?.transcript || "";
}

/**
 * Normalize transcript for chat input (Unicode-safe, Bangla/Banglish-friendly).
 * @param {string} text
 * @param {{ lang?: string }} [options]
 */
export function normalizeTranscript(text, options = {}) {
  if (!text || typeof text !== "string") return "";
  let t = text.normalize("NFC");
  t = t.replace(/\u00A0/g, " ");
  t = t.replace(/[\u200B-\u200D\uFEFF]/g, "");
  t = t.replace(/\s+/g, " ");
  t = t.replace(/\s+([,.!?;:])/g, "$1");
  t = t.replace(/([,.!?;:])([^\s])/g, "$1 $2");
  t = t.trim();
  if (!t) return "";

  let mode = options.mode;
  let lang = options.lang || resolveSpeechLanguage();
  if (SPEECH_AUTO_DETECT_LANGUAGE && mode !== SPEECH_LANG_MODES.EN) {
    const inferred = resolveTranscriptProcessingMode(t, { mode, lang });
    if (inferred.mode === SPEECH_LANG_MODES.EN && inferred.confidence >= 0.42) {
      mode = SPEECH_LANG_MODES.EN;
      lang = speechLangTagForMode(SPEECH_LANG_MODES.EN, SPEECH_LANGUAGE_PREFERENCE);
    } else if (
      inferred.mode === SPEECH_LANG_MODES.BN &&
      inferred.confidence >= 0.5
    ) {
      mode = SPEECH_LANG_MODES.BN;
      lang = speechLangTagForMode(SPEECH_LANG_MODES.BN, SPEECH_LANGUAGE_PREFERENCE);
    } else if (
      inferred.mode === SPEECH_LANG_MODES.BANGLISH &&
      inferred.confidence >= 0.45 &&
      mode !== SPEECH_LANG_MODES.EN
    ) {
      mode = SPEECH_LANG_MODES.BANGLISH;
      lang = speechLangTagForMode(SPEECH_LANG_MODES.BANGLISH, SPEECH_LANGUAGE_PREFERENCE);
    }
  }

  if (isBengaliPhoneticEnglish(t)) {
    t = repairBengaliPhoneticEnglish(t);
    mode = SPEECH_LANG_MODES.EN;
    lang = speechLangTagForMode(SPEECH_LANG_MODES.EN, SPEECH_LANGUAGE_PREFERENCE);
  }

  return postProcessBanglaTranscript(t, { lang, mode });
}

/**
 * @param {unknown} err
 */
export function speechErrorMessage(err) {
  if (!err) return "Speech recognition failed.";
  if (typeof err === "string") return err;
  const code = err.error || err.name || "";
  const messages = {
    "not-allowed": "Microphone access was denied. Allow the microphone in your browser settings and try again.",
    "service-not-allowed": "Speech recognition is blocked on this page. Use HTTPS or check browser permissions.",
    "no-speech": "No speech was detected. Try speaking again.",
    "audio-capture": "No microphone was found. Connect a microphone and try again.",
    "network": "Network error during speech recognition. Check your connection.",
    "aborted": "Speech recognition was stopped.",
    "language-not-supported": "This language is not supported for speech recognition in your browser.",
  };
  if (code && messages[code]) return messages[code];
  if (err.message && typeof err.message === "string") return err.message;
  return "Speech recognition failed. Please try again or type your message.";
}

export function canUseSpeechProvider(providerKey) {
  if (providerKey === "webSpeech") return isWebSpeechSupported();
  if (providerKey === "whisper") return isMediaRecorderSupported();
  return false;
}
