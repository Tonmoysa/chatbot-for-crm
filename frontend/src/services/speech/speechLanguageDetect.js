/**
 * Detect spoken language mode from transcript text (browser Web Speech STT).
 * Modes: bn (Bengali script), en (English), banglish (Romanized Bengali).
 */

import {
  isBengaliPhoneticEnglish,
  repairBengaliPhoneticEnglish,
} from "./bengaliPhoneticEnglish.js";

export { isBengaliPhoneticEnglish, repairBengaliPhoneticEnglish };

export const SPEECH_LANG_MODES = {
  BN: "bn",
  EN: "en",
  BANGLISH: "banglish",
  UNKNOWN: "unknown",
};

const BENGALI_RANGE = /[\u0980-\u09FF]/g;
const LATIN_WORD = /\b[a-zA-Z']+\b/g;

/** Romanized Bengali tokens — exclude HR English loanwords (leave, expense, check, etc.). */
const BANGLISH_MARKERS = new Set(
  [
    "ami", "amr", "amar", "apni", "tumi", "tomar", "ake", "ajke", "ajker", "kalke", "kal",
    "chuti", "chhuti", "chutir", "sorir", "shorir", "kharap", "lagbe", "lage", "lagto",
    "korbo", "korbe", "korte", "korchi", "korcho", "korci", "korben", "koro", "kora", "kore",
    "hobe", "hoy", "hoise", "hoyeche", "ache", "ase", "nai", "na", "ki", "koto", "keno",
    "kivabe", "kemon", "ta", "te", "er", "re", "theke", "por", "jabe", "jabo", "nite", "nibo",
    "diben", "bolun", "bolo", "bole", "bolchi", "sunun", "shunen", "kothay", "kobe",
    "khoroch", "kharcha", "kharch", "taka", "tk", "bdt", "mirpur", "motijheel", "motejheel",
    "baad", "daw", "dao", "thik", "vul", "bhul", "sothik", "protidin", "karon", "karone",
    "joma", "ghore", "ghor", "mony", "mot", "moot", "hisab", "puro", "hoye", "hoyeche",
  ].map((w) => w.toLowerCase())
);

/** High-signal English function words (rare in Banglish HR chat). */
const ENGLISH_MARKERS = new Set(
  [
    "the", "this", "that", "these", "those", "what", "when", "where", "which", "who", "whom",
    "how", "why", "please", "could", "would", "should", "have", "has", "had", "been", "being",
    "your", "mine", "ours", "their", "about", "because", "through", "during", "before", "after",
    "check", "balance", "request", "policy", "benefits", "payslip", "contact", "explain", "access",
    "company", "employee", "manager", "approval", "submit", "application",
  ].map((w) => w.toLowerCase())
);

const STORAGE_KEY = "hr_chat_speech_mode";

/**
 * @param {string} text
 */
function bengaliCharRatio(text) {
  const chars = [...text.replace(/\s/g, "")];
  if (!chars.length) return 0;
  const bn = (text.match(BENGALI_RANGE) || []).length;
  return bn / chars.length;
}

/**
 * @param {string} text
 */
function latinWords(text) {
  return (text.match(LATIN_WORD) || []).map((w) => w.toLowerCase());
}

/**
 * @param {string} text
 */
function banglishMarkerRatio(words) {
  if (!words.length) return 0;
  let hits = 0;
  for (const w of words) {
    if (BANGLISH_MARKERS.has(w)) hits += 1;
  }
  return hits / words.length;
}

/**
 * @param {string} text
 */
function englishMarkerRatio(words) {
  if (!words.length) return 0;
  let hits = 0;
  for (const w of words) {
    if (ENGLISH_MARKERS.has(w)) hits += 1;
  }
  return hits / words.length;
}

/**
 * @param {string} text
 * @param {{ minChars?: number }} [options]
 * @returns {{ mode: string, confidence: number, scores: Record<string, number> }}
 */
export function detectSpeechLanguageMode(text, options = {}) {
  const raw = (text || "").trim();
  const minChars = options.minChars ?? 6;
  const empty = {
    mode: SPEECH_LANG_MODES.UNKNOWN,
    confidence: 0,
    scores: { bn: 0, en: 0, banglish: 0 },
  };
  if (raw.length < minChars) return empty;

  const bnRatio = bengaliCharRatio(raw);
  const words = latinWords(raw);
  const banglishRatio = banglishMarkerRatio(words);
  const englishRatio = englishMarkerRatio(words);

  const scores = {
    bn: 0,
    en: 0,
    banglish: 0,
  };

  if (bnRatio >= 0.35) {
    scores.bn = 0.55 + Math.min(0.4, bnRatio);
  } else if (bnRatio >= 0.08) {
    scores.bn = 0.25 + bnRatio;
    scores.banglish += 0.15;
  }

  if (words.length) {
    scores.banglish += Math.min(0.55, banglishRatio * 2.2);
    scores.en += Math.min(0.55, englishRatio * 2.5);

    if (banglishRatio >= 0.12 && englishRatio < 0.08) {
      scores.banglish += 0.2;
    }
    if (englishRatio >= 0.12 && banglishRatio < 0.06) {
      scores.en += 0.25;
    }
    if (banglishRatio >= 0.08 && englishRatio >= 0.05) {
      scores.banglish += 0.1;
      scores.en += 0.05;
    }
  }

  if (bnRatio < 0.05 && words.length >= 3 && banglishRatio < 0.08 && englishRatio >= 0.1) {
    scores.en += 0.35;
    scores.banglish *= 0.6;
  }

  if (
    englishRatio >= 0.15 &&
    banglishRatio < 0.06 &&
    /\b(please|could|would|check|explain|access|contact)\b/i.test(raw)
  ) {
    scores.en += 0.25;
  }

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [topMode, topScore] = ranked[0];
  const secondScore = ranked[1]?.[1] ?? 0;
  const margin = topScore - secondScore;

  if (topScore < 0.28) {
    return { ...empty, scores };
  }

  const confidence = Math.min(0.98, topScore + margin * 0.35);
  return {
    mode: topMode,
    confidence,
    scores,
  };
}

/**
 * BCP-47 tag for Web Speech `lang` per mode.
 * @param {string} mode
 * @param {string[]} [preference]
 */
export function speechLangTagForMode(mode, preference = []) {
  const pref = preference.length ? preference : ["bn-BD", "bn-IN", "en-IN", "en-US"];
  const find = (prefix) =>
    pref.find((t) => t.toLowerCase() === prefix || t.toLowerCase().startsWith(`${prefix}-`));

  if (mode === SPEECH_LANG_MODES.EN) {
    return find("en-us") || find("en-in") || find("en") || "en-US";
  }
  if (mode === SPEECH_LANG_MODES.BANGLISH) {
    return find("en-in") || find("bn-bd") || find("en-us") || "en-IN";
  }
  if (mode === SPEECH_LANG_MODES.BN) {
    return find("bn-bd") || find("bn-in") || find("bn") || "bn-BD";
  }
  return pref[0] || "bn-BD";
}

/**
 * Ordered BCP-47 chain when browser rejects primary tag.
 * @param {string} mode
 * @param {string[]} [preference]
 */
export function speechLangChainForMode(mode, preference = []) {
  const pref = preference.length ? preference : ["bn-BD", "bn-IN", "en-IN", "en-US"];
  const primary = speechLangTagForMode(mode, pref);

  /** @type {string[]} */
  let extras = [];
  if (mode === SPEECH_LANG_MODES.EN) {
    extras = ["en-US", "en-IN", "en-GB"];
  } else if (mode === SPEECH_LANG_MODES.BANGLISH) {
    extras = ["en-IN", "bn-BD", "bn-IN", "en-US"];
  } else if (mode === SPEECH_LANG_MODES.BN) {
    extras = ["bn-BD", "bn-IN", "bn"];
  } else {
    extras = [...pref];
  }

  return [...new Set([primary, ...extras, ...pref])].filter(Boolean);
}

/**
 * Score how well a transcript alternative fits the active mode (0–1).
 * @param {string} transcript
 * @param {string} mode
 */
export function scoreTranscriptForMode(transcript, mode) {
  if (!transcript || !mode || mode === SPEECH_LANG_MODES.UNKNOWN) return 0.5;
  const det = detectSpeechLanguageMode(transcript, { minChars: 3 });
  if (det.mode === mode) return Math.max(0.55, det.confidence);
  if (det.mode === SPEECH_LANG_MODES.UNKNOWN) return 0.45;
  return Math.max(0.1, 0.45 - (det.confidence - 0.3));
}

export function readStoredSpeechMode() {
  if (typeof localStorage === "undefined") return null;
  const v = localStorage.getItem(STORAGE_KEY);
  if (v === SPEECH_LANG_MODES.BN || v === SPEECH_LANG_MODES.EN || v === SPEECH_LANG_MODES.BANGLISH) {
    return v;
  }
  return null;
}

/**
 * @param {string} mode
 */
export function storeSpeechMode(mode) {
  if (typeof localStorage === "undefined") return;
  if (mode === SPEECH_LANG_MODES.BN || mode === SPEECH_LANG_MODES.EN || mode === SPEECH_LANG_MODES.BANGLISH) {
    localStorage.setItem(STORAGE_KEY, mode);
  }
}

/**
 * Pick post-processing + STT mode from explicit session mode or transcript content.
 * Prevents English speech from being romanized when the mic started on bn-BD.
 * @param {string} text
 * @param {{ mode?: string, lang?: string, minChars?: number }} [options]
 */
export function resolveTranscriptProcessingMode(text, options = {}) {
  const explicit = options.mode;
  const raw = (text || "").trim();
  if (!raw) {
    return { mode: SPEECH_LANG_MODES.UNKNOWN, confidence: 0, source: "empty" };
  }

  if (BENGALI_RANGE.test(raw)) {
    if (isBengaliPhoneticEnglish(raw)) {
      return {
        mode: SPEECH_LANG_MODES.EN,
        confidence: 0.92,
        source: "phonetic_en_script",
      };
    }
    return { mode: SPEECH_LANG_MODES.BN, confidence: 0.99, source: "script" };
  }

  if (explicit === SPEECH_LANG_MODES.EN) {
    return { mode: SPEECH_LANG_MODES.EN, confidence: 1, source: "explicit" };
  }

  const minChars = options.minChars ?? 6;
  const det = detectSpeechLanguageMode(raw, { minChars });
  if (det.mode !== SPEECH_LANG_MODES.UNKNOWN && det.confidence >= 0.4) {
    return { mode: det.mode, confidence: det.confidence, source: "detected" };
  }

  if (
    explicit === SPEECH_LANG_MODES.EN ||
    explicit === SPEECH_LANG_MODES.BN ||
    explicit === SPEECH_LANG_MODES.BANGLISH
  ) {
    return { mode: explicit, confidence: 0.5, source: "explicit" };
  }

  return { mode: SPEECH_LANG_MODES.UNKNOWN, confidence: 0, source: "fallback" };
}
