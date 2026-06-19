/**
 * Post-processing for Bengali / Banglish speech transcripts (Web Speech API).
 * Handles script cleanup, stutter dedupe, and common HR-domain mis-hearings.
 */

import {
  SPEECH_LANG_MODES,
  isBengaliPhoneticEnglish,
  repairBengaliPhoneticEnglish,
  resolveTranscriptProcessingMode,
} from "./speechLanguageDetect.js";

/** @type {[RegExp, string][]} */
const ROMAN_BANGLISH_FIXES = [
  [/\bcorti\b/gi, "chuti"],
  [/\bcuti\b/gi, "chuti"],
  [/\bchhuti\b/gi, "chuti"],
  [/\bchuti\b/gi, "chuti"],
  [/\bkharcha\b/gi, "khoroch"],
  [/\bkhoroch\b/gi, "khoroch"],
  [/\bkorci\b/gi, "korci"],
  [/\bkorchi\b/gi, "korchi"],
  [/\bkorcho\b/gi, "korcho"],
  [/\bkorbo\b/gi, "korbo"],
  [/\bkorte\b/gi, "korte"],
  [/\blagbe\b/gi, "lagbe"],
  [/\blage\b/gi, "lage"],
  [/\bajke\b/gi, "ajke"],
  [/\bajker\b/gi, "ajker"],
  [/\bkalke\b/gi, "kalke"],
  [/\bmetrorail\b/gi, "metro rail"],
  [/\bmetroral\b/gi, "metro rail"],
  [/\bleave balance\b/gi, "leave balance"],
  [/\bexpense claim\b/gi, "expense claim"],
];

/** Hindi bleed-through when recognition language drifts. */
/** @type {[RegExp, string][]} */
const DRIFT_FIXES = [
  [/\bkya\b/gi, "ki"],
  [/\bnahi\b/gi, "na"],
  [/\bhai\b/gi, "ache"],
  [/\bkaise\b/gi, "kivabe"],
];

const BENGALI_RANGE = /[\u0980-\u09FF]/;

/**
 * @param {string} langTag
 */
export function isBanglaSpeechLanguage(langTag) {
  return /^bn(?:-|$)/i.test(String(langTag || ""));
}

/**
 * Collapse immediate repeated words (common with continuous STT restarts).
 * @param {string} text
 */
export function collapseStutterRepeats(text) {
  if (!text) return "";
  const tokens = text.split(/\s+/);
  if (tokens.length < 2) return text;
  const out = [tokens[0]];
  for (let i = 1; i < tokens.length; i += 1) {
    const prev = out[out.length - 1];
    const cur = tokens[i];
    if (prev.toLowerCase() === cur.toLowerCase()) continue;
    out.push(cur);
  }
  return out.join(" ");
}

/**
 * @param {string} text
 */
function applyReplacementRules(text, rules) {
  let t = text;
  for (const [pattern, replacement] of rules) {
    t = t.replace(pattern, replacement);
  }
  return t;
}

/**
 * Clean Bengali script output from Web Speech (bn-BD / bn-IN).
 * @param {string} text
 */
function cleanBengaliScript(text) {
  let t = text;
  // Normalize danda / devanagari danda to sentence spacing
  t = t.replace(/\u0964/g, "।");
  t = t.replace(/\s*([।,.!?])\s*/g, "$1 ");
  // Drop stray single Latin letters sandwiched between Bengali tokens
  t = t.replace(/([\u0980-\u09FF])\s+[a-zA-Z]\s+(?=[\u0980-\u09FF])/g, "$1 ");
  // Collapse duplicated Bengali words
  t = collapseStutterRepeats(t);
  return t.replace(/\s+/g, " ").trim();
}

/**
 * Improve Roman Banglish when English STT was used by mistake.
 * @param {string} text
 */
function cleanRomanBanglish(text) {
  let t = text;
  t = applyReplacementRules(t, DRIFT_FIXES);
  t = applyReplacementRules(t, ROMAN_BANGLISH_FIXES);
  t = collapseStutterRepeats(t);
  return t.replace(/\s+/g, " ").trim();
}

/**
 * @param {string} text
 * @param {string | { lang?: string, mode?: string }} [langOrOptions]
 */
export function postProcessBanglaTranscript(text, langOrOptions = "bn-BD") {
  if (!text || typeof text !== "string") return "";
  let t = text.normalize("NFC").trim();
  if (!t) return "";

  const opts =
    typeof langOrOptions === "string" ? { lang: langOrOptions } : langOrOptions || {};
  const langTag = opts.lang || "bn-BD";
  const mode = opts.mode || "";

  const hasBengali = BENGALI_RANGE.test(t);
  const banglaLang = isBanglaSpeechLanguage(langTag);

  if (isBengaliPhoneticEnglish(t)) {
    t = repairBengaliPhoneticEnglish(t);
    return collapseStutterRepeats(t).replace(/\s+/g, " ").trim();
  }

  const effective = resolveTranscriptProcessingMode(t, { mode, lang: langTag });
  const processMode = effective.mode;

  if (
    processMode === SPEECH_LANG_MODES.EN ||
    effective.source === "phonetic_en_script" ||
    (processMode === SPEECH_LANG_MODES.UNKNOWN && mode === "en")
  ) {
    return collapseStutterRepeats(t).replace(/\s+/g, " ").trim();
  }

  if (
    hasBengali ||
    banglaLang ||
    processMode === SPEECH_LANG_MODES.BN ||
    mode === "bn"
  ) {
    t = cleanBengaliScript(t);
  }

  const applyBanglishFixes =
    processMode === SPEECH_LANG_MODES.BANGLISH ||
    mode === "banglish" ||
    (processMode === SPEECH_LANG_MODES.BN && !hasBengali && /[a-zA-Z]{2,}/.test(t));

  if (applyBanglishFixes && processMode !== SPEECH_LANG_MODES.EN) {
    t = cleanRomanBanglish(t);
  }

  return t.trim();
}

/**
 * Merge a new final chunk without duplicating overlap from auto-restart.
 * @param {string} previous
 * @param {string} next
 */
export function mergeFinalTranscriptChunks(previous, next) {
  const prev = (previous || "").trim();
  const cur = (next || "").trim();
  if (!prev) return cur;
  if (!cur) return prev;
  if (prev === cur) return prev;
  if (cur.startsWith(prev)) return cur;
  if (prev.startsWith(cur)) return prev;

  const prevLower = prev.toLowerCase();
  const curLower = cur.toLowerCase();
  if (prevLower.endsWith(curLower) || curLower.endsWith(prevLower)) {
    return prev.length >= cur.length ? prev : cur;
  }

  // Word-level overlap at boundary (restart echo)
  const prevWords = prev.split(/\s+/);
  const curWords = cur.split(/\s+/);
  const maxOverlap = Math.min(prevWords.length, curWords.length, 8);
  for (let size = maxOverlap; size >= 2; size -= 1) {
    const tail = prevWords.slice(-size).join(" ").toLowerCase();
    const head = curWords.slice(0, size).join(" ").toLowerCase();
    if (tail === head) {
      return [...prevWords, ...curWords.slice(size)].join(" ");
    }
  }

  return `${prev} ${cur}`.replace(/\s+/g, " ").trim();
}
