/**
 * bn-BD Web Speech often writes spoken English as Bengali-script phonetics
 * (e.g. "please tell me" → "প্লিজ টেল মি"). Detect and repair to Latin English.
 */

const BENGALI_RANGE = /[\u0980-\u09FF]/;

/** Genuine Bengali words — not English spoken through bn-BD. */
const NATIVE_BENGALI_RE =
  /(?:আমার|আমি|আপনি|তুমি|ছুটি|চাই|লাগবে|খরচ|কাল|আজ|কত|কোন|কি|কেন|কিভাবে|দিছিলাম|দিয়েছি|জমা|নিয়ে|কারণ|শরীর|গত|লাস্ট|দিন|এড|করছি|বলুন|বলো|হবে|হয়|না|হ্যাঁ)/;

/** English HR terms mis-heard as Bengali script (voice STT on bn-BD). */
const PHONETIC_ENGLISH_TOKEN_RE =
  /(?:প্লিজ|প্লিজ|টেল|টোল|মি|দা|থা|এক্সপেন্স|এক্সপেনস|এক্সপেন্সে|সামারি|সামারী|লিভ|ব্যালেন্স|ব্যালান্স|পলিসি|চেক|সাবমিট|ইয়েস|স্টেটাস|রিকোয়েস্ট|হিস্টোরি|টোটাল|হিস্ট্রি|রিকোয়েস্ট|ড্রাফট|সাবমিশন|ইক্সপেন্স|এক্সপেন্সি)/gi;

/** @type {Record<string, string>} */
const TOKEN_TO_ENGLISH = {
  প্লিজ: "please",
  টেল: "tell",
  টোল: "tell",
  মি: "me",
  দা: "the",
  থা: "the",
  এক্সপেন্স: "expense",
  এক্সপেনস: "expense",
  এক্সপেন্সে: "expense",
  ইক্সপেন্স: "expense",
  এক্সপেন্সি: "expense",
  সামারি: "summary",
  সামারী: "summary",
  লিভ: "leave",
  ব্যালেন্স: "balance",
  ব্যালান্স: "balance",
  পলিসি: "policy",
  চেক: "check",
  সাবমিট: "submit",
  ইয়েস: "yes",
  স্টেটাস: "status",
  রিকোয়েস্ট: "request",
  হিস্টোরি: "history",
  হিস্ট্রি: "history",
  টোটাল: "total",
  ড্রাফট: "draft",
  সাবমিশন: "submission",
};

/**
 * True when Bengali script is almost certainly English words via bn-BD mis-transcription.
 * @param {string} text
 */
export function isBengaliPhoneticEnglish(text) {
  const raw = (text || "").trim();
  if (!raw || !BENGALI_RANGE.test(raw)) return false;
  if (NATIVE_BENGALI_RE.test(raw)) return false;

  const tokens = raw.split(/\s+/).filter(Boolean);
  if (!tokens.length) return false;

  let phoneticHits = 0;
  for (const token of tokens) {
    const core = token.replace(/[।,.!?;:'"]+$/g, "").replace(/^[।,.!?;:'"]+/g, "");
    if (TOKEN_TO_ENGLISH[core]) {
      phoneticHits += 1;
      continue;
    }
    if (PHONETIC_ENGLISH_TOKEN_RE.test(core)) {
      phoneticHits += 1;
    }
  }

  if (phoneticHits >= 2) return true;
  if (phoneticHits >= 1 && tokens.length <= 4) return true;
  return phoneticHits / tokens.length >= 0.5;
}

/**
 * Convert Bengali-script phonetic English tokens to Latin English.
 * @param {string} text
 */
export function repairBengaliPhoneticEnglish(text) {
  const raw = (text || "").trim();
  if (!raw || !isBengaliPhoneticEnglish(raw)) return raw;

  const out = raw.split(/\s+/).map((token) => {
    const trailing = (token.match(/[।,.!?;:]+$/g) || [""])[0];
    const leading = (token.match(/^[।,.!?;:]+/g) || [""])[0];
    const core = token.slice(leading.length, token.length - trailing.length || undefined);
    const english = TOKEN_TO_ENGLISH[core];
    if (english) return `${leading}${english}${trailing}`;
    return token;
  });

  return out.join(" ").replace(/\s+/g, " ").trim();
}
