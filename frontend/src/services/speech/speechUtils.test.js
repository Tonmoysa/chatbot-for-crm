import { describe, expect, it, vi } from "vitest";
import {
  collapseStutterRepeats,
  mergeFinalTranscriptChunks,
  postProcessBanglaTranscript,
} from "./banglaTranscript.js";
import {
  normalizeTranscript,
  pickBestTranscript,
  refineSpeechSessionFromTranscript,
  resolveSpeechLanguage,
  resolveSpeechSession,
  speechLanguageFallbackChain,
} from "./speechUtils.js";
import { SPEECH_LANG_MODES } from "./speechLanguageDetect.js";

describe("normalizeTranscript", () => {
  it("trims and collapses whitespace", () => {
    expect(normalizeTranscript("  hello   world  ")).toBe("hello world");
  });

  it("preserves Bengali Unicode", () => {
    expect(normalizeTranscript("আমার   ছুটি", { lang: "bn-BD" })).toBe("আমার ছুটি");
  });

  it("handles Banglish mixed text", () => {
    expect(normalizeTranscript("amar  leave balance koto", { lang: "bn-BD" })).toBe(
      "amar leave balance koto"
    );
  });

  it("fixes common roman mis-hearings for banglish mode", () => {
    expect(
      normalizeTranscript("ami corti nite chai", { lang: "en-IN", mode: "banglish" })
    ).toBe("ami chuti nite chai");
  });

  it("skips banglish fixes for english mode", () => {
    expect(
      normalizeTranscript("ami corti nite chai", { lang: "en-US", mode: "en" })
    ).toBe("ami corti nite chai");
  });

  it("keeps English phrases intact when session started on bn-BD", () => {
    expect(
      normalizeTranscript("please tell me the expense summary", {
        lang: "bn-BD",
        mode: "bn",
      })
    ).toBe("please tell me the expense summary");
  });

  it("does not map English words through Hindi drift fixes", () => {
    expect(
      normalizeTranscript("what is the policy on this", {
        lang: "bn-BD",
        mode: "bn",
      })
    ).toBe("what is the policy on this");
  });

  it("repairs bn-BD phonetic English script to Latin English", () => {
    expect(
      normalizeTranscript("প্লিজ টেল মি দা এক্সপেন্স সামারি", {
        lang: "bn-BD",
        mode: "bn",
      })
    ).toBe("please tell me the expense summary");
  });
});

describe("resolveSpeechLanguage", () => {
  it("forces primary bn-BD when auto-detect is off", () => {
    expect(
      resolveSpeechLanguage(["bn-BD", "en-US"], {
        forcePrimary: true,
        autoDetect: false,
      })
    ).toBe("bn-BD");
  });

  it("can match browser languages when forcePrimary is false", () => {
    vi.stubGlobal("navigator", { languages: ["en-US"], language: "en-US" });
    expect(
      resolveSpeechLanguage(["bn-BD", "en-US"], {
        forcePrimary: false,
        autoDetect: false,
      })
    ).toBe("en-US");
    vi.unstubAllGlobals();
  });
});

describe("resolveSpeechSession", () => {
  it("detects English from draft text when auto-detect is on", () => {
    const s = resolveSpeechSession(["bn-BD", "en-US"], {
      autoDetect: true,
      draftText: "Please check my leave balance for this month",
    });
    expect(s.mode).toBe(SPEECH_LANG_MODES.EN);
    expect(s.lang).toBe("en-US");
  });

  it("detects Banglish from draft text", () => {
    const s = resolveSpeechSession(["bn-BD", "en-IN", "en-US"], {
      autoDetect: true,
      draftText: "amar kalke chuti lagbe sick leave",
    });
    expect(s.mode).toBe(SPEECH_LANG_MODES.BANGLISH);
    expect(s.lang).toBe("en-IN");
  });

  it("defaults to en-IN banglish when auto-detect has no hints", () => {
    vi.stubGlobal("navigator", { languages: ["en-US"], language: "en-US" });
    const s = resolveSpeechSession(["bn-BD", "en-IN", "en-US"], {
      autoDetect: true,
      storedMode: null,
      draftText: "",
    });
    expect(s.mode).toBe(SPEECH_LANG_MODES.BANGLISH);
    expect(s.lang).toBe("en-IN");
    vi.unstubAllGlobals();
  });

  it("restores stored speech mode when auto-detect is on", () => {
    const s = resolveSpeechSession(["bn-BD", "en-US"], {
      autoDetect: true,
      storedMode: SPEECH_LANG_MODES.EN,
      draftText: "",
    });
    expect(s.mode).toBe(SPEECH_LANG_MODES.EN);
    expect(s.lang).toBe("en-US");
  });
});

describe("refineSpeechSessionFromTranscript", () => {
  it("locks Bengali script immediately", () => {
    const r = refineSpeechSessionFromTranscript("আমার আজকে ছুটি লাগবে কারণ শরীর খারাপ", {
      mode: SPEECH_LANG_MODES.EN,
      lang: "en-US",
    });
    expect(r.mode).toBe(SPEECH_LANG_MODES.BN);
    expect(r.lang).toBe("bn-BD");
    expect(r.modeLocked).toBe(true);
  });

  it("does not switch on short interim text", () => {
    const current = { mode: SPEECH_LANG_MODES.BN, lang: "bn-BD" };
    expect(refineSpeechSessionFromTranscript("ok", current)).toEqual(current);
  });

  it("switches to English after enough confident English text", () => {
    const r = refineSpeechSessionFromTranscript(
      "Please check my leave balance for this month and explain the policy",
      { mode: SPEECH_LANG_MODES.BN, lang: "bn-BD" }
    );
    expect(r.mode).toBe(SPEECH_LANG_MODES.EN);
    expect(r.lang).toBe("en-US");
  });

  it("switches to English on shorter clear English phrase from bn start", () => {
    const r = refineSpeechSessionFromTranscript(
      "please tell me the expense summary",
      { mode: SPEECH_LANG_MODES.BN, lang: "bn-BD", modeLocked: false }
    );
    expect(r.mode).toBe(SPEECH_LANG_MODES.EN);
    expect(r.lang).toBe("en-US");
  });
});

describe("pickBestTranscript", () => {
  it("chooses highest confidence alternative", () => {
    const result = [
      { transcript: "wrong", confidence: 0.2 },
      { transcript: "আমার ছুটি", confidence: 0.9 },
    ];
    expect(pickBestTranscript(result)).toBe("আমার ছুটি");
  });
});

describe("mergeFinalTranscriptChunks", () => {
  it("dedupes identical chunks from restart", () => {
    expect(mergeFinalTranscriptChunks("আমার ছুটি", "আমার ছুটি")).toBe("আমার ছুটি");
  });

  it("merges overlapping tail/head words", () => {
    expect(
      mergeFinalTranscriptChunks("ami ajke sick leave", "sick leave nite chai")
    ).toBe("ami ajke sick leave nite chai");
  });
});

describe("postProcessBanglaTranscript", () => {
  it("collapses stutter repeats", () => {
    expect(collapseStutterRepeats("ami ami ajke ajke chuti")).toBe("ami ajke chuti");
  });

  it("builds language fallback chain without duplicates", () => {
    expect(speechLanguageFallbackChain("bn-BD", ["bn-BD", "bn-IN", "en-US"])).toEqual([
      "bn-BD",
      "bn-IN",
      "en-US",
    ]);
  });
});

describe("speechErrorMessage", () => {
  it("maps not-allowed to user-friendly text", async () => {
    const { speechErrorMessage } = await import("./speechUtils.js");
    expect(speechErrorMessage({ error: "not-allowed" })).toMatch(/denied/i);
  });
});
