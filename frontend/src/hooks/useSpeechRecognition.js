import { useCallback, useEffect, useRef, useState } from "react";
import {
  abortRecognition,
  getSpeechCapabilities,
  setSpeechCallbacks,
  startRecognition,
  stopRecognition,
  updateRecognitionLanguage,
} from "../services/speech/speechService.js";
import { mergeFinalTranscriptChunks } from "../services/speech/banglaTranscript.js";
import {
  normalizeTranscript,
  refineSpeechSessionFromTranscript,
  resolveSpeechSession,
} from "../services/speech/speechUtils.js";
import {
  SPEECH_INTERIM_DEBOUNCE_MS,
  SPEECH_LANG_SWITCH_MAX,
} from "../services/speech/speechConfig.js";
import {
  isBengaliPhoneticEnglish,
  readStoredSpeechMode,
  storeSpeechMode,
} from "../services/speech/speechLanguageDetect.js";
import { generateTraceId, logSpeechEvent } from "../utils/trace.js";

/**
 * Dictate-mode speech hook: accumulates transcript until user confirms (✓).
 * @param {{ disabled?: boolean, draftText?: string }} [options]
 */
export function useSpeechRecognition({ disabled = false, draftText = "" } = {}) {
  const [isDictating, setIsDictating] = useState(false);
  const [sessionText, setSessionText] = useState("");
  const [interimText, setInterimText] = useState("");
  const [speechError, setSpeechError] = useState(null);
  const [speechMode, setSpeechMode] = useState("unknown");

  const traceIdRef = useRef(null);
  const finalPartsRef = useRef([]);
  const interimRef = useRef("");
  /** @type {ReturnType<typeof setTimeout> | null} */
  const interimTimerRef = useRef(null);
  const speechLangRef = useRef("bn-BD");
  const speechModeRef = useRef("bn");
  const modeLockedRef = useRef(false);
  const languageChainRef = useRef(["bn-BD"]);
  const languageSwitchCountRef = useRef(0);
  const draftTextRef = useRef(draftText);
  const capabilities = getSpeechCapabilities();

  useEffect(() => {
    draftTextRef.current = draftText;
  }, [draftText]);

  const rebuildSessionText = useCallback((interim = "") => {
    const finals = finalPartsRef.current.join(" ").trim();
    const combined = [finals, interim.trim()].filter(Boolean).join(" ");
    setSessionText(
      normalizeTranscript(combined, {
        lang: speechLangRef.current,
        mode: speechModeRef.current,
      })
    );
  }, []);

  const maybeAdaptSpeechLanguage = useCallback(async (accumulatedText) => {
    if (languageSwitchCountRef.current >= SPEECH_LANG_SWITCH_MAX) {
      return;
    }

    const refined = refineSpeechSessionFromTranscript(accumulatedText, {
      mode: speechModeRef.current,
      lang: speechLangRef.current,
      modeLocked: modeLockedRef.current,
    });

    if (refined.modeLocked) {
      modeLockedRef.current = true;
    }

    if (refined.lang === speechLangRef.current && refined.mode === speechModeRef.current) {
      return;
    }

    languageSwitchCountRef.current += 1;
    speechLangRef.current = refined.lang;
    speechModeRef.current = refined.mode;
    languageChainRef.current = refined.languageChain;
    setSpeechMode(refined.mode);

    logSpeechEvent("speech_mode_adapted", {
      traceId: traceIdRef.current,
      mode: refined.mode,
      language: refined.lang,
      confidence: refined.confidence,
      switchCount: languageSwitchCountRef.current,
    });

    await updateRecognitionLanguage({
      language: refined.lang,
      mode: refined.mode,
      languageChain: refined.languageChain,
      modeLocked: modeLockedRef.current,
    });
  }, []);

  const clearInterimTimer = useCallback(() => {
    if (interimTimerRef.current) {
      clearTimeout(interimTimerRef.current);
      interimTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    setSpeechCallbacks({
      onStart: () => {
        setSpeechError(null);
        setInterimText("");
        clearInterimTimer();
      },
      onResult: (text, isFinal) => {
        const normalized = normalizeTranscript(text, {
          lang: speechLangRef.current,
          mode: speechModeRef.current,
        });
        if (!normalized) return;

        if (isBengaliPhoneticEnglish(normalized)) {
          void maybeAdaptSpeechLanguage(normalized);
        }

        if (isFinal) {
          clearInterimTimer();
          const parts = finalPartsRef.current;
          const last = parts.length ? parts[parts.length - 1] : "";
          const merged = mergeFinalTranscriptChunks(last, normalized);
          if (parts.length) {
            parts[parts.length - 1] = merged;
          } else {
            parts.push(merged);
          }
          interimRef.current = "";
          setInterimText("");
          rebuildSessionText("");
          const accumulated = parts.join(" ").trim();
          void maybeAdaptSpeechLanguage(accumulated);
          logSpeechEvent("speech_session_final_chunk", {
            traceId: traceIdRef.current,
            chars: normalized.length,
            mode: speechModeRef.current,
          });
        } else {
          clearInterimTimer();
          interimTimerRef.current = setTimeout(() => {
            interimRef.current = normalized;
            setInterimText(normalized);
            rebuildSessionText(normalized);
            const interimAccumulated = [
              ...finalPartsRef.current,
              normalized,
            ]
              .join(" ")
              .trim();
            if (interimAccumulated.length >= 6) {
              void maybeAdaptSpeechLanguage(interimAccumulated);
            }
          }, SPEECH_INTERIM_DEBOUNCE_MS);
        }
      },
      onError: (message) => {
        clearInterimTimer();
        setSpeechError(message);
        setIsDictating(false);
        logSpeechEvent("speech_hook_error", {
          message,
          traceId: traceIdRef.current,
        });
      },
      onEnd: () => {
        clearInterimTimer();
        setInterimText("");
        rebuildSessionText("");
      },
    });

    return () => {
      clearInterimTimer();
      abortRecognition().catch(() => {});
    };
  }, [rebuildSessionText, clearInterimTimer, maybeAdaptSpeechLanguage]);

  const resetSession = useCallback(() => {
    clearInterimTimer();
    finalPartsRef.current = [];
    interimRef.current = "";
    languageSwitchCountRef.current = 0;
    modeLockedRef.current = false;
    setSessionText("");
    setInterimText("");
    setSpeechMode("unknown");
  }, [clearInterimTimer]);

  const startDictation = useCallback(async () => {
    if (disabled) return;
    if (!capabilities.supported) {
      setSpeechError(
        "Voice input is not available in this browser. Try Chrome or Edge, or type your message."
      );
      return;
    }

    resetSession();
    setSpeechError(null);
    setIsDictating(true);
    traceIdRef.current = generateTraceId();

    const session = resolveSpeechSession(undefined, {
      draftText: draftTextRef.current,
      storedMode: readStoredSpeechMode(),
    });
    speechLangRef.current = session.lang;
    speechModeRef.current = session.mode;
    languageChainRef.current = session.languageChain;
    modeLockedRef.current = Boolean(session.modeLocked);
    setSpeechMode(session.mode);

    logSpeechEvent("speech_user_start", {
      provider: capabilities.provider,
      traceId: traceIdRef.current,
      webSpeech: capabilities.webSpeech,
      whisper: capabilities.whisper,
      language: speechLangRef.current,
      mode: speechModeRef.current,
      modeLocked: modeLockedRef.current,
    });

    try {
      await startRecognition({
        traceId: traceIdRef.current,
        language: speechLangRef.current,
        mode: speechModeRef.current,
        languageChain: languageChainRef.current,
        modeLocked: modeLockedRef.current,
      });
    } catch (err) {
      setSpeechError(err?.message || "Could not start voice input.");
      setIsDictating(false);
    }
  }, [disabled, capabilities, resetSession]);

  const cancelDictation = useCallback(async () => {
    try {
      await abortRecognition();
    } catch {
      /* ignore */
    }
    resetSession();
    setIsDictating(false);
    logSpeechEvent("speech_dictate_cancel", { traceId: traceIdRef.current });
  }, [resetSession]);

  const confirmDictation = useCallback(async () => {
    try {
      await stopRecognition();
    } catch {
      /* ignore */
    }

    clearInterimTimer();
    await new Promise((r) => setTimeout(r, 250));

    const transcript = normalizeTranscript(
      [finalPartsRef.current.join(" "), interimRef.current].filter(Boolean).join(" "),
      { lang: speechLangRef.current, mode: speechModeRef.current }
    );

    logSpeechEvent("speech_dictate_confirm", {
      traceId: traceIdRef.current,
      chars: transcript.length,
      mode: speechModeRef.current,
      language: speechLangRef.current,
    });

    if (transcript.length >= 10 && speechModeRef.current !== "unknown") {
      storeSpeechMode(speechModeRef.current);
    }

    resetSession();
    setIsDictating(false);
    return transcript;
  }, [resetSession, clearInterimTimer]);

  const clearSpeechError = useCallback(() => setSpeechError(null), []);

  const displayText =
    sessionText || interimText || (isDictating ? "" : "");

  return {
    isDictating,
    sessionText: displayText,
    interimText,
    speechError,
    speechMode,
    capabilities,
    startDictation,
    cancelDictation,
    confirmDictation,
    clearSpeechError,
  };
}
