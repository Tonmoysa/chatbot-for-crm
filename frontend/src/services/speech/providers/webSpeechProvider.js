import {
  SPEECH_CONTINUOUS,
  SPEECH_INTERIM_RESULTS,
  SPEECH_MAX_ALTERNATIVES,
  SPEECH_RESTART_DELAY_MS,
} from "../speechConfig.js";
import {
  getSpeechRecognitionConstructor,
  normalizeTranscript,
  pickBestTranscript,
  resolveSpeechLanguage,
  speechErrorMessage,
  speechLanguageFallbackChain,
} from "../speechUtils.js";
import {
  SPEECH_LANG_MODES,
  resolveTranscriptProcessingMode,
} from "../speechLanguageDetect.js";
import { logSpeechEvent } from "../../../utils/trace.js";
import { BaseSpeechProvider } from "./baseProvider.js";

export class WebSpeechProvider extends BaseSpeechProvider {
  name = "webSpeech";

  /** @type {SpeechRecognition | null} */
  #recognition = null;
  #listening = false;
  #shouldRestart = false;
  #language = "bn-BD";
  #traceId = "";
  /** @type {ReturnType<typeof setTimeout> | null} */
  #restartTimer = null;
  #languageIndex = 0;
  /** @type {string[]} */
  #languageChain = ["bn-BD"];
  #speechMode = "bn";
  #modeLocked = false;

  isSupported() {
    return Boolean(getSpeechRecognitionConstructor());
  }

  #clearRestartTimer() {
    if (this.#restartTimer) {
      clearTimeout(this.#restartTimer);
      this.#restartTimer = null;
    }
  }

  #currentLanguage() {
    return this.#languageChain[this.#languageIndex] || this.#language;
  }

  #attachRecognitionHandlers(recognition) {
    recognition.onstart = () => {
      this.#listening = true;
      logSpeechEvent("speech_recognition_start", {
        provider: this.name,
        traceId: this.#traceId,
        language: this.#currentLanguage(),
      });
      this.emitStart();
    };

    recognition.onresult = (event) => {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        let pickMode = this.#speechMode;
        if (!this.#modeLocked) {
          const rawAlt = result[0]?.transcript || "";
          const inferred = resolveTranscriptProcessingMode(rawAlt, {
            mode: this.#speechMode,
          });
          if (
            inferred.mode === SPEECH_LANG_MODES.EN &&
            inferred.confidence >= 0.45
          ) {
            pickMode = SPEECH_LANG_MODES.EN;
          } else if (
            inferred.mode === SPEECH_LANG_MODES.BANGLISH &&
            inferred.confidence >= 0.45
          ) {
            pickMode = SPEECH_LANG_MODES.BANGLISH;
          }
        }
        const transcript = pickBestTranscript(result, {
          mode: pickMode,
          modeLocked: this.#modeLocked,
        });
        if (result.isFinal) {
          final += transcript;
        } else {
          interim += transcript;
        }
      }
      const raw = final || interim;
      const lang = this.#currentLanguage();
      const text = normalizeTranscript(raw, { lang, mode: this.#speechMode });
      if (!text) return;
      const isFinal = Boolean(final);
      if (isFinal) {
        logSpeechEvent("speech_transcript_final", {
          provider: this.name,
          traceId: this.#traceId,
          chars: text.length,
          language: lang,
        });
      }
      this.emitResult(text, isFinal);
    };

    recognition.onerror = (event) => {
      const msg = speechErrorMessage(event);
      logSpeechEvent("speech_recognition_error", {
        provider: this.name,
        traceId: this.#traceId,
        code: event.error,
        language: this.#currentLanguage(),
      });

      if (event.error === "aborted") return;

      if (
        event.error === "language-not-supported" &&
        this.#languageIndex < this.#languageChain.length - 1
      ) {
        this.#languageIndex += 1;
        this.#shouldRestart = true;
        this.#clearRestartTimer();
        try {
          recognition.abort();
        } catch {
          /* ignore */
        }
        this.#startRecognitionInstance();
        return;
      }

      this.#shouldRestart = false;
      this.emitError(msg);
    };

    recognition.onend = () => {
      this.#listening = false;
      logSpeechEvent("speech_recognition_stop", {
        provider: this.name,
        traceId: this.#traceId,
        language: this.#currentLanguage(),
      });

      if (this.#shouldRestart && this.#recognition === recognition) {
        this.#clearRestartTimer();
        this.#restartTimer = setTimeout(() => {
          if (!this.#shouldRestart || this.#recognition !== recognition) return;
          try {
            recognition.start();
          } catch {
            this.#shouldRestart = false;
            this.emitEnd();
          }
        }, SPEECH_RESTART_DELAY_MS);
        return;
      }
      this.emitEnd();
    };
  }

  #startRecognitionInstance() {
    const Ctor = getSpeechRecognitionConstructor();
    if (!Ctor) return;

    const lang = this.#currentLanguage();
    this.#language = lang;

    const recognition = new Ctor();
    recognition.lang = lang;
    recognition.continuous = SPEECH_CONTINUOUS;
    recognition.interimResults = SPEECH_INTERIM_RESULTS;
    recognition.maxAlternatives = SPEECH_MAX_ALTERNATIVES;

    this.#attachRecognitionHandlers(recognition);
    this.#recognition = recognition;
    recognition.start();
  }

  /**
   * @param {{ language?: string, traceId?: string }} [options]
   */
  async start(options = {}) {
    const Ctor = getSpeechRecognitionConstructor();
    if (!Ctor) {
      throw new Error(
        "Speech recognition is not supported in this browser. Try Chrome or Edge on desktop."
      );
    }

    this.#language = options.language || resolveSpeechLanguage();
    this.#speechMode = options.mode || "bn";
    this.#modeLocked = Boolean(options.modeLocked);
    this.#languageChain =
      options.languageChain?.length > 0
        ? options.languageChain
        : speechLanguageFallbackChain(this.#language);
    this.#languageIndex = 0;
    this.#traceId = options.traceId || "";
    this.#shouldRestart = true;
    this.#clearRestartTimer();

    if (this.#recognition) {
      try {
        this.#recognition.abort();
      } catch {
        /* ignore */
      }
      this.#recognition = null;
    }

    logSpeechEvent("speech_language_resolved", {
      provider: this.name,
      traceId: this.#traceId,
      language: this.#currentLanguage(),
      mode: this.#speechMode,
      fallbackChain: this.#languageChain,
    });

    this.#startRecognitionInstance();
  }

  /**
   * Switch STT language mid-session when spoken mode is detected confidently.
   * @param {{ language: string, mode?: string, languageChain?: string[] }} options
   */
  async updateLanguage(options) {
    const nextLang = options.language;
    if (!nextLang || nextLang === this.#currentLanguage()) {
      if (options.mode) this.#speechMode = options.mode;
      if (options.modeLocked) this.#modeLocked = true;
      return;
    }
    this.#language = nextLang;
    if (options.mode) this.#speechMode = options.mode;
    if (options.modeLocked) this.#modeLocked = true;
    this.#languageChain =
      options.languageChain?.length > 0
        ? options.languageChain
        : speechLanguageFallbackChain(nextLang);
    this.#languageIndex = 0;
    this.#shouldRestart = true;

    logSpeechEvent("speech_language_switch", {
      provider: this.name,
      traceId: this.#traceId,
      language: nextLang,
      mode: this.#speechMode,
    });

    const recognition = this.#recognition;
    if (!recognition) return;
    try {
      recognition.abort();
    } catch {
      /* ignore */
    }
    this.#startRecognitionInstance();
  }

  async stop() {
    this.#shouldRestart = false;
    this.#clearRestartTimer();
    const recognition = this.#recognition;
    if (!recognition) return;
    try {
      recognition.stop();
    } catch {
      try {
        recognition.abort();
      } catch {
        /* ignore */
      }
    }
    this.#recognition = null;
    this.#listening = false;
  }

  async abort() {
    this.#shouldRestart = false;
    this.#clearRestartTimer();
    const recognition = this.#recognition;
    if (!recognition) return;
    try {
      recognition.abort();
    } catch {
      /* ignore */
    }
    this.#recognition = null;
    this.#listening = false;
    this.emitEnd();
  }

  get listening() {
    return this.#listening;
  }
}
