/** Active STT provider keys (switch via VITE_SPEECH_PROVIDER). */
export const SPEECH_PROVIDERS = {
  WEB_SPEECH: "webSpeech",
  WHISPER: "whisper",
};

const envProvider = import.meta.env.VITE_SPEECH_PROVIDER?.trim();

export const ACTIVE_SPEECH_PROVIDER =
  envProvider === SPEECH_PROVIDERS.WHISPER
    ? SPEECH_PROVIDERS.WHISPER
    : SPEECH_PROVIDERS.WEB_SPEECH;

/** BCP-47 tags tried in order when browser supports them. */
export const SPEECH_LANGUAGE_PREFERENCE = (
  import.meta.env.VITE_SPEECH_LANGUAGES?.trim() || "bn-BD,bn-IN,bn,en-IN,en-US"
)
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

/**
 * When true, pick STT language from spoken content (English / Bangla / Banglish).
 * When false, always use the first tag in SPEECH_LANGUAGE_PREFERENCE (legacy Bangla-first).
 */
export const SPEECH_AUTO_DETECT_LANGUAGE =
  import.meta.env.VITE_SPEECH_AUTO_DETECT !== "false";

/**
 * Always use the first language in SPEECH_LANGUAGE_PREFERENCE for STT.
 * Ignored when SPEECH_AUTO_DETECT_LANGUAGE is true.
 */
export const SPEECH_FORCE_PRIMARY_LANGUAGE =
  import.meta.env.VITE_SPEECH_FORCE_PRIMARY_LANGUAGE !== "false";

/**
 * Start mic language: banglish (en-IN) handles English correctly; switches to bn-BD
 * when genuine Bengali script is heard. Override with VITE_SPEECH_DEFAULT_MODE=bn|en.
 */
export const SPEECH_DEFAULT_START_MODE =
  import.meta.env.VITE_SPEECH_DEFAULT_MODE?.trim() || "banglish";

/** Max STT language switches per dictation session (prevents restart thrashing). */
export const SPEECH_LANG_SWITCH_MAX = Number(
  import.meta.env.VITE_SPEECH_LANG_SWITCH_MAX || 2
);

/** Min accumulated chars before switching STT language mid-session. */
export const SPEECH_LANG_SWITCH_MIN_CHARS = Number(
  import.meta.env.VITE_SPEECH_LANG_SWITCH_MIN_CHARS || 15
);

/** Shorter threshold when leaving Bengali STT for clear English speech. */
export const SPEECH_EN_SWITCH_MIN_CHARS = Number(
  import.meta.env.VITE_SPEECH_EN_SWITCH_MIN_CHARS || 8
);

export const SPEECH_CONTINUOUS = true;
export const SPEECH_INTERIM_RESULTS = true;
export const SPEECH_MAX_ALTERNATIVES = Number(
  import.meta.env.VITE_SPEECH_MAX_ALTERNATIVES || 3
);

/** Delay before auto-restarting continuous recognition (reduces clipped Bengali words). */
export const SPEECH_RESTART_DELAY_MS = Number(
  import.meta.env.VITE_SPEECH_RESTART_DELAY_MS || 300
);

/** Debounce interim UI updates — Bengali interim text flickers heavily otherwise. */
export const SPEECH_INTERIM_DEBOUNCE_MS = Number(
  import.meta.env.VITE_SPEECH_INTERIM_DEBOUNCE_MS || 150
);

/** Whisper recording limits (Phase 2). */
export const WHISPER_MAX_RECORDING_MS = Number(
  import.meta.env.VITE_WHISPER_MAX_RECORDING_MS || 120_000
);
export const WHISPER_MIME_PREFERENCE = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];
