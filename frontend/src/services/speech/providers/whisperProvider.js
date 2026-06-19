import { AudioRecorder } from "../audioRecorder.js";
import { resolveSpeechLanguage } from "../speechUtils.js";
import { logSpeechEvent } from "../../../utils/trace.js";
import { postVoiceTranscribe } from "../../api.js";
import { BaseSpeechProvider } from "./baseProvider.js";

/**
 * Phase 2: record audio in-browser, transcribe via POST /api/voice/transcribe/.
 */
export class WhisperProvider extends BaseSpeechProvider {
  name = "whisper";

  /** @type {AudioRecorder | null} */
  #recorder = null;
  #traceId = "";
  #language = "";

  isSupported() {
    return (
      typeof window !== "undefined" &&
      typeof MediaRecorder !== "undefined" &&
      Boolean(navigator.mediaDevices?.getUserMedia)
    );
  }

  /**
   * @param {{ language?: string, traceId?: string }} [options]
   */
  async start(options = {}) {
    if (!this.isSupported()) {
      throw new Error("Audio recording is not supported in this browser.");
    }

    this.#traceId = options.traceId || "";
    this.#language = options.language || resolveSpeechLanguage();
    this.#recorder = new AudioRecorder();

    logSpeechEvent("whisper_recording_start", {
      provider: this.name,
      traceId: this.#traceId,
    });

    await this.#recorder.start({ traceId: this.#traceId });
    this.emitStart();
  }

  async stop() {
    const recorder = this.#recorder;
    this.#recorder = null;
    if (!recorder) {
      this.emitEnd();
      return;
    }

    try {
      const { blob, mimeType } = await recorder.stop();
      logSpeechEvent("whisper_upload_start", {
        traceId: this.#traceId,
        bytes: blob.size,
      });

      const { transcript } = await postVoiceTranscribe({
        blob,
        mimeType,
        language: this.#language.split("-")[0],
        traceId: this.#traceId,
      });

      if (transcript) {
        logSpeechEvent("whisper_transcript_final", {
          traceId: this.#traceId,
          chars: transcript.length,
        });
        this.emitResult(transcript, true);
      } else {
        this.emitError("No transcript was returned from the server.");
      }
    } catch (err) {
      const msg =
        err?.response?.data?.response?.message ||
        err?.message ||
        "Voice transcription failed.";
      logSpeechEvent("whisper_transcribe_error", {
        traceId: this.#traceId,
        message: msg,
      });
      this.emitError(msg);
    } finally {
      this.emitEnd();
    }
  }

  async abort() {
    this.#recorder?.cancel();
    this.#recorder = null;
    this.emitEnd();
  }
}
