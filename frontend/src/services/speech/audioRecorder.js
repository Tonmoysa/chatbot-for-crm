import { WHISPER_MAX_RECORDING_MS, WHISPER_MIME_PREFERENCE } from "./speechConfig.js";
import { logSpeechEvent } from "../../utils/trace.js";

/**
 * MediaRecorder wrapper for Phase 2 (Whisper) uploads.
 */
export class AudioRecorder {
  /** @type {MediaRecorder | null} */
  #recorder = null;
  /** @type {MediaStream | null} */
  #stream = null;
  /** @type {Blob[]} */
  #chunks = [];
  #mimeType = "audio/webm";

  /**
   * @param {{ traceId?: string, maxMs?: number }} [options]
   */
  async start(options = {}) {
    const traceId = options.traceId || "";
    const maxMs = options.maxMs ?? WHISPER_MAX_RECORDING_MS;

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone capture is not supported in this browser.");
    }

    this.#stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.#mimeType =
      WHISPER_MIME_PREFERENCE.find((m) => MediaRecorder.isTypeSupported(m)) ||
      "audio/webm";

    this.#chunks = [];
    this.#recorder = new MediaRecorder(this.#stream, { mimeType: this.#mimeType });
    this.#recorder.ondataavailable = (e) => {
      if (e.data?.size > 0) this.#chunks.push(e.data);
    };

    this.#recorder.start(250);
    logSpeechEvent("audio_recorder_start", { traceId, mimeType: this.#mimeType });

    if (maxMs > 0) {
      this.#stopTimer = window.setTimeout(() => {
        if (this.#recorder?.state === "recording") {
          this.stop().catch(() => {});
        }
      }, maxMs);
    }
  }

  /** @type {number | undefined} */
  #stopTimer;

  /**
   * @returns {Promise<{ blob: Blob, mimeType: string }>}
   */
  async stop() {
    if (this.#stopTimer) {
      clearTimeout(this.#stopTimer);
      this.#stopTimer = undefined;
    }

    const recorder = this.#recorder;
    if (!recorder || recorder.state === "inactive") {
      this.#cleanupStream();
      const blob = new Blob(this.#chunks, { type: this.#mimeType });
      return { blob, mimeType: this.#mimeType };
    }

    return new Promise((resolve, reject) => {
      recorder.onstop = () => {
        const blob = new Blob(this.#chunks, { type: this.#mimeType });
        this.#cleanupStream();
        logSpeechEvent("audio_recorder_stop", {
          bytes: blob.size,
          mimeType: this.#mimeType,
        });
        resolve({ blob, mimeType: this.#mimeType });
      };
      recorder.onerror = () => {
        this.#cleanupStream();
        reject(new Error("Recording failed."));
      };
      try {
        recorder.stop();
      } catch (e) {
        this.#cleanupStream();
        reject(e);
      }
    });
  }

  cancel() {
    if (this.#stopTimer) {
      clearTimeout(this.#stopTimer);
      this.#stopTimer = undefined;
    }
    if (this.#recorder && this.#recorder.state !== "inactive") {
      try {
        this.#recorder.stop();
      } catch {
        /* ignore */
      }
    }
    this.#chunks = [];
    this.#cleanupStream();
  }

  #cleanupStream() {
    this.#stream?.getTracks().forEach((t) => t.stop());
    this.#stream = null;
    this.#recorder = null;
  }
}
