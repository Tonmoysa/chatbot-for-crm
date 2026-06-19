/**
 * @typedef {Object} SpeechRecognitionCallbacks
 * @property {(text: string, isFinal: boolean) => void} [onResult]
 * @property {(message: string) => void} [onError]
 * @property {() => void} [onStart]
 * @property {() => void} [onEnd]
 */

/**
 * Abstract speech-to-text provider (Web Speech, Whisper, etc.).
 */
export class BaseSpeechProvider {
  /** @type {string} */
  name = "base";

  /**
   * @param {SpeechRecognitionCallbacks} [_callbacks]
   */
  constructor(_callbacks = {}) {
    /** @type {SpeechRecognitionCallbacks} */
    this.callbacks = {};
  }

  /**
   * @param {SpeechRecognitionCallbacks} callbacks
   */
  setCallbacks(callbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  /**
   * @returns {boolean}
   */
  isSupported() {
    return false;
  }

  /**
   * @param {{ language?: string, traceId?: string }} [_options]
   */
  async start(_options = {}) {
    throw new Error(`${this.name}: start() not implemented`);
  }

  async stop() {
    throw new Error(`${this.name}: stop() not implemented`);
  }

  async abort() {
    await this.stop();
  }

  /** @protected */
  emitResult(text, isFinal) {
    this.callbacks.onResult?.(text, isFinal);
  }

  /** @protected */
  emitError(message) {
    this.callbacks.onError?.(message);
  }

  /** @protected */
  emitStart() {
    this.callbacks.onStart?.();
  }

  /** @protected */
  emitEnd() {
    this.callbacks.onEnd?.();
  }
}
