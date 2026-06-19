import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createSpeechProvider,
  getActiveProviderName,
  getSpeechCapabilities,
} from "./speechService.js";
import { SPEECH_PROVIDERS } from "./speechConfig.js";

vi.mock("./providers/webSpeechProvider.js", () => ({
  WebSpeechProvider: class {
    name = "webSpeech";
    callbacks = {};
    isSupported() {
      return true;
    }
    setCallbacks(c) {
      this.callbacks = c || {};
    }
    async start() {
      this.callbacks.onStart?.();
    }
    async stop() {}
  },
}));

vi.mock("./providers/whisperProvider.js", () => ({
  WhisperProvider: class {
    name = "whisper";
    isSupported() {
      return true;
    }
    setCallbacks() {}
    async start() {}
    async stop() {}
  },
}));

describe("speechService", () => {
  beforeEach(() => {
    vi.unstubAllEnvs();
  });

  it("creates webSpeech provider by default", () => {
    const p = createSpeechProvider(SPEECH_PROVIDERS.WEB_SPEECH);
    expect(p.name).toBe("webSpeech");
  });

  it("creates whisper provider when requested", () => {
    const p = createSpeechProvider(SPEECH_PROVIDERS.WHISPER);
    expect(p.name).toBe("whisper");
  });

  it("exposes active provider name", () => {
    expect(getActiveProviderName()).toBeTruthy();
  });

  it("returns capability flags", () => {
    const caps = getSpeechCapabilities();
    expect(caps).toHaveProperty("provider");
    expect(caps).toHaveProperty("supported");
  });

  it("applies registered callbacks to new provider instances", async () => {
    globalThis.webkitSpeechRecognition = function Mock() {};
    const { setSpeechCallbacks, startRecognition } = await import("./speechService.js");
    const onStart = vi.fn();
    setSpeechCallbacks({ onStart });
    await startRecognition({ traceId: "test-trace" });
    expect(onStart).toHaveBeenCalled();
  });
});
