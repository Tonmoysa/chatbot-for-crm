import { useEffect, useRef, useState } from "react";

/**
 * Microphone level bars for dictate UI (Web Audio API).
 */
export function useVoiceWaveform(active) {
  const [levels, setLevels] = useState(() => Array(12).fill(0.15));
  const rafRef = useRef(null);
  const streamRef = useRef(null);
  const ctxRef = useRef(null);

  useEffect(() => {
    if (!active) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      ctxRef.current?.close().catch(() => {});
      ctxRef.current = null;
      setLevels(Array(12).fill(0.15));
      return undefined;
    }

    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 64;
        source.connect(analyser);
        const data = new Uint8Array(analyser.frequencyBinCount);

        const tick = () => {
          if (cancelled) return;
          analyser.getByteFrequencyData(data);
          const step = Math.floor(data.length / 12);
          const next = Array.from({ length: 12 }, (_, i) => {
            const v = data[i * step] / 255;
            return Math.max(0.12, Math.min(1, v * 1.4));
          });
          setLevels(next);
          rafRef.current = requestAnimationFrame(tick);
        };
        tick();
      } catch {
        /* waveform optional if mic already in use */
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      ctxRef.current?.close().catch(() => {});
      ctxRef.current = null;
    };
  }, [active]);

  return levels;
}
