export default function VoiceWaveform({ levels, className = "" }) {
  return (
    <div
      className={`flex h-8 flex-1 items-center justify-center gap-[3px] px-2 ${className}`}
      aria-hidden
    >
      {levels.map((level, i) => (
        <span
          key={i}
          className="w-[3px] rounded-full bg-gradient-to-t from-cyan-600 to-brand-500 transition-[height] duration-75 dark:from-cyan-400 dark:to-brand-400"
          style={{ height: `${Math.round(8 + level * 22)}px` }}
        />
      ))}
    </div>
  );
}
