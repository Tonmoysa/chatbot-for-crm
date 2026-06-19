import { BotIcon, MenuIcon, MoonIcon, PanelLeftIcon, ShieldCheckIcon, SunIcon } from "../icons.jsx";

export default function AppHeader({
  dark,
  onToggleTheme,
  onMenuClick,
  onToggleSidebar,
  sidebarCollapsed,
}) {
  return (
    <header className="relative z-30 flex shrink-0 items-center gap-3 border-b border-slate-200/60 bg-white/70 px-3 py-2.5 backdrop-blur-xl dark:border-slate-800/60 dark:bg-navy-950/70 sm:px-4 sm:py-3">
      <div className="flex min-w-0 flex-1 items-center gap-2 sm:gap-3">
        <button
          type="button"
          onClick={onMenuClick}
          className="inline-flex size-9 shrink-0 items-center justify-center rounded-xl text-slate-600 transition hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50 md:hidden dark:text-slate-300 dark:hover:bg-slate-800"
          aria-label="Open menu"
        >
          <MenuIcon />
        </button>

        <button
          type="button"
          onClick={onToggleSidebar}
          className="hidden size-9 shrink-0 items-center justify-center rounded-xl text-slate-600 transition hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50 md:inline-flex dark:text-slate-300 dark:hover:bg-slate-800"
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <PanelLeftIcon />
        </button>

        <div className="flex min-w-0 items-center gap-3">
          <div className="relative flex size-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-navy-800 via-brand-600 to-cyan-500 shadow-glow-sm ring-1 ring-white/20 dark:ring-cyan-400/20">
            <BotIcon className="size-5 text-white" />
            <span
              className="absolute -bottom-0.5 -right-0.5 size-2.5 rounded-full border-2 border-white bg-emerald-400 dark:border-navy-950"
              aria-hidden
            />
          </div>

          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold tracking-tight text-slate-900 dark:text-white sm:text-lg">
              HR AI Assistant
            </h1>
            <div className="mt-0.5 flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300">
                <span className="size-1.5 animate-pulse-soft rounded-full bg-emerald-500" aria-hidden />
                Online
              </span>
              <span className="inline-flex items-center gap-1 text-[11px] text-slate-500 dark:text-slate-400">
                <ShieldCheckIcon className="size-3.5 text-cyan-600 dark:text-cyan-400" />
                Secure HR workflow
              </span>
            </div>
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={onToggleTheme}
        className="group inline-flex size-10 shrink-0 items-center justify-center rounded-xl border border-slate-200/80 bg-slate-50/80 text-slate-700 shadow-sm transition hover:border-cyan-500/30 hover:bg-white hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50 dark:border-slate-700 dark:bg-slate-900/80 dark:text-slate-200 dark:hover:border-cyan-500/20 dark:hover:bg-slate-800"
        aria-pressed={dark}
        aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      >
        {dark ? (
          <SunIcon className="size-5 transition group-hover:rotate-12" />
        ) : (
          <MoonIcon className="size-5 transition group-hover:-rotate-12" />
        )}
      </button>
    </header>
  );
}
