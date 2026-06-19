import { formatRelativeTime } from "../../utils/formatRelativeTime.js";
import { PlusIcon, SparklesIcon } from "../icons.jsx";

function SidebarContent({
  onNewChat,
  onClose,
  activeSessionId,
  sessions,
  sessionsLoading,
  sessionsError,
  onSelectSession,
}) {
  return (
    <>
      <div className="flex items-center gap-2 border-b border-slate-200/80 px-4 py-4 dark:border-slate-800/80">
        <div className="flex size-9 items-center justify-center rounded-xl bg-gradient-to-br from-navy-800 to-brand-600 shadow-glow-sm">
          <SparklesIcon className="size-5 text-white" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-900 dark:text-white">HR AI</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">Assistant</p>
        </div>
      </div>

      <div className="p-3">
        <button
          type="button"
          onClick={() => {
            onNewChat?.();
            onClose?.();
          }}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200/90 bg-white px-4 py-2.5 text-sm font-medium text-slate-800 shadow-sm transition hover:border-cyan-500/40 hover:bg-slate-50 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50 dark:border-slate-700 dark:bg-slate-900/80 dark:text-slate-100 dark:hover:border-cyan-500/30 dark:hover:bg-slate-800"
        >
          <PlusIcon className="size-4" />
          New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <p className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
          Recent
        </p>

        {sessionsLoading ? (
          <p className="px-3 py-2 text-xs text-slate-500 dark:text-slate-400">Loading…</p>
        ) : null}

        {sessionsError ? (
          <p className="px-3 py-2 text-xs text-amber-700 dark:text-amber-300" role="alert">
            {sessionsError}
          </p>
        ) : null}

        {!sessionsLoading && !sessionsError && sessions.length === 0 ? (
          <p className="px-3 py-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            No conversations yet. Start a new chat to see history here.
          </p>
        ) : null}

        <ul className="space-y-0.5" role="list">
          {sessions.map((conv) => (
            <li key={conv.session_id}>
              <button
                type="button"
                onClick={() => {
                  onSelectSession?.(conv.session_id);
                  onClose?.();
                }}
                className={`group flex w-full flex-col rounded-lg px-3 py-2.5 text-left text-sm transition hover:bg-slate-100 dark:hover:bg-slate-800/80 ${
                  activeSessionId === conv.session_id
                    ? "bg-slate-100 dark:bg-slate-800/90"
                    : ""
                }`}
                aria-label={`Open conversation: ${conv.title}`}
                aria-current={activeSessionId === conv.session_id ? "true" : undefined}
              >
                <span className="truncate font-medium text-slate-800 group-hover:text-slate-900 dark:text-slate-200 dark:group-hover:text-white">
                  {conv.title}
                </span>
                <span className="mt-0.5 text-xs text-slate-400 dark:text-slate-500">
                  {formatRelativeTime(conv.updated_at)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

export default function Sidebar({
  open,
  collapsed,
  onClose,
  onNewChat,
  onToggleCollapse,
  activeSessionId,
  sessions,
  sessionsLoading,
  sessionsError,
  onSelectSession,
}) {
  if (collapsed) {
    return (
      <aside
        className="hidden w-14 shrink-0 flex-col border-r border-slate-200/80 bg-slate-50/95 dark:border-slate-800/80 dark:bg-navy-950/95 md:flex"
        aria-label="Sidebar"
      >
        <div className="flex flex-col items-center gap-2 p-2 pt-3">
          <button
            type="button"
            onClick={onToggleCollapse}
            className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-200/80 hover:text-slate-800 dark:hover:bg-slate-800 dark:hover:text-white"
            aria-label="Expand sidebar"
          >
            <PanelLeftExpandIcon />
          </button>
          <button
            type="button"
            onClick={onNewChat}
            className="rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-600 p-2.5 text-white shadow-md transition hover:from-emerald-400 hover:to-emerald-500"
            aria-label="New chat"
          >
            <PlusIcon className="size-4" />
          </button>
        </div>
      </aside>
    );
  }

  return (
    <>
      {open ? (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-navy-950/40 backdrop-blur-sm md:hidden"
          aria-label="Close sidebar"
          onClick={onClose}
        />
      ) : null}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[min(100%,280px)] flex-col border-r border-slate-200/80 bg-slate-50/98 shadow-xl backdrop-blur-xl transition-transform duration-300 ease-out dark:border-slate-800/80 dark:bg-navy-950/98 md:static md:z-auto md:w-72 md:translate-x-0 md:shadow-none ${
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        } ${collapsed ? "md:hidden" : ""}`}
        aria-label="Conversation sidebar"
      >
        <SidebarContent
          onNewChat={onNewChat}
          onClose={onClose}
          activeSessionId={activeSessionId}
          sessions={sessions}
          sessionsLoading={sessionsLoading}
          sessionsError={sessionsError}
          onSelectSession={onSelectSession}
        />
      </aside>
    </>
  );
}

function PanelLeftExpandIcon() {
  return (
    <svg
      className="size-5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M9 3v18M14 9l3 3-3 3" />
    </svg>
  );
}
