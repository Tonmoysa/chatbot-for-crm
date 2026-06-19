export default function ChatThreadHeader() {
  return (
    <header className="chat-header">
      <div className="header-info">
        <div className="avatar-wrap">
          <div className="avatar">HR</div>
          <div className="status-dot" aria-hidden />
        </div>
        <div className="header-text">
          <h3>HR Assistant</h3>
          <p>Online • Support</p>
        </div>
      </div>
      <div className="header-actions">
        <button type="button" aria-label="Menu">
          <svg className="hr-icon" viewBox="0 0 24 24" aria-hidden>
            <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
          </svg>
        </button>
      </div>
    </header>
  );
}
