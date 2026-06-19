export default function TypingIndicator() {
  return (
    <div className="message-row bot" aria-live="assertive">
      <div className="message-bubble typing-bubble-inner" aria-label="Assistant is typing">
        <span className="typing-dot-hr" />
        <span className="typing-dot-hr" />
        <span className="typing-dot-hr" />
      </div>
    </div>
  );
}
