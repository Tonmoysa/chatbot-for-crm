import { MarkdownContent } from "../utils/markdown.jsx";
import { getDisplayTime } from "./chat/dateUtils.js";
import WorkflowActionBar from "./chat/WorkflowActionBar.jsx";

function FileBubble({ name, sizeLabel }) {
  return (
    <div className="file-bubble">
      <div className="file-icon">
        <svg className="hr-icon" viewBox="0 0 24 24" aria-hidden>
          <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
        </svg>
      </div>
      <div className="file-info">
        <span className="file-name">{name}</span>
        {sizeLabel ? <span className="file-size">{sizeLabel}</span> : null}
      </div>
    </div>
  );
}

function AudioBubble() {
  const bars = Array.from({ length: 15 }, (_, i) => ({
    key: i,
    h: 10 + ((i * 17) % 14),
  }));
  return (
    <div className="audio-bubble">
      <button type="button" className="play-btn" aria-label="Play voice message">
        <svg className="hr-icon" viewBox="0 0 24 24" style={{ width: 16, height: 16, marginLeft: 2 }} aria-hidden>
          <path d="M8 5v14l11-7z" />
        </svg>
      </button>
      <div className="waveform">
        {bars.map(({ key, h }) => (
          <div key={key} className="bar" style={{ height: `${h}px` }} />
        ))}
      </div>
      <span className="duration">0:12</span>
    </div>
  );
}

/**
 * @param {{ message?: { role: string, text: string, at?: string, attachment?: { name: string, size: number }, type?: string }, role?: string, text?: string, at?: string }} props
 */
export default function MessageBubble(props) {
  const m = props.message || props;
  const role = m.role;
  const text = m.text ?? "";
  const at = m.at ?? props.at;
  const attachment = m.attachment;
  const type = m.type || "text";
  const actions = m.actions;
  const onAction = props.onAction;
  const actionsDisabled = Boolean(props.actionsDisabled);

  const isUser = role === "user";
  const isBot = role === "bot" || role === "assistant";
  const time = getDisplayTime(at);

  let body = null;
  if (type === "audio") {
    body = <AudioBubble />;
  } else if (type === "file" && attachment) {
    body = (
      <FileBubble
        name={attachment.name}
        sizeLabel={typeof attachment.size === "number" ? `${(attachment.size / 1024).toFixed(1)} KB` : ""}
      />
    );
  } else if (attachment && text) {
    body = (
      <>
        <FileBubble
          name={attachment.name}
          sizeLabel={typeof attachment.size === "number" ? `${(attachment.size / 1024).toFixed(1)} KB` : ""}
        />
        <p className="mt-2 whitespace-pre-wrap break-words">{text}</p>
      </>
    );
  } else if (attachment) {
    body = (
      <FileBubble
        name={attachment.name}
        sizeLabel={typeof attachment.size === "number" ? `${(attachment.size / 1024).toFixed(1)} KB` : ""}
      />
    );
  } else if (isUser || !isBot) {
    body = <p className="whitespace-pre-wrap break-words">{text}</p>;
  } else {
    body = <MarkdownContent text={text} className="prose-chat-msg" />;
  }

  return (
    <div className={`message-row ${isUser ? "user" : "bot"}`} role="article" aria-label={isUser ? "You" : "HR AI"}>
      <div className="message-bubble">
        {body}
        {isBot && actions?.length ? (
          <WorkflowActionBar
            actions={actions}
            onAction={onAction}
            disabled={actionsDisabled}
          />
        ) : null}
        {time ? <span className="message-time">{time}</span> : null}
      </div>
    </div>
  );
}
