import { useCallback, useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import ChatInput from "./chat/ChatInput";
import TypingIndicator from "./chat/TypingIndicator";
import ChatThreadHeader from "./chat/ChatThreadHeader";
import DateSeparator from "./chat/DateSeparator";
import { getDisplayDate } from "./chat/dateUtils.js";
import "../styles/chat-design.css";

/**
 * WhatsApp-style chat shell (design system under .hr-chat-card).
 * Preserves existing props: messages, loading, error, onSend, onClearError.
 */
const BOTTOM_STICK_THRESHOLD_PX = 96;

export default function ChatBox({ messages, loading, error, onSend, onClearError }) {
  const chatBodyRef = useRef(null);
  const stickToBottomRef = useRef(true);

  const handleSend = useCallback(
    (payload) => {
      stickToBottomRef.current = true;
      onSend(payload);
    },
    [onSend]
  );

  const scrollToBottom = useCallback((behavior = "smooth") => {
    const el = chatBodyRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  const updateStickToBottom = useCallback(() => {
    const el = chatBodyRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom <= BOTTOM_STICK_THRESHOLD_PX;
  }, []);

  useEffect(() => {
    const last = messages[messages.length - 1];
    const userJustSent = last?.role === "user";
    const shouldScroll = userJustSent || loading || stickToBottomRef.current;
    if (!shouldScroll) return;

    const id = requestAnimationFrame(() => {
      requestAnimationFrame(() => scrollToBottom("smooth"));
    });
    return () => cancelAnimationFrame(id);
  }, [messages, loading, scrollToBottom]);

  const showEmpty = messages.length === 0 && !loading;

  const renderThread = () => {
    let lastDate = null;
    const elements = [];

    messages.forEach((msg, i) => {
      const msgDate = getDisplayDate(msg.at);
      if (msgDate !== lastDate) {
        elements.push(<DateSeparator key={`sep-${msg.at || i}-${msgDate}`} date={msgDate} />);
        lastDate = msgDate;
      }
      const isLatestBot =
        msg.role === "bot" && !messages.slice(i + 1).some((m) => m.role === "bot");
      elements.push(
        <MessageBubble
          key={`${i}-${msg.role}-${(msg.text || "").slice(0, 24)}`}
          message={msg}
          onAction={
            isLatestBot
              ? (action) => {
                  const text = (action?.message || action?.label || "").trim();
                  if (text) handleSend(text);
                }
              : undefined
          }
          actionsDisabled={loading || !isLatestBot}
        />
      );
    });

    return elements;
  };

  return (
    <div className="hr-chat-card">
        <ChatThreadHeader />

        <div
          ref={chatBodyRef}
          className="chat-body chat-scroll"
          role="log"
          aria-live="polite"
          aria-relevant="additions"
          onScroll={updateStickToBottom}
        >
          {showEmpty ? (
            <div className="chat-empty-greeting">
              <h2 className="chat-empty-greeting-title">How can I help you today?</h2>
            </div>
          ) : (
            renderThread()
          )}

          {loading ? <TypingIndicator /> : null}

          <span className="block h-px w-full shrink-0" aria-hidden />
        </div>

        <ChatInput
          onSend={handleSend}
          busy={loading}
          error={error}
          onClearError={onClearError}
        />
    </div>
  );
}
