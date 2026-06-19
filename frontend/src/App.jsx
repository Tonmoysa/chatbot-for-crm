import { useCallback } from "react";
import ChatBox from "./components/ChatBox";
import { useChat } from "./hooks/useChat";

export default function App() {
  const {
    messages,
    loading,
    error,
    sendMessage,
    clearError,
    clearMessages,
  } = useChat();

  const handleNewChat = useCallback(() => {
    clearMessages();
  }, [clearMessages]);

  return (
    <div className="hr-chat-card-shell">
      <ChatBox
        messages={messages}
        loading={loading}
        error={error}
        onSend={sendMessage}
        onClearError={clearError}
        onNewChat={handleNewChat}
      />
    </div>
  );
}
