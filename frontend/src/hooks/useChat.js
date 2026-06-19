import { useCallback, useEffect, useRef, useState } from "react";
import { fetchChatSession, fetchChatSessions, postChat, postDocumentExtract } from "../services/api";
import {
  getClientIdentity,
  rotateSessionId,
  setSessionId,
} from "../utils/session";

function extractBotText(payload) {
  const msg = payload?.response?.message;
  return typeof msg === "string" ? msg : "";
}

function extractBotActions(payload) {
  const actions = payload?.response?.actions;
  return Array.isArray(actions) ? actions : [];
}

function friendlyAxiosMessage(err) {
  const data = err?.response?.data;
  if (data && typeof data === "object") {
    const fromEnvelope = extractBotText(data);
    if (fromEnvelope) return fromEnvelope;
    if (typeof data.detail === "string") return data.detail;
  }
  if (err?.code === "ECONNABORTED") {
    return "The request timed out. Please try again.";
  }
  if (!err?.response) {
    return "Unable to reach the server. Check that the API is running and your network connection.";
  }
  return "Something went wrong. Please try again.";
}

function mapTurnToMessage(turn) {
  const role = turn.role === "user" ? "user" : "bot";
  const at = turn.created_at || turn.at;
  return {
    role,
    text: turn.content || "",
    ...(at ? { at } : {}),
  };
}

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const sendingRef = useRef(false);
  const initialHistoryLoaded = useRef(false);

  const clearError = useCallback(() => setError(null), []);

  const refreshSessions = useCallback(async () => {
    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const identity = getClientIdentity();
      const { sessions: list } = await fetchChatSessions({ identity });
      setSessions(list);
      setActiveSessionId((prev) => prev || identity.session_id);
    } catch (err) {
      setSessionsError(friendlyAxiosMessage(err));
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  const loadSession = useCallback(async (sessionId) => {
    const sid = (sessionId || "").trim();
    if (!sid) return;
    setHistoryLoading(true);
    setError(null);
    try {
      const identity = getClientIdentity();
      setSessionId(sid, identity);
      setActiveSessionId(sid);
      const { messages: turns } = await fetchChatSession({ sessionId: sid, identity });
      const loadTime = Date.now();
      setMessages(
        turns.map((t, i) => ({
          ...mapTurnToMessage(t),
          at:
            t.created_at ||
            t.at ||
            new Date(loadTime - (turns.length - 1 - i) * 1000).toISOString(),
        }))
      );
    } catch (err) {
      setError(friendlyAxiosMessage(err));
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (initialHistoryLoaded.current || sessionsLoading || sessions.length === 0) return;
    if (messages.length > 0) {
      initialHistoryLoaded.current = true;
      return;
    }
    try {
      const identity = getClientIdentity();
      const hasCurrent = sessions.some((s) => s.session_id === identity.session_id);
      if (hasCurrent) {
        initialHistoryLoaded.current = true;
        loadSession(identity.session_id);
      }
    } catch {
      /* identity not configured */
    }
  }, [sessions, sessionsLoading, messages.length, loadSession]);

  const startNewSession = useCallback(() => {
    try {
      const identity = getClientIdentity();
      const newId = rotateSessionId(identity);
      setActiveSessionId(newId);
      setMessages([]);
      setError(null);
    } catch (err) {
      setError(err?.message || "Missing CRM identity.");
    }
  }, []);

  const sendMessage = useCallback(
    async (payload) => {
      const text = typeof payload === "string" ? payload : payload?.text || "";
      const file = typeof payload === "object" ? payload?.file : null;
      const trimmed = text.trim();
      if (!trimmed || sendingRef.current) return;
      sendingRef.current = true;

      setError(null);
      const userAt = new Date().toISOString();
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          text: trimmed,
          at: userAt,
          ...(file
            ? { attachment: { name: file.name, size: file.size } }
            : {}),
        },
      ]);
      setLoading(true);

      let identity;
      try {
        identity = getClientIdentity();
      } catch (err) {
        setError(err?.message || "Missing CRM identity.");
        setLoading(false);
        sendingRef.current = false;
        return;
      }

      setActiveSessionId(identity.session_id);

      try {
        let documentText = "";
        if (file) {
          const extracted = await postDocumentExtract({ file, identity });
          documentText = extracted.documentText || "";
        }
        const { data, sessionIdHeader } = await postChat({
          message: trimmed,
          sessionId: identity.session_id,
          documentText,
          identity,
        });

        if (sessionIdHeader) {
          setSessionId(sessionIdHeader, identity);
          setActiveSessionId(sessionIdHeader);
        }

        const topStatus = data?.status;
        const botText = extractBotText(data);
        const display =
          botText ||
          (topStatus === "failed"
            ? "We could not complete that request."
            : "No response message was returned.");

        const actions = extractBotActions(data);
        setMessages((prev) => [
          ...prev,
          {
            role: "bot",
            text: display,
            at: new Date().toISOString(),
            ...(actions.length ? { actions } : {}),
          },
        ]);
        await refreshSessions();
      } catch (err) {
        const msg = friendlyAxiosMessage(err);
        setError(msg);
      } finally {
        sendingRef.current = false;
        setLoading(false);
      }
    },
    [refreshSessions]
  );

  const clearMessages = useCallback(() => {
    startNewSession();
  }, [startNewSession]);

  return {
    messages,
    loading: loading || historyLoading,
    error,
    sendMessage,
    clearError,
    clearMessages,
    sessions,
    sessionsLoading,
    sessionsError,
    activeSessionId,
    loadSession,
    startNewSession,
    refreshSessions,
  };
}
