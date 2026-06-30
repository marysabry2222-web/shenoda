import { useState, useCallback } from 'react';
import type { Message } from '../types';
import { sendMessage } from '../services/api';

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  isSpeaking: boolean;
  error: string | null;
  sendUserMessage: (text: string) => Promise<void>;
  sendVoiceMessage: (text: string) => Promise<void>;
  addAssistantMessage: (text: string) => void;
  clearError: () => void;
}

/**
 * Manages the chat conversation state.
 * No TTS happens here at all — both typed messages and voice-note
 * messages get a TEXT-ONLY reply. TTS only happens inside the
 * real-time call flow (useCall.ts), which is full speech-to-speech.
 *
 * isSpeaking is always false here; it exists only to satisfy
 * components (like ChatWindow) that accept a shared "isSpeaking"
 * prop also used by the call flow.
 */
export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isSpeaking = false;

  const createMessage = (role: Message['role'], content: string): Message => ({
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role,
    content,
    timestamp: new Date(),
  });

  const addAssistantMessage = useCallback((text: string) => {
    setMessages((prev) => [...prev, createMessage('assistant', text)]);
  }, []);

  const handleError = (err: unknown) => {
    const isNetworkError =
      err instanceof Error &&
      (err.message.includes('Network Error') || err.message.includes('ECONNREFUSED'));
    setError(
      isNetworkError
        ? 'تعذر الاتصال بالخادم.'
        : 'حدث خطأ أثناء المعالجة. يرجى المحاولة مرة أخرى.'
    );
  };

  const send = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;
    setMessages((prev) => [...prev, createMessage('user', text)]);
    setIsLoading(true);
    setError(null);
    try {
      const answer = await sendMessage(text);
      setMessages((prev) => [...prev, createMessage('assistant', answer)]);
    } catch (err: unknown) {
      handleError(err);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const sendUserMessage = useCallback((text: string) => send(text), [send]);
  const sendVoiceMessage = useCallback((text: string) => send(text), [send]);
  const clearError = useCallback(() => setError(null), []);

  return {
    messages,
    isLoading,
    isSpeaking,
    error,
    sendUserMessage,
    sendVoiceMessage,
    addAssistantMessage,
    clearError,
  };
}
