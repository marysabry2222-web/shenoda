import { useState, useCallback } from 'react';
import type { Message } from '../types';
import { sendMessage } from '../services/api';

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
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
 */
export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    error,
    sendUserMessage,
    sendVoiceMessage,
    addAssistantMessage,
    clearError,
  };
}
