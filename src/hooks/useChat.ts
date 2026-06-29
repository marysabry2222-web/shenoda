import { useState, useCallback } from 'react';
import type { Message } from '../types';
import { sendMessage } from '../services/api';

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  sendUserMessage: (text: string) => Promise<void>;
  addAssistantMessage: (text: string) => void;
  clearError: () => void;
}

/**
 * Manages the chat conversation state.
 * Conversation lives in React state only — no persistence.
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

  const sendUserMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    // Add user message immediately
    const userMsg = createMessage('user', text);
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    try {
      const answer = await sendMessage(text);
      setMessages((prev) => [...prev, createMessage('assistant', answer)]);
    } catch (err: unknown) {
      const isNetworkError =
        err instanceof Error &&
        (err.message.includes('Network Error') || err.message.includes('ECONNREFUSED'));

      const errorMsg = isNetworkError
        ? 'تعذر الاتصال بالخادم.'
        : 'حدث خطأ أثناء المعالجة. يرجى المحاولة مرة أخرى.';

      setError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const clearError = useCallback(() => setError(null), []);

  return { messages, isLoading, error, sendUserMessage, addAssistantMessage, clearError };
}
