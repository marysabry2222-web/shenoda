import { useState, useCallback } from 'react';
import type { Message } from '../types';
import { sendMessage, textToSpeech } from '../services/api';

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  isSpeaking: boolean;
  error: string | null;
  sendUserMessage: (text: string) => Promise<void>;
  addAssistantMessage: (text: string) => void;
  clearError: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createMessage = (role: Message['role'], content: string): Message => ({
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role,
    content,
    timestamp: new Date(),
  });

  /** Play TTS audio and set speaking state */
  const playTTS = useCallback(async (text: string) => {
    const audioUrl = await textToSpeech(text);
    if (!audioUrl) return; // TTS failed silently

    const audio = new Audio(audioUrl);
    setIsSpeaking(true);

    audio.onended = () => {
      setIsSpeaking(false);
      URL.revokeObjectURL(audioUrl); // free memory
    };
    audio.onerror = () => {
      setIsSpeaking(false);
      URL.revokeObjectURL(audioUrl);
    };

    await audio.play();
  }, []);

  const addAssistantMessage = useCallback((text: string) => {
    setMessages((prev) => [...prev, createMessage('assistant', text)]);
    playTTS(text); // auto-play TTS
  }, [playTTS]);

  const sendUserMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    setMessages((prev) => [...prev, createMessage('user', text)]);
    setIsLoading(true);
    setError(null);

    try {
      const answer = await sendMessage(text);
      setMessages((prev) => [...prev, createMessage('assistant', answer)]);
      playTTS(answer); // auto-play TTS after chat response
    } catch (err: unknown) {
      const isNetworkError =
        err instanceof Error &&
        (err.message.includes('Network Error') || err.message.includes('ECONNREFUSED'));
      setError(isNetworkError ? 'تعذر الاتصال بالخادم.' : 'حدث خطأ أثناء المعالجة. يرجى المحاولة مرة أخرى.');
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, playTTS]);

  const clearError = useCallback(() => setError(null), []);

  return { messages, isLoading, isSpeaking, error, sendUserMessage, addAssistantMessage, clearError };
}
