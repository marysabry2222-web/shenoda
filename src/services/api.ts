import axios from 'axios';
import type { ChatRequest, ChatResponse, HealthResponse } from '../types';

// Base API URL from environment variable
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000, // 60s — RAG + LLM can be slow
});

/**
 * Send a text message and receive an AI answer.
 */
export async function sendMessage(message: string): Promise<string> {
  const payload: ChatRequest = { message };
  const response = await api.post<ChatResponse>('/chat', payload);
  return response.data.answer;
}

/**
 * Send recorded audio blob and receive spoken audio + transcript.
 * Returns the audio blob to play, and optionally the answer text from header.
 */
export async function sendVoice(
  audioBlob: Blob
): Promise<{ audioUrl: string; answerText: string }> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'audio.webm');

  const response = await api.post('/voice', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  });

  const audioUrl = URL.createObjectURL(response.data as Blob);
  // Answer text is sent in the response header (truncated to 500 chars)
  const answerText = decodeURIComponent(
    (response.headers['x-answer-text'] as string) || ''
  );

  return { audioUrl, answerText };
}

/**
 * Check if the backend is online.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get<HealthResponse>('/health', { timeout: 5000 });
    return response.data.status === 'ok';
  } catch {
    return false;
  }
}
