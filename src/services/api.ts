import axios from 'axios';
import type { ChatRequest, ChatResponse, HealthResponse } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
});

export async function sendMessage(message: string): Promise<string> {
  const payload: ChatRequest = { message };
  const response = await api.post<ChatResponse>('/chat', payload);
  return response.data.answer;
}

/** Convert text to speech — returns audio URL to play */
export async function textToSpeech(text: string): Promise<string | null> {
  try {
    const response = await api.post('/tts', { text }, { responseType: 'blob' });
    return URL.createObjectURL(response.data as Blob);
  } catch {
    return null; // TTS failure is non-fatal
  }
}

export async function sendVoice(
  audioBlob: Blob
): Promise<{ transcript: string; answer: string }> {
  const formData = new FormData();

  formData.append('audio', audioBlob, 'audio.webm');

  const response = await api.post('/voice', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get<HealthResponse>('/health', { timeout: 5000 });
    return response.data.status === 'ok';
  } catch {
    return false;
  }
}
