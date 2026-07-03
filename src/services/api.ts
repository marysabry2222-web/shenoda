import axios, { AxiosError } from 'axios';
import type { ChatRequest, ChatResponse, HealthResponse } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
});

// أكواد الأخطاء اللي بنعتبرها "مؤقتة" ومستاهلة إعادة محاولة
// (429 = rate limit من Groq بيتسرّب، 502/503/504 = مشاكل مؤقتة في السيرفر)
const RETRYABLE_STATUS = new Set([429, 502, 503, 504]);

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * بتعيد محاولة الطلب لو فشل بسبب خطأ مؤقت (rate limit / server hiccup)،
 * بانتظار متزايد بين كل محاولة (backoff). ده طبقة حماية إضافية جوه
 * الفرونت إند فوق الـ retry اللي المفروض يكون موجود في الباك إند نفسه.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 2,
  baseDelayMs = 3000
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;

      const status = err instanceof AxiosError ? err.response?.status : undefined;
      const isRetryable = status !== undefined && RETRYABLE_STATUS.has(status);

      if (!isRetryable || attempt === maxRetries) {
        throw err;
      }

      // backoff بسيط: 3s ثم 6s ثم 12s...
      await delay(baseDelayMs * Math.pow(2, attempt));
    }
  }

  throw lastError;
}

export async function sendMessage(message: string): Promise<string> {
  const payload: ChatRequest = { message };

  return withRetry(async () => {
    const response = await api.post<ChatResponse>('/chat', payload, {
      timeout: 90_000, // أطول من الـ default عشان تستحمل retry الباك إند لو حصل
    });
    return response.data.answer;
  });
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

  return withRetry(async () => {
    const response = await api.post('/voice', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 90_000,
    });
    return response.data;
  });
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get<HealthResponse>('/health', { timeout: 5000 });
    return response.data.status === 'ok';
  } catch {
    return false;
  }
}
