// ─── Message Types ────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}

// ─── API Types ────────────────────────────────────────────────────────────────

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  answer: string;
}

export interface HealthResponse {
  status: string;
}
