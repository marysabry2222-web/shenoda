import { useState, useRef, useCallback } from 'react';

export type CallStatus =
  | 'idle'
  | 'connecting'
  | 'listening'   // mic is open, user is speaking
  | 'processing'  // audio sent, waiting for server
  | 'speaking'    // assistant is playing TTS audio
  | 'error';

interface UseCallOptions {
  onTranscript: (text: string) => void;   // what the user said
  onAnswer: (text: string) => void;        // assistant's answer text
}

interface UseCallReturn {
  status: CallStatus;
  startCall: () => void;
  endCall: () => void;
  toggleMic: () => void;
  isCallActive: boolean;
  errorMsg: string | null;
}

const WS_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000')
  .replace(/^http/, 'ws') + '/call';

/**
 * Manages a WebSocket-based real-time voice call with the assistant.
 *
 * Flow:
 *   startCall → WebSocket connects → mic opens → user speaks
 *   → toggleMic (or auto silence) → audio sent → Whisper → RAG → TTS
 *   → audio streamed back → plays automatically
 *   → mic re-opens for next turn
 */
export function useCall({ onTranscript, onAnswer }: UseCallOptions): UseCallReturn {
  const [status, setStatus] = useState<CallStatus>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const audioQueueRef = useRef<BlobPart[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  /** Play all queued MP3 chunks by concatenating them into a Blob */
  const playQueuedAudio = useCallback(async () => {
    if (audioQueueRef.current.length === 0) return;

    const blob = new Blob(audioQueueRef.current, { type: 'audio/mpeg' });
    audioQueueRef.current = [];

    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    setStatus('speaking');

    audio.onended = () => {
      URL.revokeObjectURL(url);
      // After assistant speaks, reopen mic for next user turn
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        startMic();
      }
    };

    audio.onerror = () => {
      URL.revokeObjectURL(url);
      setStatus('listening');
    };

    await audio.play();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Open the mic and start recording */
  const startMic = useCallback(async () => {
    try {
      const stream =
        streamRef.current ||
        (await navigator.mediaDevices.getUserMedia({ audio: true }));
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        // Convert recorded chunks to base64 and send via WebSocket
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        chunksRef.current = [];

        const arrayBuffer = await blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);

        // Send in one shot (for voice turn sizes this is fine)
        const b64 = btoa(String.fromCharCode(...bytes));
        wsRef.current?.send(JSON.stringify({ type: 'audio_chunk', data: b64 }));
        wsRef.current?.send(JSON.stringify({ type: 'end_of_speech' }));
        setStatus('processing');
      };

      recorderRef.current = recorder;
      recorder.start();
      setStatus('listening');
    } catch {
      setErrorMsg('لا يمكن الوصول إلى الميكروفون.');
      setStatus('error');
    }
  }, []);

  /** Stop the current mic recording (triggers sending audio to server) */
  const stopMic = useCallback(() => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop();
    }
  }, []);

  /** Toggle mic on/off during a call */
  const toggleMic = useCallback(() => {
    if (status === 'listening') {
      stopMic();
    } else if (status === 'idle' && wsRef.current?.readyState === WebSocket.OPEN) {
      startMic();
    }
  }, [status, stopMic, startMic]);

  /** Open WebSocket and start the call */
  const startCall = useCallback(async () => {
    setErrorMsg(null);
    setStatus('connecting');
    audioQueueRef.current = [];

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      startMic();
    };

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data as string);

      switch (msg.type) {
        case 'transcript':
          onTranscript(msg.text as string);
          break;

        case 'answer_done':
          onAnswer(msg.text as string);
          break;

        case 'audio_chunk': {
          // Decode base64 chunk and queue it
          const binary = atob(msg.data as string);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          audioQueueRef.current.push(bytes);
          break;
        }

        case 'audio_done':
          await playQueuedAudio();
          break;

        case 'error':
          setErrorMsg(msg.message as string);
          setStatus('error');
          break;

        case 'pong':
          break;
      }
    };

    ws.onerror = () => {
      setErrorMsg('تعذر الاتصال بالخادم.');
      setStatus('error');
    };

    ws.onclose = () => {
      if (status !== 'idle') setStatus('idle');
    };
  }, [startMic, onTranscript, onAnswer, playQueuedAudio, status]);

  /** End the call and clean up all resources */
  const endCall = useCallback(() => {
    stopMic();

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    wsRef.current?.close();
    wsRef.current = null;

    audioContextRef.current?.close();
    audioContextRef.current = null;

    audioQueueRef.current = [];
    setStatus('idle');
    setErrorMsg(null);
  }, [stopMic]);

  return {
    status,
    startCall,
    endCall,
    toggleMic,
    isCallActive: status !== 'idle',
    errorMsg,
  };
}
