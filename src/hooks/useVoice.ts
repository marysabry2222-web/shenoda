import { useState, useRef, useCallback } from 'react';
// import { sendVoice } from '../services/api';

// =========================
// Web Speech API type declarations
// (not included in default TS DOM lib)
// =========================
interface SpeechRecognitionResultItem {
  transcript: string;
}
interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionResultItem;
}
interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}
interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}
interface SpeechRecognitionConstructor {
  new (): SpeechRecognition;
}
declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

interface UseVoiceReturn {
  isRecording: boolean;
  isProcessing: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  error: string | null;
}

export function useVoice(onAnswer: (text: string) => void): UseVoiceReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptBufferRef = useRef<string>('');   // بيجمع كل النص لحد الإيقاف
  const isStoppingRef = useRef<boolean>(false);       // لتفرقة الإيقاف اليدوي عن التوقف التلقائي

  const startRecording = useCallback(() => {
    setError(null);
    transcriptBufferRef.current = '';
    isStoppingRef.current = false;

    const SpeechRecognitionImpl =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionImpl) {
      setError('المتصفح لا يدعم التعرف على الصوت');
      return;
    }

    const recognition = new SpeechRecognitionImpl();
    recognition.lang = 'ar-EG';
    recognition.continuous = true;      // يفضل شغال لحد ما توقفيه
    recognition.interimResults = true;  // عشان يفضل يبعت partial results ومايقفلش لوحده

    recognition.onstart = () => setIsRecording(true);

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalChunk = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalChunk += result[0].transcript;
        }
      }
      if (finalChunk) {
        // transcriptBufferRef.current += finalChunk + ' ';
        transcriptBufferRef.current = finalChunk;
      }
    };

    recognition.onerror = () => {
      setError('تعذر التعرف على الصوت');
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);

      // بعض المتصفحات بتوقف الـ recognition لوحدها بعد فترة سكوت طويلة
      // فبنعيد تشغيلها تلقائي طالما المستخدم لسه مدوسش زرار الإيقاف
      if (!isStoppingRef.current) {
        try {
          recognition.start();
          return;
        } catch {
          // تجاهل، هيكمل تحت لو فشلت إعادة التشغيل
        }
      }

      // هنا يبقى المستخدم دوس إيقاف فعلاً
      const text = transcriptBufferRef.current.trim();
      transcriptBufferRef.current = '';
      isStoppingRef.current = false;

      if (text) {
        setIsProcessing(true);
        Promise.resolve(onAnswer(text)).finally(() => {
          setIsProcessing(false);
        });
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [onAnswer]);

  const stopRecording = useCallback(() => {
    isStoppingRef.current = true;
    recognitionRef.current?.stop();
  }, []);

  return { isRecording, isProcessing, startRecording, stopRecording, error };
}
