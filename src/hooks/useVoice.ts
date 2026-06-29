import { useState, useRef, useCallback } from 'react';
// import { sendVoice } from '../services/api';


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

  const startRecording = useCallback(() => {
    setError(null);
    const SpeechRecognition =
      window.SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError('المتصفح لا يدعم التعرف على الصوت');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'ar-EG';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => setIsRecording(true);

    recognition.onresult = async (event) => {
      const transcript = event.results[0][0].transcript;
      setIsProcessing(true);
      await onAnswer(transcript); // sends to chat
      setIsProcessing(false);
    };

    recognition.onerror = () => {
      setError('تعذر التعرف على الصوت');
      setIsRecording(false);
    };

    recognition.onend = () => setIsRecording(false);

    recognitionRef.current = recognition;
    recognition.start();
  }, [onAnswer]);

  const stopRecording = useCallback(() => {
    recognitionRef.current?.stop();
    setIsRecording(false);
  }, []);

  return { isRecording, isProcessing, startRecording, stopRecording, error };
}
