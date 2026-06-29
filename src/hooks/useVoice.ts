import { useState, useRef, useCallback } from 'react';
import { sendVoice } from '../services/api';

interface UseVoiceReturn {
  isRecording: boolean;
  isProcessing: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  error: string | null;
}

/**
 * Manages microphone recording and voice API calls.
 * Returns the answer text via onAnswer callback.
 */
export function useVoice(onAnswer: (text: string) => void): UseVoiceReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        // Stop all tracks to release the microphone
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setIsProcessing(true);

        try {
          const { audioUrl, answerText } = await sendVoice(blob);

          // Play the spoken response
          const audio = new Audio(audioUrl);
          audio.play();

          // Add the text answer to the chat
          if (answerText) onAnswer(answerText);
        } catch {
          setError('تعذر معالجة الصوت. يرجى المحاولة مرة أخرى.');
        } finally {
          setIsProcessing(false);
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch {
      setError('لا يمكن الوصول إلى الميكروفون. يرجى السماح بالإذن.');
    }
  }, [onAnswer]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  return { isRecording, isProcessing, startRecording, stopRecording, error };
}
