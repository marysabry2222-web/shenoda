import { useState, useRef, useCallback } from 'react';

export type CallStatus =
  | 'idle'
  | 'connecting'
  | 'listening'   // المايك فاتح، السيرفر بيسمع طول الوقت
  | 'processing'  // المستخدم سكت، السيرفر بيعالج (STT/LLM) قبل الرد
  | 'speaking'    // صوت رد المساعد بيتشغل
  | 'error';

interface UseCallOptions {
  onTranscript: (text: string) => void;   // اللي المستخدم قاله
  onAnswer: (text: string) => void;        // رد المساعد النصي
}

interface UseCallReturn {
  status: CallStatus;
  startCall: () => void;
  endCall: () => void;
  toggleMic: () => void; // كتم/إلغاء كتم المايك أثناء المكالمة (مش لازم للتشغيل العادي)
  isMicMuted: boolean;
  isCallActive: boolean;
  errorMsg: string | null;
}

const WS_URL =
  (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/^http/, 'ws') +
  '/ws/call';

const CAPTURE_SAMPLE_RATE = 16000; // لازم يطابق CALL_SAMPLE_RATE في routes.py
const PLAYBACK_SAMPLE_RATE = 24000; // نفس sample rate صوت Gemini TTS الراجع
const CAPTURE_CHUNK_SAMPLES = 2048; // ~128ms عند 16kHz قبل ما نبعت للسيرفر

// كود الـ AudioWorklet بيتحمّل كـ Blob module - بيجمع عينات الصوت الخام
// (Float32) لحد ما يوصل لحجم chunk معقول، يحولها Int16 PCM، ويبعتها
// للـ main thread عبر postMessage.
const CAPTURE_WORKLET_CODE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunks = [];
    this._bufferedSamples = 0;
    this._targetSamples = ${CAPTURE_CHUNK_SAMPLES};
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const channelData = input[0];
      this._chunks.push(new Float32Array(channelData));
      this._bufferedSamples += channelData.length;

      if (this._bufferedSamples >= this._targetSamples) {
        const merged = new Float32Array(this._bufferedSamples);
        let offset = 0;
        for (const chunk of this._chunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }

        const int16 = new Int16Array(merged.length);
        for (let i = 0; i < merged.length; i++) {
          const s = Math.max(-1, Math.min(1, merged[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        this.port.postMessage(int16.buffer, [int16.buffer]);
        this._chunks = [];
        this._bufferedSamples = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

/**
 * Manages a WebSocket-based real-time voice call with the assistant.
 *
 * الفكرة: زرار واحد بس (ابدأ/إنهاء مكالمة). المايك بيفضل فاتح طول
 * المكالمة، والسيرفر هو اللي بيكتشف لما المستخدم يتكلم ولما يسكت
 * (VAD)، فمفيش أي زرار تاني للمستخدم يدوسه بين الجمل - مكالمة طبيعية.
 *
 * لو المستخدم بدأ يتكلم والمساعد لسه بيتكلم، السيرفر بيبعت "interrupted"
 * فورًا، وإحنا بنوقف أي صوت شغال محليًا في نفس اللحظة (barge-in).
 */
export function useCall({ onTranscript, onAnswer }: UseCallOptions): UseCallReturn {
  const [status, setStatus] = useState<CallStatus>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isMicMuted, setIsMicMuted] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureContextRef = useRef<AudioContext | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | null>(null);
  const isMicMutedRef = useRef(false); // نسخة sync عشان نقراها جوه callback الـ worklet

  const playbackContextRef = useRef<AudioContext | null>(null);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const nextStartTimeRef = useRef(0);

  /** بتوقف أي صوت شغال أو متجدول للمساعد فورًا (استخدامها الأساسي: barge-in) */
  const stopPlayback = useCallback(() => {
    for (const source of activeSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // ممكن يكون خلص أصلاً - نتجاهل
      }
    }
    activeSourcesRef.current = [];

    if (playbackContextRef.current) {
      nextStartTimeRef.current = playbackContextRef.current.currentTime;
    }
  }, []);

  /** بتاخد شريحة صوت PCM16 خام (24kHz) وتضيفها لطابور التشغيل المتصل */
  const scheduleAudioChunk = useCallback((buffer: ArrayBuffer) => {
    const playbackContext = playbackContextRef.current;
    if (!playbackContext) return;

    // دفاعي: بعض المتصفحات بتعلّق (suspend) الـ AudioContext تلقائيًا
    // لو التاب راح للخلفية أو الشاشة قفلت لحظة - نتأكد إنه شغال قبل
    // ما نجدول أي صوت جديد.
    if (playbackContext.state === 'suspended') {
      playbackContext.resume().catch(() => {});
    }

    const int16 = new Int16Array(buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / (int16[i] < 0 ? 0x8000 : 0x7fff);
    }

    const audioBuffer = playbackContext.createBuffer(
      1,
      float32.length,
      PLAYBACK_SAMPLE_RATE
    );
    audioBuffer.copyToChannel(float32, 0);

    const source = playbackContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playbackContext.destination);

    const startTime = Math.max(playbackContext.currentTime, nextStartTimeRef.current);
    source.start(startTime);
    nextStartTimeRef.current = startTime + audioBuffer.duration;

    activeSourcesRef.current.push(source);
    source.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source);
    };
  }, []);

  /** بتفتح المايك وتبدأ بث الصوت الخام للسيرفر بشكل مستمر */
  const startMicStreaming = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const captureContext = new AudioContext({ sampleRate: CAPTURE_SAMPLE_RATE });
    captureContextRef.current = captureContext;

    const workletBlob = new Blob([CAPTURE_WORKLET_CODE], {
      type: 'application/javascript',
    });
    const workletUrl = URL.createObjectURL(workletBlob);
    await captureContext.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);

    const micSource = captureContext.createMediaStreamSource(stream);
    const captureNode = new AudioWorkletNode(captureContext, 'capture-processor');
    captureNodeRef.current = captureNode;

    captureNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      if (isMicMutedRef.current) return;
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(event.data);
      }
    };

    // لازم نوصّل الـ node لمخرج (حتى لو صامت) عشان المتصفح يستمر
    // ينادي process() بانتظام
    const silentGain = captureContext.createGain();
    silentGain.gain.value = 0;

    micSource.connect(captureNode);
    captureNode.connect(silentGain);
    silentGain.connect(captureContext.destination);
  }, []);

  const startCall = useCallback(async () => {
    setErrorMsg(null);
    setStatus('connecting');
    isMicMutedRef.current = false;
    setIsMicMuted(false);

    try {
      const ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = async () => {
        try {
          playbackContextRef.current = new AudioContext({
            sampleRate: PLAYBACK_SAMPLE_RATE,
          });
          // بعض المتصفحات (خصوصًا الموبايل) بتفضل الـ AudioContext
          // "suspended" لو مش استدعيناه resume() بشكل صريح، حتى لو
          // إنشاءه حصل بعد ضغطة المستخدم على الزرار - فبيتشغل الصوت
          // برمجيًا بدون أي صوت فعلي يخرج ومن غير أي error يبان.
          await playbackContextRef.current.resume();
          nextStartTimeRef.current = 0;

          await startMicStreaming();
          setStatus('listening');
        } catch {
          setErrorMsg('لا يمكن الوصول إلى الميكروفون.');
          setStatus('error');
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        if (typeof event.data === 'string') {
          const msg = JSON.parse(event.data);

          switch (msg.type) {
            case 'interrupted':
              // المستخدم بدأ يتكلم - أوقفي أي صوت شغال فورًا (barge-in)
              stopPlayback();
              setStatus('listening');
              break;

            case 'processing':
              setStatus('processing');
              break;

            case 'transcript':
              onTranscript(msg.text as string);
              break;

            case 'answer_text':
              onAnswer(msg.text as string);
              break;

            case 'answer_audio_start':
              setStatus('speaking');
              break;

            case 'answer_audio_end':
              setStatus('listening');
              break;

            case 'error':
              setErrorMsg(msg.message as string);
              setStatus('error');
              break;
          }
        } else {
          // شريحة صوت PCM خام (ArrayBuffer) من رد المساعد
          scheduleAudioChunk(event.data as ArrayBuffer);
        }
      };

      ws.onerror = () => {
        setErrorMsg('تعذر الاتصال بالخادم.');
        setStatus('error');
      };

      ws.onclose = () => {
        setStatus((prev) => (prev === 'idle' ? prev : 'idle'));
      };
    } catch {
      setErrorMsg('تعذر بدء المكالمة.');
      setStatus('error');
    }
  }, [onTranscript, onAnswer, scheduleAudioChunk, startMicStreaming, stopPlayback]);

  const endCall = useCallback(() => {
    stopPlayback();

    captureNodeRef.current?.disconnect();
    captureNodeRef.current = null;

    captureContextRef.current?.close();
    captureContextRef.current = null;

    playbackContextRef.current?.close();
    playbackContextRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    wsRef.current?.close();
    wsRef.current = null;

    setStatus('idle');
    setErrorMsg(null);
    setIsMicMuted(false);
    isMicMutedRef.current = false;
  }, [stopPlayback]);

  /** كتم/إلغاء كتم المايك - المكالمة والاستماع فاضلين شغالين، بس السيرفر
   *  مش هيستقبل صوت المستخدم لحد ما يلغي الكتم */
  const toggleMic = useCallback(() => {
    isMicMutedRef.current = !isMicMutedRef.current;
    setIsMicMuted(isMicMutedRef.current);
  }, []);

  return {
    status,
    startCall,
    endCall,
    toggleMic,
    isMicMuted,
    isCallActive: status !== 'idle',
    errorMsg,
  };
}
