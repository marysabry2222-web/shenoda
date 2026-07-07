import { useState, useRef, useCallback } from 'react';

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
interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}
interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
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

export type CallStatus =
  | 'idle'
  | 'connecting'
  | 'listening'   // المايك فاتح، بننتظر المستخدم يتكلم
  | 'processing'  // المستخدم خلص كلامه، السيرفر بيعالج (LLM) قبل الرد
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

const PLAYBACK_SAMPLE_RATE = 24000; // نفس sample rate صوت Gemini TTS الراجع

// أخطاء مؤقتة/غير قاتلة بنتجاهلها ونعيد المحاولة من غير ما نقفل الجلسة
// (نفس القائمة المستخدمة في useVoice.ts)
const RECOVERABLE_ERRORS = new Set(['no-speech', 'audio-capture', 'network']);

// محرك التعرف على الصوت (خصوصًا مع ar-EG) بيعمل أحيانًا "resegmentation":
// بيرجع يفسر جزء من الكلام اللي فات ويطلعه كـ isFinal تاني في index جديد،
// فبيتكرر جزء من النص حتى لو الـ index نفسه لم يتكرر.
// الدالة دي بتشيل أي تداخل (overlap) بين آخر كلمات في الـ buffer وأول كلمات القطعة الجديدة.
// (نفس الدالة المستخدمة في useVoice.ts)
function stripOverlap(bufferText: string, newChunk: string): string {
  const bufferWords = bufferText.trim().split(/\s+/).filter(Boolean);
  const newWords = newChunk.trim().split(/\s+/).filter(Boolean);

  if (bufferWords.length === 0 || newWords.length === 0) return newChunk;

  const maxOverlap = Math.min(bufferWords.length, newWords.length, 12);
  let overlapLen = 0;

  for (let len = maxOverlap; len > 0; len--) {
    const bufferSuffix = bufferWords.slice(-len).join(' ');
    const newPrefix = newWords.slice(0, len).join(' ');
    if (bufferSuffix === newPrefix) {
      overlapLen = len;
      break;
    }
  }

  return newWords.slice(overlapLen).join(' ');
}

/**
 * Manages a WebSocket-based real-time voice call with the assistant.
 *
 * الفكرة: زرار واحد بس (ابدأ/إنهاء مكالمة) + زرار toggleMic لكل "دور
 * كلام". التعرف على الصوت بيحصل بالكامل في المتصفح عن طريق Web Speech
 * API (نفس آلية useVoice.ts بالظبط) - السيرفر بقى بياخد نص جاهز مش
 * صوت خام، فمفيش أي بث PCM مستمر ولا AudioWorklet للمايك خالص.
 *
 * لو المستخدم بدأ يتكلم (toggleMic -> unmute) والمساعد لسه بيتكلم،
 * بنوقف صوته فورًا محليًا (barge-in) ونبلغ السيرفر بـ mic_unmuted.
 */
export function useCall({ onTranscript, onAnswer }: UseCallOptions): UseCallReturn {
  const [status, setStatus] = useState<CallStatus>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isMicMuted, setIsMicMuted] = useState(false); // المايك شغال تلقائيًا من لحظة الاتصال

  const wsRef = useRef<WebSocket | null>(null);

  const playbackContextRef = useRef<AudioContext | null>(null);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const nextStartTimeRef = useRef(0);

  // عنصر <audio> مخصص لتشغيل روابط جاهزة (زي لحن ترحيب البابا من
  // Cloudinary) - منفصل تمامًا عن مسار الـ PCM streaming (scheduleAudioChunk)
  // لأن ده ملف MP3 كامل بيتشغل عبر Audio API عادي، مش شرائح PCM خام
  const urlAudioRef = useRef<HTMLAudioElement | null>(null);

  // =========================
  // Web Speech API refs (زي useVoice.ts)
  // =========================
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptBufferRef = useRef<string>('');
  const lastFinalIndexRef = useRef<number>(-1);
  const isStoppingRef = useRef<boolean>(false); // إيقاف يدوي (toggleMic) مقابل توقف تلقائي من المتصفح
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearRestartTimeout = () => {
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
  };

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

    // وقف أي لحن/صوت رابط شغال كمان (زي لحن ترحيب البابا)
    if (urlAudioRef.current) {
      urlAudioRef.current.pause();
      urlAudioRef.current.currentTime = 0;
      urlAudioRef.current = null;
    }
  }, []);

  /** بتاخد شريحة صوت PCM16 خام (24kHz) وتضيفها لطابور التشغيل المتصل */
  const scheduleAudioChunk = useCallback((buffer: ArrayBuffer) => {
    const playbackContext = playbackContextRef.current;
    if (!playbackContext) return;

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

  /** بتشغّل رابط صوت جاهز (MP3 من Cloudinary مثلاً) بدل شرائح PCM -
   *  مسار منفصل تمامًا عن scheduleAudioChunk */
  const playAudioUrl = useCallback((url: string) => {
    if (urlAudioRef.current) {
      urlAudioRef.current.pause();
      urlAudioRef.current.currentTime = 0;
    }

    const audio = new Audio(url);
    urlAudioRef.current = audio;

    setStatus('speaking');

    audio.onended = () => {
      if (urlAudioRef.current === audio) {
        urlAudioRef.current = null;
      }
      setStatus('listening');
    };

    audio.onerror = () => {
      console.error('Failed to play audio URL:', url);
      if (urlAudioRef.current === audio) {
        urlAudioRef.current = null;
      }
      setStatus('listening');
    };

    audio.play().catch((err) => {
      console.error('audio.play() failed:', err);
      setStatus('listening');
    });
  }, []);

  /** بتبعت النص النهائي اللي اتجمع في transcriptBufferRef للسيرفر كـ user_utterance */
  const sendBufferedUtterance = useCallback(() => {
    const text = transcriptBufferRef.current.trim();
    transcriptBufferRef.current = '';

    if (text && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'user_utterance', text }));
    }
  }, []);

  /** بتبدأ جلسة Web Speech API جديدة - بتتصرف بالظبط زي useVoice.ts،
   *  بس بدل ما تنادي onAnswer محليًا لما تخلص، بتبعت النص للسيرفر */
  const startRecognition = useCallback(() => {
    const SpeechRecognitionImpl =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionImpl) {
      setErrorMsg('المتصفح لا يدعم التعرف على الصوت');
      setStatus('error');
      return;
    }

    transcriptBufferRef.current = '';
    lastFinalIndexRef.current = -1;
    isStoppingRef.current = false;
    clearRestartTimeout();

    const recognition = new SpeechRecognitionImpl();
    recognition.lang = 'ar-EG';
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const startIndex = Math.max(event.resultIndex, lastFinalIndexRef.current + 1);

      let finalChunk = '';
      for (let i = startIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalChunk += result[0].transcript;
          lastFinalIndexRef.current = i;
        }
      }
      if (finalChunk) {
        const deduped = stripOverlap(transcriptBufferRef.current, finalChunk);
        if (deduped) {
          transcriptBufferRef.current += deduped + ' ';
        }
      }
    };

    let fatalError = false;

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (RECOVERABLE_ERRORS.has(event.error)) {
        // مش قاتل - onend هيتنفذ بعده وهيعيد المحاولة عادي طالما المستخدم
        // لسه مادوسش "خلصت الكلام"
        return;
      }
      fatalError = true;
      setErrorMsg('تعذر التعرف على الصوت');
    };

    recognition.onend = () => {
      // لو المستخدم لسه مادوسش toggleMic (ماقالش خلص كلامه)، وملقيناش
      // خطأ قاتل: إعادة التشغيل تلقائيًا (المتصفح بيوقف الجلسة لوحده
      // أحيانًا بعد فترة سكوت حتى لو المستخدم لسه بيتكلم)
      if (!isStoppingRef.current && !fatalError) {
        clearRestartTimeout();
        restartTimeoutRef.current = setTimeout(() => {
          try {
            lastFinalIndexRef.current = -1;
            recognition.start();
          } catch {
            sendBufferedUtterance();
          }
        }, 250);
        return;
      }

      sendBufferedUtterance();
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [sendBufferedUtterance]);

  const stopRecognition = useCallback(() => {
    isStoppingRef.current = true;
    clearRestartTimeout();
    recognitionRef.current?.stop();
  }, []);

  const startCall = useCallback(async () => {
    if (
      wsRef.current !== null ||
      status === 'connecting' ||
      status === 'listening' ||
      status === 'processing' ||
      status === 'speaking'
    ) {
      console.warn('Call already active or connecting — ignoring duplicate startCall');
      return;
    }

    setErrorMsg(null);
    setStatus('connecting');
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
          await playbackContextRef.current.resume();
          nextStartTimeRef.current = 0;

          setStatus('listening');
          // المايك بيبدأ يسجل تلقائيًا من هنا - المستخدم مش محتاج يدوس
          // أي زرار عشان يتكلم، بالظبط زي ما كان بيحصل مع الـ PCM streaming قبل كده
          startRecognition();
        } catch {
          setErrorMsg('حصل خطأ أثناء بدء المكالمة.');
          setStatus('error');
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        if (typeof event.data === 'string') {
          const msg = JSON.parse(event.data);

          switch (msg.type) {
            case 'interrupted':
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
              if (!urlAudioRef.current) {
                setStatus('listening');
              }
              break;

            case 'play_url':
              playAudioUrl(msg.url as string);
              break;

            case 'error':
              setErrorMsg(msg.message as string);
              setStatus('error');
              break;
          }
        } else {
          console.log('Audio', event.data.byteLength);
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
  }, [status, onTranscript, onAnswer, scheduleAudioChunk, stopPlayback, playAudioUrl, startRecognition]);

  const endCall = useCallback(() => {
    stopPlayback();
    stopRecognition();
    recognitionRef.current = null;

    playbackContextRef.current?.close();
    playbackContextRef.current = null;

    wsRef.current?.close();
    wsRef.current = null;

    setStatus('idle');
    setErrorMsg(null);
    setIsMicMuted(true);
  }, [stopPlayback, stopRecognition]);

  /** زرار واحد بيتحكم في دور الكلام - زي زرار تسجيل/إرسال الفويس نوت:
   *  - بيدوس يبدأ يتكلم (unmute): لو المساعد كان بيتكلم لسه، ده يعتبر
   *    مقاطعة (barge-in) - بنوقف صوته فورًا محليًا وبنبلغ السيرفر،
   *    وبنبدأ جلسة Web Speech API جديدة.
   *  - بيدوس تاني لما يخلص كلامه (mute): بنوقف الجلسة، وبمجرد ما
   *    onend يشتغل بنبعت النص النهائي للسيرفر كـ user_utterance. */
  const toggleMic = useCallback(() => {
    const goingToMuted = !isMicMuted;
    setIsMicMuted(goingToMuted);

    if (goingToMuted) {
      // خلص كلامه - أوقفي الجلسة، والنص هيتبعت لوحده من داخل onend
      stopRecognition();
    } else {
      // بدأ يتكلم من جديد
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        stopPlayback();
        setStatus('listening');
        wsRef.current.send(JSON.stringify({ type: 'mic_unmuted' }));
      }
      startRecognition();
    }
  }, [isMicMuted, stopPlayback, stopRecognition, startRecognition]);

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
