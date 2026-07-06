import { useCallback } from 'react';
import { FaMicrophone, FaMicrophoneSlash, FaPhoneSlash } from 'react-icons/fa';
import { MdGraphicEq } from 'react-icons/md';
import { AssistantAvatar } from './AssistantAvatar';
import { useCall, type CallStatus } from '../hooks/useCall';

interface CallModalProps {
  onClose: () => void;
  onTranscript: (text: string) => void;
  onAnswer: (text: string) => void;
}

/** Human-readable Arabic status label */
function statusLabel(status: CallStatus): string {
  switch (status) {
    case 'connecting':  return 'جاري الاتصال...';
    case 'listening':   return 'تحدث الآن...';
    case 'processing':  return 'شنودة بيفكر...';
    case 'speaking':    return 'شنودة بيتكلم...';
    case 'error':       return 'حدث خطأ';
    default:            return '';
  }
}

/** Colour accent for the mic / status ring */
function ringColor(status: CallStatus): string {
  switch (status) {
    case 'listening':  return 'ring-red-400 shadow-red-300';
    case 'speaking':   return 'ring-gold-400 shadow-gold-200';
    case 'processing': return 'ring-church-400 shadow-church-200';
    default:           return 'ring-church-300 shadow-church-100';
  }
}

/**
 * Full-screen overlay for the real-time voice call.
 *
 * Layout:
 *   - Avatar + animated equalizer when assistant is speaking
 *   - Status label
 *   - Transcript / answer preview
 *   - Mic mute toggle + hang-up buttons
 */
export function CallModal({ onClose, onTranscript, onAnswer }: CallModalProps) {
  const handleAnswer = useCallback(
    (text: string) => {
      onAnswer(text);
    },
    [onAnswer]
  );

  const { status, endCall, toggleMic, isMicMuted, errorMsg } = useCall({
    onTranscript,
    onAnswer: handleAnswer,
  });

  const handleHangUp = () => {
    endCall();
    onClose();
  };

  // Start the call as soon as the modal mounts (called by parent)
  // The parent calls startCall via the button, we expose this via ref pattern below.
  // Simpler: auto-start on mount
  const isActive = status !== 'idle' && status !== 'error';
  const isListening = status === 'listening';
  const isSpeaking = status === 'speaking';

  return (
    <div className="fixed inset-0 z-50 bg-church-900/95 backdrop-blur-md flex flex-col items-center justify-between py-16 px-8 animate-fade-in">

      {/* Top: subtle cross decoration */}
      <div className="text-gold-500/30 text-4xl select-none">✝</div>

      {/* Center: avatar + waveform */}
      <div className="flex flex-col items-center gap-6">
        {/* Avatar with animated ring */}
        <div
          className={`
            rounded-full p-1 ring-4 transition-all duration-500 shadow-lg
            ${ringColor(status)}
          `}
        >
          <AssistantAvatar size="lg" />
        </div>

        {/* Animated equalizer bars when speaking */}
        {isSpeaking && (
          <div className="flex items-end gap-1 h-8">
            {[1, 2, 3, 4, 5].map((i) => (
              <span
                key={i}
                className="w-1.5 bg-gold-400 rounded-full animate-bounce-dot"
                style={{
                  height: `${12 + i * 4}px`,
                  animationDelay: `${i * 0.12}s`,
                }}
              />
            ))}
          </div>
        )}

        {/* Microphone pulse when listening */}
        {isListening && (
          <div className="relative flex items-center justify-center">
            <span className="absolute w-16 h-16 rounded-full bg-red-500/20 animate-ping" />
            <MdGraphicEq className="text-red-400 text-3xl relative z-10" />
          </div>
        )}

        {/* Status label */}
        <p className="text-gold-300 font-arabic text-lg font-medium tracking-wide">
          {statusLabel(status)}
        </p>

        {/* Error message */}
        {errorMsg && (
          <p className="text-red-400 font-arabic text-sm text-center max-w-xs">
            {errorMsg}
          </p>
        )}

        {/* Assistant name */}
        <p className="text-church-400 font-arabic text-sm">شنودة</p>
      </div>

      {/* Bottom: call controls */}
      <div className="flex items-center gap-8">
        {/* Mute / unmute mic */}
        <button
          onClick={toggleMic}
          disabled={!isActive}
          title={isMicMuted ? 'تشغيل الميكروفون' : 'إيقاف الميكروفون مؤقتاً'}
          className={`
            w-14 h-14 rounded-full flex items-center justify-center transition-all shadow-lg
            ${!isMicMuted
              ? 'bg-white/10 text-white hover:bg-white/20'
              : 'bg-white/5 text-church-400 hover:bg-white/10'
            }
            disabled:opacity-30 disabled:cursor-not-allowed
          `}
        >
          {!isMicMuted ? (
            <FaMicrophone className="text-xl text-red-400" />
          ) : (
            <FaMicrophoneSlash className="text-xl" />
          )}
        </button>

        {/* Hang up */}
        <button
          onClick={handleHangUp}
          title="إنهاء المكالمة"
          className="w-16 h-16 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center justify-center shadow-xl transition-colors"
        >
          <FaPhoneSlash className="text-2xl" />
        </button>
      </div>
    </div>
  );
}

/**
 * Call trigger button shown in the Navbar / chat area.
 * Accepts an onClick that opens the CallModal.
 */
interface CallButtonProps {
  onClick: () => void;
  isActive: boolean;
}

export function CallButton({ onClick, isActive }: CallButtonProps) {
  return (
    <button
      onClick={onClick}
      title={isActive ? 'مكالمة جارية' : 'بدء مكالمة'}
      className={`
        w-11 h-11 rounded-full flex items-center justify-center transition-all shadow
        ${isActive
          ? 'bg-green-500 text-white animate-pulse'
          : 'bg-church-700 hover:bg-church-600 text-gold-300 hover:text-gold-200'
        }
      `}
    >
      <FaMicrophone className="text-sm" />
    </button>
  );
}
