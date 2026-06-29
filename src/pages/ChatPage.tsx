import { useState, useCallback, useEffect } from 'react';
import { ChatWindow } from '../components/ChatWindow';
import { ChatInput } from '../components/ChatInput';
import { ErrorBanner } from '../components/ErrorBanner';
import { CallModal } from '../components/CallModal';
import { useChat } from '../hooks/useChat';
import { useVoice } from '../hooks/useVoice';
import { useCall } from '../hooks/useCall';

/**
 * Main chat page.
 * Manages text chat, one-shot voice, and real-time call modes.
 */
export function ChatPage() {
  const { messages, isLoading, error, sendUserMessage, addAssistantMessage, clearError } =
    useChat();

  const [callOpen, setCallOpen] = useState(false);

  // ── Callbacks shared between voice and call modes ───────────────────────
  const handleVoiceAnswer = useCallback(
    (text: string) => addAssistantMessage(text),
    [addAssistantMessage]
  );

  const handleCallTranscript = useCallback(
    (text: string) => {
      // Show what the user said as a user bubble
      sendUserMessage(text);  // won't duplicate — we only add, not re-send to API
    },
    [sendUserMessage]
  );

  const handleCallAnswer = useCallback(
    (text: string) => addAssistantMessage(text),
    [addAssistantMessage]
  );

  // ── One-shot voice ───────────────────────────────────────────────────────
  const { isRecording, isProcessing, startRecording, stopRecording, error: voiceError } =
    useVoice(handleVoiceAnswer);

  // ── Real-time call ───────────────────────────────────────────────────────
  const call = useCall({
    onTranscript: handleCallTranscript,
    onAnswer: handleCallAnswer,
  });

  // Start call as soon as modal opens
  useEffect(() => {
    if (callOpen && call.status === 'idle') {
      call.startCall();
    }
  }, [callOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleOpenCall = () => setCallOpen(true);
  const handleCloseCall = () => {
    call.endCall();
    setCallOpen(false);
  };

  const combinedError = error || voiceError;

  return (
    <div className="flex flex-col h-screen bg-parchment" dir="rtl">
      {/* Subtle diagonal pattern */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: `repeating-linear-gradient(
            45deg, #7a5c35 0px, #7a5c35 1px, transparent 1px, transparent 20px
          )`,
        }}
      />

      {/* Navbar — pass call button */}
      <NavbarWithCall onCallClick={handleOpenCall} isCallActive={call.isCallActive} />

      {/* Chat area */}
      <div className="flex flex-col flex-1 max-w-4xl mx-auto w-full pt-16 min-h-0">
        <ChatWindow messages={messages} isLoading={isLoading} />

        {combinedError && (
          <ErrorBanner message={combinedError} onDismiss={clearError} />
        )}

        <ChatInput
          onSend={sendUserMessage}
          onVoiceStart={startRecording}
          onVoiceStop={stopRecording}
          onCallStart={handleOpenCall}
          isLoading={isLoading}
          isRecording={isRecording}
          isVoiceProcessing={isProcessing}
          isCallActive={call.isCallActive}
        />
      </div>

      {/* Real-time call modal */}
      {callOpen && (
        <CallModal
          onClose={handleCloseCall}
          onTranscript={handleCallTranscript}
          onAnswer={handleCallAnswer}
        />
      )}
    </div>
  );
}

// ── Inline sub-component: Navbar with call button ─────────────────────────
import { GiChurch } from 'react-icons/gi';
import { FaPhone } from 'react-icons/fa';

function NavbarWithCall({
  onCallClick,
  isCallActive,
}: {
  onCallClick: () => void;
  isCallActive: boolean;
}) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-40 bg-church-800/95 backdrop-blur-sm border-b border-gold-500/30">
      <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-gold-500/20 border border-gold-500/50 flex items-center justify-center">
            <GiChurch className="text-gold-400 text-lg" />
          </div>
          <div className="text-right">
            <h1 className="text-gold-400 font-bold text-lg leading-none font-arabic">شنودة</h1>
            <p className="text-church-300 text-xs font-arabic">مساعد كنيسة الأنبا شنودة</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Real-time call button in navbar */}
          <button
            onClick={onCallClick}
            disabled={isCallActive}
            title="مكالمة مباشرة مع شنودة"
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-arabic
              transition-all shadow
              ${isCallActive
                ? 'bg-green-500/30 text-green-300 cursor-default'
                : 'bg-gold-500/20 hover:bg-gold-500/30 text-gold-300 hover:text-gold-200 border border-gold-500/30'
              }
            `}
          >
            <FaPhone className="text-xs" />
            {isCallActive ? 'مكالمة جارية' : 'اتصل بشنودة'}
          </button>

          <div className="text-gold-500/50 text-2xl select-none">✝</div>
        </div>
      </div>
    </nav>
  );
}
