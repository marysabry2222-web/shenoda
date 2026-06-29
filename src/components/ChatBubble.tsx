import type { Message } from '../types';
import { AssistantAvatar } from './AssistantAvatar';

interface ChatBubbleProps {
  message: Message;
  isSpeaking?: boolean;
}

export function ChatBubble({ message, isSpeaking = false }: ChatBubbleProps) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-start animate-slide-up">
        <div className="max-w-[80%] bg-gold-500 text-white rounded-2xl rounded-bl-sm px-4 py-3 shadow text-right font-arabic leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-end gap-2 animate-slide-up">
      <AssistantAvatar size="sm" isSpeaking={isSpeaking} />
      <div className={`
        max-w-[80%] bg-white/90 border text-church-800 rounded-2xl rounded-br-sm
        px-4 py-3 shadow-sm text-right font-arabic leading-relaxed transition-all
        ${isSpeaking ? 'border-gold-400 shadow-gold-100 shadow-md' : 'border-church-200'}
      `}>
        {message.content}
        {/* Speaking indicator */}
        {isSpeaking && (
          <div className="flex gap-1 mt-2 justify-end">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-gold-400 animate-bounce-dot"
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
