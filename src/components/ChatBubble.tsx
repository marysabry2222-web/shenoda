import type { Message } from '../types';
import { AssistantAvatar } from './AssistantAvatar';

interface ChatBubbleProps {
  message: Message;
}

/**
 * Renders a single chat bubble.
 * User messages appear on the right; assistant messages on the left with avatar.
 */
export function ChatBubble({ message }: ChatBubbleProps) {
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
      <AssistantAvatar size="sm" />
      <div className="max-w-[80%] bg-white/90 border border-church-200 text-church-800 rounded-2xl rounded-br-sm px-4 py-3 shadow-sm text-right font-arabic leading-relaxed">
        {message.content}
      </div>
    </div>
  );
}
