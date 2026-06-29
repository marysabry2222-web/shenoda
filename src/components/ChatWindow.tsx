import { useEffect, useRef } from 'react';
import type { Message } from '../types';
import { ChatBubble } from './ChatBubble';
import { TypingIndicator } from './TypingIndicator';
import { AssistantAvatar } from './AssistantAvatar';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
}

/**
 * Scrollable chat message list.
 * Auto-scrolls to the bottom whenever new messages arrive.
 */
export function ChatWindow({ messages, isLoading }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or loading state change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Empty state: welcome screen
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-5 text-center py-12 px-4">
        <AssistantAvatar size="lg" />
        <div>
          <h2 className="text-church-800 text-3xl font-bold font-arabic mb-2">
            شنودة
          </h2>
          <p className="text-church-500 font-arabic text-base leading-relaxed">
            المساعد الذكي لكنيسة الأنبا شنودة
          </p>
          <p className="text-church-400 font-arabic text-sm mt-3">
            اكتب سؤالك أدناه أو اضغط على الميكروفون للتحدث
          </p>
        </div>
        {/* Decorative divider */}
        <div className="flex items-center gap-3 text-gold-400/60 text-sm font-arabic mt-2">
          <span className="w-12 h-px bg-gold-300/50" />
          ✝
          <span className="w-12 h-px bg-gold-300/50" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.map((msg) => (
        <ChatBubble key={msg.id} message={msg} />
      ))}
      {isLoading && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}
