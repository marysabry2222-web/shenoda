import { useState } from 'react';
import { GiChurch } from 'react-icons/gi';

interface AssistantAvatarProps {
  size?: 'sm' | 'md' | 'lg';
}

const sizeClasses = {
  sm: 'w-8 h-8 text-base',
  md: 'w-12 h-12 text-2xl',
  lg: 'w-24 h-24 text-5xl',
};

/**
 * Displays /public/avatar/avatar.png if it exists.
 * Falls back to a church icon on error.
 * Does NOT hardcode any image data.
 */
export function AssistantAvatar({ size = 'sm' }: AssistantAvatarProps) {
  const [imgError, setImgError] = useState(false);
  const sizeClass = sizeClasses[size];

  if (!imgError) {
    return (
      <img
        src="/avatar/avatar.png"
        alt="شنودة"
        className={`${sizeClass} rounded-full object-cover border-2 border-gold-500/40 shadow`}
        onError={() => setImgError(true)}
      />
    );
  }

  // Fallback: styled icon circle
  return (
    <div
      className={`${sizeClass} rounded-full bg-gradient-to-br from-church-700 to-church-900 border-2 border-gold-500/40 flex items-center justify-center shadow`}
    >
      <GiChurch className="text-gold-400" />
    </div>
  );
}
