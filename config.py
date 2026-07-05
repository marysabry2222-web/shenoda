import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.35

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_CHAT_MODEL="llama-3.1-8b-instant"

HF_TOKEN: str = os.getenv("HF_TOKEN", "")

ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")

# =========================
# Gemini 3.1 Flash TTS (اللي بيستخدم في المكالمة الفورية /ws/call بس)
# =========================
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_TTS_MODEL: str = "gemini-3.1-flash-tts-preview"

# اسم شخصية الصوت (مش اسم لغة) - القائمة الكاملة في docs جوجل.
# ممكن تتغير لاحقًا بعد ما نسمع أكتر من واحد ونشوف أنسبهم للهجة المصرية.
GEMINI_TTS_VOICE: str = "Kore"

# بروميت التوجيه اللي بيتبعت مع كل نص عشان الموديل يتكلم باللهجة المصرية
# (الموديل بيتحكم في الأسلوب باللغة الطبيعية بدل صوت ثابت مخصص للهجة)
GEMINI_TTS_STYLE_PROMPT: str = (
    "اتكلم باللهجة المصرية العامية، بأسلوب دافئ ومحترم وطبيعي، "
    "زي حد بيرد على حد بيسأله في كنيسة."
)

GROQ_CHAT_MODEL: str = "llama-3.1-8b-instant"
HF_EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBED_URL: str = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}"

# TTS_VOICE ده كان لخيار edge-tts (لو رجعنا نستخدمه كـ fallback لاحقًا)
TTS_VOICE: str = "ar-EG-ShakirNeural"

WHISPER_MODEL: str = "tiny"
CHUNKS_PATH: str = "chunks.pkl"
CORS_ORIGINS: list = ["*"]
