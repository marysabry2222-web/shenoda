import os
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.35

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# الموديل الرئيسي للرد على الأسئلة (openai/gpt-oss-120b اتعمله deprecate
# من Groq في 17 يونيو 2026 - متبقاش نرجعله)
GROQ_CHAT_MODEL: str = "qwen/qwen3-32b"

# موديل الـ STT (تحويل الصوت لنص) للمكالمة الفورية عبر Groq - بديل
# أدق وأسرع من Whisper المحلي (tiny)، بيشتغل على سيرفرات Groq مش
# سيرفرنا. /voice (الفويس نوت) لسه بتستخدم Whisper المحلي زي ما هي.
GROQ_STT_MODEL: str = "whisper-large-v3-turbo"

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

HF_EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBED_URL: str = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}"

# TTS_VOICE ده كان لخيار edge-tts (لو رجعنا نستخدمه كـ fallback لاحقًا)
TTS_VOICE: str = "ar-EG-ShakirNeural"

WHISPER_MODEL: str = "tiny"
CHUNKS_PATH: str = "chunks.pkl"
CORS_ORIGINS: list = ["*"]

# =========================
# إعدادات الـ retrieval (rag.py بيستوردهم مباشرة)
# =========================
# عدد أقرب الـ chunks اللي بتتبعت للموديل مع كل سؤال
TOP_K: int = 7

# True = بعت بس أقرب TOP_K chunks (أسرع، أرخص، أدق)
# False = بعت كل الـ chunks كاملة مهما كان عددهم (أبطأ، أغلى)
USE_BM25_RETRIEVAL: bool = os.getenv("USE_BM25_RETRIEVAL", "true").lower() == "true"

# =========================
# Cloudinary (لسحب صور الآباء الكهنة عشوائيًا حسب فولدر كل أب)
# =========================
CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")
