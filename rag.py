import pickle
import re
import time

import numpy as np
import faiss
import requests
from config import (
    CHUNKS_PATH,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
    TOP_K,
)

_chunks: list[str] = []
_index: faiss.Index | None = None
_embeddings: np.ndarray | None = None

SYSTEM_PROMPT = """You are an AI assistant named شنودة for Anba Shenouda Church in Alexandria, Egypt.
STRICT RULES:
- Your name is شنودة. If asked who you are, say: "أنا شنودة، مساعد ذكي خاص بكنيسة الأنبا شنودة."
- Answer ONLY in Arabic.
- Answer ONLY using the provided context. Never invent information.
- If the answer is not in the context, say exactly: "عذرًا، لا أملك معلومة مؤكدة عن ذلك. يرجى الرجوع لقدس أبونا ويصا."
- Never say you are a priest or bishop.
- Never mention FAISS, embeddings, chunks, or retrieval.
- Be warm, respectful, and natural.
"""

# رسالة بديلة تتقال للمستخدم لو كل محاولات الاتصال بـ Groq فشلت
FALLBACK_MESSAGE = "في ضغط عالي على النظام دلوقتي، ممكن تجرب تاني بعد شوية؟"

MAX_RETRIES = 3
NETWORK_ERROR_BASE_DELAY = 2  # ثواني، بيتضاعف مع كل محاولة (2, 4, 8)


def load_resources():
    global _chunks, _index, _embeddings
    print("Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)
    _chunks = [c["text"] if isinstance(c, dict) else c for c in _chunks]

    print("Loading embeddings from disk...")
    _embeddings = np.load("embeddings.npy").astype(np.float32)
    faiss.normalize_L2(_embeddings)
    _index = faiss.IndexFlatIP(_embeddings.shape[1])
    _index.add(_embeddings)
    print(f"✅ Ready — {len(_chunks)} chunks")


def _embed_question(question: str) -> np.ndarray:
    """
    Embed question using the same stored embeddings via cosine similarity.
    No sentence-transformers needed — we find the closest chunk by
    keyword overlap as fallback, or use Groq to pick context directly.

    Since we can't re-embed without sentence-transformers,
    we send ALL chunks as context (only 20 chunks = small enough).
    """
    # Return None to signal: use full context
    return None


def _retrieve_context(question: str) -> str:
    """Return all chunks as context — dataset is small enough (20 chunks)."""
    return "\n\n---\n\n".join(_chunks)


def _extract_retry_seconds(response: requests.Response, default: float = 5.0) -> float:
    """بتقرأ 'Please try again in 14.27s' من رسالة خطأ Groq لو موجودة."""
    try:
        message = response.json().get("error", {}).get("message", "")
        match = re.search(r"try again in ([\d.]+)s", message)
        if match:
            return float(match.group(1)) + 0.5  # هامش أمان بسيط
    except Exception:
        pass
    return default


def _call_groq(question: str, context: str) -> requests.Response:
    return requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_CHAT_MODEL,
            "temperature": 0.2,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Church knowledge base:\n{context}\n\nQuestion: {question}\n\nAnswer in Arabic only.",
                },
            ],
        },
        timeout=60,
    )


def answer_question(question: str) -> str:
    context = _retrieve_context(question)

    for attempt in range(MAX_RETRIES + 1):
        is_last_attempt = attempt == MAX_RETRIES

        try:
            resp = _call_groq(question, context)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            # ECONNRESET / Network Error / timeout بيقعوا هنا
            print(f"NETWORK ERROR (attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}")
            if is_last_attempt:
                return FALLBACK_MESSAGE
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))  # 2s, 4s, 8s
            continue
        except requests.exceptions.RequestException as exc:
            # أي مشكلة شبكة تانية غير متوقعة (DNS، SSL، إلخ)
            print(f"UNEXPECTED REQUEST ERROR (attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}")
            if is_last_attempt:
                return FALLBACK_MESSAGE
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue

        print("STATUS:", resp.status_code)
        print("BODY:", resp.text)

        if resp.status_code == 429:
            if is_last_attempt:
                return FALLBACK_MESSAGE
            wait_seconds = _extract_retry_seconds(resp)
            print(f"RATE LIMITED — waiting {wait_seconds}s before retry")
            time.sleep(wait_seconds)
            continue

        if resp.status_code >= 500:
            # مشكلة مؤقتة من طرف Groq نفسه
            if is_last_attempt:
                return FALLBACK_MESSAGE
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue

        # أي خطأ تاني (400/401/...) مش مستاهل إعادة محاولة - خطأ حقيقي في الطلب
        resp.raise_for_status()

        return resp.json()["choices"][0]["message"]["content"].strip()

    return FALLBACK_MESSAGE
