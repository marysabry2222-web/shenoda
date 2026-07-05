import pickle
import re
import time
import math
from collections import Counter

import requests
from config import (
    CHUNKS_PATH,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
    TOP_K,
)

_chunks: list[str] = []

# =========================
# Retrieval خفيف (BM25-lite) - من غير أي مكتبات إضافية
# =========================
# بيشتغل بس بعد ما بيقارن كلمات السؤال بكلمات كل chunk، من غير الاعتماد
# على embeddings محفوظة مسبقًا - فمينفعش لو أعدتي التدريب وغيرتي البيانات
# من غير ما نحتاج نعيد بناء أي فايل تاني غير embeddings.npy/chunks.pkl
# نفسهم.
_AR_STOPWORDS = {
    "في", "من", "الى", "إلى", "على", "عن", "و", "أو", "ثم", "أن", "إن",
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "كان", "كانت", "يكون",
    "لا", "ما", "لم", "لن", "قد", "كل", "بعض", "مع", "بين", "عند",
    "التي", "الذي", "الذين", "له", "لها", "لهم", "به", "بها", "بهم",
}

_word_re = re.compile(r"[\w\u0600-\u06FF]+")


def _tokenize(text: str) -> list[str]:
    words = _word_re.findall(text.lower())
    return [w for w in words if w not in _AR_STOPWORDS and len(w) > 1]


_chunk_token_lists: list[list[str]] = []
_doc_freq: Counter = Counter()
_avg_doc_len: float = 0.0

BM25_K1 = 1.5
BM25_B = 0.75


def _build_bm25_index():
    """بتتحسب مرة واحدة بعد ما الـ chunks تتحمّل - بتجهز الـ term frequencies
    والـ document frequencies اللازمين لتسجيل BM25-lite وقت كل سؤال."""
    global _chunk_token_lists, _doc_freq, _avg_doc_len

    _chunk_token_lists = [_tokenize(chunk) for chunk in _chunks]
    _doc_freq = Counter()
    for tokens in _chunk_token_lists:
        for word in set(tokens):
            _doc_freq[word] += 1

    total_len = sum(len(tokens) for tokens in _chunk_token_lists)
    _avg_doc_len = (total_len / len(_chunk_token_lists)) if _chunk_token_lists else 0.0


def _bm25_score(query_tokens: list[str], doc_index: int) -> float:
    doc_tokens = _chunk_token_lists[doc_index]
    doc_len = len(doc_tokens) or 1
    term_freq = Counter(doc_tokens)
    n_docs = len(_chunk_token_lists) or 1

    score = 0.0
    for term in query_tokens:
        tf = term_freq.get(term, 0)
        if tf == 0:
            continue
        df = _doc_freq.get(term, 0)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / (_avg_doc_len or 1))
        score += idf * (tf * (BM25_K1 + 1)) / (denom or 1)

    return score


def _retrieve_context(question: str, top_k: int = TOP_K) -> str:
    """
    بترجع أقرب top_k chunks للسؤال بس (بدل كل الـ 20)، عشان نقلل حجم
    الـ context المبعوت لـ Groq بشكل كبير (وبالتالي نتجنب rate limiting).
    لو لأي سبب البحث مرجعش نتيجة حقيقية (كل الـ scores صفر)، بنرجع لأول
    top_k chunks كـ fallback آمن بدل ما نرجع فاضي.
    """
    if not _chunk_token_lists:
        _build_bm25_index()

    query_tokens = _tokenize(question)

    scores = [
        (_bm25_score(query_tokens, i), i) for i in range(len(_chunks))
    ]
    scores.sort(key=lambda pair: pair[0], reverse=True)

    top_indices = [i for score, i in scores[:top_k] if score > 0]

    if not top_indices:
        # مفيش تطابق كلمات حقيقي - رجّعي أول top_k chunks بدل ما ترجعي فاضي
        top_indices = list(range(min(top_k, len(_chunks))))

    selected = [_chunks[i] for i in top_indices]
    return "\n\n---\n\n".join(selected)


SYSTEM_PROMPT = """You are an AI assistant named شنودة for Anba Shenouda Church in Alexandria, Egypt.
STRICT RULES:
- Your name is شنودة. If asked who you are, say: "أنا شنودة، مساعد ذكي خاص بكنيسة الأنبا شنودة."
- Answer ONLY in Arabic. Every single word must be Arabic - no English, Spanish, French, or any other language words or letters anywhere in your answer, not even one word (e.g. never write connector words like "quienes", "who", "which" in another language).
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

MAX_HISTORY_MESSAGES = 2


def load_resources():
    """
    بتحمّل chunks.pkl بس. مش محتاجين نحمّل embeddings.npy أو نبني فهرس
    FAISS خالص - الـ retrieval الحالي (BM25-lite فوق) بيعتمد على تطابق
    كلمات النص مباشرة، مش على embeddings محفوظة. ده كان سبب البطء
    القديم على Railway (تحميل موديل من HuggingFace وقت startup) - وبما
    إننا مش مستخدمينه أصلاً في البحث، شلناه بالكامل.
    """
    global _chunks
    print("Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)
    _chunks = [c["text"] if isinstance(c, dict) else c for c in _chunks]

    _build_bm25_index()

    print(f"✅ Ready — {len(_chunks)} chunks")


def _extract_retry_seconds(response: requests.Response, default: float = 5.0) -> float:
    """بتقرأ 'Please try again in 14.27s' من رسالة خطأ Groq لو موجودة."""
    try:
        message = response.json().get("error", {}).get("message", "")
        match = re.search(r"try again in ([\d.]+)s", message)
        if match:
            return float(match.group(1)) + 0.5
    except Exception:
        pass
    return default


# لو نسبة الحروف اللاتينية في الرد عالية، غالبًا فيه "تسرب لغوي" (كلمة
# من لغة تانية اندسّت في النص) - ظاهرة معروفة مع الموديلات الصغيرة.
_LATIN_RE = re.compile(r"[a-zA-Z]")


def _has_language_leak(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return False
    latin_count = len(_LATIN_RE.findall(text))
    # نسمح بنسبة صغيرة (أسماء أو مصطلحات إنجليزية مقصودة أحيانًا)، لكن
    # أي نسبة أعلى من كده تقريبًا مؤكد إنها تسرب لغوي غير مقصود
    return latin_count > 0 and (latin_count / len(letters)) > 0.05


def _build_messages(question: str, context: str, history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        recent_history = history[-MAX_HISTORY_MESSAGES:]
        for item in recent_history:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": f"Church knowledge base:\n{context}\n\nQuestion: {question}\n\nAnswer in Arabic only.",
        }
    )

    return messages


def _call_groq(messages: list[dict]) -> requests.Response:
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
            "messages": messages,
        },
        timeout=60,
    )


def answer_question(question: str, history: list[dict] | None = None) -> str:
    context = _retrieve_context(question)
    print("CONTEXT LENGTH (chars):", len(context))
    messages = _build_messages(question, context, history)

    for attempt in range(MAX_RETRIES + 1):
        is_last_attempt = attempt == MAX_RETRIES

        try:
            resp = _call_groq(messages)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            print(f"NETWORK ERROR (attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}")
            if is_last_attempt:
                return FALLBACK_MESSAGE
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue
        except requests.exceptions.RequestException as exc:
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
            if is_last_attempt:
                return FALLBACK_MESSAGE
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue

        resp.raise_for_status()

        answer = resp.json()["choices"][0]["message"]["content"].strip()

        # بس تسجيل في اللوج للمراقبة - من غير أي retry أو تأخير إضافي،
        # لأن المكالمة الصوتية حساسة جدًا للـ latency. التعليمة المقوّاة
        # في SYSTEM_PROMPT هي خط الدفاع الأساسي. لو الظاهرة استمرت كتير
        # في اللوج بعد كده، وقتها نفكر في حل تاني (موديل أكبر مثلاً).
        if _has_language_leak(answer):
            print("LANGUAGE LEAK DETECTED (not retrying, logged for monitoring):", answer[:150])

        return answer

    return FALLBACK_MESSAGE
