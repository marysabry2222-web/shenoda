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
# Retrieval خفيف (BM25-lite) - "واعي بالسياق"
# =========================
# الفرق عن BM25 العادي: بدل ما نبحث بكلمات السؤال الحالي بس، بندمج
# كلمات آخر رسايل المحادثة (history) مع السؤال قبل التسجيل. كده أسئلة
# متابعة زي "طب عرفني عنه" أو "وهو ده مين" (من غير كلمات مفتاحية واضحة)
# لسه بتلاقي الـ chunk الصح، لأن كلمات الرسالة اللي فاتت (زي اسم
# الشخص) بتفضل موجودة في البحث.
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

# قد إيه رسايل من الهيستوري بتتضاف لكلمات البحث (غير الـ history اللي
# بيتبعت فعليًا كسياق محادثة لـ Groq - ده منفصل، شوفي MAX_HISTORY_MESSAGES)
RETRIEVAL_HISTORY_WINDOW = 2


def _build_bm25_index():
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


def _build_retrieval_query(question: str, history: list[dict] | None) -> str:
    """بتضيف كلمات آخر رسايل المحادثة لكلمات السؤال، عشان أسئلة المتابعة
    (من غير كلمات مفتاحية واضحة) لسه تلاقي الـ chunk الصح."""
    parts = [question]
    if history:
        recent = history[-RETRIEVAL_HISTORY_WINDOW:]
        for item in recent:
            content = item.get("content")
            if content:
                parts.append(content)
    return " ".join(parts)


def _retrieve_context(question: str, history: list[dict] | None = None, top_k: int = TOP_K) -> str:
    """بترجع أقرب top_k chunks بس (مش كل الـ 11) - بحث واعي بسياق
    المحادثة، مش بس بكلمات السؤال الحالي لوحدها."""
    if not _chunk_token_lists:
        _build_bm25_index()

    retrieval_query = _build_retrieval_query(question, history)
    query_tokens = _tokenize(retrieval_query)

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


SYSTEM_PROMPT = """You are شنودة, an AI assistant for Anba Shenouda Church in Alexandria, Egypt.
Rules:
- If asked who you are, say: "أنا شنودة، مساعد ذكي خاص بكنيسة الأنبا شنودة."
- Answer ONLY in Arabic. Every word must be Arabic - not even a single foreign word or letter, including connector words.
- Answer ONLY using the provided context. Never invent information.
- If the answer is not in the context, say exactly: "عذرًا، لا أملك معلومة مؤكدة عن ذلك. يرجى الرجوع لقدس أبونا ويصا."
- Never claim to be a priest or bishop.
- Never mention embeddings, FAISS, chunks, or retrieval.
- Be warm, respectful, and natural.
"""

# رسالة بديلة تتقال للمستخدم لو كل محاولات الاتصال بـ Groq فشلت
FALLBACK_MESSAGE = "في ضغط عالي على النظام دلوقتي، ممكن تجرب تاني بعد شوية؟"

MAX_RETRIES = 3
NETWORK_ERROR_BASE_DELAY = 2  # ثواني، بيتضاعف مع كل محاولة (2, 4, 8)

# عدد رسايل الهيستوري اللي بتتبعت كسياق محادثة كامل لـ Groq (منفصل عن
# RETRIEVAL_HISTORY_WINDOW اللي بيتستخدم بس لتحسين البحث عن الـ chunks)
MAX_HISTORY_MESSAGES = 4


def load_resources():
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


_LATIN_RE = re.compile(r"[a-zA-Z]")


def _has_language_leak(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return False
    latin_count = len(_LATIN_RE.findall(text))
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
    context = _retrieve_context(question, history)
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

        if _has_language_leak(answer):
            print("LANGUAGE LEAK DETECTED (not retrying, logged for monitoring):", answer[:150])

        return answer

    return FALLBACK_MESSAGE
