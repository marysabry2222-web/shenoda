import pickle
import re
import time

import requests
from config import (
    CHUNKS_PATH,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
)

_chunks: list[str] = []


def _retrieve_context(question: str) -> str:
    """بترجع كل الـ chunks كاملة دايمًا - مفيش أي بحث/فلترة خالص.
    لو عايزة تقللي حجم الـ context تاني في المستقبل، هنا المكان اللي
    نضيف فيه أي منطق retrieval (BM25 أو embeddings) بدل ما نبعت كل حاجة."""
    return "\n\n---\n\n".join(_chunks)


def _embed_question(question: str):
    """
    Placeholder لخطوة الـ embedding - مش مفعّلة حاليًا (الرد بيبعت كل
    الـ chunks كاملة من غير أي بحث). سيبناها هنا عشان لو حبينا نرجع
    نستخدم embeddings حقيقية للـ retrieval مستقبلًا، مكانها جاهز.

    Return None يعني: استخدمي الـ context الكامل (السلوك الحالي).
    """
    return None


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

MAX_HISTORY_MESSAGES = 2


def load_resources():
    """بتحمّل chunks.pkl بس. مفيش أي فهرسة أو بناء index - بنبعت كل
    الـ chunks زي ما هي مع كل سؤال."""
    global _chunks
    print("Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)
    _chunks = [c["text"] if isinstance(c, dict) else c for c in _chunks]

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
