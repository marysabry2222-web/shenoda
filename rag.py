import pickle
import re
import time
import math
import random
import json
from pathlib import Path
from collections import Counter

import requests
from config import (
    CHUNKS_PATH,
    GROQ_API_KEY,
    GROQ_CHAT_MODEL,
    TOP_K,
)

_chunks: list[str] = []

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
    parts = [question]
    if history:
        recent = history[-RETRIEVAL_HISTORY_WINDOW:]
        for item in recent:
            content = item.get("content")
            if content:
                parts.append(content)
    return " ".join(parts)


def _retrieve_context(question: str, history: list[dict] | None = None, top_k: int = TOP_K) -> str:
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
        top_indices = list(range(min(top_k, len(_chunks))))

    selected = [_chunks[i] for i in top_indices]
    return "\n\n---\n\n".join(selected)


SYSTEM_PROMPT = """You are شنودة, an AI assistant for Anba Shenouda Church in Alexandria, Egypt.
- If asked who you are, say: "أنا شنودة، مساعد ذكي خاص بكنيسة الأنبا شنودة."
- Answer ONLY in Arabic. Every word must be Arabic - not even a single foreign word or letter, including connector words.
- Answer ONLY using facts from the provided context. Never invent information that isn't in the context.
- EXCEPTION: if the context gives a date (e.g. an ordination date), you MAY calculate elapsed years/duration
  by subtracting that date from today's real date above ({today}). This is a calculation on a real fact,
  not invented information, so it is always allowed.
- If a user asks "مدة" (duration/how long) about someone's service or ordination, treat it as asking for the
  number of years since that date until today, using the calculation rule above - do not just restate the date.
- If a follow-up question refers to something discussed earlier in the conversation (e.g. "كم عدد السنين" after
  discussing an ordination date), use the conversation history to understand what is being asked, and answer using
  the same calculation rule.
- If the underlying fact itself (not just the date-math) is not in the context, say exactly:
  "عذرًا، لا أملك معلومة مؤكدة عن ذلك. يرجى الرجوع لقدس أبونا ويصا."
- Never claim to be a priest or bishop.
- Never mention embeddings, FAISS, chunks, or retrieval.
- Be warm, respectful, and natural.
"""

FALLBACK_MESSAGE = "في ضغط عالي على النظام دلوقتي، ممكن تجرب تاني بعد شوية؟"

MAX_RETRIES = 3
NETWORK_ERROR_BASE_DELAY = 2

MAX_HISTORY_MESSAGES = 4

TOPIC_FOLDERS: dict[str, list[str]] = {
    "أبونا شنودة دوس": ["الاباء/ابونا شنودة"],
    "القمص إبراهيم عطية": ["الاباء/ابونا ابراهيم عطية"],
    "أبونا إبراهيم عطية": ["الاباء/ابونا ابراهيم عطية"],
    "القمص جرجس مرقس": ["الاباء/ابونا جرجس"],
    "أبونا أغاثون حنا": ["الاباء/ابونا اغاثون"],
    "أبونا مينا زكي سليمان": ["الاباء/ابونا مينا"],
    "القمص ويصا القمص جرجس": ["الاباء/ابونا ويصا"],
    "أبونا ويصا": ["الاباء/ابونا ويصا"],

    "البابا شنودة الثالث": ["زيارات البطاركة/البابا شنودة 1977"],
    "البابا كيرلس السادس": ["زيارات البطاركة/البابا كيرلس 1960"],
    "البابا تواضروس الثاني": ["زيارات البطاركة/البابا تواضروس 2015"],

    "خدمات الكنيسة": ["خدمات"],

    "الكنيسة القديمة": [
        "صور كنيسة القديمة من 77 ل 2007",
        "الكنيسة الحالية قبل التعمير من 2012 الي 2024",
        "كنيسة خارجي 90",
    ],
    "بناء الكنيسة": [
        "صور كنيسة القديمة من 77 ل 2007",
        "الكنيسة الحالية قبل التعمير من 2012 الي 2024",
        "كنيسة خارجي 90",
    ],
    "نشأة الكنيسة": [
        "صور كنيسة القديمة من 77 ل 2007",
        "الكنيسة الحالية قبل التعمير من 2012 الي 2024",
        "كنيسة خارجي 90",
    ],
    "تعمير الكنيسة": [
        "صور كنيسة القديمة من 77 ل 2007",
        "الكنيسة الحالية قبل التعمير من 2012 الي 2024",
        "كنيسة خارجي 90",
    ],
}

IMAGES_PER_ANSWER = 2

# =========================
# قراءة الصور من assets.json (تفريغ ثابت لكل صور Cloudinary) بدل
# استدعاء Cloudinary Search API لايف - أسرع، وميحتاجش API key/secret
# وقت التشغيل خالص. الملف ده لازم يكون جنب باقي ملفات الباك اند
# (نفس مكان chunks.pkl)، وتحدّثيه بـ سكريبت التصدير كل ما تضيفي صور
# جديدة على Cloudinary.
# =========================
ASSETS_JSON_PATH = "assets.json"

# دلوقتي: dict {اسم المجلد: [رابط1, رابط2, ...]}
_folder_to_images: dict[str, list[str]] = {}


def _load_assets_json():
    global _folder_to_images

    path = Path(ASSETS_JSON_PATH)
    if not path.exists():
        print(f"ASSETS: {ASSETS_JSON_PATH} غير موجود - الصور مش هتظهر")
        _folder_to_images = {}
        return

    with open(path, "r", encoding="utf-8") as f:
        assets = json.load(f)

    grouped: dict[str, list[str]] = {}
    for asset in assets:
        folder = asset.get("folder", "")
        url = asset.get("url")
        if not folder or not url:
            continue  # نتجاهل صور samples الافتراضية (folder فاضي)
        grouped.setdefault(folder, []).append(url)

    _folder_to_images = grouped
    total_images = sum(len(v) for v in grouped.values())
    print(f"ASSETS: تم تحميل {total_images} صورة عبر {len(grouped)} مجلد من {ASSETS_JSON_PATH}")


def _normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    return text


def _match_topic(text: str) -> list[str] | None:
    normalized_text = _normalize_arabic(text)
    sorted_names = sorted(TOPIC_FOLDERS.keys(), key=len, reverse=True)
    for name in sorted_names:
        if _normalize_arabic(name) in normalized_text:
            return TOPIC_FOLDERS[name]
    return None


def _detect_topic_folders(
    question: str,
    answer: str,
    history: list[dict] | None = None,
) -> list[str] | None:
    folders = _match_topic(question)
    if folders:
        return folders

    folders = _match_topic(answer)
    if folders:
        return folders

    if history:
        for item in reversed(history[-2:]):
            content = item.get("content", "")
            folders = _match_topic(content)
            if folders:
                return folders

    return None


def _get_random_images(folders: list[str], count: int = IMAGES_PER_ANSWER) -> list[str]:
    """بتجمع روابط الصور من كل الفولدرات المطلوبة (من الكاش المحمّل من
    assets.json)، وتسحب عشوائي منها - من غير أي استدعاء شبكة."""
    if not _folder_to_images:
        _load_assets_json()

    all_urls: list[str] = []
    for folder in folders:
        all_urls.extend(_folder_to_images.get(folder, []))

    if not all_urls:
        return []

    random.shuffle(all_urls)
    return all_urls[:count]


def _detect_priest_images(
    question: str,
    answer: str,
    history: list[dict] | None = None,
) -> list[str]:
    folders = _detect_topic_folders(question, answer, history)
    if not folders:
        return []
    return _get_random_images(folders)


def load_resources():
    global _chunks
    print("Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)
    _chunks = [c["text"] if isinstance(c, dict) else c for c in _chunks]

    _build_bm25_index()
    _load_assets_json()

    print(f"✅ Ready — {len(_chunks)} chunks")


def _extract_retry_seconds(response: requests.Response, default: float = 5.0) -> float:
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


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[str]]:
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
                return FALLBACK_MESSAGE, []
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue
        except requests.exceptions.RequestException as exc:
            print(f"UNEXPECTED REQUEST ERROR (attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}")
            if is_last_attempt:
                return FALLBACK_MESSAGE, []
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue

        print("STATUS:", resp.status_code)
        print("BODY:", resp.text)

        if resp.status_code == 429:
            if is_last_attempt:
                return FALLBACK_MESSAGE, []
            wait_seconds = _extract_retry_seconds(resp)
            print(f"RATE LIMITED — waiting {wait_seconds}s before retry")
            time.sleep(wait_seconds)
            continue

        if resp.status_code >= 500:
            if is_last_attempt:
                return FALLBACK_MESSAGE, []
            time.sleep(NETWORK_ERROR_BASE_DELAY * (2 ** attempt))
            continue

        resp.raise_for_status()

        answer = resp.json()["choices"][0]["message"]["content"].strip()

        if _has_language_leak(answer):
            print("LANGUAGE LEAK DETECTED (not retrying, logged for monitoring):", answer[:150])

        images = _detect_priest_images(question, answer, history)

        return answer, images

    return FALLBACK_MESSAGE, []
