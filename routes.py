import io
import json
import base64
import asyncio
import tempfile
import os
import traceback
import wave

import numpy as np
import webrtcvad
import httpx

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from faster_whisper import WhisperModel

from models import ChatRequest, ChatResponse, HealthResponse, HistoryItem
from config import (
    WHISPER_MODEL,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    GEMINI_API_KEY,
    GEMINI_TTS_MODEL,
    GEMINI_TTS_VOICE,
    GEMINI_TTS_STYLE_PROMPT,
)
import rag

router = APIRouter()

# =========================
# إعدادات المكالمة الفورية (Real-time call)
# =========================

CALL_SAMPLE_RATE = 16000          # لازم يطابق اللي العميل بيبعته بالظبط
CALL_FRAME_MS = 20                # مدة الفريم الواحد لـ webrtcvad (لازم 10/20/30ms)
CALL_FRAME_BYTES = int(CALL_SAMPLE_RATE * CALL_FRAME_MS / 1000) * 2  # 640 بايت (PCM16 mono)
CALL_SILENCE_MS = 700             # سكوت متواصل بالقد ده = المستخدم خلص كلامه
CALL_SILENCE_FRAMES = CALL_SILENCE_MS // CALL_FRAME_MS
CALL_VAD_AGGRESSIVENESS = 2       # 0 (أقل حساسية) إلى 3 (أعلى حساسية لفلترة الضوضاء)

# أقل عدد فريمات كلام متتالية عشان نعتبرها "بداية كلام حقيقية" ونقاطع
# بيها أي رد شغال - بتحمي من إن ضوضاء عابرة أو نفس بسيط يلغي المعالجة
# غلط. 3 فريمات × 20ms = 60ms متواصلة من الكلام الفعلي.
MIN_SPEECH_FRAMES_TO_INTERRUPT = 3

# صوت الرد من Gemini بيرجع PCM16 خام 24kHz - بنقطّعه لشرائح صغيرة
# قبل الإرسال عشان نحاكي التدفق (streaming) للعميل ونسمح بالإلغاء الفوري
GEMINI_TTS_SAMPLE_RATE = 24000
GEMINI_AUDIO_CHUNK_BYTES = 4800   # ~100ms عند 24kHz/16-bit mono

# =========================
# Whisper
# =========================

_whisper = None

def get_whisper():
    global _whisper

    if _whisper is None:
        print(f"Loading Whisper model ({WHISPER_MODEL})...")

        _whisper = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print("✅ Whisper loaded successfully")

    return _whisper


# =========================
# STT
# =========================
def _speech_to_text(audio_bytes: bytes) -> str:
    whisper = get_whisper()

    with tempfile.NamedTemporaryFile(
        suffix=".webm",
        delete=False
    ) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        print("Starting transcription...")
        print("Before transcribe")

        segments, info = whisper.transcribe(
            tmp_path,
            language="ar",
            beam_size=1
        )

        print("After transcribe")

        texts = []

        for segment in segments:
            print("Segment:", segment.text)
            texts.append(segment.text)

        text = " ".join(texts).strip()

        print("Transcript:", text)

        return text

    except Exception:
        print("========== STT ERROR ==========")
        traceback.print_exc()
        raise

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _speech_to_text_pcm(pcm_bytes: bytes) -> str:
    """
    زي _speech_to_text بالظبط، لكن بتاخد صوت PCM16 خام مباشرة (من المكالمة
    الفورية) بدل ما تكتبه ملف webm مؤقت — أسرع ومناسبة أكتر للـ streaming.
    """
    whisper = get_whisper()

    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, _ = whisper.transcribe(
        audio_array,
        language="ar",
        beam_size=1
    )

    texts = [segment.text for segment in segments]
    text = " ".join(texts).strip()

    print("Call transcript:", text)

    return text


GEMINI_TTS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_TTS_MODEL}:generateContent"
)


def _build_gemini_tts_payload(text: str) -> dict:
    # بندمج تعليمات الأسلوب (اللهجة المصرية) مع النص نفسه في حقل واحد،
    # لأن الـ API ده (على عكس Cloud TTS) معندوش حقل "prompt" منفصل -
    # التوجيه بيتحقق بلغة طبيعية جوه نفس النص المُرسل.
    styled_text = f"{GEMINI_TTS_STYLE_PROMPT}\n\n{text}"

    return {
        "contents": [{"parts": [{"text": styled_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": GEMINI_TTS_VOICE}
                }
            },
        },
    }


async def _gemini_text_to_speech(text: str) -> bytes:
    """
    بترجع صوت PCM16 خام 24kHz mono جاهز للإرسال للعميل.

    بنستخدم استدعاء غير متدفق (generateContent) بدل streamGenerateContent
    لأن دعم الـ streaming الحقيقي لموديل 3.1 لسه غير مؤكد بثبات في كل
    التوثيقات وقت كتابة الكود ده. بنحاكي "التدفق" للعميل بتقطيع الصوت
    لشرائح صغيرة قبل الإرسال في _process_utterance، وده كافي لتجربة
    مستخدم سلسة وقابلية إلغاء فورية (barge-in) حتى لو التوليد نفسه
    بيحصل دفعة واحدة على السيرفر.
    """
    print("ENTERED GEMINI TTS")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GEMINI_TTS_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=_build_gemini_tts_payload(text),
        )

        # نطبع رد جوجل كامل لو مش 200 عشان نعرف سبب الفشل بالظبط
        # (مفتاح غلط، صلاحية preview، شكل الطلب، إلخ)
        if resp.status_code != 200:
            print("========== GEMINI TTS ERROR ==========")
            print("STATUS:", resp.status_code)
            print("BODY:", resp.text)

        resp.raise_for_status()
        data = resp.json()

    try:
        b64_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    except (KeyError, IndexError):
        print("========== GEMINI TTS UNEXPECTED RESPONSE ==========")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        raise RuntimeError("Gemini TTS response missing audio data")

    return base64.b64decode(b64_audio)

# =========================
# RAG / LLM
# =========================

async def get_answer(question: str, history: list[HistoryItem] | None = None) -> tuple[str, list[str]]:
    print("Question:", question)

    # بنحول الـ Pydantic models لـ plain dicts (role/content) عشان rag.py
    # يستخدمها مباشرة كـ messages جاهزة لـ Groq من غير ما يعتمد على pydantic
    history_dicts = [item.model_dump() for item in history] if history else []

    answer, images = await asyncio.to_thread(
        rag.answer_question,
        question,
        history_dicts
    )

    print("Answer:", answer)
    if images:
        print("Images:", images)

    return answer, images
# =========================
# Models
# =========================

class TTSRequest(BaseModel):
    text: str


# =========================
# Health
# =========================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


# =========================
# Debug - مؤقت، نشيله بعد ما نحل مشكلة الصوت
# =========================

@router.get("/debug/last-call-audio")
async def debug_last_call_audio():
    path = "/tmp/last_call_utterance.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No call audio recorded yet")
    return FileResponse(path, media_type="audio/wav", filename="last_call_utterance.wav")


# =========================
# Chat
# =========================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    try:
        answer, images = await get_answer(request.message, request.history)

        return ChatResponse(answer=answer, images=images)

    except Exception:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Chat failed"
        )


# =========================
# TTS Endpoint
# =========================

@router.post("/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio_data = await _text_to_speech(request.text)

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg"
        )

    except Exception:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="TTS Failed"
        )


@router.post("/voice")
async def voice(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()

        question = await asyncio.to_thread(
            _speech_to_text,
            audio_bytes
        )

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio"
            )

        # ملحوظة: /voice لسه مبيستقبلش history من الفرونت إند حاليًا
        # (sendVoice في api.ts لسه بيبعت الصوت بس). ممكن نضيفها لاحقًا
        # لو حابين محادثات الصوت تبقى فيها سياق زي الشات النصي.
        answer_text, images = await get_answer(question)

        return {
            "transcript": question,
            "answer": answer_text,
            "images": images,
        }

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Voice Failed"
        )


# =========================
# Real-time Call (WebSocket)
# =========================
#
# بروتوكول المكالمة الفورية بين العميل (useCall.ts) والسيرفر ده:
#
# Client → Server:
#   - Binary frames: صوت PCM16 خام، mono، 16000Hz. العميل يبعت الصوت أول
#     بأول من غير ما يستنى — أي حجم chunk، السيرفر بيقطّعها لفريمات
#     640 بايت (20ms) بنفسه. **مهم**: ده لازم صوت PCM خام (من
#     AudioWorklet)، مش webm/opus من MediaRecorder، لأن webrtcvad محتاج
#     فريمات ثابتة الحجم من صوت خام عشان يشتغل.
#   - أي رسالة نصية (JSON) بتتجاهل حاليًا — محجوزة لأوامر تحكم مستقبلية.
#
# Server → Client:
#   - {"type": "interrupted"}                لما المستخدم يبدأ كلام جديد أثناء تشغيل صوت؛
#                                             العميل لازم يوقف أي صوت شغال فورًا
#   - {"type": "processing"}                 لما السكوت يتاكد والسيرفر بدأ STT/LLM
#   - {"type": "transcript", "text": "..."}  بعد ما الـ STT يخلص
#   - {"type": "answer_text", "text": "..."} بعد ما الـ LLM يرد
#   - {"type": "answer_audio_start"}         قبل ما نبعت صوت الرد
#   - Binary frames: صوت PCM16 خام 24kHz mono على شرائح ~100ms
#   - {"type": "answer_audio_end"}           بعد ما الصوت يخلص
#
# التبعيات الجديدة المطلوبة: pip install webrtcvad httpx
#
# ملاحظة مهمة عن الـ barge-in:
# الإلغاء (cancel) بيحصل بس لما فيه صوت رد فعليًا بيتشغل عند العميل
# (call_state["speaking"] == True). لو المساعد لسه "بيفكر" (STT/LLM/TTS
# قاعدين شغالين بصمت، قبل ما أي صوت يوصل للعميل)، أي كلام/ضوضاء من
# المستخدم في اللحظة دي *مبيلغيش* المهمة الشغالة - لأن مفيش صوت أصلاً
# نقاطعه. المهمة بتكمل عادي وتوصل لآخرها، وبعدها لو فيه utterance جديدة
# اتجمعت في الأثناء، هتتعالج بعد ما المهمة الحالية تخلص.


async def _process_utterance(
    websocket: WebSocket,
    pcm_audio: bytes,
    call_state: dict,
) -> None:
    """
    STT → LLM → TTS لجملة واحدة كاملة. الدالة دي بتتشغل كـ asyncio.Task
    منفصلة عشان لو المستخدم قاطع بالكلام أثناء تشغيل الصوت فعليًا
    (barge-in)، السيرفر يقدر يلغيها فورًا (asyncio.CancelledError).

    call_state["speaking"] بتتحول True بس في اللحظة اللي أول شريحة صوت
    بتتبعت فعليًا للعميل، وترجع False تاني في أي مخرج من الدالة (نجاح،
    إلغاء، أو error) - عشان الـ websocket handler يعرف بالظبط إمتى
    الإلغاء يكون له معنى فعلي.
    """
    try:
        await websocket.send_json({"type": "processing"})

        question = await asyncio.to_thread(_speech_to_text_pcm, pcm_audio)
        if not question:
            return

        await websocket.send_json({"type": "transcript", "text": question})

        answer_text, images = await get_answer(question)

        if images:
            await websocket.send_json({"type": "images", "images": images})

        await websocket.send_json({"type": "answer_text", "text": answer_text})

        audio_data = await _gemini_text_to_speech(answer_text)
        print("Audio bytes:", len(audio_data))

        await websocket.send_json({"type": "answer_audio_start"})
        call_state["speaking"] = True  # من هنا بس الإلغاء بقى له معنى

        for i in range(0, len(audio_data), GEMINI_AUDIO_CHUNK_BYTES):
            chunk = audio_data[i:i + GEMINI_AUDIO_CHUNK_BYTES]
            await websocket.send_bytes(chunk)
            await asyncio.sleep(0)  # نسيب فرصة لـ event loop يعالج cancel لو حصل

        await websocket.send_json({"type": "answer_audio_end"})

    except asyncio.CancelledError:
        # اتلغت بسبب barge-in حقيقي أثناء تشغيل الصوت - مفيش داعي نبعت
        # حاجة تانية، الـ "interrupted" اتبعتت فعلًا لحظة ما المستخدم بدأ يتكلم
        raise

    except Exception:
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": "حصل خطأ أثناء معالجة الرد"
            })
        except Exception:
            pass  # الاتصال ممكن يكون اتقفل بالفعل

    finally:
        call_state["speaking"] = False


@router.websocket("/ws/call")
async def call_websocket(websocket: WebSocket):
    await websocket.accept()

    vad = webrtcvad.Vad(CALL_VAD_AGGRESSIVENESS)

    incoming_buffer = bytearray()   # بايتات وصلت لسه ماتقسمتش لفريمات
    utterance_buffer = bytearray()  # الفريمات اللي بتتجمع لحد ما الجملة تخلص
    is_user_speaking = False
    silence_frame_count = 0
    speech_frame_count = 0  # عدد فريمات الكلام المتتالية - لتفادي false positives
    current_task: asyncio.Task | None = None

    # حالة مشتركة بين الـ handler والـ _process_utterance الحالية - بتقول
    # لنا هل فيه صوت رد فعليًا بيتشغل دلوقتي عند العميل ولا لأ
    call_state = {"speaking": False}

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            chunk = message.get("bytes")
            if chunk is None:
                continue  # رسالة نصية أو نوع تاني - نتجاهله حاليًا

            incoming_buffer.extend(chunk)

            while len(incoming_buffer) >= CALL_FRAME_BYTES:
                frame = bytes(incoming_buffer[:CALL_FRAME_BYTES])
                del incoming_buffer[:CALL_FRAME_BYTES]

                is_speech = vad.is_speech(frame, CALL_SAMPLE_RATE)

                if is_speech:
                    speech_frame_count += 1

                    if not is_user_speaking:
                        # لسه ماوصلناش لعدد الفريمات الكافي - ممكن تكون
                        # ضوضاء عابرة، مننفعلش على أساسها لسه
                        if speech_frame_count < MIN_SPEECH_FRAMES_TO_INTERRUPT:
                            utterance_buffer.extend(frame)
                            continue

                        # اتأكدنا إنه كلام حقيقي (60ms+). دلوقتي نفرّق:
                        # لو فيه صوت رد بيتشغل فعليًا → دي مقاطعة حقيقية،
                        # نلغي ونبعت interrupted. لو المساعد لسه بيفكر
                        # بصمت (مفيش صوت اتبعت للعميل لسه) → متلغيش حاجة،
                        # سيبي المهمة تكمل وخلاص.
                        if current_task is not None and not current_task.done():
                            if call_state.get("speaking"):
                                print("REAL interrupt detected - cancelling in-progress speech")
                                await websocket.send_json({"type": "interrupted"})
                                current_task.cancel()
                                current_task = None
                            else:
                                print("Speech detected while still processing (no audio yet) - NOT cancelling")

                        is_user_speaking = True

                    utterance_buffer.extend(frame)
                    silence_frame_count = 0

                elif is_user_speaking:
                    utterance_buffer.extend(frame)  # نسيب شوية سكوت طبيعي في الآخر
                    silence_frame_count += 1

                    if silence_frame_count >= CALL_SILENCE_FRAMES:
                        finished_utterance = bytes(utterance_buffer)
                        utterance_buffer.clear()
                        is_user_speaking = False
                        silence_frame_count = 0
                        speech_frame_count = 0

                        # لو لسه فيه مهمة سابقة شغالة (بتفكر أو بتتكلم)،
                        # منبدأش وحدة جديدة فوقها - كده بنمنع تداخل الصوت
                        # وتضارب طلبات الـ LLM. الجملة دي هتتجاهل ببساطة
                        # (المستخدم هيحتاج يعيدها بعد ما الرد الحالي يخلص).
                        if current_task is not None and not current_task.done():
                            print("Previous response still in progress - ignoring this utterance")
                            continue

                        print(
                            f"Utterance finished, {len(finished_utterance)} bytes, "
                            "starting processing task"
                        )

                        # تشخيص مؤقت: نحفظ آخر جملة كملف WAV نقدر نسمعه
                        # عبر /debug/last-call-audio - نشيل ده بعد ما نحل المشكلة
                        try:
                            with wave.open("/tmp/last_call_utterance.wav", "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(CALL_SAMPLE_RATE)
                                wf.writeframes(finished_utterance)
                        except Exception:
                            traceback.print_exc()

                        current_task = asyncio.create_task(
                            _process_utterance(websocket, finished_utterance, call_state)
                        )
                else:
                    # سكوت والمستخدم مش بيتكلم أصلاً - نصفّر عداد الكلام
                    # ونمسح أي فريمات ضوضاء عابرة كانت اتضافت تخمينًا
                    # (قبل ما توصل لـ MIN_SPEECH_FRAMES_TO_INTERRUPT) عشان
                    # متتسربش وتتلزق في أول الجملة الحقيقية اللي بعدها
                    speech_frame_count = 0
                    if not is_user_speaking:
                        utterance_buffer.clear()

    except WebSocketDisconnect:
        pass

    finally:
        if current_task is not None and not current_task.done():
            current_task.cancel()
