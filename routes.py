import io
import json
import base64
import asyncio
import tempfile
import os
import traceback

import httpx

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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
#
# ملحوظة معمارية مهمة: المكالمة بقت بتستخدم نفس آلية useVoice.ts بالظبط -
# التعرف على الصوت (STT) بيحصل في المتصفح نفسه عن طريق Web Speech API
# (webkitSpeechRecognition, lang='ar-EG')، مش عن طريق Whisper على السيرفر.
# السيرفر بقى مسؤول بس عن: استقبال النص الجاهز -> LLM -> Gemini TTS.
# ده بيلغي تمامًا الحاجة لبعت صوت خام، الـ VAD، فحص الهدوء، وكل مشاكل
# دقة Whisper tiny مع اللهجة المصرية اللي كنا بنلف حواليها.

GEMINI_AUDIO_CHUNK_BYTES = 4800

# =========================
# Whisper (لسه مستخدم في /voice REST endpoint بس - مش في المكالمة)
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
# STT (لـ /voice REST endpoint فقط)
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


GEMINI_TTS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_TTS_MODEL}:generateContent"
)


def _build_gemini_tts_payload(text: str) -> dict:
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

async def get_answer(
    question: str, history: list[HistoryItem] | None = None
) -> tuple[str, list[str], str | None]:
    """بترجع (answer, images, audio_url). audio_url بيبقى None في الحالات
    العادية، وبيتحدد بس لما تريجر خاص (زي ترحيب البابا) يشتغل في
    rag.answer_question ويرجع رابط صوت جاهز بدل ما يمر على الـ LLM."""
    print("Question:", question)

    history_dicts = [item.model_dump() for item in history] if history else []

    answer, images, audio_url = await asyncio.to_thread(
        rag.answer_question,
        question,
        history_dicts
    )

    print("Answer:", answer)
    if images:
        print("Images:", images)
    if audio_url:
        print("Audio URL:", audio_url)

    return answer, images, audio_url


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
        answer, images, audio_url = await get_answer(request.message, request.history)

        return ChatResponse(answer=answer, images=images, audio_url=audio_url)

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
        audio_data = await _gemini_text_to_speech(request.text)

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/pcm"
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

        answer_text, images, audio_url = await get_answer(question)

        return {
            "transcript": question,
            "answer": answer_text,
            "images": images,
            "audio_url": audio_url,
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
# بروتوكول المكالمة الفورية الجديد بين العميل (useCall.ts) والسيرفر ده:
#
# Client → Server (JSON نص فقط، مفيش binary frames للمايك خالص):
#   - {"type": "mic_unmuted"}                      المستخدم بدأ يتكلم (barge-in لو المساعد بيتكلم)
#   - {"type": "user_utterance", "text": "..."}    المستخدم خلص كلامه، ده النص النهائي
#                                                    اللي طلع من Web Speech API في المتصفح
#
# Server → Client:
#   - {"type": "interrupted"}
#   - {"type": "processing"}
#   - {"type": "transcript", "text": "..."}         (نفس النص اللي بعتّه، رجعناه تأكيدي بس)
#   - {"type": "answer_text", "text": "..."}
#   - {"type": "answer_audio_start"}
#   - Binary frames: صوت PCM16 خام 24kHz mono على شرائح ~100ms
#   - {"type": "answer_audio_end"}
#   - {"type": "play_url", "url": "..."}   لما الرد عبارة عن رابط صوت جاهز
#     (زي لحن ترحيب البابا) بدل PCM متولّد من TTS


async def _process_utterance(
    websocket: WebSocket,
    question: str,
    call_state: dict,
) -> None:
    try:
        await websocket.send_json({"type": "processing"})
        await websocket.send_json({"type": "transcript", "text": question})

        answer_text, images, audio_url = await get_answer(question)

        if images:
            await websocket.send_json({"type": "images", "images": images})

        await websocket.send_json({"type": "answer_text", "text": answer_text})

        # لو فيه رابط صوت جاهز (تريجر خاص زي ترحيب البابا)، ابعتيه
        # مباشرة للعميل يشغله - من غير ما تمري على Gemini TTS خالص
        if audio_url:
            await websocket.send_json({"type": "answer_audio_start"})
            call_state["speaking"] = True
            await websocket.send_json({"type": "play_url", "url": audio_url})
            await websocket.send_json({"type": "answer_audio_end"})
            return

        audio_data = await _gemini_text_to_speech(answer_text)
        print("Audio bytes:", len(audio_data))

        await websocket.send_json({"type": "answer_audio_start"})
        call_state["speaking"] = True

        for i in range(0, len(audio_data), GEMINI_AUDIO_CHUNK_BYTES):
            chunk = audio_data[i:i + GEMINI_AUDIO_CHUNK_BYTES]
            await websocket.send_bytes(chunk)
            await asyncio.sleep(0)

        await websocket.send_json({"type": "answer_audio_end"})

    except asyncio.CancelledError:
        raise

    except Exception:
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": "حصل خطأ أثناء معالجة الرد"
            })
        except Exception:
            pass

    finally:
        call_state["speaking"] = False


@router.websocket("/ws/call")
async def call_websocket(websocket: WebSocket):
    await websocket.accept()

    current_task: asyncio.Task | None = None
    call_state = {"speaking": False}

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # مفيش صوت خام بيتبعت تاني - التعرف كله بيحصل في المتصفح.
            # أي binary frame توصل (مثلاً من نسخة فرونت إند قديمة) نتجاهلها.
            if message.get("bytes") is not None:
                continue

            if message.get("text") is None:
                continue

            try:
                control = json.loads(message["text"])
            except (TypeError, json.JSONDecodeError):
                continue

            msg_type = control.get("type")

            if msg_type == "user_utterance":
                question = (control.get("text") or "").strip()

                if not question:
                    print("Empty utterance received from client - ignoring")
                    continue

                if current_task is not None and not current_task.done():
                    print("Previous response still in progress - ignoring this utterance")
                    continue

                print(f"User utterance received: {question!r}")

                current_task = asyncio.create_task(
                    _process_utterance(websocket, question, call_state)
                )

            elif msg_type == "mic_unmuted":
                # المستخدم بدأ يتكلم من جديد. لو المساعد كان لسه بيتكلم أو
                # بيعالج، ده barge-in فعلي: نلغي أي حاجة شغالة فورًا
                if current_task is not None and not current_task.done():
                    print("Barge-in: cancelling in-progress response")
                    await websocket.send_json({"type": "interrupted"})
                    current_task.cancel()
                    current_task = None

                call_state["speaking"] = False

    except WebSocketDisconnect:
        pass

    finally:
        if current_task is not None and not current_task.done():
            current_task.cancel()
