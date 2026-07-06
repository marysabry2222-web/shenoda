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
    GROQ_API_KEY,
    GROQ_STT_MODEL,
)
import rag

router = APIRouter()

# =========================
# إعدادات المكالمة الفورية (Real-time call)
# =========================

CALL_SAMPLE_RATE = 16000
CALL_FRAME_MS = 20
CALL_FRAME_BYTES = int(CALL_SAMPLE_RATE * CALL_FRAME_MS / 1000) * 2
CALL_SILENCE_MS = 700
CALL_SILENCE_FRAMES = CALL_SILENCE_MS // CALL_FRAME_MS
CALL_VAD_AGGRESSIVENESS = 2

MIN_SPEECH_FRAMES_TO_INTERRUPT = 3

MIN_UTTERANCE_MS = 1000
MIN_UTTERANCE_BYTES = int(CALL_SAMPLE_RATE * MIN_UTTERANCE_MS / 1000) * 2

MIN_SPEECH_RMS = 500.0


def _is_too_quiet(pcm_bytes: bytes) -> bool:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return rms < MIN_SPEECH_RMS


GEMINI_TTS_SAMPLE_RATE = 24000
GEMINI_AUDIO_CHUNK_BYTES = 4800

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
    whisper = get_whisper()

    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, _ = whisper.transcribe(
        audio_array,
        language="ar",
        beam_size=1
    )

    texts = [segment.text for segment in segments]
    text = " ".join(texts).strip()

    print("Call transcript (local whisper):", text)

    return text


async def _speech_to_text_groq(pcm_bytes: bytes) -> str:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(CALL_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    wav_buffer.seek(0)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.wav", wav_buffer, "audio/wav")},
            data={
                "model": GROQ_STT_MODEL,
                "language": "ar",
            },
        )

        if resp.status_code != 200:
            print("========== GROQ STT ERROR ==========")
            print("STATUS:", resp.status_code)
            print("BODY:", resp.text)
            resp.raise_for_status()

        text = resp.json().get("text", "").strip()

    print("Call transcript (Groq):", text)
    return text


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
# بروتوكول المكالمة الفورية بين العميل (useCall.ts) والسيرفر ده:
#
# Client → Server:
#   - Binary frames: صوت PCM16 خام، mono، 16000Hz.
#   - {"type": "mic_muted"}
#
# Server → Client:
#   - {"type": "interrupted"}
#   - {"type": "processing"}
#   - {"type": "transcript", "text": "..."}
#   - {"type": "answer_text", "text": "..."}
#   - {"type": "answer_audio_start"}
#   - Binary frames: صوت PCM16 خام 24kHz mono على شرائح ~100ms
#   - {"type": "answer_audio_end"}
#   - {"type": "play_url", "url": "..."}   جديد: لما الرد عبارة عن رابط
#     صوت جاهز (زي لحن ترحيب البابا) بدل PCM متولّد من TTS - العميل
#     يشغّله مباشرة بـ Audio API عادي، مش عبر مسار الـ PCM streaming.


async def _process_utterance(
    websocket: WebSocket,
    pcm_audio: bytes,
    call_state: dict,
) -> None:
    try:
        await websocket.send_json({"type": "processing"})

        question = await _speech_to_text_groq(pcm_audio)
        if not question:
            return

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

    vad = webrtcvad.Vad(CALL_VAD_AGGRESSIVENESS)

    incoming_buffer = bytearray()
    utterance_buffer = bytearray()
    is_user_speaking = False
    silence_frame_count = 0
    speech_frame_count = 0
    current_task: asyncio.Task | None = None

    call_state = {"speaking": False}

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except (TypeError, json.JSONDecodeError):
                    continue

                if control.get("type") == "mic_muted":
                    incoming_buffer.clear()
                    utterance_buffer.clear()
                    is_user_speaking = False
                    silence_frame_count = 0
                    speech_frame_count = 0
                    print("Mic muted by client - discarded pending utterance")

                continue

            chunk = message.get("bytes")
            if chunk is None:
                continue

            incoming_buffer.extend(chunk)

            while len(incoming_buffer) >= CALL_FRAME_BYTES:
                frame = bytes(incoming_buffer[:CALL_FRAME_BYTES])
                del incoming_buffer[:CALL_FRAME_BYTES]

                is_speech = vad.is_speech(frame, CALL_SAMPLE_RATE)

                if is_speech:
                    speech_frame_count += 1

                    if not is_user_speaking:
                        if speech_frame_count < MIN_SPEECH_FRAMES_TO_INTERRUPT:
                            utterance_buffer.extend(frame)
                            continue

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
                    utterance_buffer.extend(frame)
                    silence_frame_count += 1

                    if silence_frame_count >= CALL_SILENCE_FRAMES:
                        finished_utterance = bytes(utterance_buffer)
                        utterance_buffer.clear()
                        is_user_speaking = False
                        silence_frame_count = 0
                        speech_frame_count = 0

                        if len(finished_utterance) < MIN_UTTERANCE_BYTES or _is_too_quiet(finished_utterance):
                            print(
                                f"Utterance rejected (too short/quiet - "
                                f"{len(finished_utterance)} bytes) - likely whisper/noise, ignoring"
                            )
                            continue

                        if current_task is not None and not current_task.done():
                            print("Previous response still in progress - ignoring this utterance")
                            continue

                        print(
                            f"Utterance finished, {len(finished_utterance)} bytes, "
                            "starting processing task"
                        )

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
                    speech_frame_count = 0
                    if not is_user_speaking:
                        utterance_buffer.clear()

    except WebSocketDisconnect:
        pass

    finally:
        if current_task is not None and not current_task.done():
            current_task.cancel()
