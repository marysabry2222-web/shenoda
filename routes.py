import io
import json
import base64
import asyncio
import tempfile
import os
import traceback

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import edge_tts
from faster_whisper import WhisperModel

from models import ChatRequest, ChatResponse, HealthResponse
from config import TTS_VOICE, WHISPER_MODEL
import rag

router = APIRouter()

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
# =========================
# TTS (Azure Speech)
# =========================
import azure.cognitiveservices.speech as speechsdk
from config import TTS_VOICE, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION

def _text_to_speech_sync(text: str) -> bytes:
    try:
        print(f"TTS Request: {text[:100]}")

        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = TTS_VOICE
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        # No audio output device — we want bytes in memory
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            print(f"✅ TTS Success ({len(audio_data)} bytes)")
            return audio_data

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"TTS canceled: {cancellation.reason}"
            if cancellation.reason == speechsdk.CancellationReason.Error:
                error_msg += f" - {cancellation.error_details}"
            raise RuntimeError(error_msg)

        else:
            raise RuntimeError(f"Unexpected TTS result reason: {result.reason}")

    except Exception:
        print("========== TTS ERROR ==========")
        traceback.print_exc()
        raise


async def _text_to_speech(text: str) -> bytes:
    return await asyncio.to_thread(_text_to_speech_sync, text)
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

        segments, _ = whisper.transcribe(
            tmp_path,
            language="ar"
        )

        text = " ".join(
            seg.text for seg in segments
        ).strip()

        print("Transcript:", text)

        return text

    except Exception:
        print("========== STT ERROR ==========")
        traceback.print_exc()
        raise

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
        answer = await asyncio.to_thread(
            rag.answer_question,
            request.message
        )

        return ChatResponse(answer=answer)

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


# =========================
# Voice Endpoint
# =========================

@router.post("/voice")
async def voice(audio: UploadFile = File(...)):
    try:
        print("Voice request received")

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

        answer_text = await asyncio.to_thread(
            rag.answer_question,
            question
        )

        audio_data = await _text_to_speech(
            answer_text
        )

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "X-Answer-Text": answer_text[:500]
            }
        )

    except HTTPException:
        raise

    except Exception:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Voice Failed"
        )
