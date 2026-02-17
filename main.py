from typing import Any, Dict, List, Optional
import asyncio
import io
import json
import tempfile
import wave
import os
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from starlette.websockets import WebSocketState

# Open-source STT/TTS
from faster_whisper import WhisperModel
from pydub import AudioSegment
import pyttsx3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_stream")

app = FastAPI(title="Llama 3.3 70B Voice API")

SCRIPT_QUESTIONS = [
    "Do you own your home?",
    "Is your budget over $10,000?",
    "Are you looking to start within the next 3 months?",
]

POSITIVE_KEYWORDS = {
    "yes", "yep", "yeah", "sure", "of course", "affirmative", "correct", "i do", "i am"
}
NEGATIVE_KEYWORDS = {
    "no", "nope", "nah", "not really", "i don't", "i do not", "negative"
}


# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Models ───────────────────────────────────────────────────────────
whisper_model = None

@app.on_event("startup")
async def load_models():
    """Load Whisper model at startup"""
    global whisper_model
    logger.info("Loading Whisper model...")
    try:
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("✓ Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load Whisper model: {e}")
        raise


def build_dynamic_prompt(
    query: Optional[str] = None,
    system_prompt: Optional[str] = None,
    context: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Build a dynamic prompt by combining system instructions, context, metadata, and user query."""
    prompt_parts = []
    
    if system_prompt:
        prompt_parts.append(f"System Instructions: {system_prompt}")
    else:
        prompt_parts.append(
            "System Instructions: You are a helpful, friendly, and knowledgeable AI assistant. "
            "Answer the user's questions conversationally. Keep responses concise for voice interaction. "
            "If knowledgebase context is provided, use it to give accurate answers. "
            "If no context is provided, respond using your general knowledge. "
            "Always be polite and helpful."
        )
    
    if metadata:
        metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
        prompt_parts.append(f"Metadata: {metadata_str}")
    
    if context:
        context_section = "\n".join([f"Knowledgebase Passage {i+1}: {text}" for i, text in enumerate(context)])
        prompt_parts.append(f"Knowledgebase Context:\n{context_section}")
        prompt_parts.append("Use the above knowledgebase context to answer the user's query accurately.")
    
    if query:
        prompt_parts.append(f"User Query: {query}")
    else:
        prompt_parts.append("User Query: How can I assist you today?")
    
    return "\n\n".join(prompt_parts)


# ── Request Schema ──────────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: Optional[str] = None
    query: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    context: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None


@app.post("/ask")
async def ask_llama(request: PromptRequest):
    """Text-based chat endpoint"""
    try:
        user_query = request.query or request.prompt
        
        prompt_text = build_dynamic_prompt(
            query=user_query,
            system_prompt=request.system_prompt,
            context=request.context,
            metadata=request.metadata
        )

        options: Dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.top_p is not None:
            options["top_p"] = request.top_p

        payload: Dict[str, Any] = {
            "model": "qwen3-coder:480b-cloud",
            "prompt": prompt_text,
            "stream": False,
        }

        if options:
            payload["options"] = options

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        return {"response": result.get("response", "")}
        
    except requests.RequestException as e:
        logger.error(f"Ollama connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama Connection Error: {str(e)}")


# ── TTS with pyttsx3 (open-source/offline) ───────────────────────────────────

async def text_to_speech_bytes(text: str) -> bytes:
    """Convert text to audio bytes using the offline pyttsx3 engine."""

    def _generate() -> bytes:
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)

        engine = None
        try:
            logger.info(f"Generating TTS (offline) for text: '{text[:80]}...'")
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.save_to_file(text, wav_path)
            engine.runAndWait()

            audio = AudioSegment.from_file(wav_path, format="wav")
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            wav_bytes = buf.getvalue()

            logger.info(f"Generated {len(wav_bytes)} bytes of audio")
            return wav_bytes
        except Exception as e:
            logger.error(f"Offline TTS generation error: {e}")
            raise
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {wav_path}: {e}")

    return await asyncio.to_thread(_generate)


def interpret_yes_no(text: str) -> Optional[bool]:
    """Return True for yes, False for no, or None if unclear."""
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    def contains_any(phrases: set[str]) -> bool:
        return any(phrase in cleaned for phrase in phrases)

    if contains_any(POSITIVE_KEYWORDS):
        return True
    if contains_any(NEGATIVE_KEYWORDS):
        return False

    tokens = cleaned.replace("?", "").replace("!", "").split()
    if len(tokens) == 1 and tokens[0] in {"yes", "y", "yeah", "yep"}:
        return True
    if len(tokens) == 1 and tokens[0] in {"no", "n", "nope"}:
        return False

    return None


# ── STT with Faster Whisper (FIXED) ─────────────────────────────────────────

def convert_webm_to_wav(webm_data: bytes) -> bytes:
    """
    Convert WebM/Opus audio to WAV format.
    Saves WebM to a file first, then converts it.
    """
    webm_fd, webm_path = tempfile.mkstemp(suffix=".webm")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    
    try:
        os.close(webm_fd)
        os.close(wav_fd)
        
        # Write WebM data to file
        with open(webm_path, "wb") as f:
            f.write(webm_data)
        
        logger.info(f"Converting WebM file ({len(webm_data)} bytes) to WAV...")
        
        # Load WebM and convert to WAV
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio = audio.set_sample_width(2)  # 16-bit
        audio.export(wav_path, format="wav")
        
        # Read WAV
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        
        logger.info(f"Successfully converted to WAV ({len(wav_bytes)} bytes)")
        return wav_bytes
        
    finally:
        for path in [webm_path, wav_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass


def is_webm_format(data: bytes) -> bool:
    """Check if data is WebM format"""
    # WebM/Matroska magic number
    return data[:4] == b'\x1a\x45\xdf\xa3'


def is_wav_format(data: bytes) -> bool:
    """Check if data is WAV format"""
    return data[:4] == b'RIFF' and data[8:12] == b'WAVE'


def transcribe_audio(audio_data: bytes) -> str:
    """
    Transcribe audio bytes using Faster Whisper.
    Handles WebM, WAV, and raw PCM formats.
    """
    if len(audio_data) < 100:
        logger.warning(f"Audio data too small: {len(audio_data)} bytes")
        return ""
    
    logger.info(f"Received {len(audio_data)} bytes of audio data")
    logger.info(f"First 20 bytes (hex): {audio_data[:20].hex()}")
    
    try:
        # Determine format and convert if needed
        if is_webm_format(audio_data):
            logger.info("Detected WebM format")
            wav_bytes = convert_webm_to_wav(audio_data)
        elif is_wav_format(audio_data):
            logger.info("Detected WAV format")
            wav_bytes = audio_data
        else:
            # Try to detect if it's raw PCM by attempting WebM conversion
            logger.info("Unknown format, attempting WebM conversion...")
            try:
                wav_bytes = convert_webm_to_wav(audio_data)
            except Exception as e:
                logger.warning(f"WebM conversion failed: {e}, treating as raw PCM")
                # Create WAV from raw PCM (assume 16kHz, mono, 16-bit)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                wav_bytes = buf.getvalue()
        
        # Save WAV to temp file for Whisper
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        
        try:
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(wav_bytes)
            
            logger.info(f"Transcribing WAV file: {tmp_path} ({len(wav_bytes)} bytes)")
            
            # Transcribe with Whisper
            segments, info = whisper_model.transcribe(
                tmp_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_speech_duration_ms=250,
                    threshold=0.5
                )
            )
            
            # Combine all segments
            text_parts = []
            for seg in segments:
                text_parts.append(seg.text)
                logger.info(f"Segment: {seg.text}")
            
            text = " ".join(text_parts).strip()
            
            logger.info(f"Final transcription: '{text}'")
            return text
            
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return ""


# ── WebSocket Voice Streaming ───────────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_stream(ws: WebSocket):
    """Lead-qualification voice bot."""
    await ws.accept()
    logger.info("WebSocket connection established")
    audio_buffer = bytearray()
    session = {
        "step": 0,
        "completed": False,
        "started": False,
        "waiting_for_answer": False,
        "question_task": None,
        "current_question": "",
    }

    async def stream_voice_message(text: str):
        await ws.send_json({"type": "llm_text", "text": text})
        await ws.send_json({"type": "status", "text": "Speaking..."})
        await ws.send_json({"type": "audio_start"})

        audio_wav = await text_to_speech_bytes(text)
        CHUNK_SIZE = 16 * 1024
        for i in range(0, len(audio_wav), CHUNK_SIZE):
            await ws.send_bytes(audio_wav[i : i + CHUNK_SIZE])

        await ws.send_json({"type": "audio_end"})
        await ws.send_json({"type": "status", "text": "Ready"})

    def cancel_question_timeout():
        task = session.get("question_task")
        if task and not task.done():
            task.cancel()
        session["question_task"] = None

    def schedule_question_timeout():
        cancel_question_timeout()

        if session["completed"] or not session["waiting_for_answer"]:
            return

        async def waiter():
            try:
                await asyncio.sleep(10)
                if session["completed"] or not session["waiting_for_answer"]:
                    return
                reminder = "Just checking in—could you please answer the last question?"
                await stream_voice_message(reminder)
                session["question_task"] = None
                await ask_current_question(reminder=True)
                schedule_question_timeout()
            except asyncio.CancelledError:
                pass

        session["question_task"] = asyncio.create_task(waiter())

    async def ask_current_question(reminder: bool = False):
        if session["completed"] or not session["started"]:
            return
        if session["step"] < len(SCRIPT_QUESTIONS):
            question = SCRIPT_QUESTIONS[session["step"]]
            session["current_question"] = question
            prompt = question if not reminder else f"I'll repeat the question: {question}"
            await stream_voice_message(prompt)
            session["waiting_for_answer"] = True

    async def start_flow():
        if session["started"] or session["completed"]:
            return
        session["started"] = True
        await stream_voice_message(
            "Hi, thanks for calling Northern Renovations. I just need to ask three quick yes or no questions."
        )
        await ask_current_question()
        schedule_question_timeout()

    async def wrap_up(qualified: bool):
        session["completed"] = True
        session["waiting_for_answer"] = False
        cancel_question_timeout()
        if qualified:
            msg = (
                "Fantastic! You answered yes to all three questions, so you're a hot lead. "
                "I'll transfer you to a renovation specialist right away."
            )
            await stream_voice_message(msg)
            await ws.send_json({"type": "classification", "text": "Hot Lead"})
        else:
            msg = (
                "Thanks for your time. Based on your answers we aren't the right fit right now. "
                "If things change, feel free to reach out again."
            )
            await stream_voice_message(msg)
            await ws.send_json({"type": "classification", "text": "Not Qualified"})
        await ws.send_json({"type": "done"})

    async def handle_answer(answer_text: str):
        if session["completed"]:
            return

        if not session["started"]:
            await stream_voice_message("Please press the start button so we can begin the questions.")
            return

        if not session["waiting_for_answer"]:
            await stream_voice_message("Give me just a moment and I'll ask the next question.")
            return

        session["waiting_for_answer"] = False
        cancel_question_timeout()

        verdict = interpret_yes_no(answer_text)
        if verdict is None:
            await stream_voice_message("Please answer with a simple yes or no.")
            await ask_current_question()
            schedule_question_timeout()
            return

        if not verdict:
            await wrap_up(False)
            return

        session["step"] += 1
        if session["step"] >= len(SCRIPT_QUESTIONS):
            await wrap_up(True)
        else:
            await stream_voice_message("Great, thank you.")
            await ask_current_question()
            schedule_question_timeout()

    try:
        await ws.send_json({"type": "status", "text": "Press Start Bot to begin qualification."})

        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                logger.info("WebSocket disconnected")
                break

            try:
                message = await ws.receive()
            except RuntimeError as e:
                logger.warning(f"WebSocket receive error: {e}")
                break

            if isinstance(message, dict) and message.get("type") == "websocket.disconnect":
                logger.info("Received disconnect message")
                break

            if "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])
                logger.debug(
                    "Received audio chunk: %d bytes (buffer total: %d bytes)",
                    len(message["bytes"]),
                    len(audio_buffer),
                )
                continue

            if "text" in message and message["text"]:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message['text']}")
                    ctrl = {}

                action = ctrl.get("action", "")
                logger.info(f"Received action: {action}")

                if action == "start":
                    await start_flow()

                elif action == "end" and audio_buffer:
                    await ws.send_json({"type": "status", "text": "Transcribing..."})
                    audio_bytes = bytes(audio_buffer)
                    transcript = await asyncio.to_thread(transcribe_audio, audio_bytes)
                    audio_buffer.clear()
                    await ws.send_json({"type": "transcript", "text": transcript})

                    if transcript:
                        await handle_answer(transcript)
                    else:
                        await stream_voice_message("I didn't catch that. Let's try again.")
                        await ask_current_question()
                        schedule_question_timeout()

                elif action == "text_answer":
                    text_answer = ctrl.get("text", "")
                    await handle_answer(text_answer)

                elif action == "ping":
                    await ws.send_json({"type": "pong"})
                    logger.debug("Pong sent")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        cancel_question_timeout()
        logger.info("WebSocket connection closed")


# ── HTML UI Endpoints ───────────────────────────────────────────────────────

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve the text chat interface"""
    try:
        with open("static/chat.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: static/chat.html not found</h1>",
            status_code=404
        )


@app.get("/voice", response_class=HTMLResponse)
async def voice_ui():
    """Serve the voice chat interface"""
    try:
        with open("static/voice.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: static/voice.html not found</h1>",
            status_code=404
        )


# ── Health Check ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "running",
        "service": "Llama 3.3 Voice API",
        "stt": "Faster Whisper",
        "tts": "gTTS",
        "llm": "Qwen3-Coder (via Ollama)"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "endpoints": {
            "text_chat": "/ask",
            "voice_stream": "/ws/voice",
            "chat_ui": "/chat",
            "voice_ui": "/voice"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")