from typing import Any, Dict, List, Optional
import asyncio
import io
import json
import tempfile
import wave
import os
import logging
import time
from pathlib import Path
import urllib.request

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from starlette.websockets import WebSocketState

# Open-source STT/TTS
from faster_whisper import WhisperModel
from pydub import AudioSegment
from piper import PiperVoice

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("voice_stream")

app = FastAPI(title="Open Source Voice Bot API", description="A voice-based lead qualification bot using open-source STT/TTS and Ollama LLM", version="1.0")

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
    start_time = time.time()
    try:
        user_query = request.query or request.prompt
        logger.info(f"📝 [TEXT CHAT] User query: '{user_query[:100]}...'")
        
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

        llm_start = time.time()
        logger.info("🤖 [LLM] Sending request to Ollama...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        llm_duration = time.time() - llm_start
        
        response_text = result.get("response", "")
        total_duration = time.time() - start_time
        
        logger.info(f"✅ [LLM] Response generated in {llm_duration:.2f}s")
        logger.info(f"📊 [TEXT CHAT] Total duration: {total_duration:.2f}s | Response length: {len(response_text)} chars")
        
        return {"response": response_text}
        
    except requests.RequestException as e:
        logger.error(f"❌ [LLM] Ollama connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama Connection Error: {str(e)}")


# ── TTS with Piper (Fast Local Neural TTS) ─────────────────────────────────────

# Voice models directory
PIPER_MODELS_DIR = Path.home() / ".local" / "share" / "piper-voices"
PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Available Piper voices with download URLs
PIPER_VOICES = {
    "en_US-lessac-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-amy-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-libritts_r-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
    "en_GB-alba-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alba/medium/en_GB-alba-medium.onnx",
    "en_GB-northern_english_male-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx",
}

DEFAULT_PIPER_VOICE = "en-US-AriaNeural"  # Default to match frontend dropdown

# Voice mapping for UI compatibility (Edge TTS names -> Piper names)
VOICE_MAPPING = {
    "en-US-AriaNeural": "en_US-lessac-medium",
    "en-US-GuyNeural": "en_US-libritts_r-medium",
    "en-US-JennyNeural": "en_US-amy-medium",
    "en-US-ChristopherNeural": "en_US-libritts_r-medium",
    "en-GB-SoniaNeural": "en_GB-alba-medium",
    "en-GB-RyanNeural": "en_GB-northern_english_male-medium",
    "en-AU-NatashaNeural": "en_US-amy-medium",  # Fallback to US
    "en-AU-WilliamNeural": "en_US-libritts_r-medium",  # Fallback to US
    "en-IN-NeerjaNeural": "en_US-lessac-medium",  # Fallback to US
    "en-IN-PrabhatNeural": "en_US-libritts_r-medium",  # Fallback to US
    "en-CA-ClaraNeural": "en_US-lessac-medium",  # Fallback to US
    "en-CA-LiamNeural": "en_US-libritts_r-medium",  # Fallback to US
}

# Cache loaded voices to avoid reloading
_voice_cache: Dict[str, PiperVoice] = {}

def download_voice_if_needed(voice_name: str) -> Path:
    """Download voice model if not already available."""
    model_path = PIPER_MODELS_DIR / f"{voice_name}.onnx"
    config_path = PIPER_MODELS_DIR / f"{voice_name}.onnx.json"
    
    if model_path.exists() and config_path.exists():
        return model_path
    
    if voice_name not in PIPER_VOICES:
        logger.warning(f"⚠️  [TTS] Unknown voice {voice_name}, using default")
        voice_name = DEFAULT_PIPER_VOICE
        model_path = PIPER_MODELS_DIR / f"{voice_name}.onnx"
        config_path = PIPER_MODELS_DIR / f"{voice_name}.onnx.json"
        
        if model_path.exists() and config_path.exists():
            return model_path
    
    # Download model and config
    logger.info(f"📥 [TTS] Downloading voice model: {voice_name}...")
    base_url = PIPER_VOICES[voice_name]
    
    try:
        # Download .onnx model
        urllib.request.urlretrieve(base_url, model_path)
        # Download .onnx.json config
        urllib.request.urlretrieve(base_url + ".json", config_path)
        logger.info(f"✅ [TTS] Downloaded {voice_name} successfully")
        return model_path
    except Exception as e:
        logger.error(f"❌ [TTS] Failed to download {voice_name}: {e}")
        raise

async def text_to_speech_bytes(text: str, voice: str = None) -> bytes:
    """Convert text to audio bytes using Piper TTS (fast, local, offline)."""
    tts_start = time.time()
    
    # Map voice name from Edge TTS to Piper
    selected_voice = voice or DEFAULT_PIPER_VOICE
    if selected_voice in VOICE_MAPPING:
        piper_voice = VOICE_MAPPING[selected_voice]
    else:
        piper_voice = selected_voice
    
    try:
        logger.info(f"🔊 [TTS] Generating speech with Piper ({piper_voice}) for: '{text[:80]}...'")
        
        # Load voice model (with caching)
        if piper_voice not in _voice_cache:
            model_path = await asyncio.get_event_loop().run_in_executor(
                None, download_voice_if_needed, piper_voice
            )
            _voice_cache[piper_voice] = PiperVoice.load(str(model_path))
            logger.info(f"✅ [TTS] Loaded voice model: {piper_voice}")
        
        voice_model = _voice_cache[piper_voice]
        
        # Generate speech
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        
        with wave.open(wav_path, "wb") as wav_file:
            await asyncio.get_event_loop().run_in_executor(
                None, voice_model.synthesize_wav, text, wav_file
            )
        
        # Read generated WAV
        audio = AudioSegment.from_file(wav_path, format="wav")
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        wav_bytes = buf.getvalue()
        
        # Clean up temp file
        try:
            os.unlink(wav_path)
        except:
            pass

        tts_duration = time.time() - tts_start
        logger.info(f"✅ [TTS] Generated {len(wav_bytes)} bytes in {tts_duration:.2f}s")
        return wav_bytes
        
    except Exception as e:
        logger.error(f"❌ [TTS] Piper generation error: {e}")
        raise


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


# ── Latency Reporting Functions ─────────────────────────────────────────────

def print_latency_breakdown(latencies: dict):
    """Print detailed latency breakdown for a single question-response cycle"""
    print("\n" + "="*80)
    print(f"⏱️  LATENCY BREAKDOWN - Question {latencies.get('question_num', 'N/A')}")
    print("="*80)
    
    # Question and Answer
    if "question" in latencies:
        print(f"❓ Question: {latencies['question']}")
    if "answer" in latencies:
        print(f"💬 Answer: {latencies['answer']}")
    
    print("\n📊 Timing Breakdown:")
    print("-"*80)
    
    # Individual components
    if "tts" in latencies:
        print(f"  🔊 Text-to-Speech (TTS):    {latencies['tts']:6.2f}s  (Bot speaking question)")
    if "stt" in latencies:
        print(f"  🎤 Speech-to-Text (STT):    {latencies['stt']:6.2f}s  (Transcribing user answer)")
    if "total_response" in latencies:
        print(f"  ⏰ Total Response Time:     {latencies['total_response']:6.2f}s  (Question asked → Answer received)")
    
    # Calculate waiting time (user thinking + speaking)
    if "total_response" in latencies and "tts" in latencies:
        thinking_time = latencies['total_response'] - latencies.get('stt', 0)
        print(f"  🤔 User Think + Speak Time: {thinking_time:6.2f}s  (After bot finished speaking)")
    
    print("="*80 + "\n")


def print_latency_summary(latency_log: list, total_time: float):
    """Print summary of all question-response latencies"""
    if not latency_log:
        return
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "📊 FINAL LATENCY SUMMARY" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n┌─────┬─────────────────────────────────┬────────┬────────┬────────┬──────────┐")
    print("│ Q # │ Question                        │  TTS   │  STT   │ Think  │ Response │")
    print("├─────┼─────────────────────────────────┼────────┼────────┼────────┼──────────┤")
    
    total_tts = 0
    total_stt = 0
    total_response = 0
    
    for log in latency_log:
        q_num = log.get('question_num', 'N/A')
        question = log.get('question', '')[:30] + "..." if len(log.get('question', '')) > 30 else log.get('question', '')
        tts = log.get('tts', 0)
        stt = log.get('stt', 0)
        response = log.get('total_response', 0)
        thinking = response - stt if response and stt else response
        
        total_tts += tts
        total_stt += stt
        total_response += response
        
        print(f"│  {q_num}  │ {question:<31} │ {tts:5.2f}s │ {stt:5.2f}s │ {thinking:5.2f}s │ {response:7.2f}s │")
    
    print("├─────┼─────────────────────────────────┼────────┼────────┼────────┼──────────┤")
    print(f"│ AVG │ Average per question            │ {total_tts/len(latency_log):5.2f}s │ {total_stt/len(latency_log):5.2f}s │ {(total_response-total_stt)/len(latency_log):5.2f}s │ {total_response/len(latency_log):7.2f}s │")
    print("└─────┴─────────────────────────────────┴────────┴────────┴────────┴──────────┘")
    
    print(f"\n🏁 Total Session Duration: {total_time:.2f}s")
    print(f"📝 Questions Asked: {len(latency_log)}")
    print(f"⚡ Average Response Time: {total_response/len(latency_log):.2f}s per question")
    print(f"🔊 Total TTS Time: {total_tts:.2f}s")
    print(f"🎤 Total STT Time: {total_stt:.2f}s")
    
    # Calculate efficiency
    processing_time = total_tts + total_stt
    user_time = total_response - total_stt
    print(f"\n💡 Performance Metrics:")
    print(f"   • Bot Processing: {processing_time:.2f}s ({processing_time/total_time*100:.1f}% of total)")
    print(f"   • User Interaction: {user_time:.2f}s ({user_time/total_time*100:.1f}% of total)")
    print("\n" + "="*80 + "\n")


def print_chat_latency(latencies: dict):
    """Print latency for chat mode interactions"""
    print("\n" + "="*70)
    print("💬 CHAT MODE LATENCY")
    print("="*70)
    
    if "user_message" in latencies:
        msg = latencies['user_message'][:50] + "..." if len(latencies.get('user_message', '')) > 50 else latencies.get('user_message', '')
        print(f"User: {msg}")
    
    print("\n📊 Processing Times:")
    print("-"*70)
    
    if "llm" in latencies:
        print(f"  🤖 LLM Processing:  {latencies['llm']:6.2f}s  (Generating response)")
    if "total" in latencies:
        print(f"  ⏰ Total Time:      {latencies['total']:6.2f}s  (Including TTS)")
    
    print("="*70 + "\n")


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
    stt_start = time.time()
    if len(audio_data) < 100:
        logger.warning(f"⚠️  [STT] Audio data too small: {len(audio_data)} bytes")
        return ""
    
    logger.info(f"🎤 [STT] Received {len(audio_data)} bytes of audio data")
    logger.info(f"🎤 [STT] First 20 bytes (hex): {audio_data[:20].hex()}")
    
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
            
            stt_duration = time.time() - stt_start
            logger.info(f"✅ [STT] Transcription completed in {stt_duration:.2f}s")
            logger.info(f"📝 [STT] Final transcription: '{text}'")
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
        "voice": DEFAULT_PIPER_VOICE,  # Default voice, can be changed by user
        "latency_log": [],  # Track latency for each question
        "current_latencies": {},  # Track current question's latencies
    }

    async def stream_voice_message(text: str):
        tts_start = time.time()
        await ws.send_json({"type": "llm_text", "text": text})
        await ws.send_json({"type": "status", "text": "Speaking..."})
        await ws.send_json({"type": "audio_start"})

        audio_wav = await text_to_speech_bytes(text, voice=session.get("voice", DEFAULT_PIPER_VOICE))
        CHUNK_SIZE = 16 * 1024
        for i in range(0, len(audio_wav), CHUNK_SIZE):
            await ws.send_bytes(audio_wav[i : i + CHUNK_SIZE])

        await ws.send_json({"type": "audio_end"})
        await ws.send_json({"type": "status", "text": "Ready"})
        
        # Track TTS latency
        tts_duration = time.time() - tts_start
        session["current_latencies"]["tts"] = tts_duration

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
            session["question_start_time"] = time.time()
            session["current_latencies"] = {"question_num": session["step"] + 1}  # Reset latency tracking
            prompt = question if not reminder else f"I'll repeat the question: {question}"
            logger.info(f"❓ [QUESTION {session['step'] + 1}/{len(SCRIPT_QUESTIONS)}] Asking: '{question}'")
            if reminder:
                logger.info(f"🔄 [REMINDER] Repeating question after timeout")
            await stream_voice_message(prompt)
            session["waiting_for_answer"] = True

    async def start_flow():
        if session["started"] or session["completed"]:
            return
        session["started"] = True
        session["flow_start_time"] = time.time()
        logger.info(f"🚀 [FLOW START] Beginning lead qualification process")
        await stream_voice_message(
            "Hi, thanks for calling Northern Renovations. I just need to ask three quick yes or no questions."
        )
        await ask_current_question()
        schedule_question_timeout()

    async def wrap_up(qualified: bool):
        session["completed"] = True
        session["waiting_for_answer"] = False
        cancel_question_timeout()
        
        total_flow_time = time.time() - session.get("flow_start_time", time.time())
        
        if qualified:
            msg = (
                "Fantastic! You answered yes to all three questions, so you're a hot lead. "
                "I'll transfer you to a renovation specialist right away. "
                "Thank you for your time!"
            )
            logger.info(f"🎯 [RESULT] HOT LEAD - Qualified in {total_flow_time:.2f}s")
            await stream_voice_message(msg)
            await ws.send_json({"type": "classification", "text": "Hot Lead"})
        else:
            msg = (
                "Thanks for your time. Based on your answers we aren't the right fit right now. "
                "If things change, feel free to reach out again. "
                "Have a great day!"
            )
            logger.info(f"❌ [RESULT] NOT QUALIFIED - Completed in {total_flow_time:.2f}s")
            await stream_voice_message(msg)
            await ws.send_json({"type": "classification", "text": "Not Qualified"})
        
        logger.info(f"🏁 [FLOW END] Total session duration: {total_flow_time:.2f}s")
        logger.info(f"✅ [SESSION END] Qualification complete, ending session automatically")
        
        # Print final latency summary
        print_latency_summary(session["latency_log"], total_flow_time)
        
        # End session automatically
        await ws.send_json({"type": "done"})
        await ws.send_json({"type": "status", "text": "Session ended. Refresh to start over."})

    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: LLM Chat Mode is DISABLED
    # Session ends automatically after qualification questions are complete.
    # The function below is kept for reference but is not called anymore.
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def handle_llm_chat(user_message: str):
        """Handle general conversation with LLM after qualification (DISABLED)"""
        chat_start = time.time()
        
        logger.info(f"💬 [CHAT] User: '{user_message}'")
        
        # Check for goodbye/exit phrases
        goodbye_phrases = {"bye", "goodbye", "thank you", "thanks", "that's all", "no more questions"}
        if any(phrase in user_message.lower() for phrase in goodbye_phrases) and len(user_message.split()) <= 3:
            farewell_msg = "You're welcome! Have a great day. Goodbye!"
            logger.info(f"👋 [CHAT] Ending conversation")
            await stream_voice_message(farewell_msg)
            await ws.send_json({"type": "done"})
            return
        
        # Query LLM
        try:
            llm_start = time.time()
            await ws.send_json({"type": "status", "text": "Thinking..."})
            
            system_prompt = (
                "You are a helpful renovation company assistant. You work for Northern Renovations. "
                "Keep responses conversational, friendly, and concise (2-3 sentences max for voice). "
                "If asked about services, mention home renovations, remodeling, and improvements."
            )
            
            prompt_text = build_dynamic_prompt(
                query=user_message,
                system_prompt=system_prompt
            )
            
            payload = {
                "model": "qwen3-coder:480b-cloud",
                "prompt": prompt_text,
                "stream": False,
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            llm_response = result.get("response", "I'm sorry, I didn't catch that.")
            llm_duration = time.time() - llm_start
            
            logger.info(f"🤖 [CHAT] LLM response in {llm_duration:.2f}s: '{llm_response[:100]}...'")
            await stream_voice_message(llm_response)
            
            # Print chat latency
            total_chat_time = time.time() - chat_start
            chat_latencies = {
                "user_message": user_message,
                "llm": llm_duration,
                "total": total_chat_time
            }
            print_chat_latency(chat_latencies)
            
        except Exception as e:
            logger.error(f"❌ [CHAT] LLM error: {e}")
            await stream_voice_message("I'm having trouble processing that. Could you try again?")
    
    async def handle_answer(answer_text: str):
        # Session ends automatically after qualification is complete
        if session["completed"]:
            return

        if not session["started"]:
            await stream_voice_message("Please press the start button so we can begin the questions.")
            return

        if not session["waiting_for_answer"]:
            await stream_voice_message("Give me just a moment and I'll ask the next question.")
            return

        response_time = time.time() - session.get("question_start_time", time.time())
        session["waiting_for_answer"] = False
        cancel_question_timeout()
        
        # Store latencies for this question-response cycle
        session["current_latencies"]["total_response"] = response_time
        session["current_latencies"]["question"] = session["current_question"]
        session["current_latencies"]["answer"] = answer_text

        logger.info(f"💬 [ANSWER] User said: '{answer_text}'")
        logger.info(f"⏱️  [TIMING] Response time: {response_time:.2f}s from question asked")
        
        # Print detailed latency breakdown
        print_latency_breakdown(session["current_latencies"])

        verdict = interpret_yes_no(answer_text)
        if verdict is None:
            logger.info(f"❔ [ANSWER] Unclear response, asking for clarification")
            await stream_voice_message("Please answer with a simple yes or no.")
            await ask_current_question()
            schedule_question_timeout()
            return

        logger.info(f"{'✅' if verdict else '❌'} [ANSWER] Interpreted as: {'YES' if verdict else 'NO'}")
        
        # Save latency log before moving to next question
        session["latency_log"].append(session["current_latencies"].copy())

        if not verdict:
            logger.info(f"🚫 [QUALIFICATION] User disqualified at question {session['step'] + 1}")
            await wrap_up(False)
            return

        session["step"] += 1
        if session["step"] >= len(SCRIPT_QUESTIONS):
            logger.info(f"🎉 [QUALIFICATION] All questions answered YES - Hot Lead!")
            await wrap_up(True)
        else:
            logger.info(f"➡️  [FLOW] Moving to next question ({session['step'] + 1}/{len(SCRIPT_QUESTIONS)})")
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

                elif action == "set_voice":
                    new_voice = ctrl.get("voice", DEFAULT_PIPER_VOICE)
                    session["voice"] = new_voice
                    logger.info(f"🎙️ [VOICE] Voice changed to: {new_voice}")
                    await ws.send_json({"type": "status", "text": f"Voice set to {new_voice}"})

                elif action == "end" and audio_buffer:
                    processing_start = time.time()
                    logger.info(f"🎙️  [AUDIO] Processing {len(audio_buffer)} bytes of recorded audio")
                    await ws.send_json({"type": "status", "text": "Transcribing..."})
                    audio_bytes = bytes(audio_buffer)
                    transcript = await asyncio.to_thread(transcribe_audio, audio_bytes)
                    audio_buffer.clear()
                    processing_duration = time.time() - processing_start
                    logger.info(f"⏱️  [PROCESSING] Audio-to-text took {processing_duration:.2f}s")
                    await ws.send_json({"type": "transcript", "text": transcript})
                    
                    # Track STT latency
                    session["current_latencies"]["stt"] = processing_duration

                    if transcript:
                        await handle_answer(transcript)
                    else:
                        logger.warning(f"⚠️  [STT] Empty transcript received")
                        if not session.get("completed"):
                            await stream_voice_message("I didn't catch that. Let's try again.")
                            await ask_current_question()
                            schedule_question_timeout()
                        else:
                            await ws.send_json({"type": "status", "text": "Session ended."})

                elif action == "text_answer":
                    text_answer = ctrl.get("text", "")
                    await handle_answer(text_answer)

                elif action == "stop":
                    logger.info("🛑 [STOP] User requested to stop the bot")
                    await ws.send_json({"type": "done"})
                    break

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
        "tts": "Microsoft Edge TTS (edge-tts)",
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