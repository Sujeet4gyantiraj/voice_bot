# Voice Bot - Lead Qualification System

AI-powered voice bot for automated lead qualification using speech-to-text, large language models, and high-quality text-to-speech.

## Features

- 🎤 **Speech-to-Text**: Faster Whisper (offline, high accuracy)
- 🔊 **Text-to-Speech**: Piper TTS (fast, local, offline neural voices)
- 🤖 **LLM Integration**: Ollama with Qwen3-Coder
- 📞 **Lead Qualification**: Automated 3-question screening process
- ⚡ **Real-time**: WebSocket-based voice streaming
- 📊 **Detailed Logging**: Complete timing metrics for all operations

## Tech Stack

- **Backend**: FastAPI
- **STT**: faster-whisper (small model, CPU-optimized)
- **TTS**: piper-tts (local neural TTS with automatic model download)
- **LLM**: Ollama (Qwen3-Coder 480B)
- **Audio**: pydub, AudioSegment
- **Frontend**: HTML5 + WebSocket + MediaRecorder API

## Installation

### 1. Prerequisites

```bash
# Install Ollama
curl -sSL https://ollama.com/install.sh | sh

# Pull the LLM model
ollama pull qwen3-coder:480b-cloud

# Start Ollama server
ollama serve
```

### 2. Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Using venv's python
venv/bin/python main.py

# OR after activating venv
python main.py

# OR using uvicorn directly
venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

## Usage

### Access Points

- **Voice Interface**: http://localhost:8000/voice
- **Text Chat**: http://localhost:8000/chat
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Voice Bot Flow

1. User opens `/voice` interface
2. Clicks "Start Bot"
3. Bot asks 3 qualification questions:
   - Do you own your home?
   - Is your budget over $10,000?
   - Are you looking to start within the next 3 months?
4. User responds via voice (automatically transcribed)
5. Bot classifies as "Hot Lead" or "Not Qualified"

## Customization

### Change TTS Voice

The voice bot supports multiple accents via a dropdown in the UI. To change the default voice, edit `DEFAULT_PIPER_VOICE` in `main.py`:

```python
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # Female US (default)
# OR
DEFAULT_PIPER_VOICE = "en_GB-alba-medium"   # Female UK
```

**Available Piper voices:**
- `en_US-lessac-medium` - Female US, professional
- `en_US-amy-medium` - Female US, clear
- `en_US-libritts_r-medium` - Male US, natural
- `en_GB-alba-medium` - Female UK
- `en_GB-northern_english_male-medium` - Male UK

**Voice models are downloaded automatically** on first use and cached locally in `~/.local/share/piper-voices/`

**To test voices:**
```bash
venv/bin/python3 test_piper.py
```

### Modify Questions

Edit the `SCRIPT_QUESTIONS` list in `main.py`:

```python
SCRIPT_QUESTIONS = [
    "Your first question?",
    "Your second question?",
    "Your third question?",
]
```

### Adjust Yes/No Detection

Modify keyword sets in `main.py`:

```python
POSITIVE_KEYWORDS = {"yes", "yep", "yeah", "sure", ...}
NEGATIVE_KEYWORDS = {"no", "nope", "nah", ...}
```

### Change Whisper Model

In the `load_models()` function, change model size:

```python
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
# Options: tiny, base, small, medium, large
```

## API Endpoints

### POST /ask
Text-based chat endpoint

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your company about?"}'
```

### WebSocket /ws/voice
Real-time voice interaction endpoint

## Logging

The application includes comprehensive logging:

- 🚀 Flow start/end
- ❓ Each question asked (with number: 1/3, 2/3, 3/3)
- 💬 User answers and interpretations
- ⏱️ Response times for each component:
  - STT transcription duration
  - TTS generation duration
  - LLM inference duration
  - Question-to-answer response time
  - Total session duration
- 🎯 Final qualification result

## Requirements

See `requirements.txt` for complete list:

- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- faster-whisper>=1.0.0
- piper-tts>=1.2.0
- pydub>=0.25.1
- requests>=2.31.0
- websockets>=12.0

## Troubleshooting

### piper-tts installation fails
Make sure you're using the virtual environment:
```bash
venv/bin/pip install piper-tts>=1.2.0
```

### First TTS generation is slow
Expected behavior - voice models are downloaded and cached on first use. Subsequent requests are much faster (2-3s vs 6-8s).

### Audio not playing in browser
Check browser console for errors. Ensure WebSocket connection is established.

### Whisper model download slow
First run downloads the model. Subsequent runs use cached version.

### Ollama connection refused
Ensure Ollama is running: `ollama serve`

### Voice model download fails
Check internet connection. Models are downloaded from Hugging Face on first use and cached locally.

## License

MIT

## Author

Voice Bot Lead Qualification System