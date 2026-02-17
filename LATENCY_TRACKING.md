# 📊 Latency Tracking & Performance Monitoring

## Overview

The voice bot now includes comprehensive latency tracking that prints detailed timing information for every question and response.

## What Gets Tracked

### 1. **Per-Question Breakdown** ⏱️

For each question in the qualification flow, you'll see:

- **TTS (Text-to-Speech)**: Time taken to generate and stream bot's question
- **STT (Speech-to-Text)**: Time taken to transcribe user's answer
- **User Think Time**: Time between bot finishing speaking and user responding
- **Total Response Time**: Complete time from question asked to answer received

### 2. **Final Summary Report** 📊

After qualification completes, you'll see:

- Table with all questions and their individual timings
- Average times for each metric
- Total session duration
- Performance metrics showing bot processing vs user interaction time

### 3. **Chat Mode Latency** 💬

For conversations after qualification:

- LLM processing time
- Total response time including TTS

## Example Output

### Individual Question Breakdown

```
================================================================================
⏱️  LATENCY BREAKDOWN - Question 1
================================================================================
❓ Question: Do you own your home?
💬 Answer: yes I do

📊 Timing Breakdown:
--------------------------------------------------------------------------------
  🔊 Text-to-Speech (TTS):     1.85s  (Bot speaking question)
  🎤 Speech-to-Text (STT):     1.23s  (Transcribing user answer)
  ⏰ Total Response Time:      6.45s  (Question asked → Answer received)
  🤔 User Think + Speak Time:  5.22s  (After bot finished speaking)
================================================================================
```

### Final Summary

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📊 FINAL LATENCY SUMMARY                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────┬─────────────────────────────────┬────────┬────────┬────────┬──────────┐
│ Q # │ Question                        │  TTS   │  STT   │ Think  │ Response │
├─────┼─────────────────────────────────┼────────┼────────┼────────┼──────────┤
│  1  │ Do you own your home?           │  1.85s │  1.23s │  5.22s │   6.45s │
│  2  │ Is your budget over $10,000?    │  2.10s │  1.15s │  4.80s │   5.95s │
│  3  │ Are you looking to start wit... │  2.45s │  0.98s │  5.30s │   6.28s │
├─────┼─────────────────────────────────┼────────┼────────┼────────┼──────────┤
│ AVG │ Average per question            │  2.13s │  1.12s │  5.11s │   6.23s │
└─────┴─────────────────────────────────┴────────┴────────┴────────┴──────────┘

🏁 Total Session Duration: 42.35s
📝 Questions Asked: 3
⚡ Average Response Time: 6.23s per question
🔊 Total TTS Time: 6.40s
🎤 Total STT Time: 3.36s

💡 Performance Metrics:
   • Bot Processing: 9.76s (23.0% of total)
   • User Interaction: 32.59s (77.0% of total)

================================================================================
```

### Chat Mode Latency

```
======================================================================
💬 CHAT MODE LATENCY
======================================================================
User: What services do you offer?

📊 Processing Times:
----------------------------------------------------------------------
  🤖 LLM Processing:   3.45s  (Generating response)
  ⏰ Total Time:       5.60s  (Including TTS)
======================================================================
```

## How to Use

### Run the Bot

```bash
venv/bin/python main.py
```

### Watch the Console

All latency information is automatically printed to the console (stdout) as the conversation progresses:

1. **Real-time tracking** - See breakdown after each question
2. **Final summary** - Complete report when qualification ends
3. **Chat tracking** - Individual latencies for each chat interaction

### Analyze Performance

Use the latency data to:

- **Identify bottlenecks**: Which component is slowest?
- **Optimize TTS**: Try different voices or adjust settings
- **Monitor STT**: Check if transcription is taking too long
- **User experience**: Track how long users take to respond
- **System performance**: Compare sessions over time

## Metrics Explained

### TTS (Text-to-Speech)
Time from starting to generate audio until all chunks are sent to client. Includes:
- Edge-TTS API call
- Audio format conversion
- Streaming to WebSocket

### STT (Speech-to-Text)
Time from receiving audio data until transcription is complete. Includes:
- Audio format detection and conversion
- Faster Whisper model inference
- Text extraction

### User Think Time
Time between bot finishing speaking and user starting to answer. Indicates:
- User comprehension time
- Response formulation time
- Actual speaking time

### Total Response Time
Complete cycle from question asked to answer received. Most important metric for user experience.

### LLM Processing (Chat Mode)
Time for Ollama to generate response using Qwen3-Coder model.

## Performance Tips

### Faster TTS
- Use simpler voice models (some accents are faster)
- Reduce text length where possible

### Faster STT
- Use smaller Whisper model: `tiny`, `base`, or `small`
- Ensure good audio quality (reduces retry attempts)

### Faster LLM
- Use smaller/quantized models
- Optimize system prompt
- Enable GPU acceleration if available

## Log Files

Latency information is also logged with timestamps:

```bash
2026-02-17 14:30:23 - voice_stream - INFO - ⏱️  [TIMING] Response time: 6.45s from question asked
```

Check your console output for the visual tables and complete latency reports!
