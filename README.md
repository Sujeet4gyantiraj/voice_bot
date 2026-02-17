# Voice Bot with STT and TTS

A Python-based voice bot that implements Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities for interactive voice conversations.

## Features

- **Speech-to-Text (STT)**: Converts your voice input to text using Google Speech Recognition
- **Text-to-Speech (TTS)**: Converts bot responses to speech using pyttsx3
- **Interactive Conversation**: Continuous conversation loop with natural language processing
- **Voice Commands**: Supports various commands like greetings, help, and exit

## Requirements

- Python 3.7 or higher
- Microphone for voice input
- Speakers or headphones for audio output
- Internet connection (for Google Speech Recognition API)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sujeet4gyantiraj/voice_bot.git
cd voice_bot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Note for Linux Users

On Linux, you may need to install additional system dependencies for PyAudio:

```bash
# Ubuntu/Debian
sudo apt-get install python3-pyaudio portaudio19-dev

# Fedora
sudo dnf install python3-pyaudio portaudio-devel

# Arch Linux
sudo pacman -S python-pyaudio portaudio
```

### Note for macOS Users

On macOS, you may need to install PortAudio:

```bash
brew install portaudio
pip install pyaudio
```

## Usage

Run the voice bot:

```bash
python voice_bot.py
```

### Voice Commands

Once the bot is running, you can use the following commands:

- **"Hello" / "Hi" / "Hey"**: Greet the bot
- **"What is your name?" / "Who are you?"**: Ask about the bot
- **"How are you?"**: Ask how the bot is doing
- **"Help"**: Get information about available commands
- **"Exit" / "Quit" / "Bye" / "Goodbye" / "Stop"**: Exit the application

The bot will listen to your voice, convert it to text, process your command, and respond back with speech.

## How It Works

1. **Listening**: The bot uses the `speech_recognition` library to capture audio from your microphone
2. **Speech Recognition**: Audio is sent to Google Speech Recognition API to convert speech to text
3. **Processing**: The bot processes the recognized text and generates an appropriate response
4. **Text-to-Speech**: The response is converted to speech using the `pyttsx3` library
5. **Loop**: The conversation continues until you say an exit command

## Project Structure

```
voice_bot/
├── .gitignore          # Git ignore file for Python artifacts
├── README.md           # This file
├── bot_utils.py        # Shared utility functions for command processing
├── demo.py             # Demo script for testing without microphone
├── requirements.txt    # Python dependencies
└── voice_bot.py       # Main voice bot implementation
```

## Troubleshooting

### Microphone Not Working

- Make sure your microphone is properly connected and enabled
- Check your system's audio settings and permissions
- Try running the script with administrator/sudo privileges

### Speech Recognition Errors

- Ensure you have a stable internet connection (Google Speech Recognition requires internet)
- Speak clearly and avoid background noise
- Adjust the microphone sensitivity if needed

### PyAudio Installation Issues

- On Windows: Download and install the appropriate PyAudio wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
- On Linux/macOS: Follow the platform-specific installation instructions above

## Technologies Used

- **Python 3**: Programming language
- **SpeechRecognition**: Library for performing speech recognition with various engines
- **pyttsx3**: Text-to-speech conversion library (works offline)
- **PyAudio**: Library for audio I/O

## Future Enhancements

- Add support for multiple languages
- Implement custom wake word detection
- Add integration with AI models for more intelligent responses
- Support for different TTS voices and accents
- Add conversation history and context awareness

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Sujeet4gyantiraj