#!/usr/bin/env python3
"""
Voice Bot with Speech-to-Text (STT) and Text-to-Speech (TTS)

This module implements a simple voice bot that can:
- Listen to user speech and convert it to text (STT)
- Process the text and generate responses
- Convert text responses back to speech (TTS)
"""

import speech_recognition as sr
import pyttsx3
import sys
from bot_utils import process_command, TTS_RATE, TTS_VOLUME, LISTEN_TIMEOUT, PHRASE_TIME_LIMIT


class VoiceBot:
    """A voice bot with STT and TTS capabilities."""
    
    def __init__(self):
        """Initialize the voice bot with speech recognition and text-to-speech engines."""
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS properties using constants
        self.tts_engine.setProperty('rate', TTS_RATE)
        self.tts_engine.setProperty('volume', TTS_VOLUME)
        
        print("Voice Bot initialized successfully!")
    
    def speak(self, text):
        """
        Convert text to speech.
        
        Args:
            text (str): The text to be spoken
        """
        print(f"Bot: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self):
        """
        Listen to user's speech and convert it to text.
        
        Returns:
            str: The recognized text, or None if recognition failed
        """
        with sr.Microphone() as source:
            print("Listening... (Speak now)")
            
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                # Listen to the audio using configurable timeouts
                audio = self.recognizer.listen(source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT)
                
                print("Processing speech...")
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
                
            except sr.WaitTimeoutError:
                print("Listening timed out. No speech detected.")
                return None
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service; {e}")
                return None
    
    def run(self):
        """Run the main voice bot loop."""
        self.speak("Hello! I am your voice bot. I can listen to you and respond. Say 'help' for more information or 'exit' to quit.")
        
        while True:
            # Listen to user
            user_input = self.listen()
            
            # Process the command using the shared utility
            response, continue_conversation = process_command(user_input)
            
            # Speak the response
            self.speak(response)
            
            # Check if we should exit
            if not continue_conversation:
                break


def main():
    """Main entry point for the voice bot."""
    try:
        print("=" * 50)
        print("Voice Bot with STT and TTS")
        print("=" * 50)
        print()
        
        # Create and run the voice bot
        bot = VoiceBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("\n\nVoice bot stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
