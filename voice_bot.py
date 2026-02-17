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


class VoiceBot:
    """A voice bot with STT and TTS capabilities."""
    
    def __init__(self):
        """Initialize the voice bot with speech recognition and text-to-speech engines."""
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS properties
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
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
                # Listen to the audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
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
    
    def process_command(self, text):
        """
        Process the user's command and generate a response.
        
        Args:
            text (str): The user's command in text form
            
        Returns:
            str: The bot's response
            bool: Whether to continue the conversation
        """
        if not text:
            return "I didn't catch that. Could you please repeat?", True
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for exit commands
        if any(word in words for word in ['exit', 'quit', 'bye', 'goodbye', 'stop']):
            return "Goodbye! Have a great day!", False
        
        # Check for greeting
        if any(word in words for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?", True
        
        # Check for name query
        if 'your name' in text_lower or 'who are you' in text_lower:
            return "I am a voice bot with speech to text and text to speech capabilities.", True
        
        # Check for help
        if 'help' in words:
            return "I can listen to your voice and respond back. Try saying hello, asking my name, or say goodbye to exit.", True
        
        # Check for how are you
        if 'how are you' in text_lower:
            return "I'm doing great! Thank you for asking. How can I assist you?", True
        
        # Default response
        return f"You said: {text}. I'm a simple bot, so I'm just echoing your words back to you.", True
    
    def run(self):
        """Run the main voice bot loop."""
        self.speak("Hello! I am your voice bot. I can listen to you and respond. Say 'help' for more information or 'exit' to quit.")
        
        while True:
            # Listen to user
            user_input = self.listen()
            
            # Process the command
            response, continue_conversation = self.process_command(user_input)
            
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
