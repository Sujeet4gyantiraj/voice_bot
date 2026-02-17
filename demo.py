#!/usr/bin/env python3
"""
Demo script for the voice bot that simulates voice input.

This script demonstrates the voice bot functionality without requiring
actual microphone input, useful for testing and demonstration purposes.
"""

import pyttsx3
from bot_utils import process_command


class VoiceBotDemo:
    """A demo version of the voice bot that simulates user input."""
    
    def __init__(self):
        """Initialize the demo bot with text-to-speech engine."""
        # Initialize text-to-speech engine
        self.tts_available = True
        try:
            self.tts_engine = pyttsx3.init()
            # Configure TTS properties
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            print("Voice Bot Demo initialized with TTS!")
        except Exception as e:
            self.tts_available = False
            self.tts_engine = None
            print(f"Voice Bot Demo initialized (TTS not available: {e})")
            print("Running in text-only mode for demonstration.")
    
    def speak(self, text):
        """
        Convert text to speech.
        
        Args:
            text (str): The text to be spoken
        """
        print(f"Bot: {text}")
        if self.tts_available and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS Error: {e}]")
    
    def run_demo(self):
        """Run the demo with predefined inputs."""
        print("=" * 60)
        print("Voice Bot Demo - Simulating Voice Interactions")
        print("=" * 60)
        print()
        
        # Demo conversation scenarios
        demo_inputs = [
            "Hello",
            "What is your name?",
            "How are you?",
            "This is a test message",
            "Help",
            "Goodbye"
        ]
        
        self.speak("Hello! I am your voice bot. This is a demo of my capabilities.")
        print()
        
        for user_input in demo_inputs:
            print(f"\n[Simulated User Input]: {user_input}")
            print("-" * 60)
            
            # Process the command using the shared utility
            response, continue_conversation = process_command(user_input)
            
            # Speak the response
            self.speak(response)
            
            # Check if we should exit
            if not continue_conversation:
                break
            
            print()
        
        print("\n" + "=" * 60)
        print("Demo completed!")
        print("=" * 60)


def main():
    """Main entry point for the demo."""
    try:
        demo = VoiceBotDemo()
        demo.run_demo()
        
        print("\nNote: This is a demo version.")
        print("To use the full voice bot with microphone input, run: python voice_bot.py")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
