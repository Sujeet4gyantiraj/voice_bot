"""
Shared utilities for the voice bot.

This module contains common functionality used by both the main voice bot
and the demo script.
"""


def process_command(text):
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
