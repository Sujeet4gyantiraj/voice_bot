#!/usr/bin/env python3
"""
List all available Edge TTS voices.
Run with: python list_voices.py
"""

import asyncio
import edge_tts


async def list_voices():
    """List all available voices from Edge TTS"""
    print("🎤 Available Edge TTS Voices\n")
    print("=" * 80)
    
    voices = await edge_tts.list_voices()
    
    # Filter for English voices
    en_voices = [v for v in voices if v["Locale"].startswith("en-")]
    
    print(f"\n📍 English Voices (Total: {len(en_voices)})\n")
    
    # Group by gender
    female_voices = [v for v in en_voices if v["Gender"] == "Female"]
    male_voices = [v for v in en_voices if v["Gender"] == "Male"]
    
    print(f"👩 FEMALE VOICES ({len(female_voices)}):")
    print("-" * 80)
    for voice in sorted(female_voices, key=lambda x: x["Locale"]):
        print(f"  {voice['ShortName']:<35} | {voice['Locale']:<10} | {voice['FriendlyName']}")
    
    print(f"\n👨 MALE VOICES ({len(male_voices)}):")
    print("-" * 80)
    for voice in sorted(male_voices, key=lambda x: x["Locale"]):
        print(f"  {voice['ShortName']:<35} | {voice['Locale']:<10} | {voice['FriendlyName']}")
    
    print("\n" + "=" * 80)
    print("\n💡 To change the voice, edit EDGE_TTS_VOICE in main.py")
    print("   Example: EDGE_TTS_VOICE = 'en-US-GuyNeural'")
    print("\n🎯 Recommended voices for business:")
    print("   • en-US-AriaNeural (Female, professional)")
    print("   • en-US-GuyNeural (Male, clear)")
    print("   • en-US-JennyNeural (Female, friendly)")
    print("   • en-US-ChristopherNeural (Male, authoritative)")


if __name__ == "__main__":
    asyncio.run(list_voices())
