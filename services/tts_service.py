"""
TTS Service - Text to Speech
Uses edge-tts for high quality Thai speech synthesis
"""
import edge_tts
import io
import asyncio


class TTSService:
    def __init__(self, voice="th-TH-PremwadeeNeural"):
        # Thai voices: th-TH-PremwadeeNeural (female), th-TH-NiwatNeural (male)
        self.voice = voice
        print(f"🔊 TTS Service Ready (Voice: {voice})")
    
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio (MP3 bytes)
        """
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Collect audio chunks
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
            
            return audio_data.getvalue()
        
        except Exception as e:
            print(f"TTS Error: {e}")
            return b""
    
    async def list_voices(self):
        """List available Thai voices"""
        voices = await edge_tts.list_voices()
        thai_voices = [v for v in voices if v["Locale"].startswith("th-TH")]
        return thai_voices
