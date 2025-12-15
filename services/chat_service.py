"""
Chat Service - Typhoon AI Integration
Uses OpenAI-compatible API for Thai LLM
"""
import aiohttp

TYPHOON_API_URL = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_API_KEY = "sk-lKSskWk00vQvR331ma2rRaNyHllbTBtbMM7Ix1K2sWPEht7v"

SYSTEM_PROMPT = """คุณคือ MEMO-BOT ผู้ช่วยดูแลผู้สูงอายุที่บ้าน 
คุณพูดภาษาไทยและเป็นมิตร ตอบสั้นๆ ไม่เกิน 2-3 ประโยค
ช่วยเรื่องการกินยา ความปลอดภัย และเป็นเพื่อนคุย"""


class ChatService:
    def __init__(self):
        self.conversation_history = []
        print("💬 Chat Service Ready (Typhoon AI)")
    
    async def chat(self, user_message: str) -> str:
        """
        Send message to Typhoon AI and get response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Keep only last 10 messages for context
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.conversation_history
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    TYPHOON_API_URL,
                    headers={
                        "Authorization": f"Bearer {TYPHOON_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "typhoon-v2-70b-instruct",
                        "messages": messages,
                        "max_tokens": 256,
                        "temperature": 0.7
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        ai_response = data["choices"][0]["message"]["content"]
                        
                        # Add AI response to history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": ai_response
                        })
                        
                        return ai_response
                    else:
                        error_text = await response.text()
                        print(f"Typhoon API Error: {response.status} - {error_text}")
                        return "ขออภัย ระบบมีปัญหาชั่วคราว"
        
        except Exception as e:
            print(f"Chat Error: {e}")
            return "ขออภัย ไม่สามารถเชื่อมต่อได้"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
