import aiohttp
import asyncio
import base64
import time
import cv2
import json
from secrets_config import LINE_ACCESS_TOKEN, LINE_USER_ID, IMGBB_API_KEY

class LineService:
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
        }
        self.line_api_url = "https://api.line.me/v2/bot/message/push"
        
        # Cooldown management
        self.last_sent = {
            'fall': 0,
            'medicine': 0
        }
        self.COOLDOWN = 180 # 3 minutes

    def _can_send(self, alert_type):
        """Check if enough time has passed since last alert of this type"""
        if time.time() - self.last_sent.get(alert_type, 0) > self.COOLDOWN:
            self.last_sent[alert_type] = time.time()
            return True
        return False

    async def send_message(self, text, alert_type=None):
        """Send a text message"""
        if alert_type and not self._can_send(alert_type):
            # Silently ignore during cooldown
            return

        payload = {
            "to": LINE_USER_ID,
            "messages": [{"type": "text", "text": text}]
        }
        await self._push_to_line(payload)

    async def send_image(self, frame, text, alert_type=None):
        """Upload image to ImgBB and send as image message"""
        if alert_type and not self._can_send(alert_type):
            # Silently ignore during cooldown
            return

        print(f"📤 Uploading Image for {alert_type}...")
        image_url = await self._upload_image(frame)
        
        if image_url:
            payload = {
                "to": LINE_USER_ID,
                "messages": [
                    {"type": "text", "text": text},
                    {
                        "type": "image",
                        "originalContentUrl": image_url,
                        "previewImageUrl": image_url
                    }
                ]
            }
            await self._push_to_line(payload)
        else:
            # Fallback to text if upload fails
            await self.send_message(f"{text} (Image upload failed)", alert_type=None)

    async def _upload_image(self, frame):
        """Upload frame to ImgBB and return URL"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            b64_image = base64.b64encode(buffer).decode('utf-8')
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "key": IMGBB_API_KEY,
                    "image": b64_image
                }
                async with session.post("https://api.imgbb.com/1/upload", data=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['data']['url']
                    else:
                        print(f"❌ ImgBB Upload Failed: {resp.status}")
                        return None
        except Exception as e:
            print(f"❌ Upload Error: {e}")
            return None

    async def _push_to_line(self, payload):
        """Push payload to Line Messaging API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.line_api_url, headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        print("✅ Line Message Sent")
                    else:
                        text = await resp.text()
                        print(f"❌ Line Error {resp.status}: {text}")
        except Exception as e:
            print(f"❌ Line Connection Error: {e}")
