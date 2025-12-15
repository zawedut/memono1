"""
Web Handler - Handles WebSocket connections from web control interface
Receives commands and chat messages from web UI
"""

# Message Types
import aiohttp
MSG_IMAGE = 1
MSG_TEXT = 10      # Text chat message
MSG_ACTION = 11
MSG_AUDIO = 12     # Audio response


class WebHandler:
    def __init__(self, pi_handler, web_clients, chat_service=None, tts_service=None):
        self.pi_handler = pi_handler
        self.web_clients = web_clients
        self.chat_service = chat_service
        self.tts_service = tts_service
    
    async def handle(self, ws):
        """Handle web client WebSocket connection"""
        print(f"🌐 Web Client Connected")
        self.web_clients.add(ws)
        
        try:
            # Standard aiohttp WebSocket loop
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    if len(data) > 1:
                        header = data[0]
                        payload = data[1:].decode('utf-8')
                        
                        if header == MSG_ACTION:
                            if payload == "TEST_AUDIO":  # Special test command
                                print("🔊 Triggering Test Audio to Pi...")
                                if self.pi_handler:
                                    await self.pi_handler.send_test_audio()
                            else: # Normal Robot control command
                                print(f"🎮 Command: {payload}")
                                if self.pi_handler:
                                    self.pi_handler.queue_command(payload)
                        
                        elif header == MSG_TEXT:  # Chat message
                            print(f"💬 Chat: {payload}")
                            await self._handle_chat(ws, payload)
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'Web client connection closed with exception {ws.exception()}')
        
        except Exception as e:
            print(f"Web client error: {e}")
        finally:
            self.web_clients.discard(ws)
            print(f"🌐 Web Client Disconnected")
    
    async def _handle_chat(self, ws, user_message):
        """Process chat message through Typhoon AI and respond with TTS"""
        if not self.chat_service:
            return
        
        # Get AI response
        ai_response = await self.chat_service.chat(user_message)
        print(f"🤖 AI: {ai_response}")
        
        # Send text response back to web client
        text_msg = bytes([MSG_TEXT]) + ai_response.encode('utf-8')
        await ws.send_bytes(text_msg)
        
        # Generate TTS audio if available
        if self.tts_service:
            audio_data = await self.tts_service.synthesize(ai_response)
            if audio_data:
                audio_msg = bytes([MSG_AUDIO]) + audio_data
                
                # Send to web client
                await ws.send_bytes(audio_msg)
                print(f"🔊 Sent audio to Web ({len(audio_data)} bytes)")
                
                # ALSO send to Pi if PiHandler has an active connection
                if self.pi_handler and hasattr(self.pi_handler, 'ws') and self.pi_handler.ws:
                    try:
                        await self.pi_handler.ws.send_bytes(audio_msg)
                        print(f"📤 Sent audio to Pi ({len(audio_data)} bytes)")
                    except Exception as e:
                        print(f"⚠️ Failed to send audio to Pi: {e}")

