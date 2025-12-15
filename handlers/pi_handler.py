"""
Pi Handler - Handles WebSocket connection from Raspberry Pi
Receives video frames, processes through AI services, broadcasts to web clients
"""
import asyncio
import aiohttp
import cv2
import numpy as np
import time
import speech_recognition as sr
from services.line_service import LineService


class PiHandler:
    def __init__(self, face_service, medicine_service, fall_service, web_clients, chat_service=None, tts_service=None):
        self.face_service = face_service
        self.medicine_service = medicine_service
        self.fall_service = fall_service
        self.web_clients = web_clients
        self.chat_service = chat_service
        self.tts_service = tts_service
        self.line_service = LineService()
        self.command_queue = asyncio.Queue()  # Commands from web to send to Pi
        
        # Audio Buffering
        self.audio_buffer = bytearray()
        self.last_audio_time = 0

        # Smart Scheduler State
        self.medicine_active = False
        self.medicine_start_time = 0
        self.MEDICINE_TIMEOUT = 300 # 5 Minutes Auto-Stop
        
        # Runtime Toggles
        self.use_face = True
        self.use_fall = True
        self.use_med = True
    
    async def handle(self, ws):
        """Handle Pi WebSocket connection"""
        print(f"🔗 Pi Connected")
        self.ws = ws # Store for proactively sending messages
        self.audio_buffer = bytearray() # Clear buffer on new connection
        
        # Broadcast Connection to Web
        await self.broadcast_toast("Robot Connected!", "success")
        await self.broadcast_robot_status(True)
        

        
        # Start command sender task
        sender_task = asyncio.create_task(self._send_commands(ws))
        # Start audio processor task
        audio_task = asyncio.create_task(self._process_audio_buffer(ws))
        prev_time = time.time()
        
        try:
            # Standard aiohttp WebSocket loop
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    frame_data = msg.data
                    
                    # Decode frame
                    np_data = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    
                    # Resize
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Update face service with new frame if available
                    if self.face_service and self.use_face:
                        self.face_service.update_frame(frame)
                    
                    # Process through AI services if available, otherwise just use raw frame
                    frame_med = frame.copy()
                    frame_fall = frame.copy()
                    frame_face = frame.copy()
                    
                    if self.medicine_service:
                        # Only run Medicine Service if Active (Scheduled)
                        if self.medicine_active and self.use_med:
                            frame_med = self.medicine_service.process(frame)
                            # Show Active Status
                            cv2.putText(frame_med, "Medicine Mode: ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            
                            # AUTO-STOP LOGIC
                            # 1. Check Timeout
                            if time.time() - self.medicine_start_time > self.MEDICINE_TIMEOUT:
                                await self.disable_medicine_mode("Timeout")
                                await self.line_service.send_message(
                                    "⚠️ แจ้งเตือน: ไม่ได้รับประทานยาตามเวลาที่กำหนด (เกิน 5 นาที)",
                                    alert_type='medicine'
                                )
                                
                            # 2. Check Logic Completion (If using Poom Service)
                            elif hasattr(self.medicine_service, 'state') and self.medicine_service.state in ["COMPLETED", "DONE", 5]:
                                # Wait a moment to show success message then stop
                                if time.time() - self.medicine_service.state_start_time > 3.0:
                                    # Send Success Toast
                                    await self.broadcast_toast("Medicine Taken Successfully!", "success")
                                    await self.disable_medicine_mode("Medicine Taken")
                        
                        else:
                            # If inactive, just draw status overlay
                            pass

                        
                    if self.fall_service and self.use_fall:
                        frame_fall = self.fall_service.process(frame)
                        if self.fall_service.any_fall_detected:
                            print("🚨 FALL DETECTED -> SENDING ALERT")
                            await self.line_service.send_image(
                                frame_fall,
                                "🚨 แจ้งเตือน: ตรวจพบคนล้ม! (Fall Detected)",
                                alert_type='fall'
                            )
                            # TTS Alert for Fall
                            if self.tts_service:
                                tts_audio = await self.tts_service.synthesize("แจ้งเตือน ตรวจพบการล้มค่ะ Warning! Fall Detected.")
                                if tts_audio:
                                    try: await self.ws.send_bytes(bytes([12]) + tts_audio)
                                    except: pass
                    
                    # Draw face results if available
                    # Draw face results if available
                    if self.face_service and self.use_face:
                        for (x, y, w, h, name) in self.face_service.get_results():
                            cv2.rectangle(frame_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame_face, name, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Calculate FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time
                    
                    # Draw FPS on all frames
                    frames_to_show = [(frame_med, "Medicine")]
                    if self.fall_service:
                        frames_to_show.append((frame_fall, "Fall"))
                    if self.face_service:
                        frames_to_show.append((frame_face, "Face"))
                        
                    for img, label in frames_to_show:
                        cv2.putText(img, f"{label} | FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show windows
                    try:
                        # PERSISTENT STATUS BAR
                        h, w = frame_med.shape[:2]
                        cv2.rectangle(frame_med, (0, h-40), (w, h), (0, 0, 0), -1)
                        status_text = f"[1] FACE: {'ON' if self.use_face else 'OFF'}  |  [2] FALL: {'ON' if self.use_fall else 'OFF'}  |  [3] MED: {'ON' if self.use_med else 'OFF'}"
                        cv2.putText(frame_med, status_text, (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        cv2.imshow("Screen 1: Main View", frame_med)
                        if self.fall_service:
                            cv2.imshow("Screen 2: Fall Detection", frame_fall)
                        if self.face_service:
                            cv2.imshow("Screen 3: Face Recognition", frame_face)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('1'): self.use_face = not self.use_face
                        elif key == ord('2'): self.use_fall = not self.use_fall
                        elif key == ord('3'): self.use_med = not self.use_med
                    except Exception as e:
                        print(f"Display Error: {e}")
                    
                    # Broadcast to web clients (send medicine frame)
                    await self._broadcast_frame(frame_med)

                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        import json
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "audio":
                            # Handle Audio from Pi - Add to buffer
                            payload_hex = data.get("payload", "")
                            try:
                                # Convert hex to bytes
                                audio_bytes = bytes.fromhex(payload_hex)
                                self.audio_buffer.extend(audio_bytes)
                                self.last_audio_time = time.time()
                                print(f"🎤 Rcv {len(audio_bytes)}b | Buff: {len(self.audio_buffer)}b", end="\r")
                            except Exception as e:
                                print(f"\n❌ Audio decode error: {e}")
                                
                        elif msg_type == "chat":
                            # If Pi sends text directly
                            text = data.get("payload", "")
                            print(f"📩 Chat from Pi: {text}")
                            
                            # Broadcast to web
                            if self.web_clients:
                                user_msg = bytes([10]) + f"User (Pi): {text}".encode('utf-8')
                                for client in self.web_clients:
                                    try:
                                        await client.send_bytes(user_msg)
                                    except Exception:
                                        pass
                                        
                            # Process with AI
                            if self.chat_service:
                                ai_response = await self.chat_service.chat(text)
                                print(f"🤖 AI Response: {ai_response}")
                                
                                # Broadcast AI response to Web
                                ai_msg = bytes([10]) + f"AI: {ai_response}".encode('utf-8')
                                for client in self.web_clients:
                                    try:
                                        await client.send_bytes(ai_msg)
                                    except Exception:
                                        pass

                                # Generate TTS and Send to Pi
                                if self.tts_service:
                                    tts_audio = await self.tts_service.synthesize(ai_response)
                                    if tts_audio:
                                        try:
                                            print(f"📤 Sending TTS to Pi ({len(tts_audio)} bytes)")
                                            await ws.send_bytes(bytes([12]) + tts_audio)
                                        except Exception as e:
                                            print(f"Failed to send TTS to Pi: {e}")
                                        
                                        # Also broadcast to Web
                                        if self.web_clients:
                                            web_audio_msg = bytes([12]) + tts_audio
                                            for client in self.web_clients:
                                                try:
                                                    await client.send_bytes(web_audio_msg)
                                                except Exception:
                                                    pass
                                
                    except Exception as e:
                        print(f"JSON Parse Error: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'❌ WebSocket connection closed with exception {ws.exception()}')
        
        except Exception as e:
            import traceback
            print(f"❌ Pi Error: {e}")
            traceback.print_exc()
        finally:
            sender_task.cancel()
            audio_task.cancel()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("🔴 Pi Disconnected")
            await self.broadcast_robot_status(False)
            await self.broadcast_toast("Robot Disconnected", "error")

    async def broadcast_robot_status(self, connected: bool):
        """Send Robot Status to Web"""
        # Header 21 = ROBOT_STATUS
        payload = json.dumps({"type": "status", "robot_connected": connected}).encode('utf-8')
        msg = bytes([21]) + payload
        
        disconnected = set()
        for client in set(self.web_clients):
            try: await client.send_bytes(msg)
            except: disconnected.add(client)
        self.web_clients -= disconnected

    async def _process_audio_buffer(self, ws):
        """Monitor audio buffer and process when silence detected"""
        
        recognizer = sr.Recognizer()
        
        while True:
            await asyncio.sleep(0.5) # Check every 0.5s
            
            # If we have audio and it's been > 1.5s since last packet
            if len(self.audio_buffer) > 0 and (time.time() - self.last_audio_time > 1.5):
                try:
                    print(f"\n🎤 Processing Audio Buffer ({len(self.audio_buffer)} bytes)...")
                    
                    # Convert raw bytes to AudioData
                    audio_data = sr.AudioData(bytes(self.audio_buffer), 16000, 2)
                    
                    # 1. Send Audio to Web for Playback (WAV)
                    wav_data = audio_data.get_wav_data()
                    if self.web_clients:
                        # Header=12 (MSG_AUDIO) + WAV data
                        audio_msg = bytes([12]) + wav_data
                        for client in self.web_clients:
                            try:
                                await client.send_bytes(audio_msg)
                            except Exception:
                                pass
                    
                    self.audio_buffer = bytearray() # Clear buffer immediately
                    
                    # 2. Run STT
                    text = await self._transcribe_audio(recognizer, audio_data)
                    
                    if text:
                        print(f"🗣️ Transcribed: {text}")
                        
                        # Broadcast transcribed text to Web
                        if self.web_clients:
                            user_msg = bytes([10]) + f"User (Voice): {text}".encode('utf-8')
                            for client in self.web_clients:
                                try:
                                    await client.send_bytes(user_msg)
                                except Exception:
                                    pass
                        
                        # Send to Typhoon AI
                        if self.chat_service:
                            # 2.1 Get Visual Context (Face ID)
                            visible_context = None
                            if self.face_service and self.face_service.get_results():
                                names = [r[4] for r in self.face_service.get_results()] # Index 4 is name
                                if names:
                                    visible_context = f"Visible people: {', '.join(names)}"
                                    print(f"👀 Context: {visible_context}")

                            ai_response = await self.chat_service.chat(text, context=visible_context)
                            print(f"🤖 AI Response: {ai_response}")
                            
                            # Broadcast AI response text to Web
                            ai_msg = bytes([10]) + f"AI: {ai_response}".encode('utf-8')
                            for client in self.web_clients:
                                try:
                                    await client.send_bytes(ai_msg)
                                except Exception:
                                    pass

                            # 3. Generate TTS and Send to Pi
                            if self.tts_service:
                                print(f"🔊 Generating TTS for: '{ai_response[:20]}...'")
                                tts_audio = await self.tts_service.synthesize(ai_response)
                                if tts_audio:
                                    # Send to Pi (Header 12 for Audio)
                                    try:
                                        msg = bytes([12]) + tts_audio
                                        print(f"📤 Sending TTS to Pi ({len(tts_audio)} bytes) [Header 12]")
                                        await ws.send_bytes(msg)
                                        print("✅ TTS Sent to Pi successfully")
                                    except Exception as e:
                                        print(f"❌ Failed to send TTS to Pi: {e}")
                                    
                                    # Also broadcast to Web so user can hear AI
                                    if self.web_clients:
                                        web_audio_msg = bytes([12]) + tts_audio
                                        for client in self.web_clients:
                                            try:
                                                await client.send_bytes(web_audio_msg)
                                            except Exception:
                                                pass
                                else:
                                    print("⚠️ TTS Synthesis returned empty audio")
                            else:
                                print("⚠️ TTS Service is NOT available")
                                    
                except Exception as e:
                    print(f"Error processing audio buffer: {e}")
                    import traceback
                    traceback.print_exc()
                    self.audio_buffer = bytearray() # output clearing on error

    async def _transcribe_audio(self, recognizer, audio_data):
        """Run STT in thread"""
        loop = asyncio.get_event_loop()
        try:
            # Use Google Speech Recognition (Free, works well for testing)
            # language='th-TH' for Thai
            text = await loop.run_in_executor(None, lambda: recognizer.recognize_google(audio_data, language='th-TH'))
            return text
        except Exception:
            # print(f"STT Error: {e}") # Usually "UnknownValueError" if speech not clear
            return None
            
    async def broadcast_toast(self, message, type="info"):
        """Send Toast notification to Web Clients"""
        # Protocol: Type 20 = JSON Toast
        payload = json.dumps({"type": "toast", "message": message, "level": type}).encode('utf-8')
        msg = bytes([20]) + payload
        for client in self.web_clients:
            try: await client.send_bytes(msg)
            except: pass
            
    async def broadcast_robot_status(self, connected: bool):
        """Send Robot Status to Web"""
        # Header 21 = ROBOT_STATUS
        payload = json.dumps({"type": "status", "robot_connected": connected}).encode('utf-8')
        msg = bytes([21]) + payload
        
        disconnected = set()
        for client in set(self.web_clients):
            try: await client.send_bytes(msg)
            except: disconnected.add(client)
        self.web_clients -= disconnected
    
    async def _broadcast_frame(self, frame):
        """Broadcast frame to all web clients"""
        if not self.web_clients:
            return
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # Create message: [Header=1 (IMAGE)] + [JPEG data]
        msg = bytes([1]) + buffer.tobytes()
        
        # Send to all web clients
        disconnected = set()
        # Use set copy
        for client in set(self.web_clients):
            try:
                await client.send_bytes(msg)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.web_clients -= disconnected
    
    async def _send_commands(self, ws):
        """Send queued commands to Pi"""
        while True:
            try:
                command = await self.command_queue.get()
                # Create message: [Header=11 (ACTION)] + [command string]
                msg = bytes([11]) + command.encode()
                await ws.send_bytes(msg)
                print(f"📤 Sent to Pi: {command}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Command send error: {e}")

    async def send_test_audio(self):
        """Manually trigger TTS to test audio path"""
        if not self.ws or not self.tts_service:
            print("⚠️ Cannot test audio: Pi not connected or TTS missing")
            return
        
        print("🧪 Generating Test Audio...")
        tts_audio = await self.tts_service.synthesize("ระบบทดสอบเสียง พร้อมใช้งานค่ะ")
        if tts_audio:
            try:
                msg = bytes([12]) + tts_audio
                print(f"📤 Sending Test Audio to Pi ({len(tts_audio)} bytes)")
                await self.ws.send_bytes(msg)
            except Exception as e:
                print(f"❌ Test Audio Send Failed: {e}")
    
    async def enable_medicine_mode(self):
        """Turn on Medicine Detection"""
        if self.medicine_active: return
        
        print("⏰ STARTING MEDICINE MODE")
        self.medicine_active = True
        self.medicine_start_time = time.time()
        
        # Reset Logic Service if available
        if hasattr(self.medicine_service, 'reset_state'):
            self.medicine_service.reset_state()
            
        # Voice Announcement
        if self.tts_service:
            await self.ws.send_bytes(bytes([12]) + await self.tts_service.synthesize("ได้เวลากินยาแล้วค่ะ Please take your medicine."))

    def disable_medicine_mode(self, reason="Stop"):
        """Turn off Medicine Detection"""
        if not self.medicine_active: return
        
        print(f"🛑 STOPPING MEDICINE MODE ({reason})")
        self.medicine_active = False
        
        # Notify Web? (Optional)

    def queue_command(self, command):
        """Queue a command to send to Pi"""
        self.command_queue.put_nowait(command)
