"""
MEMO-BOT Local Simulator (Web + Voice + AI)
- ACTS AS both the Server AND the Robot (Pi)
- Hosts the Web Interface on localhost:8080
- Uses Webcam as Robot Camera
- Plays TTS Audio through PC Speakers
"""
# ============= GPU SETUP =============
from gpu_config import setup_gpu, check_gpu_status
DEVICE = setup_gpu()
check_gpu_status()

import asyncio
import cv2
import os
import time
import json
import webbrowser
import tempfile
import threading
from aiohttp import web
from ultralytics import YOLO

# Import Services
from services.face_service import FaceService
from services.medicine_service import MedicineService
from services.fall_service import FallService
from services.chat_service import ChatService
from services.tts_service import TTSService
from services.line_service import LineService
from handlers.web_handler import WebHandler

# ================= CONFIG =================
SERVER_PORT = 8080
DB_PATH = "my_db"
MODEL_DIR = "models"
MED_MODEL_PATH = os.path.join(MODEL_DIR, "best2.pt")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-pose.pt")

# ================= LOCAL ROBOT HANDLER =================
class LocalRobotHandler:
    """Mocks PiHandler but runs locally on PC"""
    def __init__(self, face_service, medicine_service, fall_service, web_clients, chat_service, tts_service):
        self.face_service = face_service
        self.medicine_service = medicine_service
        self.fall_service = fall_service
        self.web_clients = web_clients
        self.chat_service = chat_service
        self.tts_service = tts_service
        
        # State
        self.medicine_active = False
        self.medicine_start_time = 0
        self.MEDICINE_TIMEOUT = 180
        
        # Toggles
        self.use_face = True
        self.use_fall = True
        self.use_med = True
        
        self.line_service = LineService()
        self.cap = None
        self.running = True
        self.ws = None # Virtual WS for compatibility

    async def enable_medicine_mode(self):
        print("\n⏰ [SIM] STARTING MEDICINE MODE")
        self.medicine_active = True
        self.medicine_start_time = time.time()
    async def enable_medicine_mode(self):
        if not self.use_med: return
        print("\n⏰ [SIM] STARTING MEDICINE MODE")
        self.medicine_active = True
        self.medicine_start_time = time.time()
        await self.speak("ทดสอบระบบ ได้เวลากินยาแล้วค่ะ Time to take medicine.")

    async def disable_medicine_mode(self, reason="Stop"):
        print(f"\n🛑 [SIM] STOPPING MEDICINE MODE ({reason})")
        self.medicine_active = False

    async def speak(self, text):
        """Generate and Play Audio Locally"""
        print(f"🔊 [SIM-TTS] '{text}'")
        if self.tts_service:
            audio_data = await self.tts_service.synthesize(text)
            if audio_data:
                # 1. Play Locally (Windows) - edge-tts outputs MP3, not WAV
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        f.write(audio_data)
                        temp_path = f.name
                    
                    # Play in thread to not block using pygame (supports MP3)
                    def _play():
                        try:
                            import pygame
                            pygame.mixer.init()
                            pygame.mixer.music.load(temp_path)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                            pygame.mixer.quit()
                        except ImportError:
                            # Fallback to ffplay if pygame not available
                            os.system(f'ffplay -nodisp -autoexit -loglevel quiet "{temp_path}"')
                        finally:
                            try: os.remove(temp_path)
                            except: pass
                    threading.Thread(target=_play).start()
                except Exception as e:
                    print(f"Audio Play Error: {e}")
                
                # 2. Send to Web (so user sees/hears it in browser too)
                msg = bytes([12]) + audio_data
                for client in self.web_clients:
                    try: await client.send_bytes(msg)
                    except: pass

    def queue_command(self, command):
        print(f"🎮 [SIM-CMD] Executing: {command}")
    def queue_command(self, command):
        print(f"🎮 [SIM-CMD] Executing: {command}")
        
        if command == "TOGGLE_FACE": self.use_face = not self.use_face
        elif command == "TOGGLE_FALL": self.use_fall = not self.use_fall
        elif command == "TOGGLE_MED": self.use_med = not self.use_med
        else:
             pass # Mock movement
        
    async def run_camera_loop(self):
        print("📷 Opening Webcam...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            # --- AI PROCESSING ---
            # 1. Face (Always On)
            # --- AI PROCESSING ---
            # 1. Face (Always On if enabled)
            if self.face_service and self.use_face:
                self.face_service.update_frame(frame)
            
            # 2. Medicine (Scheduled)
            frame_med = frame.copy()
            if self.medicine_active and self.medicine_service:
                frame_med = self.medicine_service.process(frame)
                # Auto-Stop Check
                if time.time() - self.medicine_start_time > self.MEDICINE_TIMEOUT:
                    await self.disable_medicine_mode("Timeout")
                    # Line Alert
                    await self.line_service.send_message("⚠️ SIM: Missed Medicine (Timeout)", alert_type='medicine')

                # Show Active Status (Requested)
                if self.medicine_active:
                     cv2.putText(frame_med, "Medicine Mode: ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                # Mock Success Logic (if gesture detected)
                if hasattr(self.medicine_service, 'state') and self.medicine_service.state == "COMPLETED":
                    await self.broadcast_toast("Medicine Taken!", "success")
                    await self.disable_medicine_mode("Completed")
                    
            elif not self.medicine_active and self.use_med:
                pass


            # 3. Fall (Always On if enabled)
            if self.fall_service and self.use_fall:
                 # Calculate fall on MAIN FRAME (frame_med) so we can see result
                 frame_med = self.fall_service.process(frame_med)
                 
                 if self.fall_service.any_fall_detected:
                      print("🚨 SIM: FALL DETECTED")
                      await self.line_service.send_image(frame_med, "🚨 SIM: Fall Detected!", alert_type='fall')


            # --- DRAW RESULTS ---
            # PERSISTENT STATUS BAR (Requested)
            # Draw black background bar at bottom
            h, w = frame_med.shape[:2]
            cv2.rectangle(frame_med, (0, h-40), (w, h), (0, 0, 0), -1)
            
            # Text Status
            status_text = f"[1] FACE: {'ON' if self.use_face else 'OFF'}  |  [2] FALL: {'ON' if self.use_fall else 'OFF'}  |  [3] MED: {'ON' if self.use_med else 'OFF'}"
            cv2.putText(frame_med, status_text, (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Overlay Face on Med Frame for demo

            if self.face_service:
                for (x, y, w, h, name) in self.face_service.get_results():
                    cv2.rectangle(frame_med, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_med, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # --- BROADCAST TO WEB ---
            if self.web_clients:
                _, buffer = cv2.imencode('.jpg', frame_med, [cv2.IMWRITE_JPEG_QUALITY, 50])
                msg = bytes([1]) + buffer.tobytes()
                
                disconnected = set()
                # Use list/set copy to avoid "Set changed size" runtime error
                for client in set(self.web_clients):
                    try: await client.send_bytes(msg)
                    except: disconnected.add(client)
                self.web_clients -= disconnected
            
            # Show CV2 Window (Optional)
            # Show CV2 Window
            cv2.imshow("MEMO-BOT Simulator", frame_med)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('1'):
                self.use_face = not self.use_face
                print(f"👉 Toggle Face: {self.use_face}")
            elif key == ord('2'):
                self.use_fall = not self.use_fall
                print(f"👉 Toggle Fall: {self.use_fall}")
            elif key == ord('3'):
                self.use_med = not self.use_med
                if self.use_med: self.medicine_active = True # Auto start when toggled on via key
                else: self.medicine_active = False
                print(f"👉 Toggle Med: {self.use_med}")
            elif key == ord('t'): # TEST LINE API
                print("📨 Sending Test Line Message...")
                await self.line_service.send_message("✅ MEMO-BOT API TEST: LINE OK!", alert_type='test')
                await self.broadcast_toast("Test Line Message Sent!", "info")
                
            await asyncio.sleep(0.01) # Yield
            
        self.cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera Stopped")

    async def send_test_audio(self):
         await self.speak("Test Audio System OK")

    async def broadcast_toast(self, message, type="info"):
        payload = json.dumps({"type": "toast", "message": message, "level": type}).encode('utf-8')
        msg = bytes([20]) + payload
        for client in self.web_clients:
            try: await client.send_bytes(msg)
            except: pass

# ================= SETUP =================
print("\n🚀 INITIALIZING LOCAL SIMULATOR...")
print(f"   Using Device: {DEVICE}")

# Load Models
print("📦 Loading Models...")
model_med = YOLO(MED_MODEL_PATH).to(DEVICE)
model_pose = YOLO(POSE_MODEL_PATH).to(DEVICE)

# Init Services
face_service = FaceService(DB_PATH)
medicine_service = MedicineService(model_med, DEVICE)
fall_service = FallService(model_pose, DEVICE)
chat_service = ChatService()
tts_service = TTSService()
face_service.start()

# Handlers
web_clients = set()
local_robot = LocalRobotHandler(face_service, medicine_service, fall_service, web_clients, chat_service, tts_service)
web_handler = WebHandler(local_robot, web_clients, chat_service, tts_service)

# ================= SERVER ROUTES =================
medicine_schedules = [] # Mock DB

async def index(request):
    return web.FileResponse('web/index.html')

async def serve_static(request):
    filename = request.match_info.get('filename', 'index.html')
    path = os.path.join('web', filename)
    if os.path.exists(path): return web.FileResponse(path)
    return web.Response(status=404)

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    role = request.query.get('role', 'control') # Always control in sim
    await web_handler.handle(ws)
    return ws

# Mock API
async def api_get(req): return web.json_response(medicine_schedules)
async def api_post(req):
    data = await req.json()
    data['id'] = int(time.time())
    medicine_schedules.append(data)
    return web.json_response(data)
async def api_delete(req):
    mid = int(req.match_info['id'])
    global medicine_schedules
    medicine_schedules = [m for m in medicine_schedules if m['id'] != mid]
    return web.Response(text="Deleted")
async def api_test(req):
    mid = int(req.match_info['id'])
    item = next((m for m in medicine_schedules if m['id'] == mid), None)
    if item:
        await local_robot.enable_medicine_mode()
        await local_robot.speak(f"Alert for {item['name']}")
    return web.Response(text="OK")


async def scheduler_loop():
    print("⏰ Scheduler Task Started!")
    while True:
        try:
            current_time = time.strftime("%H:%M")
            # Check Schedules
            # Structure: [{'id': 123, 'time': '14:30', 'name': 'Vit C'}]
            global medicine_schedules
            for item in medicine_schedules:
                if item.get('time') == current_time:
                    # Trigger only if not active (simple logic)
                    if not local_robot.medicine_active:
                        print(f"⏰ TRIGGER SCHEDULE: {item['name']}")
                        await local_robot.enable_medicine_mode()
                        await local_robot.speak(f"Time for {item['name']}")
                        # Prevent multiple triggers in same minute? (Ideally track last_trigger)
                        # For sim, 60s sleep is easiest hack or just re-trigger (idempotent-ish)
                        await asyncio.sleep(61) 

            await asyncio.sleep(5)
        except Exception as e:
            print(f"Scheduler Error: {e}")
            await asyncio.sleep(5)

# ================= RUN =================
async def main():
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/api/medicines', api_get)
    app.router.add_post('/api/medicines', api_post)
    app.router.add_delete('/api/medicines/{id}', api_delete)
    app.router.add_post('/api/medicines/{id}/test', api_test)
    app.router.add_get('/{filename}', serve_static)
    
    # Start Background Scheduler
    asyncio.create_task(scheduler_loop())

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', SERVER_PORT)
    await site.start()
    
    print(f"\n✅ SIMULATOR READY: http://localhost:{SERVER_PORT}")
    try:
        webbrowser.open(f"http://localhost:{SERVER_PORT}")
    except: pass
    
    # Run Tasks
    await local_robot.run_camera_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
