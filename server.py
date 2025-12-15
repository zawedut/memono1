"""
MEMO-BOT Server
- HTTP server for web interface
- WebSocket server for Pi camera and web control
- Typhoon AI Chat + TTS
- Medicine Scheduler
"""
# ============= GPU SETUP (MUST BE FIRST!) =============
from gpu_config import setup_gpu, check_gpu_status
DEVICE = setup_gpu()
check_gpu_status()

import asyncio
import os
import json
import time
import webbrowser
from datetime import datetime
from aiohttp import web
from ultralytics import YOLO

from services.face_service import FaceService
from services.medicine_service import MedicineService
from services.fall_service import FallService
from services.chat_service import ChatService
from services.tts_service import TTSService
from handlers.pi_handler import PiHandler
from handlers.web_handler import WebHandler

# ================= CONFIG =================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8765
DB_PATH = "my_db"
MODEL_DIR = "models"
MED_MODEL_PATH = os.path.join(MODEL_DIR, "best2.pt")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-pose.pt")
SCHEDULE_FILE = "medicine_schedule.json"

# ================= SCHEDULER LOGIC =================
medicine_schedules = []

def load_schedules():
    global medicine_schedules
    if os.path.exists(SCHEDULE_FILE):
        try:
            with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
                medicine_schedules = json.load(f)
            print(f"📅 Loaded {len(medicine_schedules)} schedules")
        except Exception as e:
            print(f"⚠️ Schedule load error: {e}")

def save_schedules():
    try:
        with open(SCHEDULE_FILE, 'w', encoding='utf-8') as f:
            json.dump(medicine_schedules, f, indent=4)
    except Exception as e:
        print(f"⚠️ Schedule save error: {e}")

# ================= VERIFY GPU =================
print(f"\n🚀 MEMO-BOT Server Starting... Using: {DEVICE.upper()}")
if DEVICE != 'cuda':
    print("⚠️ WARNING: GPU not available! Performance will be slow.")
    print("   Make sure CUDA is installed and GPU drivers are up to date.")

# ================= LOAD MODELS =================
print("Loading AI Models...")
model_med = YOLO(MED_MODEL_PATH)
model_pose = YOLO(POSE_MODEL_PATH)
model_med.to(DEVICE)
model_pose.to(DEVICE)

# ================= INITIALIZE SERVICES =================
face_service = FaceService(DB_PATH)
# Use original Medicine Service (model-based) as requested in last step
medicine_service = MedicineService(model_med, DEVICE) 
fall_service = FallService(model_pose, DEVICE)
chat_service = ChatService()
tts_service = TTSService()

# Start face recognition worker
face_service.start()

# ================= SHARED STATE =================
web_clients = set()
pi_handler = PiHandler(face_service, medicine_service, fall_service, web_clients, chat_service, tts_service)
web_handler = WebHandler(pi_handler, web_clients, chat_service, tts_service)

# ================= HTTP ROUTES =================
async def serve_index(request):
    """Serve the web control interface or handle implicit WebSocket connection"""
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        return await websocket_handler(request)
    return web.FileResponse('web/index.html')

async def serve_static(request):
    """Serve static files from web folder"""
    filename = request.match_info.get('filename', 'index.html')
    filepath = os.path.join('web', filename)
    if os.path.exists(filepath):
        return web.FileResponse(filepath)
    return web.Response(status=404, text="Not Found")

# --- API ---
async def api_get_medicines(request):
    return web.json_response(medicine_schedules)

async def api_add_medicine(request):
    try:
        data = await request.json()
        med_name = data.get('name')
        med_time = data.get('time') # "HH:MM"
        med_details = data.get('details', '')
        
        if not med_name or not med_time:
            return web.Response(status=400, text="Missing name or time")
            
        new_item = {
            "id": int(time.time()),
            "name": med_name,
            "time": med_time,
            "details": med_details,
            "last_triggered": 0
        }
        medicine_schedules.append(new_item)
        save_schedules()
        return web.json_response(new_item)
    except Exception as e:
        return web.Response(status=500, text=str(e))

async def api_delete_medicine(request):
    med_id = int(request.match_info['id'])
    global medicine_schedules
    medicine_schedules = [m for m in medicine_schedules if m['id'] != med_id]
    save_schedules()
    return web.Response(text="Deleted")

async def api_test_medicine(request):
    try:
        med_id = int(request.match_info['id'])
        item = next((m for m in medicine_schedules if m['id'] == med_id), None)
        
        if not item:
            return web.Response(status=404, text="Schedule not found")
            
        print(f"🧪 TESTING MEDICINE SCHEDULE: {item['name']}")
        
        # 1. Activate Mode
        await pi_handler.enable_medicine_mode()
        
        # 2. Speak
        if tts_service:
            text = f"ทดสอบระบบกินยา นี่คือเสียงเตือนสำหรับ {item['name']} ค่ะ Testing alarm for {item['name']}."
            audio = await tts_service.synthesize(text)
            if audio and pi_handler.ws:
                try:
                    await pi_handler.ws.send_bytes(bytes([12]) + audio)
                except: pass
                
        return web.Response(text="Test Triggered")
    except Exception as e:
        return web.Response(status=500, text=str(e))

# ================= WEBSOCKET ROUTE =================
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    role = request.query.get('role', 'pi')
    if role == 'control':
        await web_handler.handle(ws)
    else:
        await pi_handler.handle(ws)
    return ws

# ================= BACKGROUND SCHEDULER =================
async def scheduler_loop():
    print("⏰ Scheduler Started")
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        for item in medicine_schedules:
            if item['time'] == current_time:
                # Check safeguards (don't trigger twice in same minute)
                timestamp = time.time()
                if timestamp - item.get('last_triggered', 0) > 60:
                    print(f"🔔 TRIGGER: {item['name']} at {current_time}")
                    item['last_triggered'] = timestamp
                    save_schedules() # Save timestamp
                    
                    # 1. Activate Medicine Mode
                    await pi_handler.enable_medicine_mode()
                    
                    # 2. Speak
                    if tts_service:
                        text = f"ได้เวลากินยา {item['name']} แล้วค่ะ It is time to take your {item['name']}."
                        # Send via WebSocket to ensure correct handling
                        # But pi_handler.enable_medicine_mode already speaks generic msg.
                        # Let's override or add specific msg.
                        # Since enable_medicine_mode sends "Time to take medicine", we can send this as extra Chat msg
                        
                        # Generate specific audio
                        audio = await tts_service.synthesize(text)
                        if audio and pi_handler.ws:
                             try:
                                 await pi_handler.ws.send_bytes(bytes([12]) + audio)
                             except: pass

        await asyncio.sleep(10) # Check every 10s

# ================= MAIN =================
async def main():
    load_schedules()
    
    app = web.Application()
    app.router.add_get('/', serve_index)
    app.router.add_get('/ws', websocket_handler)
    
    # API Routes
    app.router.add_get('/api/medicines', api_get_medicines)
    app.router.add_post('/api/medicines', api_add_medicine)
    app.router.add_delete('/api/medicines/{id}', api_delete_medicine)
    app.router.add_post('/api/medicines/{id}/test', api_test_medicine)
    
    app.router.add_get('/{filename}', serve_static)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, SERVER_IP, SERVER_PORT)
    
    print(f"📡 Server: http://{SERVER_IP}:{SERVER_PORT}")
    print(f"💬 Chat: Typhoon AI + TTS enabled")
    print("✅ System Ready!")
    
    # Auto-Open Browser
    try:
        webbrowser.open(f"http://localhost:{SERVER_PORT}")
    except: pass
    
    await site.start()
    
    # Run scheduler concurrently
    await scheduler_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server Stopped")