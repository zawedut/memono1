"""
MEMO-BOT Server (TEST VERSION)
Uses the new 'Medicine Poom' Logic-Based Service (YOLO-Pose Version)
Run this to test logic-based medicine detection
"""
# ============= GPU SETUP =============
from gpu_config import setup_gpu, check_gpu_status
DEVICE = setup_gpu()
check_gpu_status()

import asyncio
import os
from aiohttp import web
from ultralytics import YOLO

from services.face_service import FaceService
from services.medicine_service import MedicineService
# from services.medicine_poom import MedicinePoomService
from services.fall_service import FallService
from services.chat_service import ChatService
from services.tts_service import TTSService
from handlers.pi_handler import PiHandler
from handlers.web_handler import WebHandler

# ================= CONFIG =================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8766 
DB_PATH = "my_db"
MODEL_DIR = "models"
MED_MODEL_PATH = os.path.join(MODEL_DIR, "best2.pt")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-pose.pt")

print(f"\n🧪 MEMO-BOT TEST SERVER (Logic-Based Medicine) Starting...")

# ================= LOAD MODELS =================
print("Loading AI Models...")
model_med = YOLO(MED_MODEL_PATH)
model_med.to(DEVICE)

model_pose = YOLO(POSE_MODEL_PATH)
model_pose.to(DEVICE)

# ================= INITIALIZE SERVICES =================
face_service = FaceService(DB_PATH)

# Original Medicine Service
medicine_service = MedicineService(model_med, device=DEVICE)

fall_service = FallService(model_pose, DEVICE)
chat_service = ChatService()
tts_service = TTSService()

# Start face worker
face_service.start()

# ================= SHARED STATE =================
web_clients = set()
pi_handler = PiHandler(face_service, medicine_service, fall_service, web_clients, chat_service, tts_service)
web_handler = WebHandler(pi_handler, web_clients, chat_service, tts_service)

# ================= HTTP ROUTES =================
async def serve_index(request):
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        return await websocket_handler(request)
    return web.FileResponse('web/index.html')

async def serve_static(request):
    filename = request.match_info.get('filename', 'index.html')
    filepath = os.path.join('web', filename)
    if os.path.exists(filepath):
        return web.FileResponse(filepath)
    return web.Response(status=404, text="Not Found")

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

# ================= MAIN =================
async def main():
    app = web.Application()
    app.router.add_get('/', serve_index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/{filename}', serve_static)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, SERVER_IP, SERVER_PORT)
    
    print(f"📡 Test Server: http://{SERVER_IP}:{SERVER_PORT}")
    print(f"ℹ️  Port {SERVER_PORT} used for testing")
    print("✅ System Ready!")
    
    await site.start()
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test Server Stopped")
