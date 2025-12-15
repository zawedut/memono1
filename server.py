"""
MEMO-BOT Server
- HTTP server for web interface
- WebSocket server for Pi camera and web control
- Typhoon AI Chat + TTS
"""
import asyncio
import os
import torch
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

# ================= GPU SETTINGS =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 MEMO-BOT Server Starting... Using: {DEVICE.upper()}")

if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True

# ================= LOAD MODELS =================
print("Loading AI Models...")
model_med = YOLO(MED_MODEL_PATH)
model_pose = YOLO(POSE_MODEL_PATH)
model_med.to(DEVICE)
model_pose.to(DEVICE)

# ================= INITIALIZE SERVICES =================
face_service = FaceService(DB_PATH)
medicine_service = MedicineService(model_med, DEVICE)
fall_service = FallService(model_pose, DEVICE)
chat_service = ChatService()
tts_service = TTSService()

# Start face recognition worker
face_service.start()

# ================= SHARED STATE =================
web_clients = set()
pi_handler = PiHandler(face_service, medicine_service, fall_service, web_clients)
web_handler = WebHandler(pi_handler, web_clients, chat_service, tts_service)

# ================= HTTP ROUTES =================
async def serve_index(request):
    """Serve the web control interface"""
    return web.FileResponse('web/index.html')

async def serve_static(request):
    """Serve static files from web folder"""
    filename = request.match_info.get('filename', 'index.html')
    filepath = os.path.join('web', filename)
    if os.path.exists(filepath):
        return web.FileResponse(filepath)
    return web.Response(status=404, text="Not Found")

# ================= WEBSOCKET ROUTE =================
async def websocket_handler(request):
    """WebSocket handler - routes to Pi or Web handler based on role"""
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
    
    print(f"📡 Server: http://{SERVER_IP}:{SERVER_PORT}")
    print(f"💬 Chat: Typhoon AI + TTS enabled")
    print("✅ System Ready!")
    
    await site.start()
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server Stopped")