"""
MEMO-BOT Server (Lite Version)
- No PyTorch/TensorFlow dependencies
- Typhoon AI Chat + TTS only
- Direct video streaming (no object detection)
"""
import asyncio
import os
from aiohttp import web

# Only import lightweight services
from services.chat_service import ChatService
from services.tts_service import TTSService
from handlers.pi_handler import PiHandler
from handlers.web_handler import WebHandler

# ================= CONFIG =================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8765
DB_PATH = "my_db"

# ================= INITIALIZE SERVICES =================
print("Starting MEMO-BOT Server (Lite Mode)...")
print("Skipping AI Models (Medicine, Fall, Face)...")

# Initialize Chat & TTS only
chat_service = ChatService()
tts_service = TTSService()

# Pass None for AI services
face_service = None
medicine_service = None
fall_service = None

# ================= SHARED STATE =================
web_clients = set()

# PiHandler will receive None for AI services and skip processing
pi_handler = PiHandler(face_service, medicine_service, fall_service, web_clients, chat_service, tts_service)
web_handler = WebHandler(pi_handler, web_clients, chat_service, tts_service)

# ================= HTTP ROUTES =================
async def serve_index(request):
    """Serve the web control interface or handle implicit WebSocket connection"""
    # Check if this is a WebSocket request hitting the root URL
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        print("⚠️ Client connected to root URL, redirecting to WebSocket handler")
        return await websocket_handler(request)
        
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
    print("✅ System Ready (Lite Mode)!")
    
    await site.start()
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server Stopped")