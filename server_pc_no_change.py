import asyncio
import websockets
import cv2
import numpy as np
import torch
import os
import threading
import time
from ultralytics import YOLO
from deepface import DeepFace

# ================= CONFIG =================
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8765
DB_PATH = "my_db"

# Model Paths (in models folder)
MODEL_MEDICINE = "models/best2.pt"       # โมเดลกินยาของคุณ
MODEL_POSE = "models/yolov8n-pose.pt"    # โมเดลจับคนล้ม

# ตั้งค่า GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 AI System Starting... Using: {DEVICE.upper()}")

# สร้างโฟลเดอร์ DB ถ้ายังไม่มี
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
    print(f"⚠️ Created {DB_PATH}. Please put images inside.")

# ================= SHARED VARIABLES =================
class SharedState:
    def __init__(self):
        self.current_frame = None
        self.face_names = []
        self.lock = threading.Lock()

state = SharedState()

# ================= WORKER: FACE RECOGNITION (ทำงานเบื้องหลัง) =================
def face_recognition_worker():
    """เธรดแยกสำหรับ Face Rec"""
    print("👤 Face Recognition Worker Started")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        try:
            img_to_process = None
            with state.lock:
                if state.current_frame is not None:
                    img_to_process = state.current_frame.copy()
            
            if img_to_process is None:
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            detected_results = []

            if len(faces) > 0:
                if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
                    for (x, y, w, h) in faces:
                        try:
                            face_roi = img_to_process[y:y+h, x:x+w]
                            results = DeepFace.find(
                                img_path=face_roi,
                                db_path=DB_PATH,
                                model_name="Facenet512",
                                detector_backend="skip",
                                enforce_detection=False,
                                silent=True
                            )
                            name = "Unknown"
                            if len(results) > 0 and not results[0].empty:
                                full_path = results[0].iloc[0]['identity']
                                name = os.path.splitext(os.path.basename(full_path))[0]
                            detected_results.append((x, y, w, h, name))
                        except: pass
                else:
                    for (x, y, w, h) in faces:
                        detected_results.append((x, y, w, h, "No DB"))

            with state.lock:
                state.face_names = detected_results
            time.sleep(0.05) 

        except Exception as e:
            print(f"Face Worker Error: {e}")
            time.sleep(1)

# เริ่ม Thread Face Rec
face_thread = threading.Thread(target=face_recognition_worker, daemon=True)
face_thread.start()

# ================= MAIN SERVER =================
print("Loading Models...")
# โหลด 2 โมเดลแยกกัน
model_med = YOLO(MODEL_MEDICINE) # AI กินยา
model_pose = YOLO(MODEL_POSE)    # AI คนล้ม

# ย้ายไป GPU ทั้งคู่
model_med.to(DEVICE)
model_pose.to(DEVICE)
print("✅ All Models Loaded on GPU.")

async def handler(ws):
    print(f"🔗 Pi Connected: {ws.remote_address}")
    prev_time = time.time()
    
    try:
        while True:
            # 1. รับภาพจาก Pi
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                np_data = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            except (asyncio.TimeoutError, Exception):
                continue

            if frame is None: continue

            # บังคับ Resize เป็น 640x480 เพื่อความชัวร์
            frame = cv2.resize(frame, (640, 480))

            # ส่งภาพเข้าตัวแปรกลางให้ Face Rec
            with state.lock:
                state.current_frame = frame.copy()

            # --- SCREEN 1: AI กินยา (best2.pt) ---
            # ใช้ model_med
            res_med = model_med(frame, device=DEVICE, verbose=False)[0]
            frame_med = res_med.plot()
            cv2.putText(frame_med, "Medicine AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # --- SCREEN 2: Fall Detection (Pose) ---
            frame_fall = frame.copy()
            # ใช้ model_pose
            res_pose = model_pose.track(frame_fall, persist=True, device=DEVICE, verbose=False, classes=0)[0]
            frame_fall = res_pose.plot()
            
            # Logic จับล้ม (ถ้านอนยาวกว่าตั้ง)
            if res_pose.boxes.id is not None:
                boxes = res_pose.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    if w > h * 1.2: # เงื่อนไขการล้ม
                        cv2.putText(frame_fall, "FALL DETECTED!", (x1, y1-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                        cv2.rectangle(frame_fall, (x1, y1), (x2, y2), (0,0,255), 4)
            cv2.putText(frame_fall, "Fall Detection AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- SCREEN 3: Face Recognition ---
            frame_face = frame.copy()
            current_faces = []
            with state.lock:
                current_faces = state.face_names
            
            for (x, y, w, h, name) in current_faces:
                color = (0, 255, 0) if name not in ["Unknown", "No DB"] else (0, 0, 255)
                cv2.rectangle(frame_face, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_face, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame_face, "Face Recognition AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- แสดงผล 3 จอ ---
            cv2.imshow("Screen 1: Medicine/Objects", frame_med)
            cv2.imshow("Screen 2: Fall Detection", frame_fall)
            cv2.imshow("Screen 3: Face Recognition", frame_face)

            # คำนวณ FPS รวม
            curr_time = time.time()
            # fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("🔴 Disconnected")

async def main():
    print(f"📡 Server started at ws://{SERVER_IP}:{SERVER_PORT}")
    async with websockets.serve(handler, SERVER_IP, SERVER_PORT, max_size=5_000_000):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped")