"""
Face Recognition Service
Uses Hybrid Approach: OpenCV (Fast Detection) + DeepFace (ArcFace for Identification)
Includes Tracking, Locking, and Stability Logic
"""
import os
import time
import threading
import cv2
import numpy as np

# Force TensorFlow to use GPU before importing DeepFace
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from deepface import DeepFace

# ================= CONFIG =================
DB_PATH = "my_db"
CONFIRM_FRAMES = 5    # Frames to confirm identity
DISTANCE_THRESH = 100 # Tracking distance threshold
LOCK_DURATION = 60    # Seconds to keep locked identity (ล็อคไว้ 60 วินาที)
MAX_MISSING_FRAMES = 30  # Max frames before removing unlocked person

class TrackedPerson:
    def __init__(self, face_box):
        self.box = face_box       # (x, y, w, h)
        self.id_name = "Unknown"  # Predicted Name
        self.confidence_count = 0 # Match Counter
        self.is_locked = False    # Name Locked?
        self.missing_frames = 0   # For handling occlusion
        self.lock_time = None     # Time when locked (for persistent lock)

    def update_position(self, new_box):
        self.box = new_box
        self.missing_frames = 0

    def update_name(self, name):
        if self.is_locked: return

        if name == self.id_name:
            self.confidence_count += 1
        else:
            self.confidence_count = max(0, self.confidence_count - 1)
            if self.confidence_count == 0:
                self.id_name = name

        if self.confidence_count >= CONFIRM_FRAMES and self.id_name != "Unknown":
            self.is_locked = True
            self.lock_time = time.time()  # บันทึกเวลาที่ล็อค
            print(f"🔒 LOCKED: {self.id_name} (will stay locked for {LOCK_DURATION}s)")

class FaceService:
    def __init__(self, db_path="my_db"):
        self.db_path = db_path
        self.active_people = [] # List of TrackedPerson
        self.current_frame = None
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        self.results = [] # Stores (x, y, w, h, name, is_locked) for drawing
        
        # Load Face Detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create DB folder if not exists
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"⚠️ Created {db_path}. Please put face images inside.")
        
        # Preload DeepFace
        print("[System] Loading AI Database (ArcFace)...")
        try:
            # Dummy run to load model into GPU memory
            DeepFace.find(img_path=np.zeros((500,500,3), np.uint8), db_path=db_path, 
                          model_name="ArcFace", enforce_detection=False, silent=True)
            print("[System] DeepFace Ready!")
        except Exception as e:
            print(f"[System] Model Load Warning: {e}")
            
    def start(self):
        """Start the background face recognition worker"""
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("👤 Face Recognition Service Started (Hybrid Mode)")
    
    def stop(self):
        self._running = False
    
    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy()
    
    def get_results(self):
        """Returns tuple: (x, y, w, h, name) for drawing by main thread"""
        # Convert internal state to simple list for external constraints
        res = []
        with self.lock:
            for p in self.active_people:
                x, y, w, h = p.box
                # Format name for display
                if p.is_locked:
                    name = f"{p.id_name}"
                else:
                    name = "Checking..." if p.id_name == "Unknown" else f"{p.id_name} {p.confidence_count}/{CONFIRM_FRAMES}"
                res.append((x, y, w, h, name))
        return res
    
    def _worker(self):
        """Background worker loop"""
        while self._running:
            try:
                # 1. Get Frame
                with self.lock:
                    if self.current_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.current_frame.copy()

                # 2. Detect Faces (OpenCV - Fast)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                current_frame_people = []

                # 3. Match with Active People (Tracking)
                for (x, y, w, h) in faces:
                    center_x, center_y = x + w//2, y + h//2
                    matched_person = None

                    with self.lock:
                        for person in self.active_people:
                            px, py, pw, ph = person.box
                            p_center_x, p_center_y = px + pw//2, py + ph//2
                            
                            dist = np.sqrt((center_x - p_center_x)**2 + (center_y - p_center_y)**2)
                            if dist < DISTANCE_THRESH:
                                matched_person = person
                                matched_person.update_position((x, y, w, h))
                                break
                    
                    if matched_person is None:
                        new_person = TrackedPerson((x, y, w, h))
                        with self.lock:
                            self.active_people.append(new_person)
                        matched_person = new_person

                    # 4. Identification (DeepFace - Slow)
                    # Only check if not locked
                    if not matched_person.is_locked:
                        try:
                            if os.path.exists(self.db_path) and len(os.listdir(self.db_path)) > 0:
                                face_img = frame[y:y+h, x:x+w]
                                dfs = DeepFace.find(img_path=face_img, db_path=self.db_path, 
                                                    model_name="ArcFace", enforce_detection=False, silent=True)
                                
                                found_name = "Unknown"
                                if len(dfs) > 0:
                                    for df in dfs:
                                        if not df.empty:
                                            path = df.iloc[0]['identity']
                                            found_name = os.path.basename(path).split('.')[0]
                                            break
                                
                                with self.lock:
                                    matched_person.update_name(found_name)
                        except Exception:
                            pass
                    
                    current_frame_people.append(matched_person)

                # 5. Cleanup - Keep locked people longer
                with self.lock:
                    current_time = time.time()
                    cleaned_people = []
                    
                    for person in self.active_people:
                        in_frame = person in current_frame_people
                        
                        if in_frame:
                            # คนที่ยังอยู่ในเฟรม - เก็บไว้
                            person.missing_frames = 0
                            cleaned_people.append(person)
                        elif person.is_locked:
                            # คนที่ล็อคแล้วแต่หายไปจากเฟรม
                            person.missing_frames += 1
                            
                            # เช็คว่าล็อคหมดเวลาหรือยัง
                            lock_expired = (current_time - person.lock_time) > LOCK_DURATION if person.lock_time else False
                            too_long_missing = person.missing_frames > MAX_MISSING_FRAMES
                            
                            if lock_expired or too_long_missing:
                                print(f"🔓 UNLOCK: {person.id_name} (expired={lock_expired}, missing={too_long_missing})")
                            else:
                                # ยังล็อคอยู่ - เก็บไว้แม้ไม่เห็นหน้า
                                cleaned_people.append(person)
                        else:
                            # คนที่ยังไม่ล็อคและหายไป - ลบออก
                            person.missing_frames += 1
                            if person.missing_frames <= 5:  # ให้โอกาส 5 เฟรม
                                cleaned_people.append(person)
                    
                    self.active_people = cleaned_people

                time.sleep(0.01) # Small sleep to prevent CPU hogging
            
            except Exception as e:
                print(f"Face Worker Error: {e}")
                time.sleep(1)
