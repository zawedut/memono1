"""
Face Recognition Service
Uses DeepFace with SSD backend for high accuracy face detection
"""
import os
import time
import threading
from deepface import DeepFace


class FaceService:
    def __init__(self, db_path="my_db"):
        self.db_path = db_path
        self.results = []
        self.current_frame = None
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        
        # Create DB folder if not exists
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"⚠️ Created {db_path}. Please put face images inside.")
    
    def start(self):
        """Start the background face recognition worker"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("👤 Face Recognition Service Started (SSD Backend)")
    
    def stop(self):
        """Stop the background worker"""
        self._running = False
    
    def update_frame(self, frame):
        """Update the current frame for processing"""
        with self.lock:
            self.current_frame = frame.copy()
    
    def get_results(self):
        """Get the latest face recognition results"""
        with self.lock:
            return self.results.copy()
    
    def _worker(self):
        """Background worker thread for face recognition"""
        while self._running:
            try:
                # Get frame copy
                frame_copy = None
                with self.lock:
                    if self.current_frame is not None:
                        frame_copy = self.current_frame.copy()
                
                if frame_copy is None:
                    time.sleep(0.1)
                    continue
                
                # Process face recognition
                if os.path.exists(self.db_path) and len(os.listdir(self.db_path)) > 0:
                    try:
                        dfs = DeepFace.find(
                            img_path=frame_copy,
                            db_path=self.db_path,
                            model_name="Facenet512",
                            detector_backend="ssd",
                            enforce_detection=False,
                            silent=True
                        )
                        
                        results_temp = []
                        if len(dfs) > 0:
                            for df in dfs:
                                if not df.empty:
                                    for _, row in df.iterrows():
                                        x = int(row['source_x'])
                                        y = int(row['source_y'])
                                        w = int(row['source_w'])
                                        h = int(row['source_h'])
                                        full_path = row['identity']
                                        name = os.path.splitext(os.path.basename(full_path))[0]
                                        results_temp.append((x, y, w, h, name))
                        
                        with self.lock:
                            self.results = results_temp
                    
                    except Exception:
                        with self.lock:
                            self.results = []
                else:
                    with self.lock:
                        self.results = []
                
                time.sleep(0.05)
            
            except Exception:
                time.sleep(1)
