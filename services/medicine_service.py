
import cv2
import numpy as np
import time

class MedicineService:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        # State Tracking
        self.state = "IDLE" # IDLE, DETECTED, CONSUMING, COMPLETED
        self.state_start_time = 0
        self.last_seen_time = 0
        self.consecutive_frames = 0
        self.CONF_THRESH = 0.5
        
        print("💊 Medicine Service (Logic-Based) Ready")

    def reset_state(self):
        self.state = "IDLE"
        self.state_start_time = 0
        self.consecutive_frames = 0

    def process(self, frame):
        """
        Process frame with logic:
        1. Detect Medicine (Bottle/Pills)
        2. If seen consistently -> DETECTED
        3. If disappears after being seen -> CONSUMING? (Simplified for now: Just detect is enough for 'success' in this test?)
        
        Actually, let's make it simple for the user:
        - If 'medicine' class is detected for 3 seconds -> COMPLETED.
        """
        
        frame_result = frame.copy()
        detected = False
        
        try:
            # Run YOLO
            results = self.model(frame, device=self.device, verbose=False, conf=self.CONF_THRESH, half=(self.device=='cuda'))[0]
            
            if results.boxes:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    # Assuming class 0 is medicine/pill/bottle based on your model training
                    # Use all detections for now
                    detected = True
                    
                    # Draw
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{results.names[cls]} {float(box.conf[0]):.2f}"
                    cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- LOGIC ---
            if detected:
                if self.state == "IDLE":
                    self.state = "DETECTED"
                    self.state_start_time = time.time()
                elif self.state == "DETECTED":
                    # If seen for > 3 seconds, mark as COMPLETED (Success)
                    if time.time() - self.state_start_time > 3.0:
                        self.state = "COMPLETED"
                        self.state_start_time = time.time() # Update time for toast timeout
            else:
                # Reset if lost for too long? (Optional)
                if self.state == "DETECTED" and (time.time() - self.state_start_time > 5.0):
                     self.state = "IDLE"

        except Exception as e:
            print(f"Med Process Error: {e}")
            
        # Draw State
        color = (0, 255, 255) # Yellow
        if self.state == "COMPLETED": color = (0, 255, 0) # Green
        
        cv2.putText(frame_result, f"MED_STATE: {self.state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame_result
