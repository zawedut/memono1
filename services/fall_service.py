"""
Fall Detection Service
Uses YOLO pose model to detect if a person has fallen
With history tracking, height drop analysis, and stabilization
"""
import cv2
import time
import numpy as np
from collections import deque, defaultdict

# Fall Logic Constants
FALL_CONFIRM_FRAMES = 5
HEIGHT_DROP_THRES = 0.80 # Relaxed from 0.65
ASPECT_RATIO_THRES = 0.9 # Relaxed from 1.1
ANGLE_THRES = 45

class FallService:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        # State Tracking
        self.history = defaultdict(lambda: deque(maxlen=150))
        self.fall_counters = defaultdict(int)
        self.is_falling_state = defaultdict(bool)
        
        print("🚨 Fall Detection Service Ready (Enhanced Logic)")

    @property
    def any_fall_detected(self):
        """Return True if any tracked person is currently falling"""
        return any(self.is_falling_state.values())
    
    def calculate_angle(self, p1, p2):
        try:
            dx = abs(p1[0] - p2[0])
            dy = abs(p1[1] - p2[1])
            if dy == 0: return 90.0
            return np.degrees(np.arctan(dx / dy))
        except:
            return 0
            
    def process(self, frame):
        """
        Process frame for fall detection
        Returns: annotated frame
        """
        frame_result = frame.copy()
        
        try:
            # Run YOLO pose tracking
            results = self.model.track(
                frame, 
                persist=True, 
                device=self.device, 
                verbose=False, 
                conf=0.5,
                classes=0, # Person class
                half=(self.device == 'cuda')
            )[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().tolist()
                keypoints = results.keypoints.data.cpu().numpy() if results.keypoints is not None else []
                
                for i, track_id in enumerate(track_ids):
                    if i >= len(boxes): break
                    
                    x1, y1, x2, y2 = map(int, boxes[i])
                    w, h = x2 - x1, y2 - y1
                    aspect_ratio = w / h
                    
                    kp = keypoints[i] if i < len(keypoints) else None
                    
                    # --- LOGIC: Freeze History ---
                    # If falling, stop updating reference height
                    if not self.is_falling_state[track_id]:
                        self.history[track_id].append(h)
                    
                    # Calculate Reference Height (90th percentile of history)
                    recent = list(self.history[track_id])
                    ref_height = np.percentile(recent, 90) if recent else h
                    
                    drop_ratio = h / ref_height if ref_height > 0 else 1.0
                    
                    # Check Conditions
                    risk_height = drop_ratio < HEIGHT_DROP_THRES
                    risk_ar = aspect_ratio > ASPECT_RATIO_THRES
                    risk_angle = False
                    
                    # Check body angle if keypoints available
                    if kp is not None and kp.shape[0] > 12:
                        # Shoulders: 5, 6 | Hips: 11, 12
                        try:
                            s_mid = ((kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2) if kp[5][2]>0.5 and kp[6][2]>0.5 else None
                            h_mid = ((kp[11][0]+kp[12][0])/2, (kp[11][1]+kp[12][1])/2) if kp[11][2]>0.5 and kp[12][2]>0.5 else None
                            
                            if s_mid and h_mid:
                                if self.calculate_angle(s_mid, h_mid) > ANGLE_THRES: 
                                    risk_angle = True
                        except:
                            pass

                    # Decision
                    is_currently_falling = (risk_height and (risk_ar or risk_angle)) or (drop_ratio < 0.5)
                    
                    # Counter Logic
                    if is_currently_falling:
                        self.fall_counters[track_id] += 1
                    else:
                        self.fall_counters[track_id] = max(0, self.fall_counters[track_id] - 1)
                        
                    # Update State
                    if self.fall_counters[track_id] >= FALL_CONFIRM_FRAMES:
                        self.is_falling_state[track_id] = True
                        status = "FALL DETECTED!"
                        color = (0, 0, 255)
                    else:
                        self.is_falling_state[track_id] = False
                        status = "Normal"
                        color = (0, 255, 0)
                        if self.fall_counters[track_id] > 2:
                            status = "Warning..."
                            color = (0, 165, 255) # Orange
                    
                    # Draw
                    cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_result, f"{status}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Debug
                    info = f"Drop:{drop_ratio:.2f} AR:{aspect_ratio:.2f}"
                    if kp is not None: info += f" Ang:{self.calculate_angle(s_mid, h_mid) if 's_mid' in locals() and s_mid else 0:.0f}"
                    cv2.putText(frame_result, info, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        except Exception as e:
            # print(f"Fall Process Error: {e}")
            pass
            
        return frame_result
