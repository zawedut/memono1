
import cv2
import numpy as np
import time
import math
import mediapipe as mp

class MedicineService:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        # --- MediaPipe Initialization ---
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        # Initialize Holistic model (Pose + Face + Hands)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            refine_face_landmarks=True # Needed for accurate lip landmarks
        )

        # --- State Tracking ---
        self.state = "IDLE" # IDLE, DETECTED, HOLDING, AT_MOUTH, COMPLETED
        self.state_start_time = 0
        self.pill_count = 0
        
        # --- Config & Thresholds ---
        self.pinch_threshold = 0.05      # Finger pinch distance
        self.hand_mouth_threshold = 0.15 # Hand to mouth distance
        self.mouth_open_threshold = 0.3  # Mouth Aspect Ratio (MAR)
        self.CONF_THRESH = 0.4
        
        print("💊 Pro Medicine Service (YOLO + MediaPipe) Ready")

    def reset_state(self):
        self.state = "IDLE"
        self.state_start_time = 0
        
    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def mouth_aspect_ratio(self, face_landmarks):
        # Mediapipe Face Mesh Landmarks:
        # Left-Right: 61, 291
        # Top-Bottom (Inner): 13, 14
        
        p_left = face_landmarks.landmark[61]
        p_right = face_landmarks.landmark[291]
        p_top = face_landmarks.landmark[13]
        p_bottom = face_landmarks.landmark[14]
        
        mouth_width = self.calculate_distance(p_left, p_right)
        mouth_height = self.calculate_distance(p_top, p_bottom)
        
        if mouth_width == 0: return 0
        return mouth_height / mouth_width

    def process(self, frame):
        """
        Process frame with Pro Logic:
        1. YOLO Detect 'Pill'
        2. MediaPipe Detect Hand & Face
        3. Check Logic: Holding -> At Mouth -> Swallowed
        """
        frame_result = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        # --- 1. Object Detection (YOLO) ---
        pill_detected = False
        pill_bbox = None
        
        try:
            results = self.model(frame, device=self.device, verbose=False, conf=self.CONF_THRESH, half=(self.device=='cuda'))[0]
            if results.boxes:
                for box in results.boxes:
                    # In custom model, likely only 1 class or just a few. 
                    # We assume ANY detection is a target for now, or check class names if known.
                    # User's script used class 67 (Cell Phone), but we have a custom trained model 'best2.pt'.
                    # We'll treat ANY detection from 'best2.pt' as the pill/bottle.
                    pill_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pill_bbox = (x1, y1, x2, y2)
                    
                    # Draw BBox
                    cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_result, f"TARGET {float(box.conf[0]):.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    break # Focus on one target
        except Exception as e:
            print(f"YOLO Error: {e}")

        # --- 2. Pose & Hand Tracking (MediaPipe) ---
        # To verify logic, we need Face AND Right Hand (Assuming right-handed for now, or check both)
        # Using Holistic for combined tracking
        
        # Optimization: Only process MP if YOLO detected OR if we are already monitoring state
        # But for "Holding" we need MP to overlap with YOLO.
        
        mp_results = self.holistic.process(rgb_frame)
        
        # Debug Draw (Optional, maybe too messy? Let's draw only hand/face if relevant)
        # self.mp_drawing.draw_landmarks(frame_result, mp_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        
        logic_active = False
        
        if mp_results.face_landmarks and mp_results.right_hand_landmarks:
            logic_active = True
            rh = mp_results.right_hand_landmarks
            face = mp_results.face_landmarks
            
            thumb_tip = rh.landmark[4]
            index_tip = rh.landmark[8]
            mouth_center = face.landmark[13] # Upper lip
            
            # Metrics
            pinch_dist = self.calculate_distance(thumb_tip, index_tip)
            hand_mouth_dist = self.calculate_distance(index_tip, mouth_center)
            mar = self.mouth_aspect_ratio(face)
            
            # Draw Line from hand to mouth
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            mouth_x, mouth_y = int(mouth_center.x * w), int(mouth_center.y * h)
            
            line_color = (0, 255, 0) if hand_mouth_dist < self.hand_mouth_threshold else (0, 0, 255)
            cv2.line(frame_result, (index_x, index_y), (mouth_x, mouth_y), line_color, 2)
            
            # --- STATE MACHINE ---
            
            # 1. HOLDING CHECK
            # Needs: Pinching + Index Finger inside YOLO Box
            is_holding_real_pill = False
            if (pinch_dist < self.pinch_threshold) and pill_detected and pill_bbox:
                if (pill_bbox[0] < index_x < pill_bbox[2]) and (pill_bbox[1] < index_y < pill_bbox[3]):
                    is_holding_real_pill = True
                    self.state = "HOLDING"
                    self.state_start_time = time.time()
            
            if self.state == "HOLDING":
                cv2.putText(frame_result, "STATUS: Holding Pill", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 2. AT MOUTH CHECK
            # Needs: Was Holding + Near Mouth + Mouth Open
            if self.state in ["HOLDING", "AT_MOUTH"] and (hand_mouth_dist < self.hand_mouth_threshold):
                if mar > self.mouth_open_threshold:
                    self.state = "AT_MOUTH"
                    cv2.putText(frame_result, f"STATUS: Opening Mouth (MAR: {mar:.2f})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                     cv2.putText(frame_result, f"STATUS: Near Mouth (MAR: {mar:.2f})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 3. CONSUMED CHECK (Completion)
            # Needs: Was At Mouth + Hand Moved Away
            if self.state == "AT_MOUTH":
                # If hand moves away significantly
                if hand_mouth_dist > (self.hand_mouth_threshold + 0.05):
                    self.state = "COMPLETED"
                    self.pill_count += 1
                    self.state_start_time = time.time()
                    print(f"*** PILL INTAKE CONFIRMED! Count: {self.pill_count} ***")

        # --- Final State Info ---
        status_color = (0, 255, 255)
        if self.state == "COMPLETED": status_color = (0, 255, 0)
        
        cv2.putText(frame_result, f"MED_STATE: {self.state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if self.state == "COMPLETED":
             cv2.putText(frame_result, "SUCCESS!", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
             # Auto-reset after 3 seconds so we don't get stuck forever (or let external handler reset)
             if time.time() - self.state_start_time > 5.0:
                 self.state = "IDLE"

        return frame_result
