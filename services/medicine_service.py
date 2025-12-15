"""
Medicine Detection Service
Uses YOLO model to detect medicine/pills
"""
import cv2


class MedicineService:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        print("💊 Medicine Detection Service Ready")
    
    def process(self, frame):
        """
        Process frame for medicine detection
        Returns: annotated frame
        """
        frame_result = frame.copy()
        
        # Run YOLO inference
        results = self.model(
            frame, 
            device=self.device, 
            verbose=False, 
            conf=0.5, 
            half=(self.device == 'cuda')
        )[0]
        
        # Draw detections
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{results.names[cls]} {conf:.2f}"
                
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_result, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_result
