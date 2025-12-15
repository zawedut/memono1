"""
Fall Detection Service
Uses YOLO pose model to detect if a person has fallen
"""
import cv2


class FallService:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        print("🚨 Fall Detection Service Ready")
    
    def process(self, frame):
        """
        Process frame for fall detection
        Returns: annotated frame
        """
        frame_result = frame.copy()
        
        # Run YOLO pose tracking
        results = self.model.track(
            frame, 
            persist=True, 
            device=self.device, 
            verbose=False, 
            classes=0,
            half=(self.device == 'cuda')
        )[0]
        
        # Draw detections with fall status
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                
                # Fall logic: width > height * 1.2
                is_fall = w > h * 1.2
                
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                label = f"ID:{track_id} FALL!" if is_fall else f"ID:{track_id} Normal"
                
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_result, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame_result
