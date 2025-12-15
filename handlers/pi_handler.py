"""
Pi Handler - Handles WebSocket connection from Raspberry Pi
Receives video frames, processes through AI services, broadcasts to web clients
"""
import asyncio
import cv2
import numpy as np
import time


class PiHandler:
    def __init__(self, face_service, medicine_service, fall_service, web_clients):
        self.face_service = face_service
        self.medicine_service = medicine_service
        self.fall_service = fall_service
        self.web_clients = web_clients  # Set of web client websockets
        self.command_queue = asyncio.Queue()  # Commands from web to send to Pi
    
    async def handle(self, ws):
        """Handle Pi WebSocket connection"""
        print(f"🔗 Pi Connected: {ws.remote_address}")
        prev_time = time.time()
        
        # Start command sender task
        sender_task = asyncio.create_task(self._send_commands(ws))
        
        try:
            while True:
                # Receive frame with buffer flush (get latest)
                frame_data = None
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                        frame_data = msg
                    except asyncio.TimeoutError:
                        break
                    except Exception:
                        return
                
                if frame_data is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Decode frame
                np_data = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                
                # Resize
                frame = cv2.resize(frame, (640, 480))
                
                # Update face service with new frame
                self.face_service.update_frame(frame)
                
                # Process through AI services
                frame_med = self.medicine_service.process(frame)
                frame_fall = self.fall_service.process(frame)
                
                # Draw face results
                frame_face = frame.copy()
                for (x, y, w, h, name) in self.face_service.get_results():
                    cv2.rectangle(frame_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_face, name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Draw FPS on all frames
                for img, label in [(frame_med, "Medicine"), (frame_fall, "Fall"), (frame_face, "Face")]:
                    cv2.putText(img, f"{label} | FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show windows
                cv2.imshow("Screen 1: Medicine", frame_med)
                cv2.imshow("Screen 2: Fall Detection", frame_fall)
                cv2.imshow("Screen 3: Face Recognition", frame_face)
                
                # Broadcast to web clients (send medicine frame)
                await self._broadcast_frame(frame_med)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"❌ Pi Error: {e}")
        finally:
            sender_task.cancel()
            cv2.destroyAllWindows()
            print("🔴 Pi Disconnected")
    
    async def _broadcast_frame(self, frame):
        """Broadcast frame to all web clients"""
        if not self.web_clients:
            return
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # Create message: [Header=1 (IMAGE)] + [JPEG data]
        msg = bytes([1]) + buffer.tobytes()
        
        # Send to all web clients
        disconnected = set()
        for client in self.web_clients:
            try:
                await client.send(msg)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.web_clients -= disconnected
    
    async def _send_commands(self, ws):
        """Send queued commands to Pi"""
        while True:
            try:
                command = await self.command_queue.get()
                # Create message: [Header=11 (ACTION)] + [command string]
                msg = bytes([11]) + command.encode()
                await ws.send(msg)
                print(f"📤 Sent to Pi: {command}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Command send error: {e}")
    
    def queue_command(self, command):
        """Queue a command to send to Pi"""
        self.command_queue.put_nowait(command)
