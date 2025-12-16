# MEMO-BOT: Technical Summary & Feature Breakdown

## 1. Project Overview
**MEMO-BOT** is an AI-powered elderly care assistant robot designed to provide safety monitoring, medication management, and companionship. It leverages a powerful PC-Server architecture with a lightweight control unit (Raspberry Pi) to deliver high-performance AI features in real-time.

---

## 2. Technology Stack

### **Backend (Central Brain)**
*   **Language:** Python 3.9+
*   **Core Framework:** `aiohttp` (Async Web Server), `asyncio` (Concurrency).
*   **Communication:** `websockets` (Real-time Video/Audio/Control streaming).
*   **AI & Machine Learning:**
    *   **PyTorch / Ultralytics:** YOLOv8 for Object and Pose Detection.
    *   **TensorFlow / Keras:** Deep learning backend for Face Recognition.
    *   **DeepFace:** Advanced Face Recognition (ArcFace model).
    *   **OpenCV:** Image processing and fast face detection (Haar Cascades).
    *   **NumPy:** High-performance numerical data processing.
*   **Generative AI:**
    *   **Typhoon AI:** Thai-specialized Large Language Model (LLM) for natural conversation.
    *   **Edge-TTS:** High-quality, low-latency Text-to-Speech synthesis.
    *   **Google Speech Recognition:** Automatic Speech Recognition (ASR) for Thai voice commands.

### **Frontend (Control Center)**
*   **Tech:** Standard HTML5, CSS3, Vanilla JavaScript (ES6+).
*   **Styling:** Custom CSS with Responsive Design (Mobile/Desktop), Flexbox/Grid layouts.
*   **UI Components:** 
    *   Real-time MJPEG Video Canvas.
    *   WebSocket-based Chat Interface.
    *   Dynamic Medicine Scheduler Dashboard.
    *   Toast Notification System.
*   **Assets:** FontAwesome Icons, Google Fonts (Inter).

### **Infrastructure & Hardware**
*   **Server:** PC with NVIDIA GPU (CUDA acceleration) for heavy AI processing.
*   **Robot Unit:** Raspberry Pi (Camera, Microphone, Speaker, Motor Driver).
*   **Connectivity:** Local Area Network (Wi-Fi) via WebSocket Protocol.
*   **External APIs:** 
    *   **LINE Messaging API:** For emergency push notifications.
    *   **ImgBB:** For hosting alert images.

---

## 3. Key Features & Functionality

### **A. Safety & Security (AI Surveillance)**
1.  **Fall Detection (Real-time)**
    *   **Tech:** YOLOv8-Pose (Keypoint Detection).
    *   **Logic:** Analyzes skeletal geometry (shoulder-hip angle), sudden height drops, and aspect ratio changes (vertical to horizontal).
    *   **Alerts:** Instantly sends a **LINE Notification** with an image of the event to family members.
    *   **Prevention:** Uses history tracking to reduce false alarms.

2.  **Face Recognition (Identity Awareness)**
    *   **Tech:** Hybrid System (OpenCV for fast tracking + DeepFace ArcFace for identity verification).
    *   **Verification:** "Locking System" requires 5 consecutive matches to confirm identity, ensuring high accuracy.
    *   **Utility:** Enables the robot to know who it is talking to and provide personalized greetings.

### **B. Health Management (Medicine Assistant)**
1.  **Smart Medicine Scheduler**
    *   **System:** JSON-based scheduling system manageable via Web UI.
    *   **Notification:** Voice reminders (TTS) when it's time to take medicine.
2.  **Visual Medicine Confirmation**
    *   **Tech:** Custom-trained YOLO Model (`best2.pt`) to detect pill bottles/medicines.
    *   **Compliance:** Monitors if the patient has actually taken the medicine.
    *   **Safety:** Warning system triggers if medicine is not taken within 5 minutes.

### **C. Companionship & Interaction**
1.  **Thai Voice Chatbot**
    *   **Brain:** Powered by **Typhoon AI**, a specialized Thai LLM, allowing for natural, empathetic conversations.
    *   **Context-Aware:** can "see" who it is talking to (via Face Recognition) and incorporate that into the conversation (e.g., "Hello Grandma, how are you?").
    *   **Interface:** Supports both Voice interaction (Speak/Listen) and Text Chat via the Web App.

2.  **Remote Control (Telepresence)**
    *   **Manual Drive:** Family members can remotely control the robot (Walk/Turn) via the Web UI.
    *   **Live Feed:** Low-latency video streaming to check on the elderly from anywhere in the house.

### **D. Notification Ecosystem**
*   **LINE Integration:** Critical alerts (Falling, Missed Medicine) are pushed directly to caregivers' phones.
*   **Smart Cooldown:** Intelligent rate-limiting to prevent spamming notifications during a single event.

---

## 4. System Architecture
1.  **Robot (Edge):** Captures Video/Audio -> Sends raw data to Server -> Plays Audio/Executes Motor Commands.
2.  **Server (Core):** 
    *   Receives Video -> Runs AI Models (Fall/Face/Med) -> Overlays Info.
    *   Manages State (Schedules, Chat History).
    *   Generates Reponses (Typhoon AI -> TTS).
3.  **Web Client:** Visualizes the processed video stream and provides control interface.
