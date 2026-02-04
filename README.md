# Webcam Behavior Analytics

**Webcam Behavior Analytics** is a real-time monitoring system that uses computer vision to detect user engagement and alertness. It is designed to enhance productivity and safety by identifying signs of drowsiness, yawning, and distraction.

## 🚀 Features

*   **Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) to detect prolonged eye closure.
*   **Yawn Detection**: Tracks Mouth Aspect Ratio (MAR) to identify yawning.
*   **Distraction Detection**: Estimates head pose to determine if the user is looking away.
*   **Real-time Alerts**: Visual status updates via a WebSocket connection.

## 🛠️ Technology Stack

### Backend
*   **Python 3.x**
*   **Flask** (Web Framework)
*   **MediaPipe** (Face Mesh & Computer Vision)
*   **OpenCV** (Image Processing)
*   **Socket.IO** (Real-time Communication)

### Frontend
*   **React**
*   **Socket.IO Client**
*   **Recharts** (Data Visualization)

## 📦 Installation & Setup

This project is a monorepo containing both backend and frontend.

### Prerequisites
*   Python 3.8+
*   Node.js 16+

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📄 License
© 2024 Webcam Behavior Analytics. All Rights Reserved.
