"""
Student Behavior Monitoring System - Backend
=============================================
A Flask server that uses MediaPipe Face Mesh to detect:
- Drowsiness (via Eye Aspect Ratio - EAR)
- Yawning (via Mouth Aspect Ratio - MAR)
- Distraction (via Head Pose Estimation)

The server receives webcam frames via WebSocket and emits status updates in real-time.
"""

import base64
import cv2
import numpy as np
import mediapipe as mp
import bcrypt
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, verify_jwt_in_request
)

# Initialize Flask app with CORS support
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sbms_secret_key_2024'
app.config['JWT_SECRET_KEY'] = 'sbms_jwt_secret_key_2024'  # Change in production!
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

# Initialize JWT Manager
jwt = JWTManager(app)

# Initialize Socket.IO with eventlet for async support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# =============================================================================
# IN-MEMORY USER STORAGE (Use a database in production!)
# =============================================================================
# Simple dict to store users: { email: { 'password_hash': '...', 'name': '...' } }
users_db = {}

# Initialize MediaPipe Face Mesh
# Face Mesh provides 468 facial landmarks for precise face tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Includes iris landmarks for better eye tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =============================================================================
# TEMPORAL DETECTION COUNTERS
# =============================================================================
# To distinguish between normal blinks and actual drowsiness, we track how many
# consecutive frames the EAR/MAR stays below/above threshold.
# At 5 FPS (200ms per frame), 10 frames = ~2 seconds of sustained behavior.

# Thresholds
EAR_THRESHOLD = 0.22  # Eyes considered closed below this value
MAR_THRESHOLD = 0.6   # Mouth considered yawning above this value

# Frame counters for temporal detection
DROWSY_FRAME_THRESHOLD = 10   # ~2 seconds at 5 FPS
YAWN_FRAME_THRESHOLD = 8      # ~1.6 seconds at 5 FPS

# Global counters (will be updated per connection)
drowsy_counter = 0
yawn_counter = 0

# =============================================================================
# FACIAL LANDMARK INDICES
# =============================================================================
# MediaPipe Face Mesh provides 468 landmarks. We use specific indices for:
# - Eyes: To calculate Eye Aspect Ratio (EAR)
# - Mouth: To calculate Mouth Aspect Ratio (MAR)
# - Nose/Face outline: For head pose estimation

# Left eye landmarks (6 points for EAR calculation)
# These form an ellipse around the left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Right eye landmarks (6 points for EAR calculation)
# These form an ellipse around the right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks (8 points for MAR calculation)
# Upper lip: 61, 291, 0, 17
# Lower lip: 405, 321, 375, 291
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78]

# Mouth corners and vertical points for MAR
MOUTH_OUTLINE = [61, 291, 0, 17, 39, 269, 181, 405]

# Nose tip for head pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: Tuple (x, y) coordinates
        point2: Tuple (x, y) coordinates
    
    Returns:
        float: Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_ear(eye_landmarks, landmarks, img_width, img_height):
    """
    Calculate Eye Aspect Ratio (EAR) for drowsiness detection.
    
    ============================================
    EYE ASPECT RATIO (EAR) - EXPLANATION
    ============================================
    
    The EAR is a formula that measures how "open" an eye is by comparing
    the vertical distance of the eye to its horizontal width.
    
    Eye landmark positions (simplified):
    
           P2    P3
            ·    ·
       P1 ·        · P4  (horizontal line through eye corners)
            ·    ·
           P6    P5
    
    Formula:
        EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)
    
    Where:
        ||P2 - P6||  = vertical distance between upper and lower eyelid (left side)
        ||P3 - P5||  = vertical distance between upper and lower eyelid (right side)
        ||P1 - P4||  = horizontal distance (width of the eye)
    
    Interpretation:
        - Open eyes: EAR ≈ 0.25 - 0.35 (vertical distances are significant)
        - Closed eyes: EAR < 0.20 (vertical distances approach zero)
        - Blinking: Rapid drop and recovery of EAR
    
    For drowsiness detection:
        - If EAR stays below threshold (0.25) for extended time → DROWSY
        - Quick dips below threshold → Normal blinking
    
    Args:
        eye_landmarks: List of 6 landmark indices for one eye
        landmarks: MediaPipe face landmarks object
        img_width: Width of the input image
        img_height: Height of the input image
    
    Returns:
        float: Eye Aspect Ratio value
    """
    # Extract (x, y) coordinates for each eye landmark
    # MediaPipe returns normalized coordinates (0-1), so we scale to pixel values
    coords = []
    for idx in eye_landmarks:
        lm = landmarks[idx]
        coords.append((lm.x * img_width, lm.y * img_height))
    
    # Calculate vertical distances (numerator)
    # P2-P6: Upper to lower on left side of eye
    vertical_1 = calculate_distance(coords[1], coords[5])
    # P3-P5: Upper to lower on right side of eye
    vertical_2 = calculate_distance(coords[2], coords[4])
    
    # Calculate horizontal distance (denominator)
    # P1-P4: Left corner to right corner
    horizontal = calculate_distance(coords[0], coords[3])
    
    # Avoid division by zero
    if horizontal == 0:
        return 0.0
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear


def calculate_mar(landmarks, img_width, img_height):
    """
    Calculate Mouth Aspect Ratio (MAR) for yawning detection.
    
    ============================================
    MOUTH ASPECT RATIO (MAR) - EXPLANATION
    ============================================
    
    Similar to EAR, MAR measures how "open" the mouth is by comparing
    vertical mouth opening to horizontal mouth width.
    
    Mouth landmark positions (simplified):
    
              P3
              ·
         P2  ·  P4
        P1 ·      · P5  (mouth corners)
         P8  ·  P6
              P7
    
    Formula:
        MAR = (||P3 - P7|| + ||P2 - P8|| + ||P4 - P6||) / (2 * ||P1 - P5||)
    
    Interpretation:
        - Closed mouth: MAR ≈ 0.1 - 0.3
        - Slightly open: MAR ≈ 0.4 - 0.5
        - Yawning: MAR > 0.6 (mouth is wide open vertically)
    
    Args:
        landmarks: MediaPipe face landmarks object
        img_width: Width of the input image
        img_height: Height of the input image
    
    Returns:
        float: Mouth Aspect Ratio value
    """
    # Key mouth landmarks for MAR calculation
    # Upper lip center: 13
    # Lower lip center: 14
    # Left corner: 61
    # Right corner: 291
    # Additional vertical points: 82, 312 (upper), 87, 317 (lower)
    
    def get_coord(idx):
        lm = landmarks[idx]
        return (lm.x * img_width, lm.y * img_height)
    
    # Mouth corners (horizontal reference)
    left_corner = get_coord(61)
    right_corner = get_coord(291)
    
    # Vertical points
    upper_center = get_coord(13)
    lower_center = get_coord(14)
    upper_left = get_coord(82)
    lower_left = get_coord(87)
    upper_right = get_coord(312)
    lower_right = get_coord(317)
    
    # Calculate vertical distances
    vertical_center = calculate_distance(upper_center, lower_center)
    vertical_left = calculate_distance(upper_left, lower_left)
    vertical_right = calculate_distance(upper_right, lower_right)
    
    # Calculate horizontal distance
    horizontal = calculate_distance(left_corner, right_corner)
    
    # Avoid division by zero
    if horizontal == 0:
        return 0.0
    
    # MAR formula (average of three vertical measurements)
    mar = (vertical_center + vertical_left + vertical_right) / (3.0 * horizontal)
    
    return mar


def estimate_head_pose(landmarks, img_width, img_height):
    """
    Estimate head pose to detect if user is looking away (distracted).
    
    Uses the relative positions of nose tip, chin, and eye corners
    to estimate if the face is turned significantly to the side.
    
    Args:
        landmarks: MediaPipe face landmarks object
        img_width: Width of the input image
        img_height: Height of the input image
    
    Returns:
        bool: True if user appears distracted (looking away), False otherwise
    """
    def get_coord(idx):
        lm = landmarks[idx]
        return (lm.x * img_width, lm.y * img_height)
    
    # Get key facial landmarks
    nose = get_coord(NOSE_TIP)
    left_eye = get_coord(LEFT_EYE_CORNER)
    right_eye = get_coord(RIGHT_EYE_CORNER)
    
    # Calculate the center of the eyes
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    
    # Calculate horizontal offset of nose from eye center
    # If the nose is significantly off-center, the person is looking away
    face_width = abs(right_eye[0] - left_eye[0])
    
    if face_width == 0:
        return False
    
    # Nose offset as percentage of face width
    nose_offset = abs(nose[0] - eye_center_x) / face_width
    
    # If nose is more than 30% off-center, consider distracted
    # This accounts for significant head turns
    distracted = nose_offset > 0.30
    
    return distracted


def process_frame(frame_data):
    """
    Process a single frame from the webcam and analyze behavior.
    
    Args:
        frame_data: Base64 encoded image data from webcam
    
    Returns:
        dict: Analysis results containing status, EAR, and MAR scores
    """
    try:
        # Decode base64 image
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        # Convert base64 to numpy array
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        # Get frame dimensions
        img_height, img_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # No face detected
            return {
                'status': 'No Face Detected',
                'ear_score': 0.0,
                'mar_score': 0.0
            }
        
        # Get the first detected face
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate Eye Aspect Ratio for both eyes
        left_ear = calculate_ear(LEFT_EYE, face_landmarks, img_width, img_height)
        right_ear = calculate_ear(RIGHT_EYE, face_landmarks, img_width, img_height)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate Mouth Aspect Ratio
        mar = calculate_mar(face_landmarks, img_width, img_height)
        
        # Check for distraction (head turned away)
        is_distracted = estimate_head_pose(face_landmarks, img_width, img_height)
        
        # Access global counters for temporal detection
        global drowsy_counter, yawn_counter
        
        # =============================================================
        # TEMPORAL DETECTION LOGIC
        # =============================================================
        # Instead of flagging drowsy on a single frame (which catches blinks),
        # we count consecutive frames where EAR/MAR exceeds threshold.
        # Only trigger alert after sustained behavior.
        
        # Update drowsy counter based on EAR
        if avg_ear < EAR_THRESHOLD:
            drowsy_counter += 1
        else:
            drowsy_counter = 0  # Reset if eyes open
        
        # Update yawn counter based on MAR
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0  # Reset if mouth closes
        
        # Determine status based on metrics with temporal smoothing
        # Priority: Distracted > Yawning > Drowsy > Attentive
        status = 'Attentive'
        
        if is_distracted:
            status = 'Distracted'
            # Reset counters when distracted
            drowsy_counter = 0
            yawn_counter = 0
        elif yawn_counter >= YAWN_FRAME_THRESHOLD:
            # Sustained high MAR indicates genuine yawning
            status = 'Yawning'
        elif drowsy_counter >= DROWSY_FRAME_THRESHOLD:
            # Sustained low EAR indicates drowsiness (not just a blink)
            status = 'Drowsy'
        
        return {
            'status': status,
            'ear_score': round(avg_ear, 3),
            'mar_score': round(mar, 3)
        }
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


# =============================================================================
# SOCKET.IO EVENT HANDLERS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle new client connections."""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections."""
    print('Client disconnected')


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handle incoming video frames from the frontend.
    
    Receives a base64-encoded frame, processes it through MediaPipe,
    and emits the analysis results back to the client.
    
    Args:
        data: Dictionary containing 'frame' key with base64 image data
    """
    frame_data = data.get('frame')
    
    if frame_data:
        result = process_frame(frame_data)
        
        if result:
            # Emit status update to the client
            emit('status_update', result)


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.route('/api/register', methods=['POST'])
def register():
    """
    Register a new user.
    
    Request body:
        { "email": "...", "password": "...", "name": "..." }
    
    Returns:
        201: User created successfully
        400: Missing fields or user exists
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    
    if not email or not password or not name:
        return jsonify({'error': 'Email, password, and name are required'}), 400
    
    if email in users_db:
        return jsonify({'error': 'User already exists'}), 400
    
    # Hash password with bcrypt
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Store user
    users_db[email] = {
        'password_hash': password_hash,
        'name': name
    }
    
    print(f"New user registered: {email}")
    return jsonify({'message': 'User registered successfully'}), 201


@app.route('/api/login', methods=['POST'])
def login():
    """
    Authenticate user and return JWT token.
    
    Request body:
        { "email": "...", "password": "..." }
    
    Returns:
        200: { "access_token": "...", "user": {...} }
        401: Invalid credentials
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    # Check if user exists
    user = users_db.get(email)
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Verify password
    if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Create JWT token
    access_token = create_access_token(identity=email)
    
    print(f"User logged in: {email}")
    return jsonify({
        'access_token': access_token,
        'user': {
            'email': email,
            'name': user['name']
        }
    }), 200


@app.route('/api/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """
    Get current authenticated user info.
    
    Requires: Authorization header with Bearer token
    
    Returns:
        200: { "email": "...", "name": "..." }
        401: Invalid or missing token
    """
    email = get_jwt_identity()
    user = users_db.get(email)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'email': email,
        'name': user['name']
    }), 200


@app.route('/')
def index():
    """Health check endpoint."""
    return {'status': 'Server is running', 'message': 'Student Behavior Monitoring System API'}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Student Behavior Monitoring System - Backend Server")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Waiting for frontend connection...")
    print("=" * 60)
    
    # Run the Flask-SocketIO server
    # Use eventlet for async WebSocket support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
