import cv2
import mediapipe as mp
import websocket
import json
import time
import socket
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# PC Connection
PC_IP = "192.168.0.4"  # Change to your PC's IP address
WS_URL = f"ws://{PC_IP}:8765"

# Performance Settings
TARGET_FPS = 30  # Balanced: 30 FPS for smooth detection
SEND_INTERVAL = 1.0 / TARGET_FPS
HEARTBEAT_INTERVAL = 2.0  # Send keepalive every 2 seconds

# MediaPipe Settings (Optimized for Accuracy)
MIN_DETECTION_CONFIDENCE = 0.7  # High confidence for initial detection
MIN_TRACKING_CONFIDENCE = 0.6   # Slightly lower for tracking stability
MAX_NUM_HANDS = 2  # Support two hands

# Display Settings
SHOW_LANDMARKS = True  # Show hand landmarks on video
SHOW_FPS = True        # Show FPS counter

# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

# Drawing specifications for visualization
landmark_spec = mp_drawing.DrawingSpec(
    color=(0, 0, 255),  # Red landmarks
    thickness=2,
    circle_radius=3
)
connection_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),  # Green connections
    thickness=2
)

# ============================================================
# GLOBALS
# ============================================================

last_send_time = 0
last_heartbeat_time = 0
frame_count = 0
fps = 0
last_fps_time = time.time()

# ============================================================
# WEBSOCKET CONNECTION
# ============================================================

def connect_ws():
    """
    Connect to WebSocket server with automatic retry
    Handles connection failures gracefully
    """
    print("=" * 60)
    print("RASPBERRY PI - SIGN LANGUAGE SENDER")
    print("=" * 60)
    print(f"Connecting to PC at {WS_URL}...")
    
    retry_count = 0
    while True:
        try:
            new_ws = websocket.create_connection(WS_URL, timeout=3)
            new_ws.settimeout(1)
            print(f"✓ Connected successfully!")
            print("=" * 60)
            return new_ws
        except (websocket.WebSocketException, socket.error) as e:
            retry_count += 1
            print(f"⚠ Connection failed (Attempt {retry_count}): {e}")
            print("  Retrying in 2 seconds...")
            time.sleep(2)

# ============================================================
# LANDMARK EXTRACTION (MATCHES TRAINING FORMAT)
# ============================================================

def extract_landmarks(results):
    """
    Extract landmarks in the EXACT format expected by the model
    
    Format: 128 values = [Hand1: 63 coords + 1 label] + [Hand2: 63 coords + 1 label]
    - 63 coords = 21 landmarks × 3 (x, y, z)
    - 1 label = hand type (0=Left, 1=Right)
    
    Returns:
        list: 128 float values, or None if no hands detected
    """
    if not results.multi_hand_landmarks:
        return None
    
    # Initialize with -1 (indicates no hand present)
    all_landmarks = [-1.0] * 128
    
    # Process up to 2 hands
    hands_data = zip(results.multi_hand_landmarks, results.multi_handedness)
    
    for hand_idx, (hand_landmarks, hand_info) in enumerate(hands_data):
        if hand_idx >= 2:  # Only process first 2 hands
            break
        
        # Extract 21 landmarks × 3 coordinates = 63 values
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([
                float(landmark.x),  # X coordinate (0-1)
                float(landmark.y),  # Y coordinate (0-1)
                float(landmark.z)   # Z coordinate (depth)
            ])
        
        # Hand label: 0 for Left, 1 for Right
        hand_label = 0.0 if hand_info.classification[0].label == 'Left' else 1.0
        
        # Place in correct position
        # Hand 1: indices 0-63, Hand 2: indices 64-127
        start_idx = hand_idx * 64
        all_landmarks[start_idx : start_idx + 63] = coords
        all_landmarks[start_idx + 63] = hand_label
    
    return all_landmarks

# ============================================================
# DISPLAY HELPERS
# ============================================================

def calculate_fps():
    """Calculate and return current FPS"""
    global frame_count, fps, last_fps_time
    
    frame_count += 1
    current_time = time.time()
    
    # Update FPS every second
    if current_time - last_fps_time >= 1.0:
        fps = frame_count / (current_time - last_fps_time)
        frame_count = 0
        last_fps_time = current_time
    
    return fps

def draw_info(image, hands_detected, data_sent):
    """Draw information overlay on the image"""
    # Background box for text
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Status text
    y_offset = 35
    cv2.putText(image, "Sign Language Sender", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    status_text = f"Hands: {hands_detected}"
    status_color = (0, 255, 0) if hands_detected > 0 else (0, 0, 255)
    cv2.putText(image, status_text, (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    y_offset += 25
    fps_value = calculate_fps()
    cv2.putText(image, f"FPS: {fps_value:.1f}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    send_status = "SENDING" if data_sent else "IDLE"
    send_color = (0, 255, 0) if data_sent else (255, 255, 255)
    cv2.putText(image, f"Status: {send_status}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, send_color, 1)
    
    # Instructions
    cv2.putText(image, "Press 'q' to quit", (20, image.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    """Main execution loop"""
    global last_send_time, last_heartbeat_time
    
    # Initialize WebSocket connection
    ws = connect_ws()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nCamera initialized successfully")
    print("Starting detection loop...\n")
    
    try:
        while cap.isOpened():
            # Read frame
            success, image = cap.read()
            if not success:
                print("WARNING: Failed to read frame")
                continue
            
            current_time = time.time()
            
            # Flip image for selfie view
            image = cv2.flip(image, 1)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(image_rgb)
            
            data_sent = False
            hands_detected = 0
            
            # ===== HAND DETECTION AND LANDMARK EXTRACTION =====
            if results.multi_hand_landmarks:
                hands_detected = len(results.multi_hand_landmarks)
                
                # Draw landmarks on image
                if SHOW_LANDMARKS:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_spec,
                            connection_spec
                        )
                
                # Extract and send landmarks (rate-limited)
                if (current_time - last_send_time) >= SEND_INTERVAL:
                    landmarks = extract_landmarks(results)
                    
                    if landmarks is not None:
                        # Validate landmark data
                        if len(landmarks) == 128:
                            # Create data packet
                            data_packet = {
                                "landmarks": landmarks,
                                "timestamp": current_time,
                                "hands": hands_detected
                            }
                            
                            try:
                                ws.send(json.dumps(data_packet))
                                data_sent = True
                                last_send_time = current_time
                                last_heartbeat_time = current_time
                            except (websocket.WebSocketConnectionClosedException, 
                                    socket.error, BrokenPipeError) as e:
                                print(f"\n⚠ Connection lost: {e}")
                                print("Reconnecting...")
                                try:
                                    ws.close()
                                except:
                                    pass
                                ws = connect_ws()
                        else:
                            print(f"WARNING: Invalid landmark size: {len(landmarks)}")
            
            # ===== HEARTBEAT (Keep connection alive) =====
            elif (current_time - last_heartbeat_time) >= HEARTBEAT_INTERVAL:
                heartbeat = {
                    "type": "ping",
                    "status": "idle",
                    "timestamp": current_time
                }
                try:
                    ws.send(json.dumps(heartbeat))
                    last_heartbeat_time = current_time
                except:
                    pass  # Silently fail for heartbeats
            
            # ===== DISPLAY =====
            draw_info(image, hands_detected, data_sent)
            cv2.imshow('Sign Language Detection (Raspberry Pi)', image)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nShutting down...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        if ws:
            try:
                ws.close()
            except:
                pass
        print("Shutdown complete")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
