"""
Enhanced Sign Language Sender for Raspberry Pi
Captures hand landmarks, frames for OCR, audio, and sends to laptop
Runs headless - no monitor required
"""

import cv2
import mediapipe as mp
import websocket
import json
import time
import socket
import numpy as np
import base64
import threading
from collections import deque

# Optional imports
try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    print("[WARNING] Audio not available. Install: pip install pyaudio")

# ============================================================
# CONFIGURATION
# ============================================================

PC_IP = "192.168.0.4"  # CHANGE THIS TO YOUR LAPTOP IP
WS_URL = f"ws://{PC_IP}:8765"

# Performance
TARGET_FPS = 30
SEND_INTERVAL = 1.0 / TARGET_FPS
HEARTBEAT_INTERVAL = 2.0

# MediaPipe
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 2

# Audio Recording
AUDIO_DURATION = 30  # seconds
AUDIO_RATE = 16000
AUDIO_CHUNK = 1024

# ============================================================
# GLOBALS
# ============================================================

last_send_time = 0
last_heartbeat_time = 0
ws = None
command_queue = deque()
is_recording_audio = False

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

# ============================================================
# WEBSOCKET CONNECTION
# ============================================================

def connect_ws():
    """Connect to laptop with retry"""
    print("=" * 60)
    print("RASPBERRY PI - SIGN LANGUAGE SENDER")
    print("=" * 60)
    print(f"Connecting to laptop at {WS_URL}...")
    
    retry_count = 0
    while True:
        try:
            new_ws = websocket.create_connection(WS_URL, timeout=3)
            new_ws.settimeout(1)
            print(f"✓ Connected to laptop!")
            print("=" * 60)
            return new_ws
        except Exception as e:
            retry_count += 1
            print(f"⚠ Connection failed (Attempt {retry_count}): {e}")
            print("  Retrying in 2 seconds...")
            time.sleep(2)

# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):
    """
    Extract landmarks in model format
    Returns: 128 float values [Hand1: 63 coords + 1 label] + [Hand2: 63 coords + 1 label]
    """
    if not results.multi_hand_landmarks:
        return None
    
    all_landmarks = [-1.0] * 128
    
    hands_data = zip(results.multi_hand_landmarks, results.multi_handedness)
    
    for hand_idx, (hand_landmarks, hand_info) in enumerate(hands_data):
        if hand_idx >= 2:
            break
        
        # Extract 21 landmarks × 3 coordinates
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ])
        
        # Hand label
        hand_label = 0.0 if hand_info.classification[0].label == 'Left' else 1.0
        
        # Place in array
        start_idx = hand_idx * 64
        all_landmarks[start_idx : start_idx + 63] = coords
        all_landmarks[start_idx + 63] = hand_label
    
    return all_landmarks

# ============================================================
# FRAME CAPTURE FOR OCR
# ============================================================

def capture_frame_for_ocr(frame):
    """Capture current frame and send to laptop for OCR"""
    try:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send to laptop
        message = {
            'type': 'frame',
            'frame': frame_base64,
            'timestamp': time.time()
        }
        
        ws.send(json.dumps(message))
        print("[OCR] Frame sent to laptop")
        return True
    except Exception as e:
        print(f"[ERROR] Capturing frame: {e}")
        return False

# ============================================================
# AUDIO RECORDING
# ============================================================

def record_audio(duration=30):
    """Record audio and send to laptop"""
    global is_recording_audio
    
    if not AUDIO_AVAILABLE:
        print("[ERROR] Audio not available")
        return False
    
    try:
        is_recording_audio = True
        print(f"[AUDIO] Recording for {duration} seconds...")
        
        # Setup audio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
        
        frames = []
        
        # Record
        for i in range(0, int(AUDIO_RATE / AUDIO_CHUNK * duration)):
            if not is_recording_audio:
                break
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            if i % 10 == 0:
                print(f"[AUDIO] Recording... {i * AUDIO_CHUNK / AUDIO_RATE:.1f}s")
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Convert to base64
        audio_data = b''.join(frames)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Send to laptop
        message = {
            'type': 'audio',
            'audio': audio_base64,
            'duration': duration,
            'rate': AUDIO_RATE,
            'timestamp': time.time()
        }
        
        ws.send(json.dumps(message))
        print("[AUDIO] Sent to laptop")
        
        is_recording_audio = False
        return True
        
    except Exception as e:
        print(f"[ERROR] Recording audio: {e}")
        is_recording_audio = False
        return False

# ============================================================
# COMMAND HANDLER
# ============================================================

def handle_command(command_data):
    """Handle commands from laptop"""
    try:
        command = command_data.get('command')
        params = command_data.get('params', {})
        
        print(f"[COMMAND] Received: {command}")
        
        if command == 'capture_frame':
            # Will be captured in next frame
            command_queue.append(('capture_frame', params))
        
        elif command == 'record_audio':
            duration = params.get('duration', 30)
            # Start audio recording in separate thread
            thread = threading.Thread(target=record_audio, args=(duration,))
            thread.daemon = True
            thread.start()
        
        elif command == 'stop_audio':
            global is_recording_audio
            is_recording_audio = False
            print("[AUDIO] Stopped")
        
        else:
            print(f"[WARNING] Unknown command: {command}")
    
    except Exception as e:
        print(f"[ERROR] Handling command: {e}")

def check_for_commands():
    """Check for incoming commands from laptop"""
    global ws
    try:
        ws.settimeout(0.001)  # Non-blocking
        message = ws.recv()
        data = json.loads(message)
        
        if data.get('type') == 'command':
            handle_command(data)
    except socket.timeout:
        pass
    except Exception as e:
        # Connection lost
        print(f"[ERROR] Command check: {e}")
    finally:
        ws.settimeout(1)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    """Main execution loop"""
    global last_send_time, last_heartbeat_time, ws
    
    # Connect to laptop
    ws = connect_ws()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return
    
    # Optimize camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n[OK] Camera initialized")
    print("[OK] Starting detection loop (headless)...\n")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            current_time = time.time()
            
            # Read frame
            success, frame = cap.read()
            if not success:
                print("[WARNING] Failed to read frame")
                continue
            
            frame_count += 1
            
            # Check for commands from laptop
            if frame_count % 10 == 0:  # Check every 10 frames
                check_for_commands()
            
            # Process command queue
            if command_queue:
                cmd, params = command_queue.popleft()
                if cmd == 'capture_frame':
                    capture_frame_for_ocr(frame)
            
            # Flip for processing
            frame_flipped = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            data_sent = False
            
            # Hand detection and landmark extraction
            if results.multi_hand_landmarks:
                # Extract and send landmarks
                if (current_time - last_send_time) >= SEND_INTERVAL:
                    landmarks = extract_landmarks(results)
                    
                    if landmarks and len(landmarks) == 128:
                        data_packet = {
                            "landmarks": landmarks,
                            "timestamp": current_time,
                            "hands": len(results.multi_hand_landmarks)
                        }
                        
                        try:
                            ws.send(json.dumps(data_packet))
                            data_sent = True
                            last_send_time = current_time
                            last_heartbeat_time = current_time
                        except Exception as e:
                            print(f"\n⚠ Connection lost: {e}")
                            print("Reconnecting...")
                            try:
                                ws.close()
                            except:
                                pass
                            ws = connect_ws()
            
            # Heartbeat
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
                    pass
            
            # Status update (every 5 seconds)
            if frame_count % 150 == 0:
                hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                print(f"[STATUS] Frame {frame_count} | Hands: {hands_detected} | Sent: {data_sent}")
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n[CLEANUP] Shutting down...")
        cap.release()
        if ws:
            try:
                ws.close()
            except:
                pass
        print("[OK] Shutdown complete")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
