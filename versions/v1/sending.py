import cv2
import mediapipe as mp
import websocket
import json
import time
import socket

PC_IP = "192.168.0.4" 
WS_URL = f"ws://{PC_IP}:8765"

# --- STABILITY CONFIG jeevan ---
TARGET_FPS = 50 
send_interval = 1.0 / TARGET_FPS
HEARTBEAT_INTERVAL = 1.0  # Send a 'ping' every 1 second if no hand is seen
last_send_time = 0
last_heartbeat_time = 0

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Changed to 2 to support two hands (matches training)
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

def connect_ws():
    while True:
        try:
            new_ws = websocket.create_connection(WS_URL, timeout=2)
            new_ws.settimeout(1)
            print("Successfully Connected to PC!")
            return new_ws
        except (websocket.WebSocketException, socket.error) as e:
            print(f"Server not ready ({e}). Retrying...")
            time.sleep(2)

ws = connect_ws()
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        current_time = time.time()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        data_to_send = None

        # 1. CHECK FOR HANDS
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_spec, connection_spec)
            
            # Prepare hand data if throttle allows
            if (current_time - last_send_time) > send_interval:
                # Extract landmarks in the SAME format as landmark_dataset.py (128 values)
                # Format: (63 coords Hand 1 + 1 Label) + (63 coords Hand 2 + 1 Label)
                all_landmarks = [-1.0] * 128  # Initialize with -1
                
                # Process up to 2 hands
                for i, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    if i < 2:  # Only process first 2 hands
                        # Extract 63 coordinates (21 landmarks × 3 coords: x, y, z)
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])  # Include z coordinate!
                        
                        # Hand Label (0 for Left, 1 for Right)
                        label = 0 if hand_info.classification[0].label == 'Left' else 1
                        
                        # Place in the correct position (Hand 1: 0-63, Hand 2: 64-127)
                        start_idx = i * 64
                        all_landmarks[start_idx : start_idx + 63] = coords
                        all_landmarks[start_idx + 63] = float(label)
                
                data_to_send = {
                    "landmarks": all_landmarks  # Send as flat array of 128 values
                }
                last_send_time = current_time
                last_heartbeat_time = current_time # Hand data counts as activity

        # 2. HEARTBEAT LOGIC (If no hand seen for 1 second)
        elif (current_time - last_heartbeat_time) > HEARTBEAT_INTERVAL:
            data_to_send = {"type": "ping", "status": "idle"}
            last_heartbeat_time = current_time
            print("Sending Keep-Alive Ping...")

        # 3. ACTUAL SENDING
        if data_to_send:
            try:
                ws.send(json.dumps(data_to_send))
            except (websocket.WebSocketConnectionClosedException, socket.error, BrokenPipeError):
                print("Connection lost. Reconnecting...")
                try: ws.close() 
                except: pass
                ws = connect_ws()

        cv2.imshow('Pi Feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as global_err:
    print(f"Fatal Error: {global_err}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if ws: ws.close()
