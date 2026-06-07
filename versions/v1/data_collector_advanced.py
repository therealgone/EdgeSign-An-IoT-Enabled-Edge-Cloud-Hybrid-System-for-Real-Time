import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque

# --- Configuration ---
DATA_PATH = 'SignData'
actions = np.array(['preset1', 'preset2', 'preset3'])

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
BATCH_SIZE_50 = 50
BATCH_SIZE_100 = 100

# MediaPipe setup with optimal settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,      # CRITICAL: tracking mode for stability
    max_num_hands=1,              # ONE hand only for consistency
             # 1 is stable and accurate
    min_detection_confidence=0.8, # High confidence threshold
    min_tracking_confidence=0.7   # Good tracking
)

# Stabilization buffer for landmark smoothing
landmark_buffer = deque(maxlen=5)

def normalize_landmarks(hand_landmarks):
    """
    Advanced normalization pipeline:
    1. Translation (wrist to origin)
    2. Scale normalization
    3. Rotation alignment
    Returns 63 normalized values (21 landmarks x 3 coordinates)
    """
    # Extract all 21 landmarks
    res = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Step 1: Translation - Move wrist to origin
    wrist = res[0].copy()
    res = res - wrist
    
    # Step 2: Scale normalization - Make hand size invariant
    distances = np.linalg.norm(res, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        res = res / max_distance
    
    # Step 3: Rotation alignment - Align hand orientation
    # Use vector from wrist to middle finger MCP as reference
    middle_mcp = res[9]  # Middle finger MCP joint
    angle = np.arctan2(middle_mcp[1], middle_mcp[0])
    
    # 2D rotation matrix
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Apply rotation to x,y coordinates
    res[:, :2] = res[:, :2] @ rotation_matrix.T
    
    return res.flatten()  # Return 63 values

def validate_hand_detection(results, frame_shape):
    """
    Validate that hand detection is reliable:
    - Hand confidence is high
    - All landmarks are within frame
    - No sudden jumps in position
    """
    if not results.multi_hand_landmarks:
        return False
    
    # Check confidence score
    if results.multi_handedness[0].classification[0].score < 0.8:
        return False
    
    # Check if landmarks are within frame bounds
    hand_landmarks = results.multi_hand_landmarks[0]
    margin = 0.05
    for lm in hand_landmarks.landmark:
        if lm.x < -margin or lm.x > 1 + margin or lm.y < -margin or lm.y > 1 + margin:
            return False
    
    return True

def get_stabilized_landmarks(hand_landmarks):
    """
    Apply temporal smoothing to reduce jitter
    Average over last 5 frames
    """
    normalized = normalize_landmarks(hand_landmarks)
    landmark_buffer.append(normalized)
    
    if len(landmark_buffer) >= 3:  # Need at least 3 frames
        return np.mean(landmark_buffer, axis=0)
    else:
        return normalized

def get_next_sequence_num(path):
    """Finds the highest number in folder and returns next index"""
    if not os.path.exists(path):
        return 0
    existing_files = [int(f.split('.')[0]) for f in os.listdir(path) 
                     if f.endswith('.npy') and f.split('.')[0].isdigit()]
    return max(existing_files) + 1 if existing_files else 0

def delete_word_data(word):
    """Delete all data for a specific word"""
    word_path = os.path.join(DATA_PATH, word)
    if os.path.exists(word_path):
        import shutil
        shutil.rmtree(word_path)
        print(f"✓ Deleted all data for '{word}'")
        return True
    return False

# UI State
action_index = 0
current_word = actions[action_index]
capturing = False
capture_count = 0
target_captures = 0
current_split = 'train'
frame_skip_counter = 0
WARMUP_FRAMES = 15  # Skip first 15 frames for stability

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("=" * 60)
print("ADVANCED SIGN LANGUAGE DATA COLLECTOR")
print("=" * 60)
print("CONTROLS:")
print("  'n' - Next word")
print("  'p' - Previous word")
print("  'd' - DELETE current word data")
print("  's' - Skip current word")
print("")
print("CAPTURE (AUTO-BATCH):")
print("  '1' - Capture 50 TRAIN samples")
print("  '2' - Capture 100 TRAIN samples")
print("  '3' - Capture 50 TEST samples")
print("  '4' - Capture 100 TEST samples")
print("")
print("  'q' - Quit")
print("=" * 60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocessing
    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False  # Performance optimization
    
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    # Draw hand landmarks
    hand_detected = False
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                 mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                                 mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))
        hand_detected = validate_hand_detection(results, frame.shape)
    
    # Auto-capture logic
    if capturing and hand_detected:
        frame_skip_counter += 1
        
        # Skip initial unstable frames
        if frame_skip_counter > WARMUP_FRAMES:
            save_path = os.path.join(DATA_PATH, current_word, current_split)
            os.makedirs(save_path, exist_ok=True)
            
            # Get stabilized landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            stabilized_data = get_stabilized_landmarks(hand_landmarks)
            
            # Save
            file_idx = get_next_sequence_num(save_path)
            np.save(os.path.join(save_path, f"{file_idx}.npy"), stabilized_data)
            
            capture_count += 1
            
            # Visual feedback
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 0), 15)
            
            # Check if batch complete
            if capture_count >= target_captures:
                print(f"✓ Completed {target_captures} {current_split} samples for '{current_word}'")
                capturing = False
                capture_count = 0
                frame_skip_counter = 0
                landmark_buffer.clear()
    
    # UI Overlay - Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 100), (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Word display
    cv2.putText(frame, f"WORD: {current_word.upper()}", 
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Status
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "HAND OK" if hand_detected else "NO HAND"
    cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Word counter
    cv2.putText(frame, f"Word {action_index + 1}/{len(actions)}", 
                (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Capture indicator
    if capturing:
        progress_text = f"CAPTURING: {capture_count}/{target_captures}"
        cv2.putText(frame, progress_text, (150, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Bottom help bar
    help_overlay = frame.copy()
    cv2.rectangle(help_overlay, (0, FRAME_HEIGHT - 60), (FRAME_WIDTH, FRAME_HEIGHT), (40, 40, 40), -1)
    frame = cv2.addWeighted(help_overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "1:50 TRAIN | 2:100 TRAIN | 3:50 TEST | 4:100 TEST", 
                (10, FRAME_HEIGHT - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "n:Next | p:Prev | d:Delete | s:Skip | q:Quit", 
                (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.imshow('Sign Language Data Collector', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('n'):  # Next word
        action_index = (action_index + 1) % len(actions)
        current_word = actions[action_index]
        landmark_buffer.clear()
        print(f"→ Switched to: {current_word}")
    
    elif key == ord('p'):  # Previous word
        action_index = (action_index - 1) % len(actions)
        current_word = actions[action_index]
        landmark_buffer.clear()
        print(f"← Switched to: {current_word}")
    
    elif key == ord('s'):  # Skip word
        action_index = (action_index + 1) % len(actions)
        current_word = actions[action_index]
        landmark_buffer.clear()
        print(f"⤭ Skipped to: {current_word}")
    
    elif key == ord('d'):  # Delete word data
        response = input(f"Delete ALL data for '{current_word}'? (yes/no): ")
        if response.lower() == 'yes':
            delete_word_data(current_word)
    
    elif key == ord('1'):  # 50 train samples
        capturing = True
        target_captures = BATCH_SIZE_50
        current_split = 'train'
        capture_count = 0
        frame_skip_counter = 0
        landmark_buffer.clear()
        print(f"▶ Capturing 50 TRAIN samples for '{current_word}'...")
    
    elif key == ord('2'):  # 100 train samples
        capturing = True
        target_captures = BATCH_SIZE_100
        current_split = 'train'
        capture_count = 0
        frame_skip_counter = 0
        landmark_buffer.clear()
        print(f"▶ Capturing 100 TRAIN samples for '{current_word}'...")
    
    elif key == ord('3'):  # 50 test samples
        capturing = True
        target_captures = BATCH_SIZE_50
        current_split = 'test'
        capture_count = 0
        frame_skip_counter = 0
        landmark_buffer.clear()
        print(f"▶ Capturing 50 TEST samples for '{current_word}'...")
    
    elif key == ord('4'):  # 100 test samples
        capturing = True
        target_captures = BATCH_SIZE_100
        current_split = 'test'
        capture_count = 0
        frame_skip_counter = 0
        landmark_buffer.clear()
        print(f"▶ Capturing 100 TEST samples for '{current_word}'...")
    
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✓ Data collection session ended")
