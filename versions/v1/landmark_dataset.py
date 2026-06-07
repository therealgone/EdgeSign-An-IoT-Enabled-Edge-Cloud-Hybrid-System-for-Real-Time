import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration ---
DATA_PATH = 'SignData'
actions = np.array(['hello', 'thanks', 'yes', 'no', 'go', 'what', 'time', 'price', 'you', 'good morning', 'me', 'eat', 'now', 'stop', 'need', 'where', 'sorry', 'help','call'])
action_index = 0
current_word = actions[action_index]

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Mode State
two_hand_mode = False 

def get_hands_model(is_two_handed):
    num = 2 if is_two_handed else 1
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num,
        min_detection_confidence=0.7, # Increased for better quality
        min_tracking_confidence=0.5
    )

hands = get_hands_model(two_hand_mode)

def extract_landmarks(results):
    """
    Returns 128 values: 
    (63 landmarks Hand 1 + 1 Label) + (63 landmarks Hand 2 + 1 Label)
    Label: 0 for Left, 1 for Right, -1 for No Hand
    """
    all_landmarks = np.zeros(128) - 1 # Initialize with -1
    
    if results.multi_hand_landmarks:
        for i, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            if i < 2:
                # 63 landmark coordinates (x, y, z)
                res = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                # Hand Label (0 for Left, 1 for Right)
                label = 0 if hand_info.classification[0].label == 'Left' else 1
                
                start_idx = i * 64
                all_landmarks[start_idx : start_idx + 63] = res
                all_landmarks[start_idx + 63] = label # The 64th value is the hand ID
    return all_landmarks

def get_next_sequence_num(path):
    """Finds the highest number in folder and returns next index to avoid overwrite"""
    if not os.path.exists(path):
        return 0
    existing_files = [int(f.split('.')[0]) for f in os.listdir(path) if f.split('.')[0].isdigit()]
    return max(existing_files) + 1 if existing_files else 0

cap = cv2.VideoCapture(0)
recording = False
current_sequence = []

print("CONTROLS:")
print("'m' - Toggle Hands | 'n' - Next Word")
print("'1' - Capture TRAIN Data | '2' - Capture TEST Data | 'q' - Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # UI Overlay
    cv2.rectangle(frame, (0,0), (FRAME_WIDTH, 70), (40, 40, 40), -1)
    mode_text = "2-HANDS" if two_hand_mode else "1-HAND"
    cv2.putText(frame, f"WORD: {current_word.upper()}", (10, 25), 0, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"MODE: {mode_text} | '1' TRAIN | '2' TEST", (10, 55), 0, 0.5, (0, 255, 255), 1)

    cv2.imshow('Advanced Data Collector', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('m'):
        two_hand_mode = not two_hand_mode
        hands.close()
        hands = get_hands_model(two_hand_mode)

    if key == ord('n'):
        action_index = (action_index + 1) % len(actions)
        current_word = actions[action_index]

    # Save logic for Train (1) and Test (2)
    if key in [ord('1'), ord('2')]:
        split = 'train' if key == ord('1') else 'test'
        save_path = os.path.join(DATA_PATH, current_word, split)
        os.makedirs(save_path, exist_ok=True)
        
        # Get next file index to prevent overwriting
        file_idx = get_next_sequence_num(save_path)
        
        # Capture current frame landmarks
        data = extract_landmarks(results)
        np.save(os.path.join(save_path, f"{file_idx}.npy"), data)
        
        print(f"Saved {split} sample {file_idx} for {current_word}")
        # Visual feedback
        cv2.rectangle(frame, (0,0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 0), 10)
        cv2.imshow('Advanced Data Collector', frame)
        cv2.waitKey(100)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
