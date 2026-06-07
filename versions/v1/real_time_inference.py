import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque, Counter

# --- Configuration ---
MODEL_PATH = 'models/sign_language_model.pkl'  # Use full ensemble model
# MODEL_PATH = 'models/sign_language_model_lite.pkl'  # Uncomment for faster inference

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to display prediction
PREDICTION_BUFFER_SIZE = 60  # Smooth predictions over N frames

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)

# Load model
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please run train_model.py first to train the model.")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
actions = model_data['actions']

print(f"✓ Model loaded successfully")
print(f"  Recognized signs: {', '.join(actions)}")
if 'test_accuracy' in model_data:
    print(f"  Model accuracy: {model_data['test_accuracy'] * 100:.2f}%")

# Landmark normalization (same as training)
landmark_buffer = deque(maxlen=5)

def normalize_landmarks(hand_landmarks):
    """
    Same normalization as training
    """
    res = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Translation
    wrist = res[0].copy()
    res = res - wrist
    
    # Scale
    distances = np.linalg.norm(res, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        res = res / max_distance
    
    # Rotation
    middle_mcp = res[9]
    angle = np.arctan2(middle_mcp[1], middle_mcp[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    res[:, :2] = res[:, :2] @ rotation_matrix.T
    
    return res.flatten()

def get_stabilized_landmarks(hand_landmarks):
    """
    Apply temporal smoothing
    """
    normalized = normalize_landmarks(hand_landmarks)
    landmark_buffer.append(normalized)
    
    if len(landmark_buffer) >= 3:
        return np.mean(landmark_buffer, axis=0)
    else:
        return normalized

# Prediction smoothing buffer
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)

def get_smooth_prediction(current_prediction, current_confidence):
    """
    Smooth predictions over time to reduce jitter
    Uses majority voting
    """
    prediction_buffer.append((current_prediction, current_confidence))
    
    if len(prediction_buffer) < 3:
        return current_prediction, current_confidence
    
    # Get most common prediction
    predictions = [p[0] for p in prediction_buffer]
    prediction_counts = Counter(predictions)
    most_common_pred = prediction_counts.most_common(1)[0][0]
    
    # Average confidence for that prediction
    confidences = [p[1] for p in prediction_buffer if p[0] == most_common_pred]
    avg_confidence = np.mean(confidences)
    
    return most_common_pred, avg_confidence

# Statistics tracking
total_predictions = 0
correct_predictions = 0
current_true_label = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("\n" + "=" * 60)
print("REAL-TIME SIGN LANGUAGE RECOGNITION")
print("=" * 60)
print("CONTROLS:")
print("  'r' - Reset prediction buffer")
print("  'c' - Clear statistics")
print("  'h' - Toggle help overlay")
print("  'q' - Quit")
print("=" * 60)
print("\nShow a sign to start recognition...\n")

show_help = True
prediction_text = ""
confidence_value = 0.0
prediction_color = (100, 100, 100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocessing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        
        # Make prediction
        if results.multi_handedness[0].classification[0].score > 0.8:
            hand_landmarks = results.multi_hand_landmarks[0]
            stabilized_data = get_stabilized_landmarks(hand_landmarks)
            
            # Preprocess for model
            data_scaled = scaler.transform([stabilized_data])
            
            # Predict
            prediction_proba = model.predict_proba(data_scaled)[0]
            predicted_class = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
            # Apply smoothing
            smooth_label, smooth_confidence = get_smooth_prediction(predicted_label, confidence)
            
            # Update display if confidence is high enough
            if smooth_confidence >= CONFIDENCE_THRESHOLD:
                prediction_text = smooth_label.upper()
                confidence_value = smooth_confidence
                
                # Color based on confidence
                if smooth_confidence > 0.9:
                    prediction_color = (0, 255, 0)  # Green - Very confident
                elif smooth_confidence > 0.75:
                    prediction_color = (0, 255, 255)  # Yellow - Confident
                else:
                    prediction_color = (0, 165, 255)  # Orange - Moderate
                
                total_predictions += 1
            else:
                prediction_text = "LOW CONFIDENCE"
                confidence_value = smooth_confidence
                prediction_color = (0, 0, 255)
    else:
        # No hand detected
        prediction_text = ""
        confidence_value = 0.0
        landmark_buffer.clear()
    
    # ===== UI RENDERING =====
    
    # Main prediction box (center)
    if prediction_text:
        # Calculate text size for dynamic sizing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5 if len(prediction_text) <= 10 else 1.8
        thickness = 4
        
        text_size = cv2.getTextSize(prediction_text, font, font_scale, thickness)[0]
        text_x = (FRAME_WIDTH - text_size[0]) // 2
        text_y = (FRAME_HEIGHT + text_size[1]) // 2
        
        # Background box
        padding = 30
        box_coords = (
            text_x - padding,
            text_y - text_size[1] - padding,
            text_x + text_size[0] + padding,
            text_y + padding
        )
        
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (box_coords[0], box_coords[1]),
                     (box_coords[2], box_coords[3]),
                     (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Border
        cv2.rectangle(frame,
                     (box_coords[0], box_coords[1]),
                     (box_coords[2], box_coords[3]),
                     prediction_color, 4)
        
        # Text
        cv2.putText(frame, prediction_text, (text_x, text_y),
                   font, font_scale, prediction_color, thickness)
        
        # Confidence bar
        bar_width = text_size[0]
        bar_height = 15
        bar_x = text_x
        bar_y = box_coords[3] + 10
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)
        
        # Confidence fill
        fill_width = int(bar_width * confidence_value)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     prediction_color, -1)
        
        # Confidence percentage
        conf_text = f"{confidence_value * 100:.0f}%"
        cv2.putText(frame, conf_text,
                   (bar_x + bar_width + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Top info bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 40), (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    status_text = "HAND DETECTED" if results.multi_hand_landmarks else "NO HAND"
    status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Prediction count
    cv2.putText(frame, f"Predictions: {total_predictions}",
               (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Help overlay
    if show_help:
        help_overlay = frame.copy()
        help_y = FRAME_HEIGHT - 120
        cv2.rectangle(help_overlay, (0, help_y), (FRAME_WIDTH, FRAME_HEIGHT), (40, 40, 40), -1)
        frame = cv2.addWeighted(help_overlay, 0.8, frame, 0.2, 0)
        
        cv2.putText(frame, f"Recognized Signs ({len(actions)}):",
                   (10, help_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show signs in multiple rows
        signs_text = ", ".join(actions)
        y_offset = help_y + 50
        max_width = FRAME_WIDTH - 20
        
        words = signs_text.split(", ")
        current_line = ""
        
        for word in words:
            test_line = current_line + word + ", "
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            
            if text_size > max_width:
                cv2.putText(frame, current_line.rstrip(", "),
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                current_line = word + ", "
                y_offset += 20
            else:
                current_line = test_line
        
        if current_line:
            cv2.putText(frame, current_line.rstrip(", "),
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Bottom control bar
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (0, FRAME_HEIGHT - 30), (FRAME_WIDTH, FRAME_HEIGHT), (40, 40, 40), -1)
    frame = cv2.addWeighted(bottom_overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "r:Reset | c:Clear Stats | h:Toggle Help | q:Quit",
               (10, FRAME_HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Sign Language Recognition', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):  # Reset buffer
        prediction_buffer.clear()
        landmark_buffer.clear()
        print("✓ Prediction buffer reset")
    
    elif key == ord('c'):  # Clear statistics
        total_predictions = 0
        correct_predictions = 0
        print("✓ Statistics cleared")
    
    elif key == ord('h'):  # Toggle help
        show_help = not show_help
    
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "=" * 60)
print("SESSION SUMMARY")
print("=" * 60)
print(f"Total predictions: {total_predictions}")
print("=" * 60)
