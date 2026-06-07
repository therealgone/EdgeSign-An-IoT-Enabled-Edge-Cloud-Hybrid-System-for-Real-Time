import asyncio
import websockets
import json
import numpy as np
import pickle
from collections import deque, Counter
import os
import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sign_language_model.pkl')

# Confidence threshold for detection (70% as requested)
CONFIDENCE_THRESHOLD = 0.70

# Prediction smoothing window (smaller = more responsive, larger = more stable)
SMOOTHING_WINDOW = 7  # Top-tier: balanced between speed and accuracy

# Minimum consensus for word prediction (out of smoothing window)
MIN_CONSENSUS = 5  # Requires strong agreement before showing word

# Debug mode
DEBUG_MODE = True  # Shows detailed prediction info in terminal

# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
scaler = None
label_encoder = None
actions = []
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
landmark_buffer = deque(maxlen=5)  # For temporal smoothing
last_prediction = None
prediction_count = 0

# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """Load the trained ensemble model with all preprocessing components"""
    global model, scaler, label_encoder, actions
    
    # Try multiple possible paths
    model_paths = [
        MODEL_PATH,
        os.path.join('models', 'sign_language_model.pkl'),
        'sign_language_model.pkl',
        os.path.join(BASE_DIR, 'sign_language_model.pkl')
    ]
    
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            break
    
    if not model_file:
        print("[ERROR] Model file not found!")
        print(f"Searched in: {model_paths}")
        return False
    
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        actions = model_data['actions']
        
        print("=" * 60)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model file: {model_file}")
        print(f"Actions ({len(actions)}): {actions}")
        print(f"Training accuracy: {model_data.get('train_accuracy', 'N/A'):.2%}")
        print(f"Test accuracy: {model_data.get('test_accuracy', 'N/A'):.2%}")
        
        # Determine expected feature count from scaler
        if hasattr(scaler, 'n_features_in_'):
            print(f"Expected features: {scaler.n_features_in_}")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# LANDMARK PREPROCESSING (MATCHES YOUR TESTING CODE EXACTLY)
# ============================================================

def normalize_landmarks(hand_landmarks_array):
    """
    EXACT normalization from your testing code
    
    Args:
        hand_landmarks_array: numpy array of shape (21, 3) with x,y,z coordinates
        
    Returns:
        Flattened normalized landmarks (63 values)
    """
    res = hand_landmarks_array.copy()
    
    # Translation - center at wrist
    wrist = res[0].copy()
    res = res - wrist
    
    # Scale - normalize by max distance
    distances = np.linalg.norm(res, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        res = res / max_distance
    
    # Rotation - align to middle finger MCP
    middle_mcp = res[9]
    angle = np.arctan2(middle_mcp[1], middle_mcp[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    res[:, :2] = res[:, :2] @ rotation_matrix.T
    
    return res.flatten()

def get_stabilized_landmarks(hand_landmarks_array):
    """
    Apply temporal smoothing (from your testing code)
    """
    normalized = normalize_landmarks(hand_landmarks_array)
    landmark_buffer.append(normalized)
    
    if len(landmark_buffer) >= 3:
        return np.mean(landmark_buffer, axis=0)
    else:
        return normalized

def preprocess_landmarks(landmarks_raw):
    """
    Preprocess landmarks from Raspberry Pi format to model format
    
    Args:
        landmarks_raw: Raw 128-value array from Raspberry Pi
                      [Hand1: 63 coords + 1 label] + [Hand2: 63 coords + 1 label]
        
    Returns:
        Preprocessed feature vector (63 values) matching training format
    """
    try:
        # Convert to numpy array
        landmarks_raw = np.array(landmarks_raw, dtype=np.float32)
        
        # Extract first hand only (indices 0-62 for x,y,z coords of 21 landmarks)
        hand1_coords = landmarks_raw[0:63].reshape(21, 3)
        
        # Check if hand is present (not all -1)
        if np.all(hand1_coords == -1):
            # Try second hand
            hand2_coords = landmarks_raw[64:127].reshape(21, 3)
            if not np.all(hand2_coords == -1):
                hand1_coords = hand2_coords
            else:
                # No valid hand data
                return None
        
        # Apply the EXACT same normalization as testing code
        features = get_stabilized_landmarks(hand1_coords)
        
        return features
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_sign(landmarks):
    """
    Predict sign language word from landmarks using ensemble model
    
    Returns:
        tuple: (predicted_word, confidence, top_3_predictions)
    """
    global prediction_count
    
    if model is None or scaler is None or label_encoder is None:
        return None, 0.0, []
    
    try:
        # Preprocess landmarks (extract features)
        features = preprocess_landmarks(landmarks)
        
        if features is None:
            return None, 0.0, []
        
        # Reshape for sklearn (needs 2D array)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict using ensemble model
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_word = label_encoder.classes_[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3 = [(label_encoder.classes_[i], probabilities[i]) for i in top_3_indices]
        
        prediction_count += 1
        
        return predicted_word, confidence, top_3
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, []

# ============================================================
# PREDICTION SMOOTHING (TOP-TIER STABILITY)
# ============================================================

def smooth_prediction(word, confidence):
    """
    Apply advanced smoothing to reduce false positives
    Uses consensus-based approach for maximum accuracy
    
    Args:
        word: Current predicted word
        confidence: Prediction confidence
        
    Returns:
        tuple: (final_word, should_display)
    """
    global prediction_history, last_prediction
    
    # Only consider predictions above threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return None, False
    
    # Add to history
    prediction_history.append(word)
    
    # Need minimum history for smoothing
    if len(prediction_history) < MIN_CONSENSUS:
        return None, False
    
    # Count occurrences in recent history
    word_counts = Counter(prediction_history)
    most_common_word, count = word_counts.most_common(1)[0]
    
    # Require strong consensus (MIN_CONSENSUS out of SMOOTHING_WINDOW)
    if count >= MIN_CONSENSUS:
        # Check if this is a new prediction
        if most_common_word != last_prediction:
            last_prediction = most_common_word
            return most_common_word, True
    
    return None, False

# ============================================================
# WEBSOCKET HANDLER
# ============================================================

async def handle_landmarks(websocket):
    """Handle incoming landmark data from Raspberry Pi"""
    global prediction_history, last_prediction, landmark_buffer
    
    print("[CONNECTION] Raspberry Pi connected!")
    print("=" * 60)
    
    try:
        async for message in websocket:
            try:
                # Parse incoming data
                data = json.loads(message)
                
                # Skip heartbeat messages
                if isinstance(data, dict) and data.get("type") == "ping":
                    continue
                
                # Extract landmarks
                landmarks = data.get("landmarks") if isinstance(data, dict) else data
                
                if landmarks is None:
                    continue
                
                # Validate landmarks
                landmarks = np.array(landmarks, dtype=np.float32)
                if landmarks.shape[0] != 128:
                    print(f"[WARNING] Invalid landmark size: {landmarks.shape[0]}")
                    continue
                
                # Predict sign
                word, confidence, top_3 = predict_sign(landmarks)
                
                if word is None:
                    continue
                
                # Apply smoothing
                final_word, should_display = smooth_prediction(word, confidence)
                
                # Display prediction (DEBUG MODE)
                if DEBUG_MODE:
                    if confidence >= CONFIDENCE_THRESHOLD:
                        print(f"\n[FRAME #{prediction_count}]")
                        print(f"Raw Prediction: {word.upper()} (Confidence: {confidence:.1%})")
                        print(f"Top 3:")
                        for w, conf in top_3:
                            # ASCII progress bar (Windows-compatible)
                            bar_length = int(conf * 30)
                            bar = "#" * bar_length + "-" * (30 - bar_length)
                            print(f"  {w:15s} [{bar}] {conf:.1%}")
                        
                        if should_display:
                            print("=" * 60)
                            print(f">>> DETECTED WORD: {final_word.upper()} <<<")
                            print(f"    Confidence: {confidence:.1%}")
                            print(f"    Consensus: {Counter(prediction_history)[final_word]}/{len(prediction_history)}")
                            print("=" * 60)
                    else:
                        print(f"[LOW CONFIDENCE] {word} ({confidence:.1%}) - Threshold: {CONFIDENCE_THRESHOLD:.1%}")
                
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] Frame processing error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except websockets.exceptions.ConnectionClosed:
        print("\n[DISCONNECTED] Raspberry Pi disconnected")
        # Reset state
        prediction_history.clear()
        landmark_buffer.clear()
        last_prediction = None
    except Exception as e:
        print(f"\n[ERROR] Connection error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# SERVER
# ============================================================

async def main():
    """Start WebSocket server"""
    print("\n")
    print("=" * 60)
    print("SIGN LANGUAGE RECOGNITION SYSTEM")
    print("WebSocket Receiver (Top-Tier Detection)")
    print("=" * 60)
    
    # Load model
    if not load_model():
        print("\n[FATAL] Cannot start without model!")
        return
    
    print(f"\n[CONFIG]")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Smoothing Window: {SMOOTHING_WINDOW} frames")
    print(f"  Consensus Required: {MIN_CONSENSUS}/{SMOOTHING_WINDOW}")
    print(f"  Debug Mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
    
    print("\n" + "=" * 60)
    print("Server listening on ws://0.0.0.0:8765")
    print("Waiting for Raspberry Pi connection...")
    print("=" * 60 + "\n")
    
    # Start WebSocket server
    async with websockets.serve(
        handle_landmarks, 
        "0.0.0.0", 
        8765,
        ping_interval=20,
        ping_timeout=20,
        max_size=10**6  # 1MB max message size
    ):
        await asyncio.Future()  # Run forever

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped by user")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()