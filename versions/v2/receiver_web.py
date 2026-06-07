"""
Enhanced Sign Language Receiver with Web Interface
Runs on Laptop - Receives landmarks from Pi, runs ML model, serves web UI
"""
from aiohttp import web
import asyncio
import websockets
import json
import numpy as np
import pickle
from collections import deque, Counter
import os
import sys
import aiohttp
import aiohttp_cors
import base64
from datetime import datetime
import threading
import io

# Optional imports for advanced features
try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False
    print("[WARNING] OCR not available. Install: pip install pillow pytesseract")

try:
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    print("[WARNING] Audio not available. Install: pip install SpeechRecognition pyaudio")

try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except:
    BLUETOOTH_AVAILABLE = False
    print("[WARNING] Bluetooth not available. Install: pip install pybluez")

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sign_language_model.pkl')

# Server Configuration
WEBSOCKET_PORT = 8765  # For Pi connection
WEB_SERVER_PORT = 3000  # For web interface

# ML Model Configuration
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7
MIN_CONSENSUS = 5
DEBUG_MODE = True

# ============================================================
# GLOBAL STATE
# ============================================================

class SystemState:
    def __init__(self):
        # ML Model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.actions = []
        
        # Prediction state
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.landmark_buffer = deque(maxlen=5)
        self.last_prediction = None
        self.prediction_count = 0
        
        # Word collection
        self.words = []
        self.context = ""
        
        # WebSocket connections
        self.pi_websocket = None
        self.web_clients = set()
        
        # Status
        self.pi_connected = False
        self.last_word = ""
        self.last_confidence = 0.0

state = SystemState()

# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """Load ML model"""
    model_paths = [
        MODEL_PATH,
        os.path.join('models', 'sign_language_model.pkl'),
        'sign_language_model.pkl',
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                
                state.model = model_data['model']
                state.scaler = model_data['scaler']
                state.label_encoder = model_data['label_encoder']
                state.actions = model_data['actions']
                
                print(f"[OK] Model loaded from {path}")
                print(f"[OK] Actions: {state.actions}")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
    
    print("[ERROR] Model file not found!")
    return False

# ============================================================
# LANDMARK PREPROCESSING
# ============================================================

def normalize_landmarks(hand_landmarks_array):
    """Normalize landmarks (same as training)"""
    res = hand_landmarks_array.copy()
    
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

def get_stabilized_landmarks(hand_landmarks_array):
    """Apply temporal smoothing"""
    normalized = normalize_landmarks(hand_landmarks_array)
    state.landmark_buffer.append(normalized)
    
    if len(state.landmark_buffer) >= 3:
        return np.mean(state.landmark_buffer, axis=0)
    else:
        return normalized

def preprocess_landmarks(landmarks_raw):
    """Preprocess landmarks from Pi"""
    try:
        landmarks_raw = np.array(landmarks_raw, dtype=np.float32)
        hand1_coords = landmarks_raw[0:63].reshape(21, 3)
        
        if np.all(hand1_coords == -1):
            hand2_coords = landmarks_raw[64:127].reshape(21, 3)
            if not np.all(hand2_coords == -1):
                hand1_coords = hand2_coords
            else:
                return None
        
        return get_stabilized_landmarks(hand1_coords)
    except Exception as e:
        print(f"[ERROR] Preprocessing: {e}")
        return None

# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_sign(landmarks):
    """Predict sign from landmarks"""
    if state.model is None:
        return None, 0.0, []
    
    try:
        features = preprocess_landmarks(landmarks)
        if features is None:
            return None, 0.0, []
        
        features = features.reshape(1, -1)
        features_scaled = state.scaler.transform(features)
        
        probabilities = state.model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_word = state.label_encoder.classes_[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3 = [(state.label_encoder.classes_[i], probabilities[i]) for i in top_3_indices]
        
        state.prediction_count += 1
        return predicted_word, confidence, top_3
    except Exception as e:
        print(f"[ERROR] Prediction: {e}")
        return None, 0.0, []

def smooth_prediction(word, confidence):
    """Apply smoothing"""
    if confidence < CONFIDENCE_THRESHOLD:
        return None, False
    
    state.prediction_history.append(word)
    
    if len(state.prediction_history) < MIN_CONSENSUS:
        return None, False
    
    word_counts = Counter(state.prediction_history)
    most_common_word, count = word_counts.most_common(1)[0]
    
    if count >= MIN_CONSENSUS:
        if most_common_word != state.last_prediction:
            state.last_prediction = most_common_word
            return most_common_word, True
    
    return None, False

# ============================================================
# WEB CLIENT COMMUNICATION
# ============================================================

async def broadcast_to_web_clients(message):
    """Send message to all connected web clients"""
    if state.web_clients:
        await asyncio.gather(
            *[client.send_str(json.dumps(message)) for client in state.web_clients],
            return_exceptions=True
        )

async def send_word_to_web(word, confidence):
    """Send detected word to web interface"""
    await broadcast_to_web_clients({
        'type': 'word_detected',
        'word': word,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })

async def send_status_to_web():
    """Send system status to web"""
    await broadcast_to_web_clients({
        'type': 'status',
        'pi_connected': state.pi_connected,
        'words_count': len(state.words),
        'last_word': state.last_word,
        'last_confidence': state.last_confidence
    })

# ============================================================
# PI WEBSOCKET HANDLER
# ============================================================

async def handle_pi_connection(websocket):
    """Handle WebSocket connection from Raspberry Pi"""
    state.pi_websocket = websocket
    state.pi_connected = True
    print("[PI] Connected!")
    await send_status_to_web()
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Handle different message types from Pi
                if isinstance(data, dict):
                    msg_type = data.get("type")
                    
                    if msg_type == "ping":
                        continue
                    elif msg_type == "frame":
                        # Frame for OCR
                        await handle_frame_from_pi(data)
                        continue
                    elif msg_type == "audio":
                        # Audio data
                        await handle_audio_from_pi(data)
                        continue
                
                # Landmark data for prediction
                landmarks = data.get("landmarks") if isinstance(data, dict) else data
                if landmarks is None:
                    continue
                
                landmarks = np.array(landmarks, dtype=np.float32)
                if landmarks.shape[0] != 128:
                    continue
                
                # Predict
                word, confidence, top_3 = predict_sign(landmarks)
                if word is None:
                    continue
                
                # Smooth and display
                final_word, should_display = smooth_prediction(word, confidence)
                
                if should_display:
                    state.last_word = final_word
                    state.last_confidence = confidence
                    print(f"[DETECTED] {final_word.upper()} ({confidence:.1%})")
                    await send_word_to_web(final_word, confidence)
                
            except Exception as e:
                print(f"[ERROR] Processing Pi message: {e}")
                continue
    
    except websockets.exceptions.ConnectionClosed:
        print("[PI] Disconnected")
    finally:
        state.pi_connected = False
        state.pi_websocket = None
        await send_status_to_web()

async def handle_frame_from_pi(data):
    """Handle frame from Pi for OCR"""
    try:
        frame_base64 = data.get("frame")
        if not frame_base64 or not TESSERACT_AVAILABLE:
            return
        
        # Decode image
        image_data = base64.b64decode(frame_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Run OCR
        text = pytesseract.image_to_string(image)
        text = text.strip()
        
        if text:
            # Send to web for confirmation
            await broadcast_to_web_clients({
                'type': 'ocr_result',
                'text': text,
                'needs_confirmation': True
            })
    except Exception as e:
        print(f"[ERROR] OCR: {e}")

async def handle_audio_from_pi(data):
    """Handle audio from Pi for transcription"""
    try:
        audio_base64 = data.get("audio")
        if not audio_base64 or not AUDIO_AVAILABLE:
            return
        
        # Decode and transcribe (implementation depends on audio format)
        # This is a placeholder - actual implementation would decode audio
        await broadcast_to_web_clients({
            'type': 'audio_result',
            'text': "Audio transcription here",
            'needs_confirmation': True
        })
    except Exception as e:
        print(f"[ERROR] Audio: {e}")

# ============================================================
# SEND COMMANDS TO PI
# ============================================================

async def send_command_to_pi(command, params=None):
    """Send command to Raspberry Pi"""
    if not state.pi_websocket:
        return {'success': False, 'error': 'Pi not connected'}
    
    try:
        message = {
            'type': 'command',
            'command': command,
            'params': params or {}
        }
        await state.pi_websocket.send(json.dumps(message))
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================
# WEB SERVER HANDLERS
# ============================================================

async def websocket_handler(request):
    """Handle WebSocket connections from web clients"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    state.web_clients.add(ws)
    print(f"[WEB] Client connected (total: {len(state.web_clients)})")
    
    # Send initial status
    await ws.send_str(json.dumps({
        'type': 'init',
        'pi_connected': state.pi_connected,
        'words': state.words,
        'context': state.context,
        'actions': state.actions
    }))
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await handle_web_command(data, ws)
                except Exception as e:
                    print(f"[ERROR] Handling web command: {e}")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f'[WEB] Connection error: {ws.exception()}')
    finally:
        state.web_clients.discard(ws)
        print(f"[WEB] Client disconnected (remaining: {len(state.web_clients)})")
    
    return ws

async def handle_web_command(data, ws):
    """Handle commands from web interface"""
    command = data.get('command')
    
    if command == 'add_word':
        word = data.get('word', state.last_word)
        if word:
            state.words.append(word)
            print(f"[ADD] Word: {word} (Total: {len(state.words)})")
            await broadcast_to_web_clients({
                'type': 'words_updated',
                'words': state.words
            })
    
    elif command == 'delete_last':
        if state.words:
            removed = state.words.pop()
            print(f"[DELETE] Removed: {removed}")
            await broadcast_to_web_clients({
                'type': 'words_updated',
                'words': state.words
            })
    
    elif command == 'clear_words':
        state.words = []
        print("[CLEAR] All words cleared")
        await broadcast_to_web_clients({
            'type': 'words_updated',
            'words': state.words
        })
    
    elif command == 'send_to_ai':
        # Send words to AI for natural rephrasing
        words_text = " ".join(state.words)
        context_text = state.context
        
        # This is where you'd call your AI API
        # For now, just echo back
        ai_response = f"AI would process: {words_text}"
        if context_text:
            ai_response += f" (with context: {context_text[:50]}...)"
        
        await ws.send_str(json.dumps({
            'type': 'ai_response',
            'response': ai_response,
            'tokens_used': len(words_text.split()) + len(context_text.split())
        }))
    
    elif command == 'remove_context':
        state.context = ""
        print("[CONTEXT] Cleared")
        await broadcast_to_web_clients({
            'type': 'context_updated',
            'context': state.context
        })
    
    elif command == 'add_context':
        text = data.get('text', '')
        state.context += " " + text
        state.context = state.context.strip()
        print(f"[CONTEXT] Added: {text[:50]}...")
        await broadcast_to_web_clients({
            'type': 'context_updated',
            'context': state.context
        })
    
    elif command == 'request_ocr':
        # Request Pi to capture frame for OCR
        result = await send_command_to_pi('capture_frame')
        await ws.send_str(json.dumps(result))
    
    elif command == 'request_audio':
        # Request Pi to record audio
        duration = data.get('duration', 30)
        result = await send_command_to_pi('record_audio', {'duration': duration})
        await ws.send_str(json.dumps(result))
    
    elif command == 'scan_bluetooth':
        # Scan for Bluetooth devices
        devices = []
        if BLUETOOTH_AVAILABLE:
            try:
                devices = bluetooth.discover_devices(duration=5, lookup_names=True)
                devices = [{'address': addr, 'name': name} for addr, name in devices]
            except Exception as e:
                print(f"[ERROR] Bluetooth scan: {e}")
        
        await ws.send_str(json.dumps({
            'type': 'bluetooth_devices',
            'devices': devices
        }))
    
    elif command == 'connect_bluetooth':
        # Connect to Bluetooth device
        address = data.get('address')
        # Implementation would go here
        await ws.send_str(json.dumps({
            'type': 'bluetooth_status',
            'connected': False,
            'message': 'Bluetooth connection not implemented'
        }))

async def serve_index(request):
    """Serve the main HTML page"""
    html_path = os.path.join(BASE_DIR, 'web', 'index.html')
    if os.path.exists(html_path):
        return web.FileResponse(html_path)
    else:
        return web.Response(text="Web interface not found. Run setup first.", status=404)

# ============================================================
# MAIN FUNCTION
# ============================================================

async def main():
    """Start all servers"""
    print("\n" + "=" * 60)
    print("SIGN LANGUAGE RECOGNITION SYSTEM")
    print("Web Interface + ML Detection")
    print("=" * 60)
    
    # Load model
    if not load_model():
        print("[FATAL] Cannot start without model!")
        return
    
    print(f"\n[CONFIG]")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  WebSocket Port (Pi): {WEBSOCKET_PORT}")
    print(f"  Web Server Port: {WEB_SERVER_PORT}")
    print(f"  OCR Available: {TESSERACT_AVAILABLE}")
    print(f"  Audio Available: {AUDIO_AVAILABLE}")
    print(f"  Bluetooth Available: {BLUETOOTH_AVAILABLE}")
    
    # Start Pi WebSocket server
    pi_server = await websockets.serve(
        handle_pi_connection,
        "0.0.0.0",
        WEBSOCKET_PORT,
        ping_interval=20,
        ping_timeout=20,
        max_size=10**7  # 10MB for frames
    )
    print(f"\n[OK] Pi WebSocket listening on port {WEBSOCKET_PORT}")
    
    # Start Web Server
    app = web.Application()
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })
    
    # Add routes
    app.router.add_get('/', serve_index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_static('/static', os.path.join(BASE_DIR, 'web', 'static'), name='static')
    
    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', WEB_SERVER_PORT)
    await site.start()
    
    print(f"[OK] Web server listening on http://localhost:{WEB_SERVER_PORT}")
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("=" * 60)
    print(f"\n1. Open browser: http://localhost:{WEB_SERVER_PORT}")
    print(f"2. Start sender.py on Raspberry Pi")
    print(f"3. Make signs and control via web interface\n")
    
    # Keep running
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped")
