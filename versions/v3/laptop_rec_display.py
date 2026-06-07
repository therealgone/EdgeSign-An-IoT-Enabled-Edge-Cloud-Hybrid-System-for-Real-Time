"""
Sign Language Receiver — Laptop
EasyOCR with full preprocessing pipeline:
  flip → grayscale → 3× resize → CLAHE → sharpen → bilateral filter → adaptive threshold → EasyOCR

Pi sends a raw 1280×720 JPEG (no filters).
All image processing happens here on the laptop.
When a word is detected it is pushed back to the Pi as a 'show_word' message
so the connected SH1106 OLED can display it.
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
import traceback
import cv2

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

EASYOCR_AVAILABLE = False
easyocr_reader = None
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("[OK] easyocr imported")
except ImportError:
    print("[WARNING] easyocr not installed — run: pip install easyocr")
except Exception as e:
    print(f"[WARNING] easyocr import error: {e}")

WHISPER_AVAILABLE = False
whisper_model = None
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("[OK] openai-whisper imported")
except ImportError:
    print("[WARNING] openai-whisper not installed — run: pip install openai-whisper")
except Exception as e:
    print(f"[WARNING] whisper import error: {e}")

try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

GEMINI_AVAILABLE = False
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
    print("[OK] google-genai imported")
except ImportError:
    print("[WARNING] google-genai not installed — run: pip install google-genai")

TTS_AVAILABLE = False
try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("[OK] pyttsx3 imported")
except ImportError:
    print("[WARNING] pyttsx3 not installed — run: pip install pyttsx3")

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sign_language_model.pkl')

WEBSOCKET_PORT  = 8765
WEB_SERVER_PORT = 3000

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW     = 7
MIN_CONSENSUS        = 5
MAX_FRAME_SIZE       = 10 * 1024 * 1024  # 10 MB
PRESETS_FILE         = os.path.join(BASE_DIR, 'presets.json')

EASYOCR_MIN_CONFIDENCE = 0.3

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL   = "gemini-2.5-flash"

GEMINI_PROMPT_NO_CONTEXT = """\
You are a sentence builder for a sign language recognition system.
The user will give you a list of words detected from sign language.
Your job is to form ONE single, complete, grammatically correct, natural-sounding sentence
using ALL of those words. You have full creative freedom to add connecting words, articles,
prepositions, verb tenses, or any other words needed to make the sentence sound fluent and natural.
Every word the user gives you MUST appear in the final sentence — do not drop any.
Reply with ONLY the sentence. No explanation, no punctuation outside the sentence, no quotes.\
"""

GEMINI_PROMPT_WITH_CONTEXT = """\
You are a sentence builder for a sign language recognition system.
The user will give you:
  1. A TOPIC / CONTEXT — this is the situation or subject being discussed.
  2. A list of WORDS detected from sign language.

Your job is to form ONE single, complete, grammatically correct, natural-sounding sentence
that fits within the given topic/context AND uses ALL of the provided words.
You have full creative freedom to add any connecting words needed to make it fluent.
Every provided word MUST appear in the final sentence — do not drop any.
Reply with ONLY the sentence. No explanation, no punctuation outside the sentence, no quotes.\
"""

TTS_RATE = 160

KEYPAD_ACTION_LABELS = {
    'scroll_down':    'Down Arrow',
    'preset_1':       'Preset Slot 1',
    'preset_2':       'Preset Slot 2',
    'add_word':       'Add Word',
    'preset_3':       'Preset Slot 3',
    'standard_key_5': 'Standard key input',
    'standard_key_6': 'Standard key input',
    'send_to_ai':     'Gemini',
    'standard_key_7': 'Standard key input',
    'standard_key_8': 'Standard key input',
    'standard_key_9': 'Standard key input',
    'request_ocr':    'OCR',
    'delete_last':    'Delete',
    'scroll_up':      'Up Arrow',
    'clear_words':    'Clear',
    'toggle_audio':   'Record Toggle',
}

# ============================================================
# GLOBAL STATE
# ============================================================

class SystemState:
    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.label_encoder = None
        self.actions       = []

        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.landmark_buffer    = deque(maxlen=5)
        self.last_prediction    = None
        self.prediction_count   = 0

        self.words   = []
        self.context = ""

        self.pi_websocket    = None
        self.web_clients     = set()
        self.pi_connected    = False
        self.last_word       = ""
        self.last_confidence = 0.0

        self.last_frame_base64 = None
        self.last_audio_base64 = None
        self.last_ocr_result   = None
        self.last_sentence     = ""
        self.last_dev_frame    = None
        self.dev_stream_enabled = False
        self.presets = {
            1: "I want a coffee",
            2: "Call me later",
            3: "Help me",
        }
        self.last_keypad_event = None

        self.error_log = deque(maxlen=50)
        self.performance_stats = {
            'frames_received':     0,
            'predictions_made':    0,
            'avg_processing_time': 0,
            'ocr_calls':           0,
            'audio_calls':         0,
            'gemini_calls':        0,
        }

state = SystemState()


def _load_presets():
    if not os.path.exists(PRESETS_FILE):
        return
    try:
        with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for slot in (1, 2, 3):
            value = str(raw.get(str(slot), raw.get(slot, state.presets[slot]))).strip()
            state.presets[slot] = value
        print(f"[PRESET] Loaded presets from {PRESETS_FILE}")
    except Exception as e:
        log_error("Loading presets", e, "Check presets.json format")


def _save_presets():
    try:
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in state.presets.items()}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log_error("Saving presets", e, "Check file permissions for presets.json")

# ============================================================
# ERROR HELPERS
# ============================================================

def log_error(context, error, solution=""):
    entry = {
        'timestamp': datetime.now().isoformat(),
        'context':   context,
        'error':     str(error),
        'type':      type(error).__name__,
        'solution':  solution,
        'traceback': traceback.format_exc(),
    }
    state.error_log.append(entry)
    print(f"\n[ERROR] {context}")
    print(f"  Type:    {type(error).__name__}")
    print(f"  Message: {error}")
    if solution:
        print(f"  Fix:     {solution}")
    return entry

async def broadcast_error(context, error, solution=""):
    entry = log_error(context, error, solution)
    await broadcast_to_web_clients({'type': 'error', 'error': entry})

# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    candidates = [
        MODEL_PATH,
        os.path.join('models', 'sign_language_model.pkl'),
        'sign_language_model.pkl',
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            state.model         = data['model']
            state.scaler        = data['scaler']
            state.label_encoder = data['label_encoder']
            state.actions       = data['actions']
            print(f"[OK] Model loaded: {path}")
            print(f"[OK] Actions: {state.actions}")
            return True
        except Exception as e:
            log_error(f"Loading model: {path}", e, "Try retraining.")
    log_error("Model not found", FileNotFoundError(), "Run: python train_model.py")
    return False

# ============================================================
# LANDMARK PROCESSING
# ============================================================

def normalize_landmarks(arr):
    try:
        res  = arr.copy() - arr[0]
        dist = np.max(np.linalg.norm(res, axis=1))
        if dist > 0:
            res /= dist
        mid   = res[9]
        angle = np.arctan2(mid[1], mid[0])
        c, s  = np.cos(-angle), np.sin(-angle)
        res[:, :2] = res[:, :2] @ np.array([[c, -s], [s, c]]).T
        return res.flatten()
    except Exception as e:
        log_error("Landmark normalization", e)
        return None

def get_stabilized_landmarks(arr):
    n = normalize_landmarks(arr)
    if n is None:
        return None
    state.landmark_buffer.append(n)
    return np.mean(state.landmark_buffer, axis=0) if len(state.landmark_buffer) >= 3 else n

def preprocess_landmarks(raw):
    try:
        raw  = np.array(raw, dtype=np.float32)
        hand = raw[0:63].reshape(21, 3)
        if np.all(hand == -1):
            hand2 = raw[64:127].reshape(21, 3)
            if np.all(hand2 == -1):
                return None
            hand = hand2
        return get_stabilized_landmarks(hand)
    except Exception as e:
        log_error("Landmark preprocessing", e)
        return None

# ============================================================
# PREDICTION
# ============================================================

def predict_sign(landmarks):
    if state.model is None:
        return None, 0.0, []
    try:
        features = preprocess_landmarks(landmarks)
        if features is None:
            return None, 0.0, []
        features = state.scaler.transform(features.reshape(1, -1))
        probs    = state.model.predict_proba(features)[0]
        idx      = np.argmax(probs)
        word     = state.label_encoder.classes_[idx]
        conf     = float(probs[idx])
        top3     = [(state.label_encoder.classes_[i], float(probs[i]))
                    for i in np.argsort(probs)[-3:][::-1]]
        state.prediction_count += 1
        state.performance_stats['predictions_made'] += 1
        return word, conf, top3
    except Exception as e:
        log_error("Prediction", e)
        return None, 0.0, []

def smooth_prediction(word, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        return None, False
    state.prediction_history.append(word)
    if len(state.prediction_history) < MIN_CONSENSUS:
        return None, False
    common, count = Counter(state.prediction_history).most_common(1)[0]
    if count >= MIN_CONSENSUS and common != state.last_prediction:
        state.last_prediction = common
        return common, True
    return None, False

# ============================================================
# EASYOCR
# ============================================================

def _preprocess_image_for_ocr(img_bgr):
    img_bgr = cv2.flip(img_bgr, 1)
    img_bgr = cv2.resize(img_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return img_bgr

def _run_easyocr(img_bgr):
    if not EASYOCR_AVAILABLE or easyocr_reader is None:
        raise RuntimeError("EasyOCR not initialised — run: pip install easyocr")
    processed  = _preprocess_image_for_ocr(img_bgr)
    results    = easyocr_reader.readtext(processed)
    detections = []
    text_parts = []
    for bbox, text, prob in results:
        print(f"  [easyocr] {prob:.2f}  {text!r}")
        if prob >= EASYOCR_MIN_CONFIDENCE and text.strip():
            text_parts.append(text.strip())
            detections.append({
                'text':       text.strip(),
                'confidence': round(float(prob), 3),
                'bbox':       [[int(p[0]), int(p[1])] for p in bbox],
            })
        else:
            print(f"  [skip]    {prob:.2f}  {text!r}")
    return ' '.join(text_parts), detections

def _encode_processed_preview(img_bgr):
    try:
        processed = _preprocess_image_for_ocr(img_bgr)
        preview = cv2.resize(processed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        return None

# ============================================================
# GEMINI
# ============================================================

def _call_gemini(words: list, context: str) -> str:
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai not installed — run: pip install google-genai")
    client    = google_genai.Client(api_key=GEMINI_API_KEY)
    words_str = ', '.join(words)
    if context.strip():
        prompt = (
            f"{GEMINI_PROMPT_WITH_CONTEXT}\n\n"
            f"TOPIC / CONTEXT: {context.strip()}\n\n"
            f"WORDS: {words_str}"
        )
    else:
        prompt = f"{GEMINI_PROMPT_NO_CONTEXT}\n\nWORDS: {words_str}"
    print(f"[GEMINI] Sending → words={words_str!r}  context={context.strip()!r}")
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    sentence = response.text.strip().strip('"').strip("'")
    print(f"[GEMINI] Response: {sentence!r}")
    return sentence

# ============================================================
# TTS
# ============================================================

def _speak(text: str):
    if not TTS_AVAILABLE:
        raise RuntimeError("pyttsx3 not installed — run: pip install pyttsx3")
    engine = pyttsx3.init()
    engine.setProperty('rate',   TTS_RATE)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    print(f"[TTS] Spoke: {text!r}")

# ============================================================
# BROADCAST  — goes to ALL web clients AND the Pi OLED
# ============================================================

async def broadcast_to_web_clients(message):
    """
    Send to every browser client AND forward to Pi so the OLED stays in sync.
    'show_word' is excluded here because send_word_to_web() handles it directly.
    """
    # ── Browser clients ───────────────────────────────────────
    dead = set()
    for client in state.web_clients:
        try:
            await client.send_str(json.dumps(message))
        except Exception:
            dead.add(client)
    state.web_clients -= dead

    # ── Pi / OLED — forward everything except show_word ───────
    if state.pi_websocket and message.get('type') != 'show_word':
        try:
            await state.pi_websocket.send(json.dumps(message))
        except Exception:
            pass


async def send_word_to_web(word, confidence):
    """Broadcast detected word to browser clients AND push show_word to Pi OLED."""
    await broadcast_to_web_clients({
        'type':       'word_detected',
        'word':       word,
        'confidence': confidence,
        'timestamp':  datetime.now().isoformat(),
    })
    # Direct show_word push to Pi for the OLED prediction strip
    if state.pi_websocket:
        try:
            await state.pi_websocket.send(json.dumps({
                'type':       'show_word',
                'word':       word,
                'confidence': confidence,
            }))
        except Exception:
            pass


async def send_status_to_web():
    await broadcast_to_web_clients({
        'type':            'status',
        'pi_connected':    state.pi_connected,
        'words_count':     len(state.words),
        'last_word':       state.last_word,
        'last_confidence': state.last_confidence,
        'stats':           state.performance_stats,
    })


async def send_presets_to_pi():
    if not state.pi_websocket:
        return
    try:
        await state.pi_websocket.send(json.dumps({
            'type': 'presets_updated',
            'presets': {str(slot): text for slot, text in state.presets.items()},
        }))
    except Exception as e:
        log_error("Send presets to Pi", e)

# ============================================================
# PI WEBSOCKET HANDLER
# ============================================================

async def handle_pi_connection(websocket):
    state.pi_websocket = websocket
    state.pi_connected = True
    print("[PI] Connected!")
    await send_presets_to_pi()
    await send_status_to_web()

    try:
        async for message in websocket:
            try:
                state.performance_stats['frames_received'] += 1
                data     = json.loads(message)
                msg_type = data.get('type') if isinstance(data, dict) else None

                if msg_type == 'ping':
                    continue
                elif msg_type == 'frame':
                    await handle_frame_from_pi(data)
                    continue
                elif msg_type == 'audio':
                    await handle_audio_from_pi(data)
                    continue
                elif msg_type == 'audio_status':
                    await broadcast_to_web_clients(data)
                    continue
                elif msg_type == 'dev_frame':
                    frame_b64 = data.get('frame')
                    if frame_b64:
                        state.last_dev_frame = frame_b64
                        await broadcast_to_web_clients({
                            'type': 'dev_frame',
                            'image': frame_b64,
                            'timestamp': data.get('timestamp'),
                        })
                    continue
                elif msg_type == 'dev_stream_status':
                    state.dev_stream_enabled = bool(data.get('enabled', False))
                    await broadcast_to_web_clients({
                        'type': 'dev_stream_status',
                        'enabled': state.dev_stream_enabled,
                    })
                    continue
                elif msg_type == 'keypad_press':
                    state.last_keypad_event = {
                        'key': data.get('key', ''),
                        'action': data.get('action', ''),
                        'timestamp': datetime.now().isoformat(),
                    }
                    await broadcast_to_web_clients({
                        'type': 'keypad_press',
                        'key': data.get('key', ''),
                        'action': data.get('action', ''),
                        'action_label': KEYPAD_ACTION_LABELS.get(data.get('action', ''), data.get('action', '')),
                    })
                    continue

                if isinstance(data, dict) and data.get('command'):
                    await handle_web_command(data, None, source='pi')
                    continue

                # ── Landmark / sign data ───────────────────────
                landmarks = data.get('landmarks') if isinstance(data, dict) else data
                if landmarks is None:
                    continue
                landmarks = np.array(landmarks, dtype=np.float32)
                if landmarks.shape[0] != 128:
                    continue
                word, conf, _ = predict_sign(landmarks)
                if word is None:
                    continue
                final, show = smooth_prediction(word, conf)
                if show:
                    state.last_word       = final
                    state.last_confidence = conf
                    print(f"[DETECTED] {final.upper()} ({conf:.1%})")
                    await send_word_to_web(final, conf)

            except json.JSONDecodeError as e:
                await broadcast_error("JSON parse from Pi", e, "Pi must send valid JSON")
            except Exception as e:
                await broadcast_error("Processing Pi message", e)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[PI] Connection closed: {getattr(e, 'reason', 'unknown')}")
    except Exception as e:
        await broadcast_error("Pi connection handler", e)
    finally:
        state.pi_connected = False
        state.pi_websocket = None
        print("[PI] Disconnected")
        await send_status_to_web()

# ============================================================
# OCR FRAME HANDLER
# ============================================================

async def handle_frame_from_pi(data):
    try:
        state.performance_stats['ocr_calls'] += 1

        if not EASYOCR_AVAILABLE or easyocr_reader is None:
            raise RuntimeError("EasyOCR not initialised — run: pip install easyocr")

        frame_b64 = data.get('frame', '')
        if not frame_b64:
            raise ValueError("Pi sent an empty 'frame' field")

        clean = frame_b64.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        if not clean:
            raise ValueError("base64 string is empty")

        try:
            img_bytes = base64.b64decode(clean + '==')
        except Exception as e:
            raise ValueError(f"base64 decode failed: {e}")

        if len(img_bytes) < 500:
            raise ValueError(f"Decoded frame only {len(img_bytes)} bytes")

        print(f"[OCR] Frame received: {len(img_bytes) // 1024} KB")
        state.last_frame_base64 = frame_b64

        arr     = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed — not a valid JPEG")

        h, w = img_bgr.shape[:2]
        print(f"[OCR] Image decoded: {w}×{h} → EasyOCR ...")

        loop = asyncio.get_event_loop()
        try:
            text, detections = await asyncio.wait_for(
                loop.run_in_executor(None, _run_easyocr, img_bgr),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("OCR timed out after 60 s")

        print(f"[OCR] Extracted ({len(text)} chars): {text[:200]!r}")

        processed_preview_b64 = await loop.run_in_executor(
            None, _encode_processed_preview, img_bgr
        )

        state.last_ocr_result = {
            'text':       text,
            'detections': detections,
            'timestamp':  datetime.now().isoformat(),
        }

        if text:
            await broadcast_to_web_clients({
                'type':               'ocr_result',
                'text':               text,
                'detections':         detections,
                'image_preview':      frame_b64,
                'processed_preview':  processed_preview_b64,
                'needs_confirmation': True,
            })
        else:
            await broadcast_to_web_clients({
                'type': 'error',
                'error': {
                    'context':  'OCR',
                    'error':    'No text detected in image',
                    'solution': 'Ensure text is clear, well-lit and camera is steady.',
                },
            })

    except ValueError as e:
        await broadcast_error("OCR - bad image data", e, "Check Pi is sending a valid base64 JPEG")
    except RuntimeError as e:
        await broadcast_error("OCR - engine error", e, "run: pip install easyocr")
    except Exception as e:
        await broadcast_error("OCR processing", e, "Check easyocr installation")

# ============================================================
# AUDIO HANDLER
# ============================================================

def _transcribe_with_whisper(audio_bytes, rate, channels):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        audio_np = audio_np.reshape(-1, 2).mean(axis=1)
    if rate != 16000:
        target_len = int(len(audio_np) * 16000 / rate)
        audio_np   = np.interp(
            np.linspace(0, len(audio_np) - 1, target_len),
            np.arange(len(audio_np)),
            audio_np
        ).astype(np.float32)
    print(f"[AUDIO] Running Whisper on {len(audio_np)/16000:.1f}s ...")
    result = whisper_model.transcribe(audio_np, fp16=False, language='en')
    return result.get('text', '').strip()

async def handle_audio_from_pi(data):
    try:
        state.performance_stats['audio_calls'] += 1

        if not WHISPER_AVAILABLE or whisper_model is None:
            raise RuntimeError("Whisper not initialised — run: pip install openai-whisper")

        audio_b64 = data.get('audio', '')
        duration  = data.get('duration', 0)
        rate      = data.get('rate', 16000)
        channels  = data.get('channels', 1)

        if not audio_b64:
            raise ValueError("No audio data received from Pi")

        state.last_audio_base64 = audio_b64
        print(f"[AUDIO] Received {duration:.1f}s at {rate} Hz ({channels}ch)")

        audio_bytes = base64.b64decode(audio_b64)

        await broadcast_to_web_clients({
            'type':     'audio_status',
            'status':   'transcribing',
            'duration': duration,
        })

        loop = asyncio.get_event_loop()
        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _transcribe_with_whisper, audio_bytes, rate, channels
                ),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Whisper transcription timed out after 120 s")

        print(f"[AUDIO] Transcribed: {text!r}")

        if text:
            await broadcast_to_web_clients({
                'type':               'audio_result',
                'text':               text,
                'duration':           duration,
                'needs_confirmation': True,
            })
        else:
            await broadcast_to_web_clients({
                'type': 'error',
                'error': {
                    'context':  'Audio',
                    'error':    'Whisper returned empty transcription',
                    'solution': 'Speak clearly — check microphone and try again.',
                },
            })

    except ValueError as e:
        await broadcast_error("Audio - bad data", e, "Check Pi is sending valid audio")
    except RuntimeError as e:
        await broadcast_error("Audio - engine error", e, "run: pip install openai-whisper")
    except Exception as e:
        await broadcast_error("Audio processing", e)

# ============================================================
# SEND COMMAND TO PI
# ============================================================

async def send_command_to_pi(command, params=None):
    if not state.pi_websocket:
        return {'success': False, 'error': 'Pi not connected'}
    try:
        await state.pi_websocket.send(json.dumps({
            'type':      'command',
            'command':   command,
            'params':    params or {},
            'timestamp': datetime.now().isoformat(),
        }))
        print(f"[CMD] -> Pi: {command}")
        return {'success': True}
    except Exception as e:
        log_error("Send command to Pi", e)
        return {'success': False, 'error': str(e)}

# ============================================================
# BROWSER AUDIO HANDLER
# ============================================================

async def handle_browser_audio(data, ws):
    try:
        state.performance_stats['audio_calls'] += 1

        if not WHISPER_AVAILABLE or whisper_model is None:
            raise RuntimeError("Whisper not loaded — run: pip install openai-whisper")

        audio_b64 = data.get('audio', '')
        duration  = data.get('duration', 0)

        if not audio_b64:
            raise ValueError("No audio data in browser_audio command")

        await broadcast_to_web_clients({
            'type':     'audio_status',
            'status':   'transcribing',
            'duration': duration,
        })

        audio_bytes = base64.b64decode(audio_b64)
        audio_np    = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        audio_np    = np.clip(audio_np, -1.0, 1.0)

        if len(audio_np) < 1600:
            raise ValueError("Audio too short — hold the button longer")

        loop = asyncio.get_event_loop()
        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(None, _transcribe_numpy, audio_np),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Whisper timed out after 120 s")

        print(f"[BROWSER AUDIO] Transcribed: {text!r}")

        if text:
            await broadcast_to_web_clients({
                'type':               'audio_result',
                'text':               text,
                'duration':           duration,
                'source':             'browser',
                'needs_confirmation': True,
            })
        else:
            await broadcast_to_web_clients({
                'type': 'error',
                'error': {
                    'context':  'Browser Audio',
                    'error':    'Whisper returned empty result',
                    'solution': 'Speak clearly and close to the microphone.',
                },
            })

    except ValueError as e:
        await broadcast_to_web_clients({'type': 'error', 'error': {
            'context': 'Browser Audio', 'error': str(e), 'solution': ''}})
    except RuntimeError as e:
        await broadcast_to_web_clients({'type': 'error', 'error': {
            'context': 'Browser Audio', 'error': str(e),
            'solution': 'pip install openai-whisper'}})
    except Exception as e:
        log_error("Browser audio", e)
        await broadcast_to_web_clients({'type': 'error', 'error': {
            'context': 'Browser Audio', 'error': str(e), 'solution': ''}})


def _transcribe_numpy(audio_np):
    result = whisper_model.transcribe(audio_np, fp16=False, language='en')
    return result.get('text', '').strip()

# ============================================================
# WEB SERVER
# ============================================================

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    state.web_clients.add(ws)
    print(f"[WEB] Client connected ({len(state.web_clients)} total)")

    await ws.send_str(json.dumps({
        'type':         'init',
        'pi_connected': state.pi_connected,
        'words':        state.words,
        'context':      state.context,
        'actions':      state.actions,
        'presets':      {str(slot): text for slot, text in state.presets.items()},
        'keypad_map':   KEYPAD_ACTION_LABELS,
        'last_keypad_event': state.last_keypad_event,
        'dev_stream_enabled': state.dev_stream_enabled,
        'stats':        state.performance_stats,
        'features': {
            'ocr':       EASYOCR_AVAILABLE,
            'audio':     WHISPER_AVAILABLE,
            'bluetooth': BLUETOOTH_AVAILABLE,
            'gemini':    GEMINI_AVAILABLE,
            'tts':       TTS_AVAILABLE,
        },
    }))

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    await handle_web_command(json.loads(msg.data), ws)
                except Exception as e:
                    await broadcast_error("Web command", e)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"[WEB] Error: {ws.exception()}")
    finally:
        state.web_clients.discard(ws)
        print(f"[WEB] Client disconnected ({len(state.web_clients)} remaining)")
    return ws


async def _send_ws_if_present(ws, payload):
    if ws is None:
        return
    await ws.send_str(json.dumps(payload))


async def handle_web_command(data, ws, source='web'):
    cmd = data.get('command')
    print(f"[CMD:{source.upper()}] {cmd}")

    # ── Preset Trigger (from Pi keypad) ─────────────────────
    if cmd == 'trigger_preset':
        try:
            slot = int(data.get('slot', 0) or 0)
        except (TypeError, ValueError):
            slot = 0
        sentence = str(data.get('sentence', '')).strip()
        if not sentence and slot in state.presets:
            sentence = state.presets[slot]
        if not sentence:
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context': 'Preset',
                'error': f'Preset slot {slot} is empty',
                'solution': 'Set preset text in the UI and save it',
            }})
            return
        state.words.append(sentence)
        await broadcast_to_web_clients({
            'type': 'preset_triggered',
            'slot': slot,
            'sentence': sentence,
            'source': source,
        })
        await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})
        return

    # ── Standard Key Input (5/6/7/8/9) ──────────────────────
    if cmd == 'standard_key_input':
        key = str(data.get('key', '')).strip()
        if key:
            state.words.append(key)
            await broadcast_to_web_clients({
                'type': 'standard_key_input',
                'key': key,
                'source': source,
            })
            await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})
        return

    # ── Preset Update (from Web UI) ──────────────────────────
    if cmd == 'update_preset':
        try:
            slot = int(data.get('slot', 0) or 0)
        except (TypeError, ValueError):
            slot = 0
        text = str(data.get('text', '')).strip()
        if slot not in (1, 2, 3):
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context': 'Preset',
                'error': f'Invalid preset slot: {slot}',
                'solution': 'Use only preset slots 1, 2 or 3',
            }})
            return
        state.presets[slot] = text
        _save_presets()
        await broadcast_to_web_clients({
            'type': 'presets_updated',
            'presets': {str(k): v for k, v in state.presets.items()},
            'updated_slot': slot,
        })
        await send_presets_to_pi()
        return

    # ── Add Word ─────────────────────────────────────────────
    # Accepts an explicit 'word' field (from Pi keypad) OR falls
    # back to state.last_word (from web UI Add Word button).
    if cmd == 'add_word':
        word = data.get('word', '').strip()
        if not word:
            word = state.last_word.strip()
        if word:
            state.words.append(word)
            print(f"[ADD WORD] '{word}' → words now: {state.words}")
            await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})
        else:
            # Nothing to add — tell the client
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context':  'Add Word',
                'error':    'No word detected yet — sign a word first',
                'solution': 'Make a sign and wait for the prediction to appear',
            }})
        return

    # ── Delete Last ──────────────────────────────────────────
    elif cmd == 'delete_last':
        if state.words:
            removed = state.words.pop()
            print(f"[DELETE] Removed '{removed}' → words now: {state.words}")
        await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})

    # ── Clear All ────────────────────────────────────────────
    elif cmd == 'clear_words':
        state.words = []
        print("[CLEAR] All words cleared")
        await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})

    # ── Send to Gemini ───────────────────────────────────────
    elif cmd == 'send_to_ai':
        if not state.words:
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    'No words to send — add some words first',
                'solution': 'Sign some words and add them before sending to AI',
            }})
            return

        if not GEMINI_AVAILABLE:
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    'google-generativeai not installed',
                'solution': 'pip install google-generativeai',
            }})
            return

        # Notify ALL clients (browser + Pi OLED) that Gemini is generating
        await broadcast_to_web_clients({'type': 'gemini_status', 'status': 'generating'})

        state.performance_stats['gemini_calls'] += 1
        words_snapshot   = list(state.words)
        context_snapshot = state.context

        loop = asyncio.get_event_loop()
        try:
            sentence = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _call_gemini, words_snapshot, context_snapshot
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    'Request timed out after 30 s',
                'solution': 'Check your internet connection and API key',
            }})
            await broadcast_to_web_clients({'type': 'gemini_status', 'status': 'idle'})
            return
        except Exception as e:
            log_error("Gemini call", e, "Check GEMINI_API_KEY in laptop_receiver.py")
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    str(e),
                'solution': 'Check GEMINI_API_KEY in laptop_receiver.py',
            }})
            await broadcast_to_web_clients({'type': 'gemini_status', 'status': 'idle'})
            return

        state.last_sentence = sentence

        # Broadcast sentence to ALL clients (browser + Pi OLED)
        await broadcast_to_web_clients({
            'type':     'gemini_sentence',
            'sentence': sentence,
            'words':    words_snapshot,
            'context':  context_snapshot,
        })

        if TTS_AVAILABLE:
            loop.run_in_executor(None, _speak, sentence)
        else:
            print("[TTS] Skipped — pyttsx3 not installed")

    # ── Speak Sentence ───────────────────────────────────────
    elif cmd == 'speak_sentence':
        text = data.get('sentence', state.last_sentence)
        if text and TTS_AVAILABLE:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _speak, text)
        elif not TTS_AVAILABLE:
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context': 'TTS', 'error': 'pyttsx3 not installed',
                'solution': 'pip install pyttsx3'}})

    # ── Context ───────────────────────────────────────────────
    elif cmd == 'remove_context':
        state.context = ""
        await broadcast_to_web_clients({'type': 'context_updated', 'context': state.context})

    elif cmd == 'add_context':
        state.context = (state.context + ' ' + data.get('text', '')).strip()
        await broadcast_to_web_clients({'type': 'context_updated', 'context': state.context})

    # ── OCR ──────────────────────────────────────────────────
    elif cmd == 'request_ocr':
        result = await send_command_to_pi('capture_frame')
        await _send_ws_if_present(ws, result)

    # ── Audio ─────────────────────────────────────────────────
    elif cmd == 'browser_audio':
        await handle_browser_audio(data, ws)

    elif cmd == 'request_audio':
        duration = data.get('duration', 10)
        result = await send_command_to_pi('record_audio', {'duration': duration})
        await _send_ws_if_present(ws, result)

    elif cmd == 'start_audio':
        result = await send_command_to_pi('start_audio')
        await _send_ws_if_present(ws, result)

    elif cmd == 'stop_audio':
        result = await send_command_to_pi('stop_audio')
        await _send_ws_if_present(ws, result)

    elif cmd == 'toggle_dev_stream':
        enabled = bool(data.get('enabled', False))
        result = await send_command_to_pi('toggle_dev_stream', {'enabled': enabled})
        await _send_ws_if_present(ws, result)
        if result.get('success'):
            state.dev_stream_enabled = enabled
            await broadcast_to_web_clients({'type': 'dev_stream_status', 'enabled': enabled})
        else:
            state.dev_stream_enabled = False
            await broadcast_to_web_clients({'type': 'dev_stream_status', 'enabled': False})
            await broadcast_to_web_clients({'type': 'error', 'error': {
                'context': 'Dev Stream',
                'error': result.get('error', 'Pi not connected'),
                'solution': 'Start pi_sender.py and ensure Pi websocket is connected',
            }})

    # ── Bluetooth ─────────────────────────────────────────────
    elif cmd == 'scan_bluetooth':
        devices = []
        if BLUETOOTH_AVAILABLE:
            try:
                raw     = bluetooth.discover_devices(duration=5, lookup_names=True)
                devices = [{'address': a, 'name': n} for a, n in raw]
            except Exception as e:
                log_error("Bluetooth scan", e)
        await _send_ws_if_present(ws, {'type': 'bluetooth_devices', 'devices': devices})

    # ── Debug ─────────────────────────────────────────────────
    elif cmd == 'get_debug_data':
        await _send_ws_if_present(ws, {
            'type':       'debug_data',
            'last_frame': state.last_frame_base64,
            'last_dev_frame': state.last_dev_frame,
            'last_audio': state.last_audio_base64,
            'last_ocr':   state.last_ocr_result,
            'dev_stream_enabled': state.dev_stream_enabled,
            'error_log':  list(state.error_log)[-10:],
            'stats':      state.performance_stats,
        })


async def serve_index(request):
    path = os.path.join(BASE_DIR, 'web', 'index.html')
    return web.FileResponse(path) if os.path.exists(path) else \
           web.Response(text="Web interface not found", status=404)

# ============================================================
# MAIN
# ============================================================

async def main():
    print("\n" + "=" * 60)
    print("SIGN LANGUAGE RECOGNITION SYSTEM")
    print("=" * 60)

    if not load_model():
        print("\n[FATAL] No model — run: python train_model.py")
        return

    _load_presets()

    print(f"\n[CONFIG]")
    print(f"  WebSocket port : {WEBSOCKET_PORT}")
    print(f"  Web server port: {WEB_SERVER_PORT}")
    print(f"  EasyOCR        : {'available' if EASYOCR_AVAILABLE else 'not available'}")
    print(f"  Whisper        : {'available' if WHISPER_AVAILABLE else 'not available'}")
    print(f"  Gemini         : {'available' if GEMINI_AVAILABLE else 'not available'}")
    print(f"  TTS (pyttsx3)  : {'available' if TTS_AVAILABLE else 'not available'}")

    global easyocr_reader
    if EASYOCR_AVAILABLE:
        try:
            print("[OCR] Loading EasyOCR model ...")
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            easyocr_reader.readtext(dummy)
            print("[OCR] EasyOCR ready ✓")
        except Exception as e:
            print(f"[OCR] WARNING: {e}")
            easyocr_reader = None

    global whisper_model
    if WHISPER_AVAILABLE:
        try:
            print("[AUDIO] Loading Whisper 'base' model ...")
            whisper_model = whisper.load_model('base')
            print("[AUDIO] Whisper ready ✓")
        except Exception as e:
            print(f"[AUDIO] WARNING: {e}")
            whisper_model = None

    pi_server = await websockets.serve(
        handle_pi_connection, "0.0.0.0", WEBSOCKET_PORT,
        ping_interval=20, ping_timeout=20,
        max_size=MAX_FRAME_SIZE,
    )
    print(f"[OK] Pi WebSocket on port {WEBSOCKET_PORT}")

    app  = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*")
    })
    app.router.add_get('/', serve_index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_static('/static', os.path.join(BASE_DIR, 'web', 'static'), name='static')
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', WEB_SERVER_PORT).start()

    print(f"[OK] Web server on http://localhost:{WEB_SERVER_PORT}")
    print("=" * 60)
    print(f"  EasyOCR : {'✓ loaded' if easyocr_reader else '✗ not loaded'}")
    print(f"  Whisper : {'✓ loaded' if whisper_model else '✗ not loaded'}")
    print(f"  Gemini  : {'✓ ready' if GEMINI_AVAILABLE else '✗ not installed'}")
    print(f"  TTS     : {'✓ ready' if TTS_AVAILABLE else '✗ not installed'}")
    print("Open browser and start Pi sender")
    print("=" * 60 + "\n")

    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopped")
    except Exception as e:
        log_error("Main", e)