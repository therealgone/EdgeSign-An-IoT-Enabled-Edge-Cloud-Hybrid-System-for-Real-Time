"""
Sign Language Receiver — Laptop
EasyOCR with full preprocessing pipeline:
  flip → grayscale → 3× resize → CLAHE → sharpen → bilateral filter → adaptive threshold → EasyOCR

Pi sends a raw 1280×720 JPEG (no filters).
All image processing happens here on the laptop.
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
import io
import traceback
import tempfile
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

# Gemini
GEMINI_AVAILABLE = False
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
    print("[OK] google-genai imported")
except ImportError:
    print("[WARNING] google-genai not installed — run: pip install google-genai")

# TTS — pyttsx3 works offline, plays on laptop speakers
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

# EasyOCR confidence threshold — detections below this are discarded
EASYOCR_MIN_CONFIDENCE = 0.3

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL   = "gemini-2.5-flash"           # matches the working example

# System prompt used when NO context is set
GEMINI_PROMPT_NO_CONTEXT = """\
You are a sentence builder for a sign language recognition system.
The user will give you a list of words detected from sign language.
Your job is to form ONE single, complete, grammatically correct, natural-sounding sentence
using ALL of those words. You have full creative freedom to add connecting words, articles,
prepositions, verb tenses, or any other words needed to make the sentence sound fluent and natural.
Every word the user gives you MUST appear in the final sentence — do not drop any.
Reply with ONLY the sentence. No explanation, no punctuation outside the sentence, no quotes.\
"""

# System prompt used when context IS set — context is the topic/situation
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

# TTS speed (words per minute) — adjust to taste
TTS_RATE = 160

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

        self.pi_websocket = None
        self.web_clients  = set()
        self.pi_connected    = False
        self.last_word       = ""
        self.last_confidence = 0.0

        self.last_frame_base64 = None
        self.last_audio_base64 = None
        self.last_ocr_result   = None
        self.last_sentence     = ""   # last Gemini-generated sentence

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
# EASYOCR — Preprocessing pipeline (laptop-side)
#
# EasyOCR has its own internal grayscale + binarisation + detection
# pipeline trained on colour images, so we do NOT manually grayscale,
# threshold, or apply bilateral/CLAHE filters — those hurt EasyOCR.
#
# All we do before passing to EasyOCR:
#   1. Flip horizontally  (mirror correction — camera reverses left↔right)
#   2. 3× upscale (INTER_CUBIC) — helps with small/blurry text
# ============================================================

def _preprocess_image_for_ocr(img_bgr):
    """
    Minimal preprocessing for EasyOCR — flip + upscale only.
    Returns a colour BGR numpy array ready for easyocr.readtext().
    """
    # Step 1 — Flip: camera mirrors text left↔right
    img_bgr = cv2.flip(img_bgr, 1)

    # Step 2 — Upscale 3× so small/blurry characters have more pixels
    img_bgr = cv2.resize(img_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    return img_bgr


def _run_easyocr(img_bgr):
    """
    Full pipeline: preprocess → EasyOCR.
    Runs in a thread executor (blocking call).
    Returns (text_string, detections_list).
    """
    if not EASYOCR_AVAILABLE or easyocr_reader is None:
        raise RuntimeError("EasyOCR not initialised — run: pip install easyocr")

    processed = _preprocess_image_for_ocr(img_bgr)

    # EasyOCR accepts numpy BGR array directly
    results = easyocr_reader.readtext(processed)

    detections = []
    text_parts  = []

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

    full_text = ' '.join(text_parts)
    return full_text, detections


def _encode_processed_preview(img_bgr):
    """
    Return base64 JPEG of the preprocessed image for dev-mode UI.
    Shows the flipped + upscaled colour image (what EasyOCR actually sees).
    Downscaled back to 50% so it's not huge in the browser.
    """
    try:
        processed = _preprocess_image_for_ocr(img_bgr)
        # Scale back down for preview — 3× upscale is too big for the UI
        preview = cv2.resize(processed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        return None

# ============================================================
# GEMINI — sentence generation
# ============================================================

def _call_gemini(words: list, context: str) -> str:
    """
    Send words (and optional context) to Gemini.
    Returns the generated sentence string.
    Runs in thread executor — network call blocks.
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai not installed — run: pip install google-genai")

    client = google_genai.Client(api_key=GEMINI_API_KEY)

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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    sentence = response.text.strip().strip('"').strip("'")
    print(f"[GEMINI] Response: {sentence!r}")
    return sentence


# ============================================================
# TTS — speak sentence on laptop speaker
# ============================================================

def _speak(text: str):
    """
    Speak text through the laptop's default speaker using pyttsx3.
    Runs in thread executor — blocks until speech finishes.
    """
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
# WEB CLIENT BROADCAST
# ============================================================

async def broadcast_to_web_clients(message):
    if not state.web_clients:
        return
    dead = set()
    for client in state.web_clients:
        try:
            await client.send_str(json.dumps(message))
        except Exception:
            dead.add(client)
    state.web_clients -= dead

async def send_word_to_web(word, confidence):
    await broadcast_to_web_clients({
        'type':       'word_detected',
        'word':       word,
        'confidence': confidence,
        'timestamp':  datetime.now().isoformat(),
    })

async def send_status_to_web():
    await broadcast_to_web_clients({
        'type':            'status',
        'pi_connected':    state.pi_connected,
        'words_count':     len(state.words),
        'last_word':       state.last_word,
        'last_confidence': state.last_confidence,
        'stats':           state.performance_stats,
    })

# ============================================================
# PI WEBSOCKET HANDLER
# ============================================================

async def handle_pi_connection(websocket):
    state.pi_websocket = websocket
    state.pi_connected = True
    print("[PI] Connected!")
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
                    # Pi notifies us recording started — forward to UI
                    await broadcast_to_web_clients(data)
                    continue

                # --- Landmark data (sign language) ---
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
# Receives raw 1280×720 JPEG from Pi, applies full preprocessing,
# runs EasyOCR, sends result + processed preview + bounding boxes to UI.
# ============================================================

async def handle_frame_from_pi(data):
    """
    Pi sends raw base64 JPEG (no filters).
    Laptop applies full pipeline then runs EasyOCR.
    """
    try:
        state.performance_stats['ocr_calls'] += 1

        if not EASYOCR_AVAILABLE or easyocr_reader is None:
            raise RuntimeError("EasyOCR not initialised — run: pip install easyocr")

        frame_b64 = data.get('frame', '')
        if not frame_b64:
            raise ValueError("Pi sent an empty 'frame' field")

        # ── Decode base64 → numpy BGR ──────────────────────────
        clean = frame_b64.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        if not clean:
            raise ValueError("base64 string is empty")

        try:
            img_bytes = base64.b64decode(clean + '==')
        except Exception as e:
            raise ValueError(f"base64 decode failed: {e}")

        if len(img_bytes) < 500:
            raise ValueError(
                f"Decoded frame only {len(img_bytes)} bytes — Pi may not have sent the full image"
            )

        print(f"[OCR] Frame received: {len(img_bytes) // 1024} KB")
        state.last_frame_base64 = frame_b64

        # Decode JPEG bytes → numpy BGR array
        arr     = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed — not a valid JPEG")

        h, w = img_bgr.shape[:2]
        print(f"[OCR] Image decoded: {w}×{h}  →  preprocessing + EasyOCR ...")

        # ── Run full pipeline in thread executor ───────────────
        loop = asyncio.get_event_loop()
        try:
            text, detections = await asyncio.wait_for(
                loop.run_in_executor(None, _run_easyocr, img_bgr),
                timeout=60.0   # EasyOCR is slower than Tesseract on first run
            )
        except asyncio.TimeoutError:
            raise RuntimeError("OCR timed out after 60 s")

        print(f"[OCR] Extracted ({len(text)} chars): {text[:200]!r}")
        print(f"[OCR] {len(detections)} detections above confidence {EASYOCR_MIN_CONFIDENCE}")

        # Build processed preview for dev-mode UI
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
                'detections':         detections,        # per-word bbox + confidence
                'image_preview':      frame_b64,         # raw original from Pi
                'processed_preview':  processed_preview_b64,  # after all filters
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
        await broadcast_error("OCR - bad image data", e,
                              "Check Pi is sending a valid base64 JPEG")
    except RuntimeError as e:
        await broadcast_error("OCR - engine error", e,
                              "run: pip install easyocr")
    except Exception as e:
        await broadcast_error("OCR processing", e,
                              "Check easyocr installation")

# ============================================================
# AUDIO HANDLER — OpenAI Whisper transcription
#
# Pi sends:  raw int16 PCM bytes (base64), sample rate, channels, duration
# Laptop:    wraps in a WAV, saves to temp file, runs whisper.transcribe()
# ============================================================

def _transcribe_with_whisper(audio_bytes, rate, channels):
    """
    Transcribe raw int16 PCM using Whisper.
    Bypasses ffmpeg entirely by feeding Whisper a float32 numpy array directly
    (whisper.transcribe() accepts a numpy array — no temp file needed).
    """
    # Convert raw int16 PCM bytes → float32 numpy array in [-1.0, 1.0]
    # This is exactly what Whisper expects internally
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # If stereo, average down to mono
    if channels == 2:
        audio_np = audio_np.reshape(-1, 2).mean(axis=1)

    # Whisper requires 16000 Hz — resample if Pi sent a different rate
    if rate != 16000:
        import math
        target_len = int(len(audio_np) * 16000 / rate)
        audio_np   = np.interp(
            np.linspace(0, len(audio_np) - 1, target_len),
            np.arange(len(audio_np)),
            audio_np
        ).astype(np.float32)

    print(f"[AUDIO] Running Whisper on {len(audio_np)/16000:.1f}s of audio ...")

    # Pass numpy array directly — no file, no ffmpeg
    result = whisper_model.transcribe(audio_np, fp16=False, language='en')
    text   = result.get('text', '').strip()
    return text


async def handle_audio_from_pi(data):
    """
    Receives raw int16 PCM audio from Pi (base64 encoded).
    Decodes → WAV → Whisper → broadcasts transcription as audio_result.
    """
    try:
        state.performance_stats['audio_calls'] += 1

        if not WHISPER_AVAILABLE or whisper_model is None:
            raise RuntimeError(
                "Whisper not initialised — run: pip install openai-whisper"
            )

        audio_b64 = data.get('audio', '')
        duration  = data.get('duration', 0)
        rate      = data.get('rate', 16000)
        channels  = data.get('channels', 1)

        if not audio_b64:
            raise ValueError("No audio data received from Pi")

        state.last_audio_base64 = audio_b64
        print(f"[AUDIO] Received {duration:.1f}s at {rate} Hz ({channels}ch)")

        audio_bytes = base64.b64decode(audio_b64)
        print(f"[AUDIO] Decoded: {len(audio_bytes)//1024} KB raw PCM")

        # Notify UI that transcription is in progress
        await broadcast_to_web_clients({
            'type':     'audio_status',
            'status':   'transcribing',
            'duration': duration,
        })

        # Run Whisper in thread executor (CPU-heavy, blocks)
        loop = asyncio.get_event_loop()
        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _transcribe_with_whisper, audio_bytes, rate, channels
                ),
                timeout=120.0   # Whisper on long audio can take a while
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
        await broadcast_error("Audio - bad data", e,
                              "Check Pi is sending valid audio")
    except RuntimeError as e:
        await broadcast_error("Audio - engine error", e,
                              "run: pip install openai-whisper")
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
#
# The browser records via Web Audio API (AudioContext + ScriptProcessor),
# collects raw float32 PCM samples at 16000 Hz (mono), base64-encodes them,
# and sends:  { command: 'browser_audio', audio: '<base64>', duration: N }
#
# We decode float32 bytes → numpy array → whisper.transcribe(array)
# No ffmpeg, no temp files, works from laptop browser AND phone browser.
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

        # Notify UI transcription is starting
        await ws.send_str(json.dumps({
            'type':   'audio_status',
            'status': 'transcribing',
            'duration': duration,
        }))

        audio_bytes = base64.b64decode(audio_b64)
        print(f"[BROWSER AUDIO] {duration:.1f}s, {len(audio_bytes)//1024} KB float32 PCM")

        # Browser sends float32 little-endian samples at 16000 Hz mono
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()

        # Clamp to [-1, 1] in case of any float overflow
        audio_np = np.clip(audio_np, -1.0, 1.0)

        if len(audio_np) < 1600:  # less than 0.1s — ignore
            raise ValueError("Audio too short — hold the button longer")

        print(f"[BROWSER AUDIO] {len(audio_np)/16000:.1f}s samples → Whisper ...")

        loop = asyncio.get_event_loop()
        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _transcribe_numpy, audio_np
                ),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Whisper timed out after 120 s")

        print(f"[BROWSER AUDIO] Transcribed: {text!r}")

        if text:
            await ws.send_str(json.dumps({
                'type':               'audio_result',
                'text':               text,
                'duration':           duration,
                'source':             'browser',
                'needs_confirmation': True,
            }))
        else:
            await ws.send_str(json.dumps({'type': 'error', 'error': {
                'context':  'Browser Audio',
                'error':    'Whisper returned empty result',
                'solution': 'Speak clearly and close to the microphone.',
            }}))

    except ValueError as e:
        await ws.send_str(json.dumps({'type': 'error',
                                      'error': {'context': 'Browser Audio',
                                                'error': str(e), 'solution': ''}}))
    except RuntimeError as e:
        await ws.send_str(json.dumps({'type': 'error',
                                      'error': {'context': 'Browser Audio',
                                                'error': str(e),
                                                'solution': 'pip install openai-whisper'}}))
    except Exception as e:
        log_error("Browser audio", e)
        await ws.send_str(json.dumps({'type': 'error',
                                      'error': {'context': 'Browser Audio',
                                                'error': str(e), 'solution': ''}}))


def _transcribe_numpy(audio_np):
    """Whisper transcription from float32 numpy array. Runs in thread executor."""
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


async def handle_web_command(data, ws):
    cmd = data.get('command')
    print(f"[WEB CMD] {cmd}")

    if cmd == 'add_word':
        word = data.get('word', state.last_word)
        if word:
            state.words.append(word)
            await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})

    elif cmd == 'delete_last':
        if state.words:
            state.words.pop()
            await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})

    elif cmd == 'clear_words':
        state.words = []
        await broadcast_to_web_clients({'type': 'words_updated', 'words': state.words})

    elif cmd == 'send_to_ai':
        # ── Gemini sentence generation ──────────────────────────
        if not state.words:
            await ws.send_str(json.dumps({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    'No words to send — add some words first',
                'solution': 'Sign some words and add them before sending to AI',
            }}))
            return

        if not GEMINI_AVAILABLE:
            await ws.send_str(json.dumps({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    'google-generativeai not installed',
                'solution': 'pip install google-generativeai',
            }}))
            return

        # Tell UI we're working
        await ws.send_str(json.dumps({'type': 'gemini_status', 'status': 'generating'}))

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
            await ws.send_str(json.dumps({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    'Request timed out after 30 s',
                'solution': 'Check your internet connection and API key',
            }}))
            await ws.send_str(json.dumps({'type': 'gemini_status', 'status': 'idle'}))
            return
        except Exception as e:
            log_error("Gemini call", e, "Check GEMINI_API_KEY in laptop_receiver.py")
            await ws.send_str(json.dumps({'type': 'error', 'error': {
                'context':  'Gemini',
                'error':    str(e),
                'solution': 'Check GEMINI_API_KEY in laptop_receiver.py',
            }}))
            await ws.send_str(json.dumps({'type': 'gemini_status', 'status': 'idle'}))
            return

        state.last_sentence = sentence

        # Broadcast sentence to ALL web clients (laptop + phone)
        await broadcast_to_web_clients({
            'type':     'gemini_sentence',
            'sentence': sentence,
            'words':    words_snapshot,
            'context':  context_snapshot,
        })

        # Speak on laptop speaker in background (don't block the web response)
        if TTS_AVAILABLE:
            loop.run_in_executor(None, _speak, sentence)
        else:
            print("[TTS] Skipped — pyttsx3 not installed")

    elif cmd == 'speak_sentence':
        # Re-speak the last generated sentence on demand
        text = data.get('sentence', state.last_sentence)
        if text and TTS_AVAILABLE:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _speak, text)
        elif not TTS_AVAILABLE:
            await ws.send_str(json.dumps({'type': 'error', 'error': {
                'context': 'TTS', 'error': 'pyttsx3 not installed',
                'solution': 'pip install pyttsx3'}}))

    elif cmd == 'remove_context':
        state.context = ""
        await broadcast_to_web_clients({'type': 'context_updated', 'context': state.context})

    elif cmd == 'add_context':
        state.context = (state.context + ' ' + data.get('text', '')).strip()
        await broadcast_to_web_clients({'type': 'context_updated', 'context': state.context})

    elif cmd == 'request_ocr':
        # Tell Pi to capture and send a frame
        result = await send_command_to_pi('capture_frame')
        await ws.send_str(json.dumps(result))

    elif cmd == 'browser_audio':
        # Audio recorded directly in the browser (laptop or phone mic)
        # Payload: base64-encoded raw float32 PCM at 16000 Hz, mono
        await handle_browser_audio(data, ws)

    elif cmd == 'request_audio':
        # Fixed-duration recording
        duration = data.get('duration', 10)
        result = await send_command_to_pi('record_audio', {'duration': duration})
        await ws.send_str(json.dumps(result))

    elif cmd == 'start_audio':
        # Open-ended recording — Pi records until stop_audio
        result = await send_command_to_pi('start_audio')
        await ws.send_str(json.dumps(result))

    elif cmd == 'stop_audio':
        # Tell Pi to stop recording and send what it has
        result = await send_command_to_pi('stop_audio')
        await ws.send_str(json.dumps(result))

    elif cmd == 'scan_bluetooth':
        devices = []
        if BLUETOOTH_AVAILABLE:
            try:
                raw     = bluetooth.discover_devices(duration=5, lookup_names=True)
                devices = [{'address': a, 'name': n} for a, n in raw]
            except Exception as e:
                log_error("Bluetooth scan", e)
        await ws.send_str(json.dumps({'type': 'bluetooth_devices', 'devices': devices}))

    elif cmd == 'get_debug_data':
        await ws.send_str(json.dumps({
            'type':       'debug_data',
            'last_frame': state.last_frame_base64,
            'last_audio': state.last_audio_base64,
            'last_ocr':   state.last_ocr_result,
            'error_log':  list(state.error_log)[-10:],
            'stats':      state.performance_stats,
        }))


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

    print(f"\n[CONFIG]")
    print(f"  WebSocket port : {WEBSOCKET_PORT}")
    print(f"  Web server port: {WEB_SERVER_PORT}")
    print(f"  EasyOCR        : {'available' if EASYOCR_AVAILABLE else 'not available'}")
    print(f"  Whisper        : {'available' if WHISPER_AVAILABLE else 'not available'}")
    print(f"  Gemini         : {'available' if GEMINI_AVAILABLE else 'not available'}")
    print(f"  TTS (pyttsx3)  : {'available' if TTS_AVAILABLE else 'not available'}")

    # ── EasyOCR: load model + warmup so first real call is instant ──
    global easyocr_reader
    if EASYOCR_AVAILABLE:
        try:
            print("[OCR] Loading EasyOCR model (may download ~100 MB on first run)...")
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[OCR] Model loaded — running warmup inference...")
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            easyocr_reader.readtext(dummy)
            print("[OCR] Warmup done — EasyOCR is hot and ready ✓")
        except Exception as e:
            print(f"[OCR] WARNING: EasyOCR failed to load: {e}")
            easyocr_reader = None

    # ── Whisper: load model at startup so first transcription is fast ──
    global whisper_model
    if WHISPER_AVAILABLE:
        try:
            # 'base' model — good balance of speed and accuracy for real-world use
            # Other options: 'tiny' (fastest), 'small', 'medium', 'large' (most accurate)
            print("[AUDIO] Loading Whisper 'base' model (downloads ~145 MB on first run)...")
            whisper_model = whisper.load_model('base')
            print("[AUDIO] Whisper ready ✓")
        except Exception as e:
            print(f"[AUDIO] WARNING: Whisper failed to load: {e}")
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