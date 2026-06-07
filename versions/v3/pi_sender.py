import cv2
import mediapipe as mp
import websocket
import json
import time
import socket
import numpy as np
import base64
import threading

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARNING] sounddevice not installed — run: pip install sounddevice")

try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import sh1106
    from luma.core.render import canvas
    from PIL import ImageFont
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    print("[WARNING] luma.oled not installed — run: pip install luma.oled")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARNING] RPi.GPIO not available")

# ============================================================
# CONFIGURATION
# ============================================================

PC_IP                = "YOUR_LAPTOP_IP_OR_HOSTNAME"  # e.g. 192.168.1.100 or my-laptop.local
WS_URL               = f"ws://{PC_IP}:8765"

TARGET_FPS           = 30
SEND_INTERVAL        = 1.0 / TARGET_FPS
HEARTBEAT_INTERVAL   = 2.0

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.6
MAX_NUM_HANDS            = 2

SHOW_LANDMARKS       = True
CAPTURE_WIDTH        = 1280
CAPTURE_HEIGHT       = 720
CAPTURE_JPEG_QUALITY = 85
DEV_STREAM_FPS       = 8
DEV_STREAM_QUALITY   = 55
DEV_STREAM_WIDTH     = 640

AUDIO_SAMPLE_RATE    = 16000
AUDIO_CHANNELS       = 1
AUDIO_MAX_SECONDS    = 120

# ============================================================
# KEYPAD SETUP
# ============================================================

ROWS = [5, 6, 13, 19]
COLS = [12, 16, 20, 21]
KEYS = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
]

# Key -> action mapping
# Required mapping:
# [1] down, [0] up, [2/3/4] preset slots 1/2/3,
# [5/6/7/8/9] standard key input, [A/B/C/D/*/#] actions.
KEY_ACTIONS = {
    'A': 'add_word',
    'B': 'send_to_ai',
    'C': 'request_ocr',
    'D': 'toggle_audio',
    '*': 'delete_last',
    '#': 'clear_words',
    '0': 'scroll_up',
    '1': 'scroll_down',
    '2': 'preset_1',
    '3': 'preset_2',
    '4': 'preset_3',
    '5': 'standard_key_5',
    '6': 'standard_key_6',
    '7': 'standard_key_7',
    '8': 'standard_key_8',
    '9': 'standard_key_9',
}

# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
landmark_spec   = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

# ============================================================
# GLOBALS
# ============================================================

last_send_time      = 0
last_heartbeat_time = 0
frame_count         = 0
fps_val             = 0
last_fps_time       = time.time()

latest_frame_lock = threading.Lock()
latest_frame      = None
dev_stream_enabled   = False
last_dev_stream_time = 0.0

audio_lock            = threading.Lock()
audio_recording       = False
audio_stop_event      = threading.Event()
audio_recorded_chunks = []

# ── OLED state ───────────────────────────────────────────────
display_lock       = threading.Lock()
oled_prediction    = ""
oled_words         = []
oled_context       = ""
oled_status        = "READY"
oled_flash_text    = ""
oled_flash_until   = 0.0
oled_scroll_offset = 0
oled_is_recording  = False
oled_boot_mode     = True
oled_boot_text     = "Booting..."

oled_event    = threading.Event()
ws_ref_holder = [None]

# -- Preset cache (populated from laptop on connect) ----------
preset_cache = {i: "" for i in range(1, 4)}   # 1-3
preset_lock  = threading.Lock()

# ============================================================
# FONTS
# ============================================================

font_large  = None
font_medium = None
font_small  = None


def _load_fonts():
    global font_large, font_medium, font_small
    bold   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    normal = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font_large  = ImageFont.truetype(bold,   15)
        font_medium = ImageFont.truetype(normal, 11)
        font_small  = ImageFont.truetype(normal, 10)
    except Exception:
        font_large = font_medium = font_small = None

# ============================================================
# OLED LAYOUT
# ============================================================

DISPLAY_W = 128
DISPLAY_H = 64
STATUS_H  = 12
DIV_Y     = STATUS_H
PRED_H    = 16
PRED_Y    = DISPLAY_H - PRED_H
SENT_Y    = DIV_Y + 2
SENT_H    = PRED_Y - SENT_Y - 2
LINE_H    = 12


def _tw(draw, text, font):
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0]
    except Exception:
        return len(text) * 6


def _wrap(draw, text, font, max_w):
    if not text:
        return [""]
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if _tw(draw, test, font) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_normal_ui(draw, status, is_rec, words, scroll, prediction):
    f_s = font_small  or ImageFont.load_default()
    f_m = font_medium or ImageFont.load_default()
    f_l = font_large  or ImageFont.load_default()

    # Status bar
    draw.rectangle([0, 0, DISPLAY_W - 1, STATUS_H - 1], fill="black")
    status = str(status or "")
    label = ("* " if is_rec else "") + status
    while _tw(draw, label, f_s) > DISPLAY_W - 4 and len(label) > 1:
        label = label[:-1]
    draw.text((2, 1), label, font=f_s, fill="white")
    draw.line([0, DIV_Y, DISPLAY_W - 1, DIV_Y], fill="white")

    # Sentence area
    draw.rectangle([0, SENT_Y, DISPLAY_W - 1, PRED_Y - 1], fill="black")
    if not words:
        draw.text((2, SENT_Y + 2), "No words yet", font=f_m, fill="white")
    else:
        sentence = " ".join(str(w) for w in words if w is not None)
        lines    = _wrap(draw, sentence, f_m, DISPLAY_W - 4)
        visible  = max(1, SENT_H // LINE_H)
        off      = min(scroll, max(0, len(lines) - visible))
        y = SENT_Y + 2
        for line in lines[off: off + visible]:
            draw.text((2, y), line, font=f_m, fill="white")
            y += LINE_H

    # Prediction strip
    draw.rectangle([0, PRED_Y, DISPLAY_W - 1, DISPLAY_H - 1], fill="black")
    draw.line([0, PRED_Y, DISPLAY_W - 1, PRED_Y], fill="white")
    if prediction:
        word = str(prediction).upper()
        tw   = _tw(draw, word, f_l)
        tx   = max(0, (DISPLAY_W - tw) // 2)
        draw.text((tx, PRED_Y + 1), word, font=f_l, fill="white")
    else:
        draw.text((4, PRED_Y + 3), "Waiting...", font=f_s, fill="white")


def _draw_flash_ui(draw, text):
    f = font_medium or ImageFont.load_default()
    text = str(text or "")
    draw.rectangle([0, 0, DISPLAY_W - 1, DISPLAY_H - 1], fill="black")
    lines = _wrap(draw, text, f, DISPLAY_W - 6)
    total = len(lines) * LINE_H
    y     = max(2, (DISPLAY_H - total) // 2)
    for line in lines:
        draw.text((3, y), line, font=f, fill="white")
        y += LINE_H


def _draw_context_ui(draw, ctx):
    f_h = font_small  or ImageFont.load_default()
    f_t = font_medium or ImageFont.load_default()
    ctx = str(ctx or "")
    draw.rectangle([0, 0, DISPLAY_W - 1, DISPLAY_H - 1], fill="black")
    draw.text((2, 1), "CONTEXT:", font=f_h, fill="white")
    draw.line([0, DIV_Y, DISPLAY_W - 1, DIV_Y], fill="white")
    lines = _wrap(draw, ctx, f_t, DISPLAY_W - 4)
    y = DIV_Y + 3
    for line in lines[:4]:
        draw.text((2, y), line, font=f_t, fill="white")
        y += LINE_H

# ============================================================
# OLED DISPLAY THREAD
# ============================================================

def oled_display_thread():
    if not OLED_AVAILABLE:
        print("[OLED] Not available — thread exiting")
        return
    try:
        serial = i2c(port=1, address=0x3C)
        device = sh1106(serial)
        print("[OLED] SH1106 ready ✓")
    except Exception as e:
        print(f"[OLED] Init failed: {e}")
        return

    _load_fonts()

    try:
        with canvas(device) as draw:
            f = font_medium or ImageFont.load_default()
            draw.text((10, 20), "Sign Lang", font=f, fill="white")
            draw.text((10, 34), "System Ready", font=f, fill="white")
    except Exception:
        pass
    time.sleep(1.5)

    while True:
        oled_event.wait(timeout=0.3)
        oled_event.clear()

        with display_lock:
            prediction  = oled_prediction
            words       = list(oled_words)
            context     = oled_context
            status      = oled_status
            flash_text  = oled_flash_text
            flash_until = oled_flash_until
            scroll      = oled_scroll_offset
            is_rec      = oled_is_recording
            boot_mode   = oled_boot_mode
            boot_text   = oled_boot_text

        now = time.time()

        try:
            with canvas(device) as draw:
                if boot_mode:
                    _draw_flash_ui(draw, boot_text or "Booting...")
                elif flash_text and now < flash_until:
                    _draw_flash_ui(draw, flash_text)
                elif context and now < flash_until:
                    _draw_context_ui(draw, context)
                else:
                    _draw_normal_ui(draw, status, is_rec, words, scroll, prediction)
        except Exception as e:
            import traceback
            print(f"[OLED] Render error: {e}")
            traceback.print_exc()

# ============================================================
# OLED STATE SETTERS
# ============================================================

def oled_set_prediction(word):
    global oled_prediction
    with display_lock:
        oled_prediction = word
    oled_event.set()

def oled_set_words(words):
    global oled_words
    with display_lock:
        oled_words = list(words)
    oled_event.set()

def oled_set_status(s):
    global oled_status
    with display_lock:
        oled_status = s
    oled_event.set()

def oled_set_recording(v):
    global oled_is_recording
    with display_lock:
        oled_is_recording = v
    oled_event.set()

def oled_flash(text, duration=4.0):
    global oled_flash_text, oled_flash_until
    with display_lock:
        oled_flash_text  = str(text or "")
        oled_flash_until = time.time() + duration
    oled_event.set()

def oled_show_context(text, duration=6.0):
    global oled_context, oled_flash_until
    with display_lock:
        oled_context     = str(text or "")
        oled_flash_until = time.time() + duration
    oled_event.set()

def oled_scroll(direction):
    global oled_scroll_offset
    with display_lock:
        oled_scroll_offset = max(0, oled_scroll_offset + direction)
    oled_event.set()


def oled_boot_step(text, hold=0.7):
    """Show short startup step text on OLED (headless quick-terminal style)."""
    global oled_boot_mode, oled_boot_text
    if OLED_AVAILABLE:
        with display_lock:
            oled_boot_mode = True
            oled_boot_text = str(text or "")
        oled_event.set()
    print(f"[BOOT] {text.replace(chr(10), ' | ')}")
    time.sleep(max(hold, 0.2))


def oled_boot_finish():
    global oled_boot_mode
    with display_lock:
        oled_boot_mode = False
    oled_event.set()


def oled_force_clear():
    """Hard-clear OLED so stale text is not left after process exit."""
    if not OLED_AVAILABLE:
        return
    try:
        serial = i2c(port=1, address=0x3C)
        device = sh1106(serial)
        with canvas(device) as draw:
            draw.rectangle([0, 0, DISPLAY_W - 1, DISPLAY_H - 1], fill="black")
    except Exception as e:
        print(f"[OLED] Clear failed: {e}")

# ============================================================
# WEBSOCKET CONNECTION
# ============================================================

def connect_ws():
    print(f"Connecting to {WS_URL} ...")
    oled_set_status("CONNECTING")
    retry = 0
    while True:
        try:
            ws = websocket.create_connection(WS_URL, timeout=3)
            ws.settimeout(1)
            print("✓ Connected")
            oled_set_status("READY")
            return ws
        except (websocket.WebSocketException, socket.error) as e:
            retry += 1
            oled_set_status(f"RETRY {retry}")
            time.sleep(2)

# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    all_lm = [-1.0] * 128
    for idx, (hl, hi) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)):
        if idx >= 2:
            break
        coords = []
        for lm in hl.landmark:
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])
        label  = 0.0 if hi.classification[0].label == 'Left' else 1.0
        start  = idx * 64
        all_lm[start: start + 63] = coords
        all_lm[start + 63]        = label
    return all_lm

# ============================================================
# CAPTURE FRAME
# ============================================================

def capture_frame_as_base64():
    with latest_frame_lock:
        if latest_frame is None:
            return None
        fc = latest_frame.copy()
    ok, buf = cv2.imencode('.jpg', fc, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else None


def capture_dev_preview_as_base64():
    with latest_frame_lock:
        if latest_frame is None:
            return None
        fc = latest_frame.copy()
    h, w = fc.shape[:2]
    if w > DEV_STREAM_WIDTH:
        nh = int(h * (DEV_STREAM_WIDTH / float(w)))
        fc = cv2.resize(fc, (DEV_STREAM_WIDTH, nh))
    ok, buf = cv2.imencode('.jpg', fc, [cv2.IMWRITE_JPEG_QUALITY, DEV_STREAM_QUALITY])
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else None

# ============================================================
# AUDIO RECORDING
# ============================================================

def _audio_cb(indata, frames, time_info, status):
    with audio_lock:
        if audio_recording:
            audio_recorded_chunks.append(indata.copy())

def record_audio_fixed(duration_seconds, ws):
    global audio_recording, audio_recorded_chunks
    if not AUDIO_AVAILABLE:
        _send_ws(ws, {'type': 'error', 'error': 'sounddevice not installed'})
        return
    duration_seconds = min(duration_seconds, AUDIO_MAX_SECONDS)
    with audio_lock:
        audio_recording = True
        audio_recorded_chunks = []
    try:
        _send_ws(ws, {'type': 'audio_status', 'status': 'recording',
                      'duration': duration_seconds})
        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS,
                            dtype='int16', callback=_audio_cb):
            sd.sleep(int(duration_seconds * 1000))
    except Exception as e:
        _send_ws(ws, {'type': 'error', 'error': str(e)})
        with audio_lock:
            audio_recording = False
        oled_set_recording(False)
        return
    with audio_lock:
        audio_recording = False
        chunks = list(audio_recorded_chunks)
    oled_set_recording(False)
    _send_audio_chunks(chunks, duration_seconds, ws)

def record_audio_manual(ws):
    global audio_recording, audio_recorded_chunks
    if not AUDIO_AVAILABLE:
        _send_ws(ws, {'type': 'error', 'error': 'sounddevice not installed'})
        return
    audio_stop_event.clear()
    with audio_lock:
        audio_recording = True
        audio_recorded_chunks = []
    try:
        _send_ws(ws, {'type': 'audio_status', 'status': 'recording', 'duration': 0})
        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS,
                            dtype='int16', callback=_audio_cb):
            while not audio_stop_event.is_set():
                sd.sleep(100)
    except Exception as e:
        _send_ws(ws, {'type': 'error', 'error': str(e)})
        with audio_lock:
            audio_recording = False
        oled_set_recording(False)
        return
    with audio_lock:
        audio_recording = False
        chunks = list(audio_recorded_chunks)
    oled_set_recording(False)
    # Derive duration from actual captured samples, not assumed chunk size.
    total_samples = int(sum(chunk.shape[0] for chunk in chunks)) if chunks else 0
    duration = total_samples / float(AUDIO_SAMPLE_RATE) if total_samples > 0 else 0.0
    _send_audio_chunks(chunks, duration, ws)

def _send_audio_chunks(chunks, duration, ws):
    if not chunks:
        _send_ws(ws, {'type': 'error', 'error': 'No audio recorded'})
        return
    audio_np  = np.concatenate(chunks, axis=0)
    audio_b64 = base64.b64encode(audio_np.tobytes()).decode('utf-8')
    _send_ws(ws, {
        'type': 'audio', 'audio': audio_b64,
        'rate': AUDIO_SAMPLE_RATE, 'channels': AUDIO_CHANNELS,
        'duration': round(duration, 2), 'encoding': 'int16_pcm',
    })
    print("[AUDIO] Sent ✓")

# ============================================================
# WS HELPER
# ============================================================

def _send_ws(ws, payload):
    if ws is None:
        return
    try:
        ws.send(json.dumps(payload))
    except Exception as e:
        print(f"[WS] Send error: {e}")

# ============================================================
# KEYPAD THREAD
# ============================================================

def keypad_thread(ws_ref):
    if not GPIO_AVAILABLE:
        print("[KEYPAD] GPIO not available — disabled")
        return

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    for r in ROWS:
        GPIO.setup(r, GPIO.OUT)
        GPIO.output(r, GPIO.HIGH)
    for c in COLS:
        GPIO.setup(c, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print("[KEYPAD] Ready ✓")
    last_key  = None
    last_time = 0.0

    while True:
        try:
            for i, r in enumerate(ROWS):
                GPIO.output(r, GPIO.LOW)
                for j, c in enumerate(COLS):
                    if GPIO.input(c) == 0:
                        key = KEYS[i][j]
                        now = time.time()
                        if key != last_key or (now - last_time) > 0.4:
                            last_key  = key
                            last_time = now
                            # Notify laptop of keypress for UI highlight
                            _send_ws(ws_ref[0], {
                                'type':   'keypad_press',
                                'key':    key,
                                'action': KEY_ACTIONS.get(key, ''),
                            })
                            _handle_key(key, ws_ref)
                        time.sleep(0.05)
                GPIO.output(r, GPIO.HIGH)
            time.sleep(0.02)
        except Exception as e:
            print(f"[KEYPAD] Error: {e}")
            time.sleep(0.1)


def _handle_key(key, ws_ref):
    global oled_scroll_offset, oled_context

    ws     = ws_ref[0]
    action = KEY_ACTIONS.get(key)
    if not action:
        return

    print(f"[KEY] {key} → {action}")

    # ── Scroll ───────────────────────────────────────────────
    if action == 'scroll_up':
        oled_scroll(+1)
        return
    if action == 'scroll_down':
        oled_scroll(-1)
        return

    # ── Preset 1-3 ───────────────────────────────────────────
    if action.startswith('preset_'):
        slot = int(action.split('_')[1])
        with preset_lock:
            sentence = preset_cache.get(slot, "")
        if sentence:
            _send_ws(ws, {'command': 'trigger_preset', 'slot': slot, 'sentence': sentence})
            oled_set_status(f"P{slot} SENT")
            oled_flash(f"Preset {slot}:\n{sentence[:40]}", 4.0)
            print(f"[PRESET {slot}] {sentence}")
        else:
            oled_flash(f"Preset {slot}\nnot set", 2.0)
        return

    # ── Standard keypad inputs (5/6/7/8/9) ──────────────────
    if action.startswith('standard_key_'):
        key_input = action.rsplit('_', 1)[-1]
        _send_ws(ws, {'command': 'standard_key_input', 'key': key_input})
        oled_set_status(f"KEY {key_input}")
        return

    # ── Add Word ─────────────────────────────────────────────
    if action == 'add_word':
        with display_lock:
            word = oled_prediction
        if word:
            _send_ws(ws, {'command': 'add_word', 'word': word})
            oled_set_status(f"+ {word.upper()}")
        else:
            oled_flash("No word detected", 2.0)
        return

    # ── Delete Last ──────────────────────────────────────────
    if action == 'delete_last':
        _send_ws(ws, {'command': 'delete_last'})
        oled_set_status("DELETED LAST")
        return

    # ── Clear All ────────────────────────────────────────────
    if action == 'clear_words':
        _send_ws(ws, {'command': 'clear_words'})
        oled_set_words([])
        with display_lock:
            oled_scroll_offset = 0
        oled_set_status("CLEARED")
        return

    # ── Send to Gemini ───────────────────────────────────────
    if action == 'send_to_ai':
        with display_lock:
            has_words = len(oled_words) > 0
        if not has_words:
            oled_flash("Add words first", 2.5)
            return
        _send_ws(ws, {'command': 'send_to_ai'})
        oled_set_status("GEMINI...")
        oled_flash("Sending to\nGemini...", 30.0)
        return

    # ── OCR ──────────────────────────────────────────────────
    if action == 'request_ocr':
        _send_ws(ws, {'command': 'request_ocr'})
        oled_set_status("OCR...")
        oled_flash("Capturing &\nReading...", 30.0)
        return

    # ── Toggle Recording ─────────────────────────────────────
    if action == 'toggle_audio':
        with audio_lock:
            rec = audio_recording
        if rec:
            audio_stop_event.set()
            _send_ws(ws, {'command': 'stop_audio'})
            oled_set_status("REC STOPPED")
            oled_set_recording(False)
        else:
            _send_ws(ws, {'command': 'start_audio'})
            oled_set_status("RECORDING")
            oled_set_recording(True)
        return

# ============================================================
# INCOMING MESSAGE LISTENER
# ============================================================

def listen_for_commands(ws_ref):
    global oled_prediction, oled_context, oled_scroll_offset, dev_stream_enabled

    print("[CMD] Listener started")
    while True:
        ws = ws_ref[0]
        try:
            raw = ws.recv()
            if not raw:
                continue

            data     = json.loads(raw)
            msg_type = data.get('type')

            # ── Predicted word ────────────────────────────────
            if msg_type == 'show_word':
                word = data.get('word', '')
                with display_lock:
                    oled_prediction = word
                oled_event.set()
                continue

            # ── Words list updated ────────────────────────────
            if msg_type == 'words_updated':
                oled_set_words(data.get('words', []))
                oled_set_status("READY")
                continue

            # ── Preset data from laptop ───────────────────────
            if msg_type == 'presets_updated':
                presets = data.get('presets', {})
                with preset_lock:
                    for k, v in presets.items():
                        try:
                            preset_cache[int(k)] = v
                        except (ValueError, KeyError):
                            pass
                print(f"[PRESET] Cache updated: {preset_cache}")
                continue

            # ── Gemini generating ─────────────────────────────
            if msg_type == 'gemini_status':
                if data.get('status') == 'generating':
                    oled_set_status("GEMINI...")
                    oled_flash("Generating\nsentence...", 30.0)
                else:
                    oled_set_status("READY")
                continue

            # ── Gemini sentence result ────────────────────────
            if msg_type == 'gemini_sentence':
                sentence = data.get('sentence', '')
                oled_set_status("GEMINI DONE")
                oled_flash(sentence, 8.0)
                continue

            # ── OCR result ────────────────────────────────────
            if msg_type == 'ocr_result':
                text = data.get('text', '')
                if text:
                    with display_lock:
                        oled_context       = text
                        oled_scroll_offset = 0
                    oled_set_status("OCR DONE")
                    oled_show_context(text, 6.0)
                    _send_ws(ws_ref[0], {'command': 'add_context', 'text': text})
                continue

            # ── Audio status ──────────────────────────────────
            if msg_type == 'audio_status':
                s = data.get('status', '')
                if s == 'recording':
                    oled_set_status("RECORDING")
                    oled_set_recording(True)
                elif s == 'transcribing':
                    oled_set_status("TRANSCRIBING")
                continue

            # ── Audio / Whisper result ────────────────────────
            if msg_type == 'audio_result':
                text = data.get('text', '')
                oled_set_recording(False)
                if text:
                    with display_lock:
                        oled_context       = text
                        oled_scroll_offset = 0
                    oled_set_status("AUDIO DONE")
                    oled_show_context(text, 6.0)
                    _send_ws(ws_ref[0], {'command': 'add_context', 'text': text})
                continue

            # ── Context updated ───────────────────────────────
            if msg_type == 'context_updated':
                ctx = data.get('context', '')
                with display_lock:
                    oled_context = ctx
                if not ctx:
                    oled_set_status("CTX CLEARED")
                oled_event.set()
                continue

            # ── Error ─────────────────────────────────────────
            if msg_type == 'error':
                err = data.get('error', {})
                ctx_str = err.get('context', 'Error') if isinstance(err, dict) else 'Error'
                msg_str = err.get('error', str(err))  if isinstance(err, dict) else str(err)
                oled_set_status("ERROR")
                oled_flash(f"{ctx_str}:\n{msg_str}", 5.0)
                print(f"[ERR] {ctx_str}: {msg_str}")
                continue

            # ── Commands from laptop ──────────────────────────
            if msg_type != 'command':
                continue

            command = data.get('command', '')

            if command == 'capture_frame':
                b64 = capture_frame_as_base64()
                if b64:
                    _send_ws(ws, {
                        'type': 'frame', 'frame': b64,
                        'width': CAPTURE_WIDTH, 'height': CAPTURE_HEIGHT,
                        'timestamp': time.time(),
                    })
                else:
                    _send_ws(ws, {'type': 'error', 'error': 'No frame available'})

            elif command == 'record_audio':
                duration = data.get('params', {}).get('duration', 10)
                with audio_lock:
                    rec = audio_recording
                if not rec:
                    threading.Thread(target=record_audio_fixed,
                                     args=(duration, ws), daemon=True).start()
                else:
                    _send_ws(ws, {'type': 'error', 'error': 'Already recording'})

            elif command == 'start_audio':
                with audio_lock:
                    rec = audio_recording
                if not rec:
                    threading.Thread(target=record_audio_manual,
                                     args=(ws,), daemon=True).start()

            elif command == 'stop_audio':
                with audio_lock:
                    rec = audio_recording
                if rec:
                    audio_stop_event.set()

            elif command == 'toggle_dev_stream':
                enabled = bool(data.get('params', {}).get('enabled', False))
                dev_stream_enabled = enabled
                _send_ws(ws, {'type': 'dev_stream_status', 'enabled': dev_stream_enabled})

        except websocket.WebSocketTimeoutException:
            continue
        except (websocket.WebSocketConnectionClosedException, BrokenPipeError, OSError):
            time.sleep(0.5)
        except json.JSONDecodeError as e:
            print(f"[CMD] Bad JSON: {e}")
        except Exception as e:
            print(f"[CMD] Error: {e}")
            time.sleep(0.1)

# ============================================================
# CV2 WINDOW
# ============================================================

def calculate_fps():
    global frame_count, fps_val, last_fps_time
    frame_count += 1
    now = time.time()
    if now - last_fps_time >= 1.0:
        fps_val       = frame_count / (now - last_fps_time)
        frame_count   = 0
        last_fps_time = now
    return fps_val

def draw_info(image, hands_detected, data_sent):
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (270, 155), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    y = 35
    cv2.putText(image, "Sign Language Sender", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25
    cv2.putText(image, f"Hands: {hands_detected}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if hands_detected > 0 else (0, 0, 255), 1)
    y += 25
    cv2.putText(image, f"FPS: {calculate_fps():.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25
    cv2.putText(image, "SENDING" if data_sent else "IDLE", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if data_sent else (255, 255, 255), 1)
    y += 25
    with audio_lock:
        rec = audio_recording
    if rec:
        cv2.putText(image, "* REC AUDIO", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(image, "S=Capture  Q=Quit", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(image, "Press 'q' to quit", (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ============================================================
# MAIN
# ============================================================

def main():
    global last_send_time, last_heartbeat_time, latest_frame, last_dev_stream_time

    threading.Thread(target=oled_display_thread, daemon=True).start()
    time.sleep(0.2)
    oled_boot_step("Booting...\nOLED ready", hold=0.8)

    oled_boot_step("Init network\nConnecting WS", hold=0.7)
    ws = connect_ws()
    ws_ref = [ws]
    ws_ref_holder[0] = ws
    oled_boot_step("WebSocket\nConnected", hold=0.6)

    oled_boot_step("Starting\nKeypad thread", hold=0.5)
    threading.Thread(target=keypad_thread, args=(ws_ref,), daemon=True).start()
    oled_boot_step("Keypad\nReady", hold=0.5)

    oled_boot_step("Starting\nListener", hold=0.5)
    threading.Thread(target=listen_for_commands, args=(ws_ref,), daemon=True).start()
    oled_boot_step("Listener\nReady", hold=0.5)

    oled_boot_step("Starting\nCamera", hold=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        oled_flash("Camera failed", 2.5)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Camera: {int(cap.get(3))}×{int(cap.get(4))}")
    print("Detection loop started\n")
    oled_boot_step("Camera ready\nSystem live", hold=0.8)
    oled_boot_finish()
    oled_set_status("READY")

    try:
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                continue

            now   = time.time()
            image = cv2.flip(image, 1)

            with latest_frame_lock:
                latest_frame = image.copy()

            rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            data_sent      = False
            hands_detected = 0

            if results.multi_hand_landmarks:
                hands_detected = len(results.multi_hand_landmarks)
                if SHOW_LANDMARKS:
                    for hl in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS,
                                                  landmark_spec, connection_spec)
                if (now - last_send_time) >= SEND_INTERVAL:
                    lm = extract_landmarks(results)
                    if lm and len(lm) == 128:
                        try:
                            ws.send(json.dumps({'landmarks': lm, 'timestamp': now,
                                                'hands': hands_detected}))
                            data_sent           = True
                            last_send_time      = now
                            last_heartbeat_time = now
                            ws_ref[0]           = ws
                            ws_ref_holder[0]    = ws
                        except (websocket.WebSocketConnectionClosedException,
                                socket.error, BrokenPipeError):
                            ws = connect_ws()
                            ws_ref[0]        = ws
                            ws_ref_holder[0] = ws

            elif (now - last_heartbeat_time) >= HEARTBEAT_INTERVAL:
                try:
                    ws.send(json.dumps({'type': 'ping', 'status': 'idle', 'timestamp': now}))
                    last_heartbeat_time = now
                except Exception:
                    pass

            if dev_stream_enabled and (now - last_dev_stream_time) >= (1.0 / DEV_STREAM_FPS):
                preview_b64 = capture_dev_preview_as_base64()
                if preview_b64:
                    _send_ws(ws, {
                        'type': 'dev_frame',
                        'frame': preview_b64,
                        'timestamp': now,
                    })
                    last_dev_stream_time = now

            # Headless mode: keep pipeline running without OpenCV preview window.
            draw_info(image, hands_detected, data_sent)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        oled_set_recording(False)
        oled_set_status("SHUTDOWN")
        oled_flash("Shutting down...", 0.6)
        time.sleep(0.15)
        oled_force_clear()
        try:
            ws.close()
        except Exception:
            pass
        print("Shutdown complete")


if __name__ == "__main__":
    main()