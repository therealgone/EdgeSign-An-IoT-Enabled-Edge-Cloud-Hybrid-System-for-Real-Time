"""
Sign Language Sender — Raspberry Pi
Headless (no monitor required)
Streams hand landmarks + sends high-quality camera frames for PaddleOCR
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

# Optional audio
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[INFO] pyaudio not installed — audio disabled (pip install pyaudio)")

# ============================================================
# CONFIGURATION — edit these
# ============================================================

PC_IP  = "YOUR_LAPTOP_IP_OR_HOSTNAME"  # e.g. 192.168.1.100 or my-laptop.local
WS_URL = f"ws://{PC_IP}:8765"

# Landmark streaming
TARGET_FPS         = 30
SEND_INTERVAL      = 1.0 / TARGET_FPS
HEARTBEAT_INTERVAL = 5.0

# MediaPipe hand detection
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.6
MAX_NUM_HANDS            = 2

# OCR frame capture settings
# Higher resolution → better OCR on real-world text
# 1280×720 is a good balance between quality and transmission speed
OCR_WIDTH   = 1280
OCR_HEIGHT  = 720
OCR_QUALITY = 92   # JPEG quality 0–100

# Audio
AUDIO_RATE  = 16000
AUDIO_CHUNK = 1024

# ============================================================
# GLOBALS
# ============================================================

last_send_time      = 0
last_heartbeat_time = 0
ws_conn             = None       # active WebSocket connection
command_queue       = deque()
is_recording_audio  = False
connection_attempts = 0

last_error_time = 0.0
ERROR_COOLDOWN  = 5.0   # seconds between printing the same error

status = {
    'connected':         False,
    'frames_sent':       0,
    'heartbeats_sent':   0,
    'commands_received': 0,
    'errors':            0,
}

# ============================================================
# LOGGING
# ============================================================

def log_error(context, error, solution=""):
    global last_error_time
    now = time.time()
    if now - last_error_time < ERROR_COOLDOWN:
        return
    last_error_time  = now
    status['errors'] += 1
    print(f"\n[ERROR] {context}")
    print(f"  Type:    {type(error).__name__}")
    print(f"  Message: {error}")
    if solution:
        print(f"  Fix:     {solution}")
    print()

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# ============================================================
# WEBSOCKET CONNECTION
# ============================================================

def connect_ws():
    """Connect to laptop receiver with automatic retry."""
    global connection_attempts
    connection_attempts += 1

    print("=" * 60)
    print("RASPBERRY PI — SIGN LANGUAGE SENDER")
    print("=" * 60)
    print(f"Connecting to {WS_URL}  (attempt #{connection_attempts})")

    for attempt in range(1, 11):
        try:
            conn = websocket.create_connection(WS_URL, timeout=5)
            conn.settimeout(1)
            # Send initial ping so receiver knows Pi is alive
            conn.send(json.dumps({'type': 'ping', 'source': 'pi'}))
            status['connected'] = True
            print(f"[OK] Connected to laptop!")
            print("=" * 60)
            return conn

        except socket.timeout:
            print(f"  Timeout (attempt {attempt}/10) — is receiver running?")
            time.sleep(2)

        except ConnectionRefusedError:
            print(f"  Refused (attempt {attempt}/10) — is receiver_web_v2.py running?")
            time.sleep(2)

        except socket.gaierror:
            print(f"[FATAL] Cannot resolve IP: {PC_IP}")
            print("  → Update PC_IP at the top of this file to your laptop's IP")
            return None

        except Exception as e:
            log_error(f"Connection attempt {attempt}", e,
                      "Check network and laptop firewall (port 8765)")
            time.sleep(2)

    print("[FATAL] 10 connection attempts failed.")
    print("  1. Verify PC_IP is correct (laptop: ipconfig / ifconfig)")
    print("  2. Confirm receiver_web_v2.py is running on the laptop")
    print("  3. Both devices must be on the same Wi-Fi network")
    print("  4. Laptop firewall must allow inbound TCP port 8765")
    return None

# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):
    """Pack both detected hands into a flat 128-element array."""
    if not results.multi_hand_landmarks:
        return None

    flat = [-1.0] * 128   # -1 = slot not used

    for idx, (hand_lm, hand_info) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)):
        if idx >= 2:
            break

        coords = []
        for lm in hand_lm.landmark:
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])

        label = 0.0 if hand_info.classification[0].label == 'Left' else 1.0
        start = idx * 64
        flat[start : start + 63] = coords
        flat[start + 63]         = label

    return flat

# ============================================================
# FRAME CAPTURE FOR OCR — real-world camera
# ============================================================

def capture_frame_for_ocr(cap):
    """
    Capture one frame and send it to the laptop for PaddleOCR.

    Pipeline (optimised for real-world text detection):
      1. Raise camera resolution to OCR_WIDTH x OCR_HEIGHT
      2. CLAHE contrast enhancement — lifts text in uneven lighting
         without blowing out highlights (better than raw brightness add)
      3. Unsharp mask — sharpens text edges without ringing artefacts
      4. High-quality JPEG encode (OCR_QUALITY)
      5. Clean ASCII base64 — no line breaks, no padding issues
    """
    try:
        print("[OCR] Capturing frame...")

        # ---- save current resolution, switch to OCR resolution ----
        orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  OCR_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OCR_HEIGHT)
        time.sleep(0.15)   # let camera sensor settle at new resolution

        ok, frame = cap.read()

        # restore streaming resolution immediately
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  orig_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)

        if not ok or frame is None:
            raise ValueError("cap.read() returned no frame — camera may be busy")

        h, w = frame.shape[:2]
        print(f"[OCR] Raw frame: {w}x{h}")

        # ---- CLAHE contrast enhancement ----
        # Works in LAB colour space so only luminance (L) is adjusted,
        # leaving colours (A, B) unchanged — prevents colour shift.
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        frame   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # ---- unsharp mask — sharpens text edges ----
        # Formula: sharpened = original * 1.5 - blurred * 0.5
        # Equivalent to adding high-frequency detail back to the image.
        blur  = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

        # ---- JPEG encode ----
        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, OCR_QUALITY])
        if not ok:
            raise ValueError("cv2.imencode failed — cannot encode frame as JPEG")

        # ---- base64 encode — NO newlines ----
        # decode('ascii') gives a pure ASCII string.
        # base64.b64encode never inserts newlines by default.
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        print(f"[OCR] Encoded: {w}x{h} | "
              f"JPEG {len(buf)//1024} KB | "
              f"base64 {len(b64)} chars")

        ws_conn.send(json.dumps({
            'type':       'frame',
            'frame':      b64,
            'resolution': f"{w}x{h}",
            'quality':    OCR_QUALITY,
            'timestamp':  time.time(),
        }))
        print("[OCR] Frame sent to laptop ✓")
        return True

    except ValueError as e:
        log_error("Frame capture", e, "Check camera is connected and not in use")
        return False
    except websocket.WebSocketConnectionClosedException:
        raise   # let main loop handle reconnection
    except Exception as e:
        log_error("Frame capture (unexpected)", e, "Check OpenCV and camera")
        return False

# ============================================================
# AUDIO RECORDING
# ============================================================

def record_audio(duration=30):
    global is_recording_audio

    if not AUDIO_AVAILABLE:
        print("[ERROR] pyaudio not installed — run: pip install pyaudio")
        return False

    is_recording_audio = True
    print(f"[AUDIO] Recording {duration}s...")

    try:
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK,
            )
        except OSError as e:
            print(f"[ERROR] Cannot open microphone: {e}")
            pa.terminate()
            is_recording_audio = False
            return False

        frames       = []
        total_chunks = int(AUDIO_RATE / AUDIO_CHUNK * duration)
        chunks_per_s = AUDIO_RATE // AUDIO_CHUNK

        for i in range(total_chunks):
            if not is_recording_audio:
                print("[AUDIO] Stopped early by command")
                break
            try:
                frames.append(stream.read(AUDIO_CHUNK, exception_on_overflow=False))
                if i % chunks_per_s == 0:
                    print(f"[AUDIO] {i // chunks_per_s}/{duration}s")
            except Exception as e:
                print(f"[AUDIO] Read error: {e}")
                break

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not frames:
            print("[ERROR] No audio captured")
            is_recording_audio = False
            return False

        audio_bytes = b''.join(frames)
        b64         = base64.b64encode(audio_bytes).decode('ascii')

        ws_conn.send(json.dumps({
            'type':      'audio',
            'audio':     b64,
            'duration':  duration,
            'rate':      AUDIO_RATE,
            'timestamp': time.time(),
        }))
        print(f"[AUDIO] Sent {len(audio_bytes)//1024} KB ✓")

    except websocket.WebSocketConnectionClosedException:
        raise
    except Exception as e:
        log_error("Audio recording", e, "Check microphone and pyaudio")

    is_recording_audio = False
    return True

# ============================================================
# COMMAND HANDLER
# ============================================================

def handle_command(cmd_data):
    global is_recording_audio

    command = cmd_data.get('command', '')
    params  = cmd_data.get('params', {})
    status['commands_received'] += 1
    print(f"[CMD] Received: {command}")

    if command == 'capture_frame':
        command_queue.append(('capture_frame', params))
        print("[CMD] Frame capture queued")

    elif command == 'record_audio':
        duration = int(params.get('duration', 30))
        threading.Thread(
            target=record_audio, args=(duration,), daemon=True
        ).start()
        print(f"[CMD] Audio recording started ({duration}s)")

    elif command == 'stop_audio':
        is_recording_audio = False
        print("[CMD] Audio recording stopping...")

    else:
        print(f"[CMD] Unknown command: {command!r}")


def check_commands():
    """Non-blocking poll for incoming commands from the laptop."""
    global ws_conn
    try:
        ws_conn.settimeout(0.001)
        msg  = ws_conn.recv()
        data = json.loads(msg)
        if isinstance(data, dict) and data.get('type') == 'command':
            handle_command(data)
    except socket.timeout:
        pass    # nothing waiting — normal
    except websocket.WebSocketConnectionClosedException:
        raise   # propagate so main loop can reconnect
    except json.JSONDecodeError as e:
        print(f"[WARNING] Bad JSON from laptop: {e}")
    except Exception:
        pass    # other transient errors — ignore
    finally:
        try:
            ws_conn.settimeout(1)
        except Exception:
            pass

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global last_send_time, last_heartbeat_time, ws_conn

    ws_conn = connect_ws()
    if ws_conn is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera — check it is connected and not in use by another app")
        return

    # Normal streaming resolution (fast, low latency)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,         30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce buffer lag

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera ready at {actual_w}x{actual_h}")
    print("[OK] Streaming landmarks... (headless mode)\n")

    frame_count      = 0
    last_status_time = time.time()

    try:
        while cap.isOpened():
            now = time.time()

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_count += 1

            # ---- check for commands every 5 frames ----
            if frame_count % 5 == 0:
                try:
                    check_commands()
                except websocket.WebSocketConnectionClosedException:
                    print("\n[WARNING] Connection lost — reconnecting...")
                    status['connected'] = False
                    try:
                        ws_conn.close()
                    except Exception:
                        pass
                    ws_conn = connect_ws()
                    if ws_conn is None:
                        break
                    continue

            # ---- execute queued OCR capture ----
            if command_queue:
                cmd, params = command_queue.popleft()
                if cmd == 'capture_frame':
                    try:
                        capture_frame_for_ocr(cap)
                    except websocket.WebSocketConnectionClosedException:
                        print("\n[WARNING] Connection lost during OCR capture — reconnecting...")
                        status['connected'] = False
                        try:
                            ws_conn.close()
                        except Exception:
                            pass
                        ws_conn = connect_ws()
                        if ws_conn is None:
                            break

            # ---- hand detection ----
            rgb     = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hands_detected = 0

            if results.multi_hand_landmarks:
                hands_detected = len(results.multi_hand_landmarks)

                if (now - last_send_time) >= SEND_INTERVAL:
                    landmarks = extract_landmarks(results)
                    if landmarks and len(landmarks) == 128:
                        try:
                            ws_conn.send(json.dumps({
                                'landmarks': landmarks,
                                'timestamp': now,
                                'hands':     hands_detected,
                            }))
                            status['frames_sent']  += 1
                            last_send_time          = now
                            last_heartbeat_time     = now
                        except websocket.WebSocketConnectionClosedException:
                            print("\n[WARNING] Disconnected during send — reconnecting...")
                            status['connected'] = False
                            try:
                                ws_conn.close()
                            except Exception:
                                pass
                            ws_conn = connect_ws()
                            if ws_conn is None:
                                break
                        except Exception as e:
                            log_error("Sending landmarks", e, "Check network stability")

            # ---- heartbeat when idle ----
            elif (now - last_heartbeat_time) >= HEARTBEAT_INTERVAL:
                try:
                    ws_conn.send(json.dumps({'type': 'ping', 'status': 'idle'}))
                    status['heartbeats_sent'] += 1
                    last_heartbeat_time        = now
                except Exception:
                    pass   # heartbeat failure is non-fatal

            # ---- status log every 10 s ----
            if now - last_status_time >= 10:
                state_str = f"{hands_detected} hand(s)" if hands_detected else "idle"
                print(
                    f"[STATUS] frame={frame_count} | "
                    f"sent={status['frames_sent']} | "
                    f"cmds={status['commands_received']} | "
                    f"errors={status['errors']} | "
                    f"{state_str}"
                )
                last_status_time = now

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    except Exception as e:
        log_error("Main loop", e, "Check camera and network")
    finally:
        print("[CLEANUP] Shutting down...")
        cap.release()
        hands.close()
        if ws_conn:
            try:
                ws_conn.close()
            except Exception:
                pass
        print(
            f"[SUMMARY] "
            f"Sent: {status['frames_sent']} frames | "
            f"Commands: {status['commands_received']} | "
            f"Errors: {status['errors']}"
        )
        print("[OK] Done")


if __name__ == "__main__":
    main()
