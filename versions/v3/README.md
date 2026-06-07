# EdgeSign V3 — Final Production Version

This is the **recommended version** of EdgeSign. It runs gesture detection on a Raspberry Pi and processes predictions, OCR, audio, and AI on a laptop with a web dashboard.

## Entry Points

| Script | Runs On | Purpose |
|--------|---------|---------|
| `laptop_rec.py` | Laptop | Main receiver + web server (ports 8765 & 3000) |
| `laptop_rec_display.py` | Laptop | Receiver with local OpenCV display window |
| `pi_sender.py` | Raspberry Pi | Camera capture + landmark streaming |
| `pi_sender_headless.py` | Raspberry Pi | Same as above, no GUI (for SSH/headless) |
| `receiver_web_v2.py` | Laptop | Alternative receiver (V2 web stack) |
| `sender_web_v2.py` | Raspberry Pi | Alternative sender (V2 web stack) |

## Setup

### Before You Run

1. Place a trained model at `models/sign_language_model.pkl` (train with V1 or copy an existing model).
2. Set your laptop IP in `pi_sender.py`:
   ```python
   PC_IP = "YOUR_LAPTOP_IP_OR_HOSTNAME"
   ```
3. Optionally set a Gemini API key for AI sentence generation:
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```

### Install Dependencies

**Laptop:**
```bash
pip install -r requirements_laptop.txt
```

**Raspberry Pi:**
```bash
pip install -r requirements_pi.txt
```

### Run

```bash
# Terminal 1 — Laptop (start first)
python laptop_rec.py

# Terminal 2 — Raspberry Pi
python pi_sender.py
```

Open **http://localhost:3000** in your browser.

## Configuration Files

- `presets.json` — Quick phrase presets for the web UI
- `INSTRUCTIONS.txt` — Detailed software list and troubleshooting
- `V2_UPGRADE_GUIDE.md` — Changes from V2 to V3

## Ports

| Port | Service |
|------|---------|
| 8765 | WebSocket (Pi ↔ Laptop) |
| 3000 | Web dashboard |

## Optional Features

All optional integrations degrade gracefully if packages are missing:

- **EasyOCR** — text extraction from camera frames
- **Whisper** — audio transcription
- **Gemini** — AI sentence rephrasing
- **pyttsx3** — text-to-speech on laptop
- **pybluez** — Bluetooth scanning (Linux/Mac)
- **luma.oled** — OLED display on Pi
