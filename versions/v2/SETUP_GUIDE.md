# Sign Language Recognition System with Web Interface
## Complete Setup Guide

---

## 🎯 System Overview

This is a complete sign language recognition system with:
- **ML-powered detection** running on your laptop
- **Real-time hand tracking** on Raspberry Pi
- **Web interface** for control and visualization
- **OCR** for extracting text from images
- **Audio** transcription for voice context
- **Bluetooth** device scanning and connection
- **AI integration** for natural language rephrasing

### Architecture

```
Raspberry Pi (sender_web.py)
    ↓ WebSocket (port 8765)
Laptop (receiver_web.py)
    ↓ WebSocket (port 3000)
Web Browser (index.html)
```

**All processing happens locally** - no external dependencies except AI API (optional).

---

## 📋 Prerequisites

### Laptop
- Python 3.8+
- Trained ML model in `models/sign_language_model.pkl`
- Modern web browser (Chrome, Firefox, Edge)

### Raspberry Pi
- Python 3.8+
- Camera module or USB webcam
- **Runs headless** (no monitor needed)

---

## 🚀 Installation

### 1. Laptop Setup

```bash
# Navigate to project directory
cd /path/to/project

# Install Python dependencies
pip install -r requirements_laptop.txt

# Create web directory structure
mkdir -p web/static

# Place files:
# - receiver_web.py → project root
# - web_index.html → web/index.html
# - web_app.js → web/static/app.js
```

#### Optional Features (Laptop):

**For OCR (text extraction from images):**
```bash
# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Mac:
brew install tesseract

# Windows:
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

**For Audio Transcription:**
```bash
# Already included in requirements_laptop.txt
# On Linux, you may need:
sudo apt-get install portaudio19-dev python3-pyaudio
```

**For Bluetooth (Linux only):**
```bash
# Already included in requirements_laptop.txt
sudo apt-get install bluetooth libbluetooth-dev
```

### 2. Raspberry Pi Setup

```bash
# Navigate to project directory
cd /path/to/project

# Install Python dependencies
pip install -r requirements_pi.txt

# Place file:
# - sender_web.py → project root

# Update laptop IP address
nano sender_web.py
# Change line 13: PC_IP = "YOUR.LAPTOP.IP.HERE"
```

#### Find Your Laptop IP:

**Windows:**
```cmd
ipconfig
# Look for "IPv4 Address"
```

**Mac/Linux:**
```bash
ifconfig
# Look for "inet" under active connection
```

---

## 🏃 Running the System

### Step 1: Start Receiver (Laptop)

```bash
python receiver_web.py
```

You should see:
```
============================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Web Interface + ML Detection
============================================================
[OK] Model loaded from models/sign_language_model.pkl
[OK] Actions: ['hello', 'thanks', 'help', ...]
...
[OK] Pi WebSocket listening on port 8765
[OK] Web server listening on http://localhost:3000
============================================================
SYSTEM READY!
============================================================

1. Open browser: http://localhost:3000
2. Start sender.py on Raspberry Pi
3. Make signs and control via web interface
```

### Step 2: Open Web Interface (Laptop)

Open your browser and go to:
```
http://localhost:3000
```

You should see the web interface with:
- Live detection display
- Words collection area
- Context management
- Bluetooth controls

### Step 3: Start Sender (Raspberry Pi)

```bash
python sender_web.py
```

You should see:
```
============================================================
RASPBERRY PI - SIGN LANGUAGE SENDER
============================================================
Connecting to laptop at ws://192.168.0.4:8765...
✓ Connected to laptop!
============================================================

[OK] Camera initialized
[OK] Starting detection loop (headless)...
```

The web interface will show "Raspberry Pi: Connected" ✅

---

## 🎮 Using the Web Interface

### Main Features

#### 1. Live Detection
- Shows currently detected word
- Confidence level with color-coded bar
- **Add Current Word** button to add to collection

#### 2. Collected Words
- Displays all collected words as chips
- **Delete Last**: Remove most recent word
- **Clear All**: Remove all words
- **Send to AI**: Process words with AI for natural rephrasing

#### 3. Context Management
- Add contextual information to enrich AI understanding
- **Capture & OCR**: Takes photo from Pi, extracts text
- **Record Audio**: Records speech (30 sec default), converts to text
- **Remove Context**: Clears all context

#### 4. Bluetooth Devices
- **Scan for Devices**: Find nearby Bluetooth devices
- Click device to select, then **Connect Selected**

---

## 📖 Button Functions Explained

### Add Word
Adds the currently detected word to your collection. Words are collected sequentially.

### Delete Last
Removes the most recently added word from your collection.

### Clear All
Removes all words from your collection. Useful to start fresh.

### Send to AI
Takes your collected words and sends them to AI for natural rephrasing. The AI combines your words into fluent human speech with **minimal token usage** - just sends the words, not elaborate prompts.

Example:
- Input: ["I", "need", "help", "tomorrow"]
- AI Output: "I need help tomorrow." (natural, concise)

### Capture & OCR
1. Captures current frame from Raspberry Pi camera
2. Sends frame to laptop
3. Runs OCR locally to extract text
4. Shows extracted text with confirmation dialog
5. If confirmed, adds text as context

**Use Case**: Point camera at sign, document, menu - extract text as context.

### Record Audio
1. Records audio for specified duration (default 30 seconds)
2. Converts audio to text using speech recognition (locally)
3. Shows transcription with confirmation dialog
4. If confirmed, adds transcription as context

**Use Case**: Describe your situation verbally to add rich context.

### Remove Context
Clears all previously added context. Use this before starting a new conversation to avoid AI confusion.

---

## 🔄 How It Works

### Detection Pipeline

```
1. Pi Camera → MediaPipe Hand Detection
2. Hand Landmarks (128 values) → WebSocket → Laptop
3. Laptop: Preprocess → ML Model → Word Prediction
4. Smooth & Filter → Confidence Check (70%)
5. Display in Web Interface
```

### Context Enrichment

**OCR Pipeline:**
```
User clicks "Capture & OCR"
  ↓
Web UI → WebSocket → Laptop → Command → Pi
  ↓
Pi captures frame → WebSocket → Laptop
  ↓
Laptop runs Tesseract OCR → Extracts text
  ↓
Text → WebSocket → Web UI (confirmation)
  ↓
User confirms → Added to context
```

**Audio Pipeline:**
```
User clicks "Record Audio"
  ↓
Web UI → WebSocket → Laptop → Command → Pi
  ↓
Pi records audio (30 sec) → WebSocket → Laptop
  ↓
Laptop transcribes (SpeechRecognition) → Text
  ↓
Text → WebSocket → Web UI (confirmation)
  ↓
User confirms → Added to context
```

### AI Integration

When you click "Send to AI":
```
Collected Words: ["hello", "need", "help"]
Context: "I am at the hospital"

Sent to AI API:
{
  "words": "hello need help",
  "context": "I am at the hospital"
}

AI Response:
"Hello, I need help at the hospital."
```

**Token Efficiency**: Only sends raw words + context, lets AI do the work.

---

## 🎨 UI Features

### Status Indicators
- **Green dot**: Raspberry Pi connected
- **Red dot**: Raspberry Pi disconnected
- **Word count**: Number of collected words
- **Last detection**: Most recent word and confidence

### Color-Coded Confidence
- **Green**: >90% confidence (excellent)
- **Orange**: 75-90% confidence (good)
- **Red**: 70-75% confidence (acceptable)

### Keyboard Shortcuts
- `Ctrl/Cmd + A`: Add current word
- `Ctrl/Cmd + D`: Delete last word
- `Ctrl/Cmd + Enter`: Send to AI

---

## 🔧 Configuration

### Change Detection Threshold

Edit `receiver_web.py`:
```python
CONFIDENCE_THRESHOLD = 0.70  # 70% (default)
```

Lower = more sensitive, Higher = more accurate

### Change Ports

Edit `receiver_web.py`:
```python
WEBSOCKET_PORT = 8765  # Pi connection
WEB_SERVER_PORT = 3000  # Web interface
```

### Change Camera FPS

Edit `sender_web.py`:
```python
TARGET_FPS = 30  # Frames per second
```

Lower = less CPU, Higher = faster detection

---

## 🐛 Troubleshooting

### Web interface won't open
- Check receiver is running: `python receiver_web.py`
- Try: `http://127.0.0.1:3000` instead of `localhost`
- Check firewall isn't blocking port 3000

### Pi won't connect
- Verify IP address in `sender_web.py`
- Check both devices on same network
- Ensure laptop firewall allows port 8765

### OCR not working
- Install Tesseract: See installation section
- Check Pi camera is working
- Ensure good lighting for text

### Audio not working
- Install pyaudio: `pip install pyaudio`
- On Linux: `sudo apt-get install python3-pyaudio`
- Check Pi has microphone (USB or built-in)

### No words detected
- Check model is trained and loaded
- Ensure good lighting on hands
- Hold signs steady for 1-2 seconds
- Check confidence threshold (might be too high)

### Bluetooth not working
- Only supported on Linux/Mac
- Install: `sudo apt-get install bluetooth libbluetooth-dev`
- Enable Bluetooth: `sudo systemctl start bluetooth`

---

## 📊 Performance Tips

### For Better Accuracy
1. Good, even lighting on hands
2. Plain background
3. Hold signs steady for 1-2 seconds
4. Train model with more samples

### For Better Speed
1. Lower TARGET_FPS on Pi (20-25)
2. Close other applications
3. Use Ethernet instead of WiFi
4. Use lightweight model variant

---

## 🔐 Security Notes

- System runs **entirely locally**
- No data sent to external servers (except AI API if configured)
- OCR and audio processing done on laptop
- WebSocket only between laptop and Pi on local network
- No authentication (assumes trusted local network)

---

## 🆘 Getting Help

### Check Logs
- Receiver logs: Terminal running `receiver_web.py`
- Sender logs: Terminal running `sender_web.py`
- Browser console: Press F12 in browser

### Common Issues

**"Model not found"**
- Train model first: `python train_model.py`
- Check path: `models/sign_language_model.pkl`

**"Pi keeps disconnecting"**
- Check network stability
- Increase ping timeout in receiver
- Use wired connection

**"OCR returns gibberish"**
- Ensure text is clear and well-lit
- Hold camera steady
- Use high-contrast text (black on white)

---

## 🎓 Advanced Usage

### Integrating AI API

Edit `receiver_web.py`, find `handle_web_command` function:

```python
elif command == 'send_to_ai':
    words_text = " ".join(state.words)
    context_text = state.context
    
    # YOUR AI API CALL HERE
    import requests
    
    response = requests.post('https://api.your-ai-service.com/chat', json={
        'messages': [
            {'role': 'system', 'content': 'Rephrase sign language words naturally.'},
            {'role': 'user', 'content': f"Words: {words_text}. Context: {context_text}"}
        ]
    })
    
    ai_response = response.json()['message']
    
    await ws.send_str(json.dumps({
        'type': 'ai_response',
        'response': ai_response,
        'tokens_used': len(words_text.split())
    }))
```

### Custom Styling

Edit `web/index.html` CSS section to change colors, fonts, layout.

### Adding Custom Features

1. Add button in HTML
2. Add JavaScript function in `app.js`
3. Add handler in `receiver_web.py`
4. Optionally add Pi handler in `sender_web.py`

---

## 📝 Summary

**What You Have:**
- ✅ Full-stack sign language recognition system
- ✅ Web interface for control and visualization
- ✅ OCR for text extraction from images
- ✅ Audio transcription for voice context
- ✅ Bluetooth device scanning
- ✅ AI integration for natural rephrasing
- ✅ Runs entirely locally (headless on Pi)
- ✅ Bidirectional communication (web ↔ laptop ↔ Pi)
- ✅ Production-ready with error handling

**System is ready to use!** 🚀
