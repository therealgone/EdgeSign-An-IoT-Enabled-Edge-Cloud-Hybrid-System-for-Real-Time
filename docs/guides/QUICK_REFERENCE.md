# QUICK REFERENCE GUIDE
## Top-Tier Sign Language Detection System

---

## ⚡ QUICK START (60 seconds)

### Laptop:
```bash
python receiver_improved.py
```

### Raspberry Pi:
```bash
python sender_improved.py
```

**That's it!** System should connect and start detecting.

---

## 🎛️ QUICK SETTINGS

### Make Detection More Sensitive (detect easier)
**Edit `receiver_improved.py` line 16:**
```python
CONFIDENCE_THRESHOLD = 0.60  # Was 0.70
```

### Make Detection More Stable (less false positives)
**Edit `receiver_improved.py` line 22:**
```python
MIN_CONSENSUS = 7  # Was 5
```

### Make Detection Faster (more responsive)
**Edit `receiver_improved.py` line 19:**
```python
SMOOTHING_WINDOW = 5  # Was 7
```

---

## 🔥 PRESETS

### Ultra Fast (for demos)
```python
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW = 3
MIN_CONSENSUS = 2
```

### Balanced (default - recommended)
```python
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7
MIN_CONSENSUS = 5
```

### Ultra Stable (production)
```python
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW = 10
MIN_CONSENSUS = 7
```

---

## 🚨 COMMON ISSUES (90% of problems)

### Problem: No connection
**Fix:** Change PC IP in `sender_improved.py` line 13
```python
PC_IP = "YOUR.LAPTOP.IP.HERE"
```

### Problem: Words not showing
**Fix:** Hold sign for 2 seconds, or enable debug mode:
```python
DEBUG_MODE = True  # Line 25 in receiver_improved.py
```

### Problem: Wrong words detected
**Fix:** Lower confidence threshold:
```python
CONFIDENCE_THRESHOLD = 0.60  # Line 16 in receiver_improved.py
```

### Problem: Too many false detections
**Fix:** Increase consensus requirement:
```python
MIN_CONSENSUS = 7  # Line 22 in receiver_improved.py
```

---

## 📊 UNDERSTANDING OUTPUT

### Normal Output:
```
[FRAME #45]
Raw Prediction: HELLO (Confidence: 85.3%)
Top 3:
  HELLO          ████████████████████ 85.3%
  HELP           ███████             34.2%
  YES            ████                18.7%
```
✅ This is normal - just monitoring

### Detected Word:
```
============================================================
🎯 DETECTED WORD: HELLO
📊 Confidence: 85.3%
📈 Consensus: 6/7
============================================================
```
✅ This means word was recognized!

### Low Confidence:
```
[LOW CONFIDENCE] HELLO (62.5%) - Threshold: 70.0%
```
⚠️ Sign detected but below threshold - hold longer or adjust threshold

---

## 🎯 OPTIMIZATION GUIDE

### Your Goal: Fast Response
```python
TARGET_FPS = 40           # sender
SMOOTHING_WINDOW = 5      # receiver
MIN_CONSENSUS = 3         # receiver
CONFIDENCE_THRESHOLD = 0.65  # receiver
```

### Your Goal: High Accuracy
```python
TARGET_FPS = 25           # sender
SMOOTHING_WINDOW = 10     # receiver
MIN_CONSENSUS = 7         # receiver
CONFIDENCE_THRESHOLD = 0.80  # receiver
```

### Your Goal: Balanced (Recommended)
```python
TARGET_FPS = 30           # sender
SMOOTHING_WINDOW = 7      # receiver
MIN_CONSENSUS = 5         # receiver
CONFIDENCE_THRESHOLD = 0.70  # receiver
```

---

## 🔍 DEBUG MODE

### Enable:
```python
DEBUG_MODE = True  # Line 25 in receiver_improved.py
```

### What you'll see:
- Every single prediction
- Confidence bars for top 3 words
- Consensus tracking
- When words get detected

### Use when:
- Testing new signs
- Debugging detection issues
- Checking model performance
- Finding confidence sweet spot

---

## 🌐 NETWORK SETUP

### Find Your Laptop IP:

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

**Update in sender_improved.py:**
```python
PC_IP = "192.168.0.XXX"  # Your laptop's IP
```

---

## 🎨 VISUAL FEATURES

### Raspberry Pi Display Shows:
- ✅ Hand landmarks (red dots, green lines)
- ✅ FPS counter
- ✅ Number of hands detected
- ✅ Send status (SENDING/IDLE)

### Laptop Terminal Shows:
- ✅ Model information
- ✅ Live predictions
- ✅ Detected words with confidence
- ✅ Top 3 alternatives

---

## 🔧 ADVANCED TUNING

### Camera Settings (sender_improved.py line 252):
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)            # FPS
```

### MediaPipe Settings (sender_improved.py line 21):
```python
MIN_DETECTION_CONFIDENCE = 0.7  # Initial detection
MIN_TRACKING_CONFIDENCE = 0.6   # Tracking stability
```

---

## ⚠️ CRITICAL REQUIREMENTS

### Must Have:
- ✅ Model trained with `train_model_improved.py`
- ✅ Model file in `models/sign_language_model.pkl`
- ✅ Both devices on same WiFi network
- ✅ Port 8765 open on laptop

### Optional (but helps):
- Good lighting on hands
- Plain background
- Hands clearly visible
- Steady hand movements

---

## 💡 PRO TIPS

1. **Better Detection**: Collect 50+ samples per word during training
2. **Faster Response**: Use lighter background, better lighting
3. **More Stable**: Increase smoothing window and consensus
4. **Less CPU**: Lower FPS to 20-25
5. **Debug Issues**: Always start with `DEBUG_MODE = True`

---

## 📞 EMERGENCY FIXES

### System Frozen?
1. Press `Ctrl+C` to stop
2. Restart both scripts
3. Check network connection

### Constant Wrong Words?
1. Retrain model with more data
2. Lower confidence threshold
3. Check lighting conditions

### No Hands Detected?
1. Better lighting
2. Check camera is working
3. Lower detection confidence in sender

---

## ✅ HEALTH CHECK

Run this checklist if issues occur:

- [ ] Model file exists and loads
- [ ] Connection established (see "Connected successfully!")
- [ ] Hand landmarks visible on Pi screen
- [ ] FPS > 20 on Pi display
- [ ] Terminal shows predictions (even if low confidence)
- [ ] No Python errors in terminal

If all ✅ → System is healthy!
If any ❌ → Check documentation section for that issue

---

## 🎓 REMEMBER

**The 70% threshold means:**
- Model must be 70% sure before prediction
- Reduces false positives significantly
- May need to hold sign for 1-2 seconds
- Perfect for production use

**The consensus system means:**
- Word must appear 5 out of 7 times
- Eliminates jitter and false triggers
- Ensures stable, reliable detection
- Creates professional user experience

---

## 📱 CONTROLS

### Raspberry Pi Window:
- **Q** = Quit application

### Terminal:
- **Ctrl+C** = Stop server

---

**That's it! You now have a production-ready sign language detection system! 🚀**
