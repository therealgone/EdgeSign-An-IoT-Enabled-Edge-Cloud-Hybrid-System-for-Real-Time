# Sign Language Web Interface

A real-time web interface for displaying sign language predictions from your Raspberry Pi.

## Features

- **Real-time Updates**: Live display of predicted words from the receiver
- **Sentence Builder**: Add words to build sentences
- **AI Integration**: Send sentences to Gemini AI for correction/completion
- **Text-to-Speech**: Placeholder for TTS functionality
- **Modern UI**: Clean, responsive interface

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Start the Web Server

```bash
python web_server.py
```

The server will start on `http://localhost:8000`

### 3. Start the Receiver (in a separate terminal)

```bash
python Reciver/reciver.py
```

The receiver will:
- Listen for landmarks from your Raspberry Pi on port 8765
- Process predictions using `sign_model.pkl`
- Send predictions to the web server automatically

### 4. Open the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. **Current Prediction**: Shows the latest predicted word from your Pi
2. **Add Button**: Adds the current word to the sentence buffer
3. **Delete Prev**: Removes the last word from the sentence
4. **Delete All**: Clears the entire sentence
5. **Listen**: Placeholder for text-to-speech (currently shows alert)
6. **Send to AI**: Sends the sentence to Gemini AI for correction

## Configuration

### Web Server URL

In `Reciver/reciver.py`, you can change the web server URL:

```python
WEB_SERVER_URL = "http://localhost:8000"  # Change if needed
SEND_TO_WEB = True  # Set to False to disable
```

### Gemini AI Integration

To use Gemini AI, set your API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then update the `call_gemini_ai()` function in `web_server.py` with the actual API implementation.

## API Endpoints

- `GET /` - Serves the web interface
- `POST /update_word` - Receives word updates from receiver
- `GET /get_sentence` - Returns current sentence buffer
- `POST /add_word` - Adds current word to buffer
- `POST /delete_prev` - Removes last word
- `POST /delete_all` - Clears sentence buffer
- `POST /send` - Sends sentence to AI
- `POST /listen` - Text-to-speech (placeholder)
- `WS /ws` - WebSocket for real-time updates

## Architecture

```
Raspberry Pi → Receiver (reciver.py) → Web Server (web_server.py) → Web Interface (HTML/JS)
```

1. Pi sends landmarks via WebSocket to receiver
2. Receiver processes landmarks and makes predictions
3. Receiver sends predictions to web server via HTTP POST
4. Web server broadcasts updates to connected web clients via WebSocket
5. Web interface displays predictions and manages sentence buffer

## Troubleshooting

- **Web interface not updating**: Check that the receiver is running and sending to the web server
- **Connection status shows disconnected**: Make sure the web server is running on port 8000
- **Predictions not appearing**: Verify that `sign_model.pkl` exists and the receiver loaded it successfully


