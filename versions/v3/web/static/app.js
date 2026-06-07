/**
 * Sign Language Recognition System - Enhanced Frontend with Debugging
 */

// ============================================================
// GLOBAL STATE
// ============================================================

let ws = null;
let words = [];
let context = "";
let currentWord = "";
let currentConfidence = 0;
let piConnected = false;
let pendingOCRText = "";
let pendingAudioText = "";
let pendingOCRImage = "";
let debugMode = false;
let debugData = {
    lastFrame: null,
    lastAudio: null,
    lastOCR: null,
    errors: [],
    stats: {}
};

// ============================================================
// WEBSOCKET CONNECTION
// ============================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('[WS] Connecting to:', wsUrl);
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('[WS] Connected successfully');
        showNotification('Connected to server', 'success');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        showNotification('Connection error', 'error');
    };
    
    ws.onclose = () => {
        console.log('[WS] Disconnected - Reconnecting in 3s...');
        showNotification('Disconnected - Reconnecting...', 'warning');
        setTimeout(connectWebSocket, 3000);
    };
}

// ============================================================
// MESSAGE HANDLING
// ============================================================

function handleMessage(data) {
    console.log('[MSG]', data.type, data);
    
    switch(data.type) {
        case 'init':
            words = data.words || [];
            context = data.context || "";
            piConnected = data.pi_connected || false;
            debugData.stats = data.stats || {};
            updateWordsDisplay();
            updateContextDisplay();
            updatePiStatus(piConnected);
            updateDebugStats();
            console.log('[INIT] Features:', data.features);
            break;
        
        case 'word_detected':
            currentWord = data.word;
            currentConfidence = data.confidence;
            updateDetection(data.word, data.confidence);
            break;
        
        case 'words_updated':
            words = data.words || [];
            updateWordsDisplay();
            break;
        
        case 'context_updated':
            context = data.context || "";
            updateContextDisplay();
            break;
        
        case 'status':
            piConnected = data.pi_connected || false;
            debugData.stats = data.stats || {};
            updatePiStatus(piConnected);
            updateDebugStats();
            document.getElementById('wordCount').textContent = `Words: ${data.words_count}`;
            if (data.last_word) {
                document.getElementById('lastDetection').textContent = 
                    `Last: ${data.last_word} (${(data.last_confidence * 100).toFixed(0)}%)`;
            }
            break;
        
        case 'ocr_result':
            pendingOCRText = data.text;
            pendingOCRImage = data.image_preview;
            debugData.lastFrame = data.image_preview;
            showOCRModal(data.text, data.image_preview);
            updateDebugPanel();
            break;
        
        case 'audio_result':
            pendingAudioText = data.text;
            showAudioModal(data.text);
            break;
        
        case 'ai_response':
            showAIResponse(data.response, data.tokens_used);
            break;
        
        case 'error':
            handleError(data.error);
            break;
        
        case 'debug_data':
            updateDebugData(data);
            break;
    }
}

// ============================================================
// UI UPDATES
// ============================================================

function updateDetection(word, confidence) {
    document.getElementById('detectedWord').textContent = word.toUpperCase();
    document.getElementById('confidence').textContent = 
        `Confidence: ${(confidence * 100).toFixed(1)}%`;
    document.getElementById('confidenceFill').style.width = `${confidence * 100}%`;
    
    // Color code
    const fill = document.getElementById('confidenceFill');
    if (confidence > 0.9) {
        fill.style.background = '#10b981';
    } else if (confidence > 0.75) {
        fill.style.background = '#f59e0b';
    } else {
        fill.style.background = '#ef4444';
    }
}

function updateWordsDisplay() {
    const container = document.getElementById('wordsList');
    
    if (words.length === 0) {
        container.innerHTML = '<span class="word-item empty">No words yet - start signing!</span>';
    } else {
        container.innerHTML = words.map(word => 
            `<span class="word-item">${word}</span>`
        ).join('');
    }
    
    document.getElementById('wordCount').textContent = `Words: ${words.length}`;
}

function updateContextDisplay() {
    const container = document.getElementById('contextArea');
    
    if (!context || context.trim() === '') {
        container.innerHTML = '<span class="empty">No context added yet</span>';
        container.classList.add('empty');
    } else {
        container.textContent = context;
        container.classList.remove('empty');
    }
}

function updatePiStatus(connected) {
    piConnected = connected;
    const dot = document.getElementById('piStatus');
    const text = document.getElementById('piStatusText');
    
    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Raspberry Pi: Connected ✓';
    } else {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Raspberry Pi: Disconnected ✗';
    }
}

function showAIResponse(response, tokensUsed) {
    const responseDiv = document.getElementById('aiResponse');
    const responseText = document.getElementById('aiResponseText');
    
    responseText.textContent = response;
    responseDiv.classList.add('show');
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}]`, message);
    // Could add toast notifications here
}

// ============================================================
// DEBUGGING PANEL
// ============================================================

function toggleDebug() {
    debugMode = !debugMode;
    const panel = document.getElementById('debugPanel');
    panel.classList.toggle('show');
    
    if (debugMode) {
        refreshDebug();
    }
}

function refreshDebug() {
    sendCommand('get_debug_data');
}

function updateDebugStats() {
    document.getElementById('statFrames').textContent = debugData.stats.frames_received || 0;
    document.getElementById('statPredictions').textContent = debugData.stats.predictions_made || 0;
    document.getElementById('statProcessing').textContent = 
        `${(debugData.stats.avg_processing_time || 0).toFixed(1)}ms`;
    document.getElementById('statOCR').textContent = debugData.stats.ocr_calls || 0;
}

function updateDebugData(data) {
    debugData = {
        lastFrame: data.last_frame,
        lastAudio: data.last_audio,
        lastOCR: data.last_ocr,
        errors: data.error_log || [],
        stats: data.stats || {}
    };
    
    updateDebugPanel();
}

function updateDebugPanel() {
    // Update stats
    updateDebugStats();
    
    // Update OCR image
    const ocrImage = document.getElementById('lastOCRImage');
    if (debugData.lastFrame) {
        ocrImage.src = `data:image/jpeg;base64,${debugData.lastFrame}`;
        ocrImage.alt = 'Last captured image';
    } else {
        ocrImage.src = '';
        ocrImage.alt = 'No image captured yet';
    }
    
    // Update error log
    const errorLog = document.getElementById('errorLog');
    if (debugData.errors && debugData.errors.length > 0) {
        errorLog.innerHTML = debugData.errors.map(err => `
            <div class="error-entry">
                <div class="error-time">${new Date(err.timestamp).toLocaleTimeString()}</div>
                <div class="error-context">${err.context}</div>
                <div class="error-message">${err.type}: ${err.error}</div>
                ${err.solution ? `<div class="error-solution">💡 ${err.solution}</div>` : ''}
            </div>
        `).reverse().join('');
    } else {
        errorLog.innerHTML = '<div style="color: #9ca3af; text-align: center; padding: 20px;">No errors yet - system running smoothly!</div>';
    }
}

function handleError(error) {
    console.error('[ERROR]', error);
    
    // Add to debug data
    if (!debugData.errors) {
        debugData.errors = [];
    }
    debugData.errors.push(error);
    if (debugData.errors.length > 50) {
        debugData.errors.shift();
    }
    
    // Update panel if open
    if (debugMode) {
        updateDebugPanel();
    }
    
    // Show notification
    showNotification(`${error.context}: ${error.error}`, 'error');
    
    // Log to console with solution
    if (error.solution) {
        console.log('[SOLUTION]', error.solution);
    }
}

// ============================================================
// WORD MANAGEMENT
// ============================================================

function addWord() {
    if (!currentWord) {
        alert('No word detected yet!');
        return;
    }
    
    sendCommand('add_word', { word: currentWord });
}

function deleteLast() {
    if (words.length === 0) {
        alert('No words to delete!');
        return;
    }
    
    sendCommand('delete_last');
}

function clearWords() {
    if (words.length === 0) {
        return;
    }
    
    if (confirm('Clear all words?')) {
        sendCommand('clear_words');
    }
}

function sendToAI() {
    if (words.length === 0) {
        alert('No words to send!');
        return;
    }
    
    document.getElementById('aiResponse').classList.remove('show');
    sendCommand('send_to_ai');
}

// ============================================================
// CONTEXT MANAGEMENT
// ============================================================

function removeContext() {
    if (!context || confirm('Remove all context?')) {
        sendCommand('remove_context');
    }
}

function addContext(text) {
    sendCommand('add_context', { text: text });
}

// ============================================================
// OCR FUNCTIONALITY
// ============================================================

function requestOCR() {
    if (!piConnected) {
        alert('Raspberry Pi not connected!');
        return;
    }
    
    sendCommand('request_ocr');
    showNotification('Capturing high-quality frame for OCR...', 'info');
}

function showOCRModal(text, imageBase64) {
    document.getElementById('ocrText').textContent = text;
    
    // Show preview image
    const preview = document.getElementById('ocrPreviewImage');
    if (imageBase64) {
        preview.src = `data:image/jpeg;base64,${imageBase64}`;
        preview.style.display = 'block';
    } else {
        preview.style.display = 'none';
    }
    
    document.getElementById('ocrModal').classList.add('show');
}

function confirmOCR() {
    if (pendingOCRText) {
        addContext(pendingOCRText);
        closeModal('ocrModal');
        pendingOCRText = "";
        pendingOCRImage = "";
    }
}

// ============================================================
// AUDIO FUNCTIONALITY
// ============================================================

function requestAudio() {
    if (!piConnected) {
        alert('Raspberry Pi not connected!');
        return;
    }
    
    const duration = prompt('Recording duration in seconds (1-60):', '30');
    if (duration === null) return;
    
    const seconds = parseInt(duration);
    if (isNaN(seconds) || seconds < 1 || seconds > 60) {
        alert('Invalid duration!');
        return;
    }
    
    sendCommand('request_audio', { duration: seconds });
    showNotification(`Recording for ${seconds} seconds...`, 'info');
}

function showAudioModal(text) {
    document.getElementById('audioText').textContent = text;
    document.getElementById('audioModal').classList.add('show');
}

function confirmAudio() {
    if (pendingAudioText) {
        addContext(pendingAudioText);
        closeModal('audioModal');
        pendingAudioText = "";
    }
}

// ============================================================
// BLUETOOTH FUNCTIONALITY
// ============================================================

function scanBluetooth() {
    document.getElementById('bluetoothStatus').innerHTML = 
        '<div style="text-align: center;"><div class="loading" style="display: inline-block;"></div> Scanning...</div>';
    sendCommand('scan_bluetooth');
}

// ============================================================
// MODAL MANAGEMENT
// ============================================================

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('show');
}

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('show');
    }
});

// ============================================================
// WEBSOCKET COMMAND SENDER
// ============================================================

function sendCommand(command, params = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('Not connected to server!');
        return;
    }
    
    const message = {
        command: command,
        ...params
    };
    
    ws.send(JSON.stringify(message));
    console.log('[SEND]', command, params);
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + A: Add word
    if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        addWord();
    }
    
    // Ctrl/Cmd + D: Delete last
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        deleteLast();
    }
    
    // Ctrl/Cmd + Enter: Send to AI
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        sendToAI();
    }
    
    // F12: Toggle debug
    if (e.key === 'F12') {
        e.preventDefault();
        toggleDebug();
    }
});

// ============================================================
// INITIALIZATION
// ============================================================

window.addEventListener('load', () => {
    connectWebSocket();
    console.log('[APP] Initialized');
    
    // Auto-refresh debug panel every 5 seconds if open
    setInterval(() => {
        if (debugMode) {
            refreshDebug();
        }
    }, 5000);
});

window.addEventListener('beforeunload', (e) => {
    if (words.length > 0) {
        e.preventDefault();
        e.returnValue = '';
        return '';
    }
});

console.log(`
╔═══════════════════════════════════════════════════════════╗
║   Sign Language Recognition System - Enhanced Version    ║
║                                                           ║
║   Features:                                               ║
║   • Real-time ML detection                               ║
║   • High-quality OCR with image preview                  ║
║   • Audio transcription                                  ║
║   • Debugging panel (F12 or button)                      ║
║   • Detailed error handling with solutions               ║
║                                                           ║
║   Shortcuts:                                              ║
║   • Ctrl+A: Add word                                      ║
║   • Ctrl+D: Delete last                                   ║
║   • Ctrl+Enter: Send to AI                                ║
║   • F12: Toggle debug panel                               ║
╚═══════════════════════════════════════════════════════════╝
`);
