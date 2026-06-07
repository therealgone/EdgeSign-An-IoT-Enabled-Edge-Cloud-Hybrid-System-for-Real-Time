/**
 * Sign Language Recognition System - Frontend
 * Handles WebSocket communication, UI updates, and user interactions
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
let selectedBluetoothDevice = null;

// ============================================================
// WEBSOCKET CONNECTION
// ============================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('Connecting to:', wsUrl);
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('Connected to server');
        updateStatus('Connected to server', true);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Connection error', false);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from server');
        updateStatus('Disconnected - Reconnecting...', false);
        setTimeout(connectWebSocket, 3000);
    };
}

// ============================================================
// MESSAGE HANDLING
// ============================================================

function handleMessage(data) {
    console.log('Received:', data);
    
    switch(data.type) {
        case 'init':
            // Initial state
            words = data.words || [];
            context = data.context || "";
            piConnected = data.pi_connected || false;
            updateWordsDisplay();
            updateContextDisplay();
            updatePiStatus(piConnected);
            break;
        
        case 'word_detected':
            // New word detected
            currentWord = data.word;
            currentConfidence = data.confidence;
            updateDetection(data.word, data.confidence);
            break;
        
        case 'words_updated':
            // Words list changed
            words = data.words || [];
            updateWordsDisplay();
            break;
        
        case 'context_updated':
            // Context changed
            context = data.context || "";
            updateContextDisplay();
            break;
        
        case 'status':
            // System status
            piConnected = data.pi_connected || false;
            updatePiStatus(piConnected);
            document.getElementById('wordCount').textContent = `Words: ${data.words_count}`;
            if (data.last_word) {
                document.getElementById('lastDetection').textContent = 
                    `Last: ${data.last_word} (${(data.last_confidence * 100).toFixed(0)}%)`;
            }
            break;
        
        case 'ocr_result':
            // OCR completed
            pendingOCRText = data.text;
            showOCRModal(data.text);
            break;
        
        case 'audio_result':
            // Audio transcription completed
            pendingAudioText = data.text;
            showAudioModal(data.text);
            break;
        
        case 'bluetooth_devices':
            // Bluetooth scan results
            showBluetoothDevices(data.devices);
            break;
        
        case 'bluetooth_status':
            // Bluetooth connection status
            alert(data.message);
            break;
        
        case 'ai_response':
            // AI response
            showAIResponse(data.response, data.tokens_used);
            break;
    }
}

// ============================================================
// UI UPDATES
// ============================================================

function updateStatus(message, connected) {
    // Update connection status if needed
}

function updateDetection(word, confidence) {
    document.getElementById('detectedWord').textContent = word.toUpperCase();
    document.getElementById('confidence').textContent = 
        `Confidence: ${(confidence * 100).toFixed(1)}%`;
    document.getElementById('confidenceFill').style.width = `${confidence * 100}%`;
    
    // Color code based on confidence
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
        text.textContent = 'Raspberry Pi: Connected';
    } else {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Raspberry Pi: Disconnected';
    }
}

function showAIResponse(response, tokensUsed) {
    const responseDiv = document.getElementById('aiResponse');
    const responseText = document.getElementById('aiResponseText');
    const tokenCount = document.getElementById('tokenCount');
    
    responseText.textContent = response;
    tokenCount.textContent = `Tokens used: ~${tokensUsed}`;
    responseDiv.classList.add('show');
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
    
    // Hide previous response
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
    alert('Capturing frame for OCR... Please wait.');
}

function showOCRModal(text) {
    document.getElementById('ocrText').textContent = text;
    document.getElementById('ocrModal').classList.add('show');
}

function confirmOCR() {
    if (pendingOCRText) {
        addContext(pendingOCRText);
        closeModal('ocrModal');
        pendingOCRText = "";
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
    alert(`Recording for ${seconds} seconds... Please wait.`);
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
        '<div class="loading"></div> Scanning for devices...';
    sendCommand('scan_bluetooth');
}

function showBluetoothDevices(devices) {
    if (devices.length === 0) {
        document.getElementById('bluetoothStatus').textContent = 
            'No devices found. Make sure Bluetooth is enabled.';
        return;
    }
    
    const listHtml = devices.map((device, index) => `
        <div class="bluetooth-device" onclick="selectBluetoothDevice(${index})">
            <strong>${device.name || 'Unknown Device'}</strong><br>
            <small>${device.address}</small>
        </div>
    `).join('');
    
    document.getElementById('bluetoothList').innerHTML = listHtml;
    document.getElementById('bluetoothModal').classList.add('show');
    
    // Store devices for selection
    window.bluetoothDevices = devices;
}

function selectBluetoothDevice(index) {
    // Remove previous selection
    document.querySelectorAll('.bluetooth-device').forEach(el => {
        el.classList.remove('selected');
    });
    
    // Select new device
    document.querySelectorAll('.bluetooth-device')[index].classList.add('selected');
    selectedBluetoothDevice = window.bluetoothDevices[index];
}

function connectBluetooth() {
    if (!selectedBluetoothDevice) {
        alert('Please select a device first!');
        return;
    }
    
    sendCommand('connect_bluetooth', { 
        address: selectedBluetoothDevice.address 
    });
    
    closeModal('bluetoothModal');
}

// ============================================================
// MODAL MANAGEMENT
// ============================================================

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('show');
}

// Close modals when clicking outside
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
});

// ============================================================
// INITIALIZATION
// ============================================================

// Connect when page loads
window.addEventListener('load', () => {
    connectWebSocket();
    console.log('Application initialized');
});

// Prevent accidental page close
window.addEventListener('beforeunload', (e) => {
    if (words.length > 0) {
        e.preventDefault();
        e.returnValue = '';
        return '';
    }
});
