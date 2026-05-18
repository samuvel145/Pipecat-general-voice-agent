// JLL Voice Agent Frontend
// Simple test interface for WebSocket-based voice agent

class VoiceAgentClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.scriptProcessor = null;
        this.isConnected = false;
        this.isCallActive = false;
        
        this.connectBtn = document.getElementById('connectBtn');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.status = document.getElementById('status');
        this.logs = document.getElementById('logs');
        this.audioIndicator = document.getElementById('audioIndicator');
        this.audioStatus = document.getElementById('audioStatus');
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        this.connectBtn.addEventListener('click', () => this.connect());
        this.startBtn.addEventListener('click', () => this.startCall());
        this.stopBtn.addEventListener('click', () => this.stopCall());
    }
    
    log(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        this.logs.appendChild(entry);
        this.logs.scrollTop = this.logs.scrollHeight;
    }
    
    updateStatus(text, className) {
        this.status.textContent = text;
        this.status.className = `status ${className}`;
    }
    
    async connect() {
        if (this.isConnected) {
            this.disconnect();
            return;
        }
        
        this.updateStatus('Connecting...', 'connecting');
        this.log('Connecting to WebSocket...', 'info');
        
        try {
            // Update this URL to your Render deployment
            const wsUrl = 'ws://localhost:8000/ws'; // Change this for Render
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus('Connected', 'connected');
                this.connectBtn.textContent = 'Disconnect';
                this.startBtn.disabled = false;
                this.log('WebSocket connected', 'success');
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };
            
            this.ws.onerror = (error) => {
                this.log('WebSocket error', 'error');
                console.error('WebSocket error:', error);
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus('Disconnected', 'disconnected');
                this.connectBtn.textContent = 'Connect';
                this.startBtn.disabled = true;
                this.stopBtn.disabled = true;
                this.log('WebSocket disconnected', 'info');
            };
        } catch (error) {
            this.log(`Connection failed: ${error.message}`, 'error');
            this.updateStatus('Connection Failed', 'disconnected');
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
    
    async startCall() {
        if (!this.isConnected) {
            this.log('Not connected', 'error');
            return;
        }
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Get microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            this.isCallActive = true;
            this.updateStatus('Call Active', 'active');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.audioIndicator.classList.add('show', 'recording');
            this.audioStatus.textContent = '🎤 Recording...';
            this.log('Call started', 'success');
            
            // Start sending audio
            this.startAudioCapture();
            
        } catch (error) {
            this.log(`Failed to start call: ${error.message}`, 'error');
        }
    }
    
    stopCall() {
        this.isCallActive = false;
        
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.updateStatus('Connected', 'connected');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.audioIndicator.classList.remove('show', 'recording', 'playing');
        this.log('Call stopped', 'info');
    }
    
    startAudioCapture() {
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (e) => {
            if (!this.isCallActive || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                return;
            }
            
            const inputData = e.inputBuffer.getChannelData(0);
            const pcmData = this.floatTo16BitPCM(inputData);
            
            // Send as base64
            const base64 = this.arrayBufferToBase64(pcmData);
            
            // Format according to Exotel/Vodafone protocol
            const message = JSON.stringify({
                event: 'media',
                media: {
                    payload: base64
                }
            });
            
            this.ws.send(message);
        };
        
        source.connect(processor);
        processor.connect(this.audioContext.destination);
        this.scriptProcessor = processor;
    }
    
    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            
            if (message.event === 'media' && message.media && message.media.payload) {
                // Play audio
                this.playAudio(message.media.payload);
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    }
    
    async playAudio(base64Audio) {
        try {
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);
            
            this.audioIndicator.classList.add('playing');
            this.audioStatus.textContent = '🔊 Playing...';
            
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.onended = () => {
                this.audioIndicator.classList.remove('playing');
                this.audioStatus.textContent = '🎤 Recording...';
            };
            source.start();
            
        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }
    
    floatTo16BitPCM(float32Array) {
        const l = float32Array.length;
        const buffer = new Int16Array(l);
        for (let i = 0; i < l; i++) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return buffer.buffer;
    }
    
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new VoiceAgentClient();
});
