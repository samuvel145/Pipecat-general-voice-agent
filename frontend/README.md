# JLL Voice Agent Frontend

Simple test interface for the JLL Voice Agent. Can be deleted anytime - no backend changes required.

## Files

- `index.html` - Main UI with Connect/Start/Stop buttons
- `app.js` - WebSocket client with audio capture/playback

## Usage

### Local Testing

1. Start your backend server
2. Open `index.html` in a browser
3. Update the WebSocket URL in `app.js` line 24 if needed (default: `ws://localhost:8000/ws`)
4. Click "Connect" to connect to WebSocket
5. Click "Start Call" to begin voice interaction
6. Click "Stop Call" to end

### Render Deployment

Option 1: Serve from same Render app
- Add a static file route to your FastAPI app to serve these files
- Update WebSocket URL to use your Render domain

Option 2: Separate static hosting
- Deploy to Netlify/Vercel/GitHub Pages
- Update WebSocket URL in `app.js` to your Render backend URL
- Enable CORS on your backend if needed

## Configuration

Update the WebSocket URL in `app.js` line 24:
```javascript
const wsUrl = 'ws://your-render-app-url.onrender.com/ws';
```

## Notes

- Uses Web Audio API for microphone capture and playback
- Sends audio in Exotel/Vodafone protocol format (base64-encoded PCM)
- Sample rate: 16kHz, mono
- Requires HTTPS for microphone access on production

## Cleanup

To delete this frontend:
```bash
rm -rf frontend/
```

This will not affect your backend in any way.
