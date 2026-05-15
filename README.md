# JLL Voice Sales Agent

Local voice pipeline: Microphone → Azure Speech STT → Azure OpenAI LLM → Cartesia TTS → Speaker.

**Repository:** [github.com/samuvel145/cubikey-jll](https://github.com/samuvel145/cubikey-jll)

---

## How it works

Two processes must run together:

| Process | File | What it does |
|---------|------|-------------|
| **Node proxy** | `proxy-server.js` | Express server on port 3000. Receives tool calls from the Python agent, forwards them to the JLL backend API. Loads `integration (1).js` automatically — **do not run that file directly**. |
| **Python agent** | `main.py` | Pipecat voice pipeline. Mic → Azure STT → Azure OpenAI → Cartesia TTS → Speaker. |

---

## Prerequisites

- **Node.js** v18+ — for the proxy server
- **Python 3.11 or 3.12** — PyAudio has prebuilt wheels for these versions. Avoid Python 3.14 (no prebuilt PyAudio wheel, requires manual build with vcpkg).
- API keys: Azure Speech, Azure OpenAI, Cartesia, JLL backend

---

## Setup

### 1. Node.js dependencies

```powershell
npm install
```

### 2. Python virtual environment

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Purpose |
|----------|---------|
| `AZURE_STT_KEY` | Azure Speech Services API key |
| `AZURE_SPEECH_REGION` | Azure region, e.g. `centralindia` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name, e.g. `gpt-4o-mini` |
| `CARTESIA_API_KEY` | Cartesia TTS API key |
| `CARTESIA_VOICE_ID` | Cartesia voice ID |
| `JLL_PROXY_URL` | `http://localhost:3000/api/integration` (default) |
| `JLL_BASE_URL` | JLL backend base URL |

---

## Running the agent

Open **two terminals** and run these in order:

### Terminal 1 — Start the Node proxy first

```powershell
node proxy-server.js
```

You should see:
```
✅ JLL Integration Proxy listening on http://localhost:3000
   Routes mounted at: /api/integration/proxy/*
```

> **Note:** `integration (1).js` is the router module — it is loaded automatically by `proxy-server.js`.
> You never run it directly. Only `node proxy-server.js` is needed.

### Terminal 2 — Start the Python agent

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

Wait for the greeting, then speak. Press **Ctrl+C** to stop.

---

## File structure

```
proxy-server.js        ← Node entry point (run this)
integration (1).js     ← JLL API router (loaded by proxy-server.js, do NOT run directly)
pipeline/
  agent.py             ← Pipecat pipeline wiring
  tools.py             ← LLM tool schemas and handlers
  jll_client.py        ← HTTP client calling the Node proxy
  processors.py        ← Frame processors (STT log, TTS normalizer, function filter)
  prompts.py           ← System prompt builder
main.py                ← Entry point for the Python agent
config.py              ← Settings from .env
logger.py              ← Structured logging
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `SyntaxError: Invalid or unexpected token` in integration (1).js | Smart quotes crept in during editing. Run: `node -e "const fs=require('fs');let c=fs.readFileSync('integration (1).js','utf8');c=c.replace(/‘/g,\"'\").replace(/’/g,\"'\");fs.writeFileSync('integration (1).js',c,'utf8');console.log('fixed')"` |
| `No module named 'pyaudio'` | Use Python 3.11/3.12. Or build from source with vcpkg (see below). |
| `DLL load failed while importing _portaudio` | Copy `portaudio.dll` from vcpkg into `venv\Lib\site-packages\pyaudio\`. |
| Agent returns "No properties found" | Ensure the Node proxy is running and was restarted after any edits to `integration (1).js`. |
| `UnicodeEncodeError` on startup | Windows terminal encoding issue — already fixed in `main.py`. |
