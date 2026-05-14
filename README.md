# Pipecat general voice agent

Local voice pipeline: microphone → Deepgram STT → Groq LLM → Cartesia TTS → speaker.

**Repository:** [github.com/samuvel145/Pipecat-general-voice-agent](https://github.com/samuvel145/Pipecat-general-voice-agent)

```powershell
git clone https://github.com/samuvel145/Pipecat-general-voice-agent.git
cd Pipecat-general-voice-agent
copy .env.example .env
# Edit .env with your API keys, then follow Setup below.
```

## Prerequisites

- **Python 3.11 or 3.12** on Windows (PyAudio has prebuilt wheels; **avoid 3.14** for this project—slow or broken installs and a mismatched default `python` if your shell still points at 3.14).
- API keys: [Deepgram](https://deepgram.com/), [Groq](https://console.groq.com/), [Cartesia](https://cartesia.ai/).

## Setup (virtual environment)

From the project root, create the venv with **3.11** if the `py` launcher has it:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If you only have one Python installed:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If you prefer not to activate the venv, use the venv’s Python explicitly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and add your keys. Required variables (see `config.py` for defaults and optional fields):

| Variable | Purpose |
|----------|---------|
| `DEEPGRAM_API_KEY` | Speech-to-text |
| `GROQ_API_KEY` | LLM |
| `CARTESIA_API_KEY` | Text-to-speech |
| `CARTESIA_VOICE_ID` | Cartesia voice id (not `default` unless that is your real id) |
| `STARTUP_GREETING` | Optional. Spoken once at startup (TTS). Set to empty to disable. |

## Run the agent

After `pip install` finishes successfully:

```powershell
.\venv\Scripts\python.exe main.py
```

With an activated venv:

```powershell
python main.py
```

Speak into the microphone. Stop with **Ctrl+C**. Logs also go to `logs/agent.log` (see `LOG_FILE` in `config.py`).

## Troubleshooting

- **Install hangs or PyAudio fails:** Use Python 3.11/3.12 for a new `venv`, or install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if you must build PyAudio from source.
- **Blank pip output:** Run `python -m pip install -r requirements.txt -v` to see progress.

## New Commands
```powershell
npm install
node proxy-server.js
```
