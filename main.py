"""
main.py
Entry point for the Pipecat JLL Voice Agent.

  python main.py         — FastAPI WebSocket server on port 8000
                           Phone / remote clients connect via ws://host:8000/ws
                           using the Exotel / Vodafone media-streams JSON protocol.

  python main.py --local — LocalAudioTransport (mic + speaker on this machine).
                           Use this for local development / testing.

Both modes auto-start the Node.js JLL proxy (proxy-server.js) on port 3000.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket

from logger import get_logger, setup_logging
from config import settings
from pipeline import jll_client
from pipeline.agent import run_agent, run_agent_ws, _prefetch_ack_phrases, _gen_keyboard_pcm

log = get_logger("agent")

_PROXY_PORT = 3000
_PROXY_READY_TIMEOUT = 15
_WS_PORT = 8000

# Holds the Node proxy process so the FastAPI lifespan can clean it up.
_proxy_proc: subprocess.Popen | None = None


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ── Startup: pre-warm expensive per-call resources once ───────────────────
    # Pre-warm Silero VAD — loads PyTorch model into process memory so the first
    # call does not pay the ~300 ms cold-start penalty.
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        log.info("[Startup] Pre-warming Silero VAD model…")
        _warmup_vad = await asyncio.to_thread(SileroVADAnalyzer)
        del _warmup_vad   # model is now in OS page cache; each call creates its own instance
        log.info("[Startup] Silero VAD model warm.")
    except Exception as exc:
        log.warning("[Startup] Silero warm-up skipped: %s", exc)

    # Pre-generate keyboard typing PCM (deterministic, ~4 s, CPU-only).
    log.info("[Startup] Pre-generating keyboard PCM…")
    _app.state.keyboard_pcm = _gen_keyboard_pcm(settings.SAMPLE_RATE)
    log.info("[Startup] Keyboard PCM ready  %d bytes", len(_app.state.keyboard_pcm))

    # Pre-synthesize ack phrases via Cartesia REST once at startup so every
    # WebSocket call can start the typing sound immediately without waiting for
    # 10 HTTP round-trips.
    log.info("[Startup] Pre-fetching acknowledgment phrases…")
    try:
        _app.state.ack_pcms = await _prefetch_ack_phrases(
            settings.CARTESIA_API_KEY,
            settings.CARTESIA_VOICE_ID,
            settings.SAMPLE_RATE,
        )
        log.info("[Startup] %d ack phrases ready.", len(_app.state.ack_pcms))
    except Exception as exc:
        log.warning("[Startup] Ack phrase prefetch failed (non-fatal): %s", exc)
        _app.state.ack_pcms = []

    yield

    # ── Shutdown: close shared HTTP client and stop Node proxy ────────────────
    await jll_client.close_client()
    if _proxy_proc and _proxy_proc.poll() is None:
        print("[Proxy] Stopping Node proxy …", flush=True)
        _proxy_proc.terminate()
        try:
            _proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proxy_proc.kill()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Exotel / Vodafone media-streams WebSocket endpoint.

    The carrier sends a JSON "start" event first:
      {"event": "start", "streamSid": "...", "callSid": "...", ...}
    followed by "media" events carrying base64-encoded PCM audio.
    We extract streamSid and pass it to the serializer so outbound
    audio is tagged correctly.
    """
    await websocket.accept()
    stream_sid = ""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        msg = json.loads(raw)
        if msg.get("event") == "start":
            stream_sid = msg.get("streamSid", "")
            log.info("[WS] Incoming call  stream_sid=%s", stream_sid)
    except Exception:
        pass

    ack_pcms     = getattr(app.state, "ack_pcms", None)
    keyboard_pcm = getattr(app.state, "keyboard_pcm", None)
    await run_agent_ws(websocket, stream_sid, ack_pcms=ack_pcms, keyboard_pcm=keyboard_pcm)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "websocket"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def start_proxy() -> subprocess.Popen:
    here = os.path.dirname(os.path.abspath(__file__))
    proxy_script = os.path.join(here, "proxy-server.js")
    proc = subprocess.Popen(
        ["node", proxy_script],
        cwd=here,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(f"[Proxy] Starting Node proxy (pid={proc.pid}) …", flush=True)
    if _wait_for_port(_PROXY_PORT, _PROXY_READY_TIMEOUT):
        print(f"[Proxy] Ready on http://localhost:{_PROXY_PORT}", flush=True)
    else:
        proc.terminate()
        raise RuntimeError(
            f"Node proxy did not bind port {_PROXY_PORT} within {_PROXY_READY_TIMEOUT}s. "
            "Make sure Node.js is installed and 'npm install' has been run."
        )
    return proc


def _banner_ws() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        "\n"
        "+----------------------------------------------+\n"
        "|   JLL Voice Agent  —  WebSocket Server       |\n"
        "|   STT : Azure Speech (en-IN)                 |\n"
        "|   LLM : Azure OpenAI gpt-4o-mini             |\n"
        "|   TTS : Cartesia sonic-2                     |\n"
        f"|   WS  : ws://0.0.0.0:{_WS_PORT}/ws              |\n"
        "+----------------------------------------------+\n"
    )


def _banner_local() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        "\n"
        "+----------------------------------------------+\n"
        "|   [MIC]  Pipecat JLL Voice Agent  [MIC]      |\n"
        "|   STT : Azure Speech (en-IN)                 |\n"
        "|   LLM : Azure OpenAI gpt-4o-mini             |\n"
        "|   TTS : Cartesia sonic-2                     |\n"
        "|   MODE: LocalAudio (mic + speaker)           |\n"
        "+----------------------------------------------+\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    local_mode = "--local" in sys.argv

    if local_mode:
        # ── Local audio mode (mic + speaker on this machine) ──────────────────
        _banner_local()
        proxy_proc = None
        try:
            proxy_proc = start_proxy()
            asyncio.run(run_agent())
        except Exception as exc:
            log.exception("Fatal error: %s", exc)
            sys.exit(1)
        finally:
            if proxy_proc and proxy_proc.poll() is None:
                print("[Proxy] Stopping Node proxy …", flush=True)
                proxy_proc.terminate()
                try:
                    proxy_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proxy_proc.kill()
    else:
        # ── WebSocket server mode (phone / remote clients) ────────────────────
        _banner_ws()
        try:
            _proxy_proc = start_proxy()
        except RuntimeError as exc:
            log.error(str(exc))
            sys.exit(1)
        # uvicorn takes over the event loop; lifespan() handles cleanup on exit.
        uvicorn.run(app, host="0.0.0.0", port=_WS_PORT)
