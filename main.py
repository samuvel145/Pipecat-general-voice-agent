"""
main.py
Entry point for the Pipecat Voice Agent.
Run with:  python main.py

Automatically starts the Node.js proxy server (proxy-server.js) as a
background subprocess, waits for it to be ready on port 3000, then launches
the voice pipeline.  Both processes shut down together on Ctrl+C.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time

from logger import get_logger, setup_logging
from pipeline.agent import run_agent

log = get_logger("agent")

_PROXY_PORT = 3000
_PROXY_READY_TIMEOUT = 15   # seconds to wait for Node to bind port 3000


def _wait_for_port(port: int, timeout: float) -> bool:
    """Return True once something is listening on localhost:port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def start_proxy() -> subprocess.Popen:
    """
    Launch `node proxy-server.js` from the directory containing main.py.
    Returns the Popen handle so the caller can terminate it on exit.
    """
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


def banner() -> None:
    # Force UTF-8 on Windows to avoid cp1252 UnicodeEncodeError
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        "\n"
        "+----------------------------------------------+\n"
        "|   [MIC]  Pipecat JLL Voice Agent  [MIC]      |\n"
        "|   STT : Azure Speech (en-IN)                 |\n"
        "|   LLM : Azure OpenAI gpt-4o-mini             |\n"
        "|   TTS : Cartesia sonic-2                     |\n"
        "+----------------------------------------------+\n"
    )


if __name__ == "__main__":
    setup_logging()
    banner()

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
