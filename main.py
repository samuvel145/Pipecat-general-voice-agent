"""
main.py
Entry point for the Pipecat Voice Agent.
Run with:  python main.py

Repository: https://github.com/samuvel145/Pipecat-general-voice-agent
"""

import asyncio
import sys

from logger import get_logger, setup_logging
from pipeline.agent import run_agent

log = get_logger("agent")


def banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║   🎙  Pipecat AI Voice Agent  🎙             ║\n"
        "║   STT : Deepgram nova-2                      ║\n"
        "║   LLM : Groq llama-3.3-70b-versatile         ║\n"
        "║   TTS : Cartesia sonic-2                     ║\n"
        "╚══════════════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    setup_logging()
    banner()

    try:
        asyncio.run(run_agent())
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)
