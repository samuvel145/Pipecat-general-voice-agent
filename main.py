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

    try:
        asyncio.run(run_agent())
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)
