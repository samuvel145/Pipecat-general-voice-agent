"""
logger.py
Comprehensive structured logging for the Pipecat Voice Agent.

Every pipeline stage emits named log records so you can trace the full
lifecycle of each utterance:

  [AUDIO-IN]   Raw microphone frames received
  [VAD]        Speech / silence detection events
  [STT]        Transcript received from Deepgram
  [LLM]        Prompt sent / tokens streamed from Groq
  [TTS]        Audio chunks synthesised by Cartesia
  [AUDIO-OUT]  Audio frames delivered to speaker
  [PIPELINE]   Pipecat pipeline lifecycle events
  [AGENT]      High-level agent state changes
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from config import settings

# ── Constants ────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
MAX_LOG_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5

# ── Named child loggers (one per pipeline stage) ──────────────────────────────
LOGGERS = {
    "audio_in":  "AUDIO-IN",
    "vad":       "VAD",
    "stt":       "STT",
    "llm":       "LLM",
    "tts":       "TTS",
    "audio_out": "AUDIO-OUT",
    "pipeline":  "PIPELINE",
    "agent":     "AGENT",
}


def _make_rich_handler() -> RichHandler:
    """Returns a colourful terminal handler powered by rich."""
    console = Console(stderr=True)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt=DATE_FORMAT))
    return handler


def _make_file_handler(log_file: str) -> RotatingFileHandler:
    """Returns a rotating file handler."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def setup_logging() -> None:
    """
    Configure the root logger and all stage-specific child loggers.
    Call this once at application startup.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # Root captures everything; handlers filter
    root.handlers.clear()

    # Terminal — INFO+ only (our own clean events)
    rich_handler = _make_rich_handler()
    rich_handler.setLevel(logging.INFO)
    root.addHandler(rich_handler)

    # File — keep DEBUG for post-mortem analysis
    file_handler = _make_file_handler(settings.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # ── Silence standard-library third-party loggers ─────────────────────
    _SILENT = [
        "httpx", "httpcore", "urllib3",
        "openai", "openai._base_client", "openai.resources",
        "websockets", "websockets.client", "websockets.server",
        "pipecat",
        "pipecat.pipeline", "pipecat.pipeline.task", "pipecat.pipeline.runner",
        "pipecat.processors", "pipecat.services", "pipecat.transports",
        "pipecat.audio", "pipecat.utils",
        "asyncio", "loguru",
    ]
    for name in _SILENT:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── Silence loguru (Pipecat's internal logger) ───────────────────────
    # Pipecat uses loguru (not logging), so setLevel above has no effect.
    # We remove all loguru sinks and add one WARNING-only sink.
    try:
        from loguru import logger as _loguru
        _loguru.remove()   # remove the default stderr sink (which dumps everything)
        _loguru.add(
            sys.stderr,
            level="WARNING",
            format="<yellow>{level}</>: {message}",
            colorize=True,
        )
    except ImportError:
        pass

    root.info("[bold green]Logging initialised[/bold green] — "
              f"level={settings.LOG_LEVEL}  file={settings.LOG_FILE}")



def get_logger(stage: str) -> logging.Logger:
    """
    Return a named child logger for a pipeline stage.

    Usage:
        log = get_logger("stt")
        log.info("Transcript received: %s", text)
    """
    name = LOGGERS.get(stage, stage.upper())
    return logging.getLogger(name)


# ── Convenience helpers ───────────────────────────────────────────────────────

def log_pipeline_event(event: str, detail: str = "") -> None:
    get_logger("pipeline").info("⚙  %-20s  %s", event, detail)


def log_vad_event(speech_detected: bool) -> None:
    logger = get_logger("vad")
    if speech_detected:
        logger.info("🎤 [bold]Speech START[/bold] detected")
    else:
        logger.info("🔇 Speech END (silence threshold reached)")


def log_stt_result(transcript: str, is_final: bool) -> None:
    tag = "[FINAL]" if is_final else "[partial]"
    get_logger("stt").info("📝 %s  %r", tag, transcript)


def log_llm_prompt(messages: list) -> None:
    get_logger("llm").debug("📨 Sending %d message(s) to Groq", len(messages))


def log_llm_token(token: str) -> None:
    get_logger("llm").debug("🔤 Token: %r", token)


def log_llm_complete(full_response: str) -> None:
    get_logger("llm").info("✅ LLM complete — %d chars", len(full_response))


def log_tts_chunk(chunk_bytes: int) -> None:
    get_logger("tts").debug("🔊 TTS audio chunk: %d bytes", chunk_bytes)


def log_tts_complete() -> None:
    get_logger("tts").info("✅ TTS synthesis complete")
