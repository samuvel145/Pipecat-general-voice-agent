"""
config.py
Centralised settings for the Pipecat Voice Agent.
Repository: https://github.com/samuvel145/Pipecat-general-voice-agent
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────
    DEEPGRAM_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    CARTESIA_API_KEY: str = ""
    CARTESIA_VOICE_ID: str = "default"

    # ── Audio ─────────────────────────────────────────────────
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1

    @property
    def FRAME_DURATION_MS(self) -> int:
        return 20

    @property
    def FRAME_SIZE(self) -> int:
        return self.SAMPLE_RATE * 2 * self.FRAME_DURATION_MS // 1000

    # ── VAD ───────────────────────────────────────────────────
    VAD_AGGRESSIVENESS: int = 2
    SILENCE_THRESHOLD_MS: int = 800

    # ── LLM ───────────────────────────────────────────────────
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = 300
    MAX_HISTORY_TURNS: int = 5

    # ── Startup ───────────────────────────────────────────────
    # Spoken once when the pipeline is ready (TTS). Set to empty string to disable.
    STARTUP_GREETING: str = (
        "Hi! I'm your voice assistant. Go ahead whenever you're ready — what would you like to talk about?"
    )

    # ── Logging ────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "logs/agent.log"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
