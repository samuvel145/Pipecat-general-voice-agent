"""
config.py
Centralised settings for the Pipecat Voice Agent.
Repository: https://github.com/samuvel145/Pipecat-general-voice-agent
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Azure STT (Speech-to-Text) ────────────────────────────
    AZURE_STT_KEY: str = ""
    AZURE_SPEECH_REGION: str = "centralindia"

    # ── Azure OpenAI LLM ──────────────────────────────────────
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o-mini"

    # ── Previous STT/LLM (kept for fallback) ──────────────────
    DEEPGRAM_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── TTS ────────────────────────────────────────────────────
    CARTESIA_API_KEY: str = ""
    CARTESIA_VOICE_ID: str = "default"
    SARVAM_API_KEY: str = ""
    SARVAM_VOICE_ID: str = "priya"

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
    # How long (ms) of silence before VAD considers speech ended
    SILENCE_THRESHOLD_MS: int = 600

    # ── LLM ───────────────────────────────────────────────────
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 250
    LLM_TEMPERATURE: float = 0.4
    MAX_HISTORY_TURNS: int = 10

    # ── JLL Integration ───────────────────────────────────────
    JLL_PROXY_URL: str = "http://localhost:3000/api/integration"
    JLL_ASSISTANT_NAME: str = "Riya"

    # ── Startup ───────────────────────────────────────────────
    STARTUP_GREETING: str = ""

    # ── Logging ────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "logs/agent.log"


    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
