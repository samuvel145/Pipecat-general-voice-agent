"""
pipeline/agent.py
Assembles the full Pipecat voice pipeline:

  LocalAudioTransport (mic)
    → AudioInputLogProcessor
    → VADProcessor (Silero)
    → VADLogProcessor
    → DeepgramSTTService  (nova-2, streaming)
    → STTLogProcessor
    → GroqLLMService      (llama-3.3-70b-versatile)
    → LLMLogProcessor
    → CartesiaTTSService  (sonic-2, streaming)
    → TTSLogProcessor
    → LocalAudioTransport (speaker)

Repository: https://github.com/samuvel145/Pipecat-general-voice-agent
"""

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import StartFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.transcriptions.language import Language

from config import settings
from logger import get_logger, log_pipeline_event
from pipeline.processors import (
    AudioInputLogProcessor,
    LLMLogProcessor,
    STTLogProcessor,
    TTSLogProcessor,
    VADLogProcessor,
)

log = get_logger("agent")


SYSTEM_PROMPT = """You are a helpful, concise AI voice assistant.
You are talking to a user through a microphone. Keep responses short and natural —
no more than 2-3 sentences unless the user explicitly asks for more detail.
Do not use markdown, bullet points, or special formatting in your responses.
The user has already heard a short spoken greeting when the session started; do not repeat that same opening unless they greet you first."""


async def run_agent() -> None:
    """Build and run the Pipecat voice pipeline in the terminal."""

    log.info("[bold cyan]Pipecat Voice Agent starting…[/bold cyan]")
    log_pipeline_event("INIT", "Building pipeline components")

    # ── Transport (microphone → speaker) ─────────────────────────────────────
    log_pipeline_event("TRANSPORT", "Initialising LocalAudioTransport")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.SAMPLE_RATE,
            audio_out_sample_rate=settings.SAMPLE_RATE,
            audio_in_channels=settings.CHANNELS,
            audio_out_channels=settings.CHANNELS,
            audio_in_passthrough=True,
        )
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    # ── STT — Deepgram nova-2 ─────────────────────────────────────────────────
    log_pipeline_event("STT", "Initialising Deepgram nova-2")
    stt = DeepgramSTTService(
        api_key=settings.DEEPGRAM_API_KEY,
        sample_rate=settings.SAMPLE_RATE,
        channels=settings.CHANNELS,
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            language=Language.EN_US,
        ),
    )

    # ── LLM — Groq ────────────────────────────────────────────────────────────
    log_pipeline_event("LLM", f"Initialising Groq model={settings.LLM_MODEL}")
    llm = GroqLLMService(
        api_key=settings.GROQ_API_KEY,
        settings=GroqLLMService.Settings(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
        ),
    )

    # Conversation context (system prompt + rolling history)
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    # ── TTS — Cartesia sonic-2 ────────────────────────────────────────────────
    log_pipeline_event("TTS", "Initialising Cartesia sonic-2")
    tts = CartesiaTTSService(
        api_key=settings.CARTESIA_API_KEY,
        sample_rate=settings.SAMPLE_RATE,
        settings=CartesiaTTSService.Settings(
            model="sonic-2",
            voice=settings.CARTESIA_VOICE_ID,
        ),
    )

    # ── Logging processors ────────────────────────────────────────────────────
    audio_log_proc = AudioInputLogProcessor()
    vad_log_proc = VADLogProcessor()
    stt_log_proc = STTLogProcessor()
    llm_log_proc = LLMLogProcessor()
    tts_log_proc = TTSLogProcessor()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    log_pipeline_event("PIPELINE", "Assembling pipeline stages")
    pipeline = Pipeline(
        [
            transport.input(),           # 1. Raw mic frames
            audio_log_proc,              # LOG: audio-in
            vad,                         # 2. Silero VAD → VAD* frames for STT
            vad_log_proc,                # LOG: VAD events
            stt,                         # 3. Deepgram STT
            stt_log_proc,                # LOG: transcript
            context_aggregator.user(),   # 4. Accumulate user turn
            llm,                         # 5. Groq LLM
            llm_log_proc,                # LOG: LLM tokens
            tts,                         # 6. Cartesia TTS
            tts_log_proc,                # LOG: TTS audio
            transport.output(),          # 7. Speaker output
            context_aggregator.assistant(),  # 8. Store assistant turn
        ]
    )

    # ── Task & Runner ─────────────────────────────────────────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    greeting = (settings.STARTUP_GREETING or "").strip()
    if greeting:

        @task.event_handler("on_pipeline_started")
        async def _speak_startup_greeting(t: PipelineTask, _frame: StartFrame) -> None:
            log_pipeline_event("GREET", "Playing startup greeting via TTS")
            await t.queue_frame(
                TTSSpeakFrame(text=greeting, append_to_context=True),
            )

    log_pipeline_event("READY", "Pipeline assembled — starting runner")
    log.info(
        "[bold green]✅ Agent ready.[/bold green] "
        "You will hear a short greeting first, then speak. Press [bold]Ctrl+C[/bold] to exit."
    )

    runner = PipelineRunner()

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        log_pipeline_event("SHUTDOWN", "KeyboardInterrupt received")
        log.info("[yellow]Shutting down…[/yellow]")
    finally:
        log_pipeline_event("CLEANUP", "Pipeline task cancelled")
        await task.cancel()
        log.info("Agent stopped cleanly.")
