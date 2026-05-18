"""
pipeline/agent.py
JLL Voice Sales Agent — local terminal pipeline.

Replaces the general-purpose agent.py.
Wires: Mic -> Deepgram STT -> Groq LLM (JLL tools) -> Cartesia TTS -> Speaker

Tool call flow (derived from IntegrationToolHandler in bot (1).py):
  LLM emits function_call -> llm.register_function handler -> jll_client HTTP call
  -> result returned to Pipecat -> LLM speaks the result

References:
  - bot (1).py: pipeline assembly, TranscriptionLogger, IntegrationToolHandler
  - integration (1).js: proxy routes, field names
"""

from __future__ import annotations

import json
import logging
import math
import random
import struct

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.processors.audio.vad_processor import VADProcessor

from config import settings
from logger import get_logger, log_pipeline_event
from pipeline import jll_client
from pipeline.prompts import build_gather_hint, build_system_prompt
from pipeline.processors import ConversationLogProcessor, EchoCancelGate, FunctionCallFilter, PhoneticCorrectorProcessor, STTLogProcessor, TextNormalizerProcessor, TTSLogProcessor, TTSSpeakingTracker, TypingSoundGate, TypingSoundProcessor, VADLogProcessor
from pipeline.tools import TOOL_SCHEMAS, JLLToolHandler

log = get_logger("agent")

_ACK_PHRASES = [
    "Okay…", "Sure…", "Got it…", "Yeah…", "Alright…",
    "Moving forward…", "Uh-huh…", "Understood…", "Let me check…", "One moment…",
]


async def _prefetch_ack_phrases(api_key: str, voice_id: str, sample_rate: int) -> list[bytes]:
    """Pre-synthesize acknowledgment phrases via Cartesia /tts/bytes at startup."""
    import httpx
    pcm_list: list[bytes] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for phrase in _ACK_PHRASES:
            try:
                resp = await client.post(
                    "https://api.cartesia.ai/tts/bytes",
                    headers={
                        "X-API-Key": api_key,
                        "Cartesia-Version": "2024-06-10",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model_id": "sonic-2",
                        "transcript": phrase,
                        "voice": {"mode": "id", "id": voice_id},
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": sample_rate,
                        },
                    },
                )
                resp.raise_for_status()
                pcm_list.append(resp.content)
                log.info("[ACK] Pre-synthesized %r — %d bytes", phrase, len(resp.content))
            except Exception as exc:
                log.warning("[ACK] Skipped %r — %s", phrase, exc)
    log.info("[ACK] %d / %d phrases ready", len(pcm_list), len(_ACK_PHRASES))
    return pcm_list


def _gen_keyboard_pcm(sample_rate: int) -> bytes:
    """Synthesise ~4 s of keyboard-typing sound as 16-bit mono PCM (fixed seed)."""
    rng = random.Random(42)
    n   = int(4.0 * sample_rate)
    buf = [0.0] * n
    t   = 0
    while t < n:
        click_len = int(rng.uniform(0.008, 0.014) * sample_rate)
        vol       = rng.uniform(0.28, 0.50)
        for i in range(min(click_len, n - t)):
            noise      = rng.gauss(0, 1.0)
            decay      = math.exp(-i / max(1, click_len * 0.18))
            buf[t + i] = max(-1.0, min(1.0, buf[t + i] + noise * decay * vol))
        t += int(rng.uniform(0.08, 0.20) * sample_rate)
    peak  = max(abs(x) for x in buf) or 1.0
    scale = 0.45 / peak
    return struct.pack(
        f'<{n}h',
        *(max(-32768, min(32767, int(x * scale * 32767))) for x in buf),
    )


async def run_agent() -> None:
    log.info("[bold cyan]JLL Voice Agent starting…[/bold cyan]")
    log_pipeline_event("INIT", "Building pipeline components")

    # ── Transport (mic + speaker) ─────────────────────────────────────────────
    log_pipeline_event("TRANSPORT", "Initialising LocalAudioTransport")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.SAMPLE_RATE,
            audio_out_sample_rate=settings.SAMPLE_RATE,
            audio_in_channels=settings.CHANNELS,
            audio_out_channels=settings.CHANNELS,
            # Keep True so mic stays active. Pipecat 1.1.0 LocalAudioTransport
            # may not properly re-enable mic input when set to False.
            audio_in_passthrough=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    # Lowered from 0.7 → more tolerant of softer speech
                    confidence=0.5,
                    start_secs=0.2,
                    stop_secs=float(settings.SILENCE_THRESHOLD_MS) / 1000,
                    # Lowered to 0.2 → ensures mic is picked up after bot stops
                    min_volume=0.2,
                )
            ),
        )
    )

    # ── STT — Azure Speech (en-IN) ────────────────────────────────────
    log_pipeline_event("STT", f"Initialising Azure Speech region={settings.AZURE_SPEECH_REGION}")
    from pipecat.transcriptions.language import Language
    stt = AzureSTTService(
        api_key=settings.AZURE_STT_KEY,
        region=settings.AZURE_SPEECH_REGION,
        sample_rate=settings.SAMPLE_RATE,
        settings=AzureSTTService.Settings(language=Language.EN_IN),
    )

    # ── LLM — Azure OpenAI (gpt-4o-mini) ────────────────────────────────
    log_pipeline_event("LLM", f"Initialising Azure OpenAI deployment={settings.AZURE_OPENAI_DEPLOYMENT}")
    llm = AzureLLMService(
        api_key=settings.AZURE_OPENAI_API_KEY,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        settings=AzureLLMService.Settings(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        ),
    )

    # ── TTS — Cartesia sonic-2 ────────────────────────────────────────
    log_pipeline_event("TTS", f"Initialising Cartesia voice_id={settings.CARTESIA_VOICE_ID[:8]}...")
    from pipecat.services.cartesia.tts import CartesiaTTSSettings
    tts = CartesiaTTSService(
        api_key=settings.CARTESIA_API_KEY,
        sample_rate=settings.SAMPLE_RATE,
        settings=CartesiaTTSSettings(
            voice=settings.CARTESIA_VOICE_ID,
            model="sonic-2",
        ),
    )

    # ── LLM Context (system prompt + tool schemas) ────────────────────────────
    system_prompt = build_system_prompt(settings.JLL_ASSISTANT_NAME)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=ToolsSchema(
            standard_tools=[],
            custom_tools={AdapterType.OPENAI: TOOL_SCHEMAS},
        ),
    )
    context_aggregator = LLMContextAggregatorPair(context=context)

    # ── Tool handler ──────────────────────────────────────────────────────────
    tool_handler = JLLToolHandler()

    # ── Pipeline assembly ─────────────────────────────────────────────────────
    log_pipeline_event("PIPELINE", "Assembling pipeline stages")
    func_filter        = FunctionCallFilter()
    text_normalizer    = TextNormalizerProcessor()
    stt_log            = STTLogProcessor()          # logs user speech + stt latency
    conv_log           = ConversationLogProcessor() # logs LLM response + TTS label
    tts_log            = TTSLogProcessor()          # logs TTS first chunk latency
    vad_log            = VADLogProcessor()          # resets latency clock on speech start
    echo_gate          = EchoCancelGate()           # mutes mic while bot audio is playing
    # TTSSpeakingTracker: closes gate on BotStartedSpeakingFrame, opens after
    # BotStoppedSpeakingFrame (200 ms delay). No longer calls stop_kb() —
    # TypingSoundGate handles keyboard stop on first real TTS audio.
    tts_tracker        = TTSSpeakingTracker(gate=echo_gate)
    phonetic_corrector = PhoneticCorrectorProcessor(context=context)  # Soundex+Metaphone name/location fix

    # ── Acknowledgment phrases (pre-synthesized via Cartesia at startup) ─────
    log_pipeline_event("ACK", "Pre-synthesizing acknowledgment phrases via Cartesia")
    ack_pcms = await _prefetch_ack_phrases(
        settings.CARTESIA_API_KEY,
        settings.CARTESIA_VOICE_ID,
        settings.SAMPLE_RATE,
    )

    # ── Typing sound (pre-generate PCM once at startup) ───────────────────────
    _keyboard_pcm = _gen_keyboard_pcm(settings.SAMPLE_RATE)
    log.info("[SOUND] Keyboard PCM ready  %.1fs  %d bytes", 4.0, len(_keyboard_pcm))
    typing_sound_gate = TypingSoundGate(
        keyboard_pcm=_keyboard_pcm,
        sample_rate=settings.SAMPLE_RATE,
        ack_pcms=ack_pcms,
    )
    # TypingSoundProcessor pre-closes echo_gate before any audio plays, then
    # signals typing_sound_gate to start ack phrase + keyboard loop.
    typing_sound = TypingSoundProcessor(
        typing_sound_gate=typing_sound_gate,
        echo_gate=echo_gate,
    )

    pipeline = Pipeline(
        [
            transport.input(),               # 1.  Raw mic
            echo_gate,                       # 2.  Drop mic frames while bot speaks
            vad_log,                         # 3.  Reset latency clock on VAD speech start
            stt,                             # 4.  Azure STT → TranscriptionFrame
            stt_log,                         # 5.  STT log + stt_latency stamp
            typing_sound,                    # 6.  Signals typing_sound_gate on TranscriptionFrame
            phonetic_corrector,              # 7.  Phonetic correction for names + locations
            context_aggregator.user(),       # 8.  Accumulate user turn
            llm,                             # 9.  Azure OpenAI LLM
            func_filter,                     # 10. Drop function-call markup
            conv_log,                        # 11. LLM log + llm_first_token / llm_done stamps
            text_normalizer,                 # 12. Number normalisation
            tts,                             # 13. Cartesia TTS
            tts_log,                         # 14. TTS first chunk stamp
            typing_sound_gate,               # 15. Keyboard PCM loop + seamless handoff to real TTS
            transport.output(),              # 16. Speaker (fires BotStarted/StoppedSpeakingFrame)
            tts_tracker,                     # 17. Echo gate control + latency report
            context_aggregator.assistant(),  # 18. Store assistant turn
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ── Tool call handlers ────────────────────────────────────────────────────
    # The LLM is instructed (system prompt SEARCH ANNOUNCEMENT) to speak a brief
    # status phrase before every tool call — no separate TTSSpeakFrame filler needed.
    def _make_tool_handler(tool_name: str):
        async def _handler(params) -> None:
            args = params.arguments
            _update_gather_hint(context, tool_handler)
            t0 = __import__("time").monotonic()
            result_text = await tool_handler.handle(tool_name, args)
            elapsed = __import__("time").monotonic() - t0
            log.info(
                "[TOOL] %-22s | %s | %.2fs",
                tool_name,
                json.dumps({k: v for k, v in args.items() if k in ("city", "location", "property_type", "min_price", "max_price")}, ensure_ascii=False),
                elapsed,
            )
            log.info("[TOOL-RESULT] %s", result_text[:120])
            await params.result_callback(result_text)
        return _handler

    for schema in TOOL_SCHEMAS:
        func_name: str = schema["function"]["name"]
        llm.register_function(func_name, _make_tool_handler(func_name))

    # ── Startup: let LLM speak the opening from system_prompt.txt ────────────
    @task.event_handler("on_pipeline_started")
    async def _trigger_opening(t: PipelineTask, _frame: StartFrame) -> None:
        log_pipeline_event("GREET", "Triggering opening from system_prompt.txt")
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await t.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "[BEGIN]"}],
                run_llm=True,
            )
        )

    # ── Runner ────────────────────────────────────────────────────────────────
    log_pipeline_event("READY", "Pipeline assembled — starting runner")
    log.info(
        "[bold green]✅ JLL Agent ready.[/bold green] "
        "Hear the greeting, then speak. Press [bold]Ctrl+C[/bold] to exit."
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
        await jll_client.close_client()
        log.info("Agent stopped cleanly.")


async def run_agent_ws(
    websocket,
    stream_sid: str = "",
    ack_pcms: list[bytes] | None = None,
    keyboard_pcm: bytes | None = None,
) -> None:
    """WebSocket pipeline for phone/remote clients (Exotel/Vodafone protocol).

    Audio flows over WebSocket as base64-encoded PCM JSON frames:
      {"event": "media", "media": {"payload": "<base64_pcm>"}}
    This is the same Exotel Media Streams format used by Vodafone India.
    ExotelFrameSerializer handles resampling between pipeline rate and 8 kHz.
    """
    log.info("[bold cyan]JLL Voice Agent (WebSocket) starting…[/bold cyan]")
    log_pipeline_event("INIT", "Building WebSocket pipeline components")

    # ── Transport — FastAPI WebSocket + Exotel serializer ────────────────────
    serializer = ExotelFrameSerializer(stream_sid=stream_sid)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.SAMPLE_RATE,
            audio_out_sample_rate=settings.SAMPLE_RATE,
            audio_in_channels=settings.CHANNELS,
            audio_out_channels=settings.CHANNELS,
            audio_in_passthrough=True,  # pass frames downstream for VADProcessor
            serializer=serializer,
            # Split large ack-phrase / TTS frames into 20 ms chunks so the
            # carrier (Exotel/Vodafone) receives properly sized payloads and
            # the _write_audio_sleep clock stays accurate.
            fixed_audio_packet_size=640,  # 20 ms × 16 kHz × 2 bytes = 640 bytes
        ),
    )

    # ── VAD — explicit processor (WebSocket transport has no built-in VAD) ───
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.5,
                start_secs=0.2,
                stop_secs=float(settings.SILENCE_THRESHOLD_MS) / 1000,
                min_volume=0.2,
            )
        )
    )

    # ── STT ───────────────────────────────────────────────────────────────────
    from pipecat.transcriptions.language import Language
    stt = AzureSTTService(
        api_key=settings.AZURE_STT_KEY,
        region=settings.AZURE_SPEECH_REGION,
        sample_rate=settings.SAMPLE_RATE,
        settings=AzureSTTService.Settings(language=Language.EN_IN),
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = AzureLLMService(
        api_key=settings.AZURE_OPENAI_API_KEY,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        settings=AzureLLMService.Settings(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        ),
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    from pipecat.services.cartesia.tts import CartesiaTTSSettings
    tts = CartesiaTTSService(
        api_key=settings.CARTESIA_API_KEY,
        sample_rate=settings.SAMPLE_RATE,
        settings=CartesiaTTSSettings(
            voice=settings.CARTESIA_VOICE_ID,
            model="sonic-2",
        ),
    )

    # ── Context ───────────────────────────────────────────────────────────────
    system_prompt = build_system_prompt(settings.JLL_ASSISTANT_NAME)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=ToolsSchema(
            standard_tools=[],
            custom_tools={AdapterType.OPENAI: TOOL_SCHEMAS},
        ),
    )
    context_aggregator = LLMContextAggregatorPair(context=context)
    tool_handler = JLLToolHandler()

    # ── Processors ────────────────────────────────────────────────────────────
    func_filter        = FunctionCallFilter()
    text_normalizer    = TextNormalizerProcessor()
    stt_log            = STTLogProcessor()
    conv_log           = ConversationLogProcessor()
    tts_log            = TTSLogProcessor()
    vad_log            = VADLogProcessor()
    echo_gate          = EchoCancelGate()
    tts_tracker        = TTSSpeakingTracker(gate=echo_gate)
    phonetic_corrector = PhoneticCorrectorProcessor(context=context)

    # ── Ack phrases (use pre-warmed pool from lifespan; fall back per-call) ──────
    if ack_pcms is None:
        log_pipeline_event("ACK", "Pre-synthesizing acknowledgment phrases (per-call fallback)")
        ack_pcms = await _prefetch_ack_phrases(
            settings.CARTESIA_API_KEY,
            settings.CARTESIA_VOICE_ID,
            settings.SAMPLE_RATE,
        )
    else:
        log.info("[ACK] Using %d pre-warmed phrases from startup", len(ack_pcms))

    # ── Typing sound (use pre-warmed PCM from lifespan; fall back per-call) ──────
    if keyboard_pcm is None:
        keyboard_pcm = _gen_keyboard_pcm(settings.SAMPLE_RATE)
        log.info("[SOUND] Keyboard PCM generated on-demand  %d bytes", len(keyboard_pcm))
    else:
        log.info("[SOUND] Using pre-warmed keyboard PCM  %d bytes", len(keyboard_pcm))

    typing_sound_gate = TypingSoundGate(
        keyboard_pcm=keyboard_pcm,
        sample_rate=settings.SAMPLE_RATE,
        ack_pcms=ack_pcms,
    )
    typing_sound = TypingSoundProcessor(
        typing_sound_gate=typing_sound_gate,
        echo_gate=echo_gate,
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # VAD is placed before echo_gate so interruption detection works even
    # while bot is speaking. Echo_gate drops audio frames (not VAD events),
    # so STT won't transcribe echo but the pipeline still sees speech start.
    pipeline = Pipeline(
        [
            transport.input(),               # 1.  WebSocket audio in
            vad,                             # 2.  Silero VAD (explicit — no built-in for WS)
            echo_gate,                       # 3.  Drop mic frames while bot speaks
            vad_log,                         # 4.  Reset latency clock on VAD speech start
            stt,                             # 5.  Azure STT → TranscriptionFrame
            stt_log,                         # 6.  STT log + stt_latency stamp
            typing_sound,                    # 7.  Typing sound trigger on VAD stop
            phonetic_corrector,              # 8.  Phonetic correction for names + locations
            context_aggregator.user(),       # 9.  Accumulate user turn
            llm,                             # 10. Azure OpenAI LLM
            func_filter,                     # 11. Drop function-call markup
            conv_log,                        # 12. LLM log
            text_normalizer,                 # 13. Number normalisation
            tts,                             # 14. Cartesia TTS
            tts_log,                         # 15. TTS first chunk stamp
            typing_sound_gate,               # 16. Keyboard PCM → seamless handoff to real TTS
            transport.output(),              # 17. WebSocket audio out (ExotelSerializer paces chunks)
            tts_tracker,                     # 18. Echo gate control + latency report
            context_aggregator.assistant(),  # 19. Store assistant turn
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ── Tool handlers ─────────────────────────────────────────────────────────
    def _make_tool_handler(tool_name: str):
        async def _handler(params) -> None:
            args = params.arguments
            _update_gather_hint(context, tool_handler)
            t0 = __import__("time").monotonic()
            result_text = await tool_handler.handle(tool_name, args)
            elapsed = __import__("time").monotonic() - t0
            log.info(
                "[TOOL] %-22s | %s | %.2fs",
                tool_name,
                json.dumps({k: v for k, v in args.items() if k in ("city", "location", "property_type", "min_price", "max_price")}, ensure_ascii=False),
                elapsed,
            )
            log.info("[TOOL-RESULT] %s", result_text[:120])
            await params.result_callback(result_text)
        return _handler

    for schema in TOOL_SCHEMAS:
        func_name: str = schema["function"]["name"]
        llm.register_function(func_name, _make_tool_handler(func_name))

    # ── Opening greeting ──────────────────────────────────────────────────────
    @task.event_handler("on_pipeline_started")
    async def _trigger_opening(t: PipelineTask, _frame: StartFrame) -> None:
        log_pipeline_event("GREET", "Triggering opening from system_prompt.txt")
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await t.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "[BEGIN]"}],
                run_llm=True,
            )
        )

    # ── Runner ────────────────────────────────────────────────────────────────
    log_pipeline_event("READY", "WebSocket pipeline assembled — waiting for audio")
    log.info("[bold green]✅ JLL Agent (WS) ready.[/bold green] stream_sid=%s", stream_sid)

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except Exception as exc:
        log.warning("[WS] Pipeline ended: %s", exc)
    finally:
        log_pipeline_event("CLEANUP", "WebSocket session ended")
        await task.cancel()
        log.info("[WS] Agent session closed. stream_sid=%s", stream_sid)


def _update_gather_hint(context: LLMContext, tool_handler: JLLToolHandler) -> None:
    """
    Inject a gather-state hint into the system message so the LLM always
    knows what has been collected and what to ask next.
    Mirrors GatherStateHint processor in bot (1).py.
    """
    hint = build_gather_hint(tool_handler.gathered)
    messages = context.messages

    # Replace existing gather hint system message if present
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and "[GATHER STATE]" in msg.get("content", ""):
            messages[i] = {"role": "system", "content": f"[GATHER STATE]\n{hint}"}
            return

    # Insert after the main system prompt
    messages.insert(1, {"role": "system", "content": f"[GATHER STATE]\n{hint}"})
