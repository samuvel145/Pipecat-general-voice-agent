"""
pipeline/processors.py
Custom Pipecat frame processors for logging and tracing.

Each processor sits between two pipeline stages and logs the frames
passing through without modifying them (transparent pass-through).
"""

import re
import time

from pipecat.frames.frames import (
    AudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from logger import (
    get_logger,
    log_llm_complete,
    log_llm_token,
    log_stt_result,
    log_tts_chunk,
    log_tts_complete,
    log_vad_event,
)

audio_log = get_logger("audio_in")

_func_log = get_logger("filter")

# Matches a complete <function=name>{...}</function> block
_FUNC_COMPLETE_RE = re.compile(r"<function=[^>]+>.*?</function>", re.DOTALL)

# The marker we're trying to detect across streamed tokens
_FUNC_MARKER = "<function="


class FunctionCallFilter(FrameProcessor):
    """
    Buffers streamed LLMTextFrame tokens and strips any raw function-call
    markup (``<function=...>...</function>``) before it reaches TTS.

    Because the LLM may stream tokens one character at a time (``<``, ``f``,
    ``u``, ``n``, …), this filter accumulates ALL tokens in a buffer and
    only emits text once it's confirmed NOT to be the start of a function
    call pattern.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_func: bool = False    # True while inside <function=...>...</function>
        self._buf: str = ""            # rolling buffer of un-emitted text

    def _flush_buffer(self) -> str | None:
        """
        Process the buffer and return any safe-to-emit text.
        Returns None if nothing should be emitted yet.
        """
        # ── Inside a function call: swallow everything until </function> ──
        if self._in_func:
            if "</function>" in self._buf:
                _, after = self._buf.split("</function>", 1)
                _func_log.debug("FunctionCallFilter: dropped function call block")
                self._buf = after
                self._in_func = False
                # Recursively process any text after the closing tag
                if self._buf:
                    return self._flush_buffer()
                return None
            # Still waiting for closing tag
            return None

        # ── Check for a COMPLETE function call block in buffer ────────────
        match = _FUNC_COMPLETE_RE.search(self._buf)
        if match:
            before = self._buf[:match.start()]
            after = self._buf[match.end():]
            _func_log.debug("FunctionCallFilter: stripped complete call")
            self._buf = after
            result = before.strip()
            if self._buf:
                more = self._flush_buffer()
                if more:
                    result = (result + " " + more).strip() if result else more
            return result if result else None

        # ── Check if buffer contains the START of a function call ─────────
        if _FUNC_MARKER in self._buf:
            idx = self._buf.index(_FUNC_MARKER)
            before = self._buf[:idx]
            self._buf = self._buf[idx:]
            self._in_func = True
            _func_log.debug("FunctionCallFilter: detected function call start")
            return before.strip() if before.strip() else None

        # ── Check if the buffer ENDS with a partial prefix of "<function=" ─
        # e.g. buffer ends with "<" or "<func" — we can't tell yet if it's
        # the start of a function call, so hold those chars back.
        marker = _FUNC_MARKER
        for i in range(min(len(self._buf), len(marker)), 0, -1):
            if self._buf.endswith(marker[:i]):
                # Hold back the partial match, emit everything before it
                emit = self._buf[:-i]
                self._buf = self._buf[-i:]
                return emit if emit.strip() else None

        # ── No function-call markers at all — safe to emit everything ─────
        emit = self._buf
        self._buf = ""
        return emit if emit else None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            self._buf += frame.text
            result = self._flush_buffer()
            if result:
                await self.push_frame(LLMTextFrame(text=result), direction)
            return

        # On end of LLM response, flush any remaining buffered text
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._buf and not self._in_func:
                await self.push_frame(LLMTextFrame(text=self._buf), direction)
            self._buf = ""
            self._in_func = False

        await self.push_frame(frame, direction)


_norm_log = get_logger("normalizer")


class TextNormalizerProcessor(FrameProcessor):
    """
    Converts comma-formatted numbers in LLMTextFrame text to spoken words
    before the frame reaches TTS.

    Examples (Indian real-estate context):
      ₹1,00,00,000  →  "1 crore"
      ₹50,00,000    →  "50 lakhs"
      ₹5,00,000     →  "5 lakhs"
      50,000        →  "50 thousand"
      1,500         →  "1500"

    Streaming-safe: processes each token independently; no buffering needed
    because the comma pattern is almost never split across tokens.
    """

    # Currency prefix  (₹ / Rs / Rs.)  followed by optional whitespace
    _CURRENCY_RE = re.compile(
        r'[₹]\s*(\d{1,3}(?:,\d{2,3})+)|'
        r'(?:Rs\.?\s*)(\d{1,3}(?:,\d{2,3})+)'
    )
    # Bare comma-separated numbers NOT preceded by a currency symbol
    _NUMBER_RE = re.compile(r'(?<![₹\d])(\d{1,3}(?:,\d{2,3})+)(?!\d)')

    @staticmethod
    def _to_words(num_str: str) -> str:
        """Convert a comma-formatted number string to Indian-style spoken words."""
        try:
            n = int(num_str.replace(',', ''))
        except ValueError:
            return num_str

        if n >= 10_000_000:          # crore
            val = n / 10_000_000
            label = f"{int(val)} crore" if val == int(val) else f"{val:.1f} crore"
        elif n >= 100_000:           # lakh
            val = n / 100_000
            label = f"{int(val)} lakhs" if val == int(val) else f"{val:.1f} lakhs"
        elif n >= 1_000:             # thousand
            val = n / 1_000
            label = f"{int(val)} thousand" if val == int(val) else f"{val:.1f} thousand"
        else:
            label = str(n)
        return label

    def _normalise(self, text: str) -> str:
        # Replace currency + number (e.g. ₹50,00,000 → "50 lakhs")
        def _repl_currency(m: re.Match) -> str:
            num_str = m.group(1) or m.group(2)
            return self._to_words(num_str)

        text = self._CURRENCY_RE.sub(_repl_currency, text)

        # Replace any remaining bare comma-numbers (e.g. 50,000 → "50 thousand")
        text = self._NUMBER_RE.sub(lambda m: self._to_words(m.group(1)), text)
        return text

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            normalised = self._normalise(frame.text)
            if normalised != frame.text:
                _norm_log.debug("Normalised: %r → %r", frame.text, normalised)
                frame = LLMTextFrame(text=normalised)

        await self.push_frame(frame, direction)


class VADLogProcessor(FrameProcessor):
    """Logs VAD speech-start / speech-end frames."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            log_vad_event(speech_detected=True)
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            log_vad_event(speech_detected=False)
        await self.push_frame(frame, direction)


# ── Shared turn state between STT and Conv log processors ────────────────────
class _TurnState:
    """Module-level shared state so processors at different pipeline positions
    can coordinate turn IDs and pipeline state."""
    turn_id: int = 0
    state: str = "idle"

    @classmethod
    def transition(cls, new_state: str) -> None:
        if new_state != cls.state:
            get_logger("agent").info("[STATE] %s → %s", cls.state, new_state)
            cls.state = new_state


class STTLogProcessor(FrameProcessor):
    """Bolt-style STT + turn tracking.
    Must be placed BEFORE context_aggregator.user() (position 3) so it sees
    TranscriptionFrame before the aggregator consumes it.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            _TurnState.turn_id += 1
            _TurnState.transition("listening")
            get_logger("agent").info(
                "[TURN] id=%d", _TurnState.turn_id
            )
            get_logger("agent").info(
                "[STT]  final=%r", frame.text.strip()
            )
        await self.push_frame(frame, direction)


class LLMLogProcessor(FrameProcessor):
    """Logs LLM response start, streaming tokens, and completion."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            get_logger("llm").info("🤖 LLM response stream started")
        elif isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text)
            log_llm_token(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            log_llm_complete("".join(self._buffer))
            self._buffer = []
        await self.push_frame(frame, direction)


class TTSLogProcessor(FrameProcessor):
    """Logs TTS audio chunks and synthesis lifecycle."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            get_logger("tts").info("🗣  TTS synthesis started")
        elif isinstance(frame, TTSAudioRawFrame):
            log_tts_chunk(len(frame.audio))
        elif isinstance(frame, TTSStoppedFrame):
            log_tts_complete()
        await self.push_frame(frame, direction)


class AudioInputLogProcessor(FrameProcessor):
    """Logs raw microphone frame counts (at DEBUG level to avoid spam)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frame_count = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            if self._frame_count % 50 == 0:   # log every 50 frames (~1 sec)
                audio_log.debug("🎙  %d audio frames captured", self._frame_count)
        await self.push_frame(frame, direction)


_conv_log = get_logger("agent")


class ConversationLogProcessor(FrameProcessor):
    """
    Bolt-style LLM + TTS logging (position 7, after func_filter).
    TranscriptionFrame never reaches here — STTLogProcessor handles that.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._agent_buf: list[str] = []
        self._llm_start_ts: float = 0.0
        self._llm_first_token_ts: float = 0.0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # ── LLM starts generating ───────────────────────────────────────────
        if isinstance(frame, LLMFullResponseStartFrame):
            self._agent_buf = []
            self._llm_start_ts = time.monotonic()
            self._llm_first_token_ts = 0.0
            _TurnState.transition("thinking")

        # ── First LLM token + accumulate ────────────────────────────────────
        elif isinstance(frame, LLMTextFrame) and frame.text:
            if not self._llm_first_token_ts:
                self._llm_first_token_ts = time.monotonic()
                first_token_ms = (self._llm_first_token_ts - self._llm_start_ts) * 1000
                _conv_log.info("[LLM]  first_token=%.0fms", first_token_ms)
            self._agent_buf.append(frame.text)

        # ── LLM done → log full response + TTS ──────────────────────────────
        elif isinstance(frame, LLMFullResponseEndFrame):
            full_text = "".join(self._agent_buf).strip()
            if full_text:
                total_ms = (time.monotonic() - self._llm_start_ts) * 1000
                _conv_log.info(
                    "[LLM]  done | total=%.0fms chars=%d",
                    total_ms, len(full_text),
                )
                _conv_log.info("[TTS]  speaking | %r", full_text[:120])
                _TurnState.transition("speaking")
                _conv_log.info(
                    "[TURN_SUMMARY] turn_id=%d llm=%.2fs",
                    _TurnState.turn_id, total_ms / 1000,
                )
            self._agent_buf = []

        await self.push_frame(frame, direction)
