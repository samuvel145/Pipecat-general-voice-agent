"""
pipeline/processors.py
Custom Pipecat frame processors for logging and tracing.

Each processor sits between two pipeline stages and logs the frames
passing through without modifying them (transparent pass-through).
"""

import dataclasses
import re
import time

from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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

_echo_log = get_logger("echo_gate")


class EchoCancelGate(FrameProcessor):
    """
    Placed immediately after transport.input().
    Drops AudioRawFrames while the bot's TTS is playing so the
    microphone cannot pick up the speaker output and cause the STT
    to transcribe the bot's own voice.

    TTSSpeakingTracker (placed after the TTS service) calls
    set_bot_speaking() directly to open/close the gate.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking: bool = False

    def set_bot_speaking(self, active: bool) -> None:
        if active != self._bot_speaking:
            _echo_log.debug("EchoCancelGate: bot_speaking=%s", active)
        self._bot_speaking = active

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and self._bot_speaking:
            return  # swallow mic audio while bot is speaking
        await self.push_frame(frame, direction)


class TTSSpeakingTracker(FrameProcessor):
    """
    Placed after the TTS service in the pipeline.
    Keys off BotStartedSpeakingFrame / BotStoppedSpeakingFrame — fired by the
    transport output when audio actually starts/stops playing (not when synthesis
    finishes), so the gate stays closed for the full duration of playback.
    A 400 ms tail delay absorbs speaker ring-down after playback ends.
    """

    def __init__(self, gate: "EchoCancelGate", **kwargs):
        super().__init__(**kwargs)
        self._gate = gate
        self._stop_handle = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            if self._stop_handle:
                self._stop_handle.cancel()
                self._stop_handle = None
            self._gate.set_bot_speaking(True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            import asyncio
            loop = asyncio.get_event_loop()
            if self._stop_handle:
                self._stop_handle.cancel()
            self._stop_handle = loop.call_later(
                0.4, lambda: self._gate.set_bot_speaking(False)
            )
        await self.push_frame(frame, direction)

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
    Converts comma-formatted Indian numbers in LLM output to spoken words
    before the frame reaches TTS.

    Examples:
      ₹1,00,00,000  →  "1 crore"
      ₹50,00,000    →  "50 lakhs"
      ₹20,00,000    →  "20 lakhs"
      50,000        →  "50 thousand"

    Buffers tokens into complete sentences before normalising so that
    prices like ₹20,00,000 streamed as "₹20" + ",00,000" across multiple
    tokens are always seen as a whole.
    """

    # Currency prefix (₹ / Rs / Rs.) + Indian comma-formatted number
    _CURRENCY_RE = re.compile(
        r'[₹]\s*(\d{1,3}(?:,\d{2,3})+)|'
        r'(?:Rs\.?\s*)(\d{1,3}(?:,\d{2,3})+)',
        re.IGNORECASE,
    )
    # Bare comma-separated Indian numbers not already preceded by a currency symbol
    _NUMBER_RE = re.compile(r'(?<![₹\d,])(\d{1,3}(?:,\d{2,3})+)(?!\d)')

    # Sentence-ending punctuation — flush buffer when we see one of these
    _SENTENCE_END_RE = re.compile(r'[.!?]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buf: str = ""

    @staticmethod
    def _to_words(num_str: str) -> str:
        try:
            n = int(num_str.replace(',', ''))
        except ValueError:
            return num_str
        if n >= 10_000_000:
            val = n / 10_000_000
            return f"{int(val)} crore" if val == int(val) else f"{val:.1f} crore"
        if n >= 100_000:
            val = n / 100_000
            return f"{int(val)} lakh" if val == int(val) else f"{val:.1f} lakh"
        if n >= 1_000:
            val = n / 1_000
            return f"{int(val)} thousand" if val == int(val) else f"{val:.1f} thousand"
        return str(n)

    def _normalise(self, text: str) -> str:
        def _repl_currency(m: re.Match) -> str:
            return self._to_words(m.group(1) or m.group(2))
        text = self._CURRENCY_RE.sub(_repl_currency, text)
        text = self._NUMBER_RE.sub(lambda m: self._to_words(m.group(1)), text)
        return text

    async def _flush(self, direction: FrameDirection) -> None:
        if self._buf:
            normalised = self._normalise(self._buf)
            if normalised != self._buf:
                _norm_log.debug("Normalised: %r → %r", self._buf, normalised)
            await self.push_frame(LLMTextFrame(text=normalised), direction)
            self._buf = ""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            self._buf += frame.text
            # Flush on sentence boundary so Cartesia still starts early
            if self._SENTENCE_END_RE.search(frame.text):
                await self._flush(direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._flush(direction)

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


# ─────────────────────────────────────────────────────────────────────────────
# Phonetic correction: Soundex + Metaphone
# Pure-Python, no external deps, ~microseconds per call — zero latency impact.
# ─────────────────────────────────────────────────────────────────────────────

_SDEX: dict[str, str] = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6',
}


def _soundex(word: str) -> str:
    w = re.sub(r'[^A-Z]', '', word.upper())
    if not w:
        return '0000'
    result = w[0]
    prev = _SDEX.get(w[0], '0')
    for ch in w[1:]:
        code = _SDEX.get(ch, '0')
        if code != '0' and code != prev:
            result += code
        prev = code
        if len(result) == 4:
            break
    return result.ljust(4, '0')


def _metaphone(word: str) -> str:
    w = re.sub(r'[^A-Z]', '', word.upper())
    if not w:
        return ''
    if w[:2] in ('AE', 'GN', 'KN', 'PN', 'WR'):
        w = w[1:]
    if len(w) > 1 and w.endswith('E'):
        w = w[:-1]
    result: list[str] = []
    i = 0
    while i < len(w):
        c = w[i]
        if i > 0 and c == w[i - 1] and c != 'C':
            i += 1
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ''
        if c in 'AEIOU':
            if i == 0:
                result.append(c)
        elif c == 'B':
            if not (nxt == '' and i > 0 and w[i - 1] == 'M'):
                result.append('B')
        elif c == 'C':
            if nxt in 'EIY':
                result.append('S')
            elif w[i:i + 2] == 'CH':
                result.append('X'); i += 1
            elif w[i:i + 2] == 'CK':
                result.append('K'); i += 1
            else:
                result.append('K')
        elif c == 'D':
            if w[i:i + 2] == 'DG' and nxt in 'EIY':
                result.append('J'); i += 1
            else:
                result.append('T')
        elif c == 'G':
            if nxt == 'H':
                if i == 0 or w[i - 1] not in 'AEIOU':
                    result.append('K')
                i += 1
            elif nxt == 'N':
                pass
            elif nxt in 'EIY':
                result.append('J')
            else:
                result.append('K')
        elif c == 'H':
            if nxt in 'AEIOU' and (i == 0 or w[i - 1] not in 'AEIOU'):
                result.append('H')
        elif c in 'FJLMNR':
            result.append(c)
        elif c == 'K':
            if i == 0 or w[i - 1] != 'C':
                result.append('K')
        elif c == 'P':
            if nxt == 'H':
                result.append('F'); i += 1
            else:
                result.append('P')
        elif c == 'Q':
            result.append('K')
        elif c == 'S':
            if w[i:i + 2] == 'SH' or w[i:i + 3] in ('SIA', 'SIO'):
                result.append('X')
            else:
                result.append('S')
        elif c == 'T':
            if w[i:i + 2] == 'TH':
                result.append('0'); i += 1
            elif w[i:i + 3] in ('TIA', 'TIO'):
                result.append('X')
            else:
                result.append('T')
        elif c == 'V':
            result.append('F')
        elif c == 'W':
            if nxt in 'AEIOU':
                result.append('W')
        elif c == 'X':
            result.append('KS')
        elif c == 'Y':
            if nxt in 'AEIOU':
                result.append('Y')
        elif c == 'Z':
            result.append('S')
        i += 1
    return ''.join(result)


def _phonetic_score(token: str, candidate: str) -> int:
    """Score 0-100: exact=100, both algorithms=80, metaphone only=60, soundex only=40."""
    if token.upper() == candidate.upper():
        return 100
    sdx = _soundex(token) == _soundex(candidate)
    meta = _metaphone(token) == _metaphone(candidate)
    if sdx and meta:
        return 80
    if meta:
        return 60
    if sdx:
        return 40
    return 0


_LOCATION_LIST: list[str] = [
    # Chennai
    "Anna Nagar", "T Nagar", "Adyar", "Velachery", "Porur", "Perambur",
    "Tambaram", "Sholinganallur", "Pallavaram", "Chromepet", "Medavakkam",
    "Ambattur", "Avadi", "Mogappair", "Nungambakkam", "Mylapore",
    "Kodambakkam", "Guindy", "Thoraipakkam", "Perungudi", "Siruseri",
    "Kelambakkam", "Maraimalai Nagar", "Perungalathur", "Navallur", "Padur",
    # Bengaluru
    "Whitefield", "Koramangala", "Indiranagar", "Sarjapur", "Bellandur",
    "Marathahalli", "HSR Layout", "JP Nagar", "Bannerghatta", "Electronic City",
    "Yelahanka", "Hebbal", "Rajajinagar", "Malleshwaram", "Jayanagar",
    "BTM Layout", "Basavanagudi", "Devanahalli", "Kengeri", "Banashankari",
    "Vijayanagar", "Frazer Town",
    # Hyderabad
    "Gachibowli", "Kondapur", "Madhapur", "Hitech City", "Banjara Hills",
    "Jubilee Hills", "Kukatpally", "Miyapur", "Kompally", "Secunderabad",
    "Ameerpet", "Begumpet", "Manikonda", "Narsingi", "Tellapur",
    "Nallagandla", "Kokapet", "Nanakramguda", "Financial District",
    "Gandipet", "Tolichowki", "Mehdipatnam", "LB Nagar", "Dilsukhnagar",
    "Uppal", "Bachupally",
]

_NAME_LIST: list[str] = [
    "Aarav", "Aditya", "Akash", "Amith", "Anand", "Anil", "Anita", "Anitha",
    "Anjali", "Arjun", "Arun", "Ashok", "Bala", "Balaji", "Bharath",
    "Chitra", "Deepa", "Deepak", "Divya", "Ganesh", "Gopal", "Harish",
    "Haritha", "Hari", "Karthik", "Kavitha", "Kumar", "Lakshmi", "Lavanya",
    "Madhavan", "Mahesh", "Meena", "Meenakshi", "Mohan", "Muthu",
    "Nithya", "Pooja", "Prabhu", "Pradeep", "Prasad", "Priya", "Rajesh",
    "Rajan", "Ramesh", "Rekha", "Rohit", "Sanjay", "Santhosh", "Senthil",
    "Shobha", "Sridhar", "Suresh", "Swathi", "Uma", "Usha", "Vani",
    "Venkat", "Vijay", "Vikram", "Vinay", "Vishal", "Yuvaraj",
]

# Pure function words — never phonetically correct these
_STOPWORDS: frozenset[str] = frozenset({
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she',
    'they', 'them', 'it', 'its', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'the',
    'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'how', 'when', 'where',
    'why', 'all', 'some', 'any', 'both', 'yes', 'no', 'not', 'just',
    'also', 'so', 'then', 'than', 'too', 'very', 'here', 'there', 'now',
    'please', 'okay', 'ok', 'sir', 'hi', 'hello', 'name', 'want',
})

_phon_log = get_logger("phonetic")


class PhoneticCorrectorProcessor(FrameProcessor):
    """
    Intercepts TranscriptionFrame between STT and the context aggregator.
    Corrects misheard location and customer name tokens using Soundex+Metaphone.

    Context rules (derived from the last assistant message in LLMContext):
      - NAME context  : bot last asked for the user's name → score names at >=60
      - LOCATION context: bot last asked for area/city → score locations at >=60
      - PASSIVE       : always correct locations at >=80 regardless of context

    Highest-scoring candidate wins. Bigrams are tried before single tokens so
    multi-word places like "Anna Nagar" beat individual-word false matches.
    """

    _NAME_TRIGGERS = (
        'your name', 'may i have', 'name please', 'good name',
        'who am i speaking', 'your good name',
    )
    _LOC_TRIGGERS = (
        'which area', 'which location', 'preferred location', 'which city',
        'where are you', 'looking in', 'looking at', 'interested in',
        'area are you', 'location are you', 'city are you',
        'what area', 'what location', 'what city',
    )

    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        # Pre-split location list once at startup
        self._multi_locs: list[tuple[str, list[str]]] = [
            (loc, loc.split()) for loc in _LOCATION_LIST if ' ' in loc
        ]
        self._single_locs: list[str] = [loc for loc in _LOCATION_LIST if ' ' not in loc]

    def _last_assistant_text(self) -> str:
        for msg in reversed(self._context.messages):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, list):
                    return ' '.join(
                        c.get('text', '') for c in content if isinstance(c, dict)
                    )
                return str(content)
        return ''

    def _detect_context(self) -> tuple[bool, bool]:
        text = self._last_assistant_text().lower()
        awaiting_name = any(t in text for t in self._NAME_TRIGGERS)
        awaiting_location = any(t in text for t in self._LOC_TRIGGERS)
        return awaiting_name, awaiting_location

    @staticmethod
    def _best_match(token: str, candidates: list[str], threshold: int) -> tuple[str | None, int]:
        best_score, best = 0, None
        for c in candidates:
            s = _phonetic_score(token, c)
            if s > best_score:
                best_score, best = s, c
        return (best, best_score) if best_score >= threshold else (None, 0)

    def _correct(self, text: str, awaiting_name: bool, awaiting_location: bool) -> str:
        words = text.split()
        if not words:
            return text

        result = list(words)
        skip = [False] * len(words)
        loc_threshold = 60 if awaiting_location else 80

        # ── Bigram pass (multi-word locations: "Anna Nagar", "Hitech City" …) ──
        for i in range(len(words) - 1):
            if skip[i]:
                continue
            best_score, best = 0, None
            for loc, parts in self._multi_locs:
                if len(parts) == 2:
                    s = (_phonetic_score(words[i], parts[0]) +
                         _phonetic_score(words[i + 1], parts[1])) // 2
                    if s > best_score:
                        best_score, best = s, loc
            if best and best_score >= loc_threshold:
                _phon_log.debug(
                    "Bigram %r+%r → %r (score=%d)", words[i], words[i + 1], best, best_score
                )
                result[i] = best
                result[i + 1] = ''
                skip[i] = skip[i + 1] = True

        # ── Single-token pass ──────────────────────────────────────────────────
        for i, word in enumerate(words):
            if skip[i]:
                continue
            if len(word) <= 2 or re.search(r'\d', word) or word.lower() in _STOPWORDS:
                continue

            corrected = None

            # Name correction — only when bot explicitly asked for name
            if awaiting_name:
                corrected, score = self._best_match(word, _NAME_LIST, 60)
                if corrected:
                    _phon_log.debug("Name %r → %r (score=%d)", word, corrected, score)

            # Location correction
            if not corrected:
                corrected, score = self._best_match(word, self._single_locs, loc_threshold)
                if corrected:
                    _phon_log.debug(
                        "Location %r → %r (score=%d, thr=%d)", word, corrected, score, loc_threshold
                    )

            if corrected:
                result[i] = corrected

        return ' '.join(w for w in result if w)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            awaiting_name, awaiting_location = self._detect_context()
            corrected = self._correct(frame.text, awaiting_name, awaiting_location)
            if corrected != frame.text:
                _phon_log.info("[PHONETIC] %r → %r", frame.text, corrected)
                try:
                    frame = dataclasses.replace(frame, text=corrected)
                except Exception:
                    pass
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
