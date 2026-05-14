# JLL Voice Sales Agent — Complete Build Guide

Convert your existing **Pipecat general voice agent** into a fully working **JLL real-estate voice sales agent** that runs locally (terminal only, no phone line needed).

---

## What You Are Building

```
Microphone → Deepgram STT → Groq LLM (with JLL tools) → Cartesia TTS → Speaker
                                      ↕
                          JLL Integration Proxy (HTTP)
                          search_properties / get_property_details /
                          areas_by_budget / submit_callback / schedule_site_visit
```

The agent will:
- Greet the caller as a JLL property sales assistant
- Gather city, property type, budget, and location from the user
- Call the JLL backend to search and present properties
- Handle "show more", property details, site visit bookings, and callback requests
- Print everything to the terminal

---

## Reference Files You Already Have

| File | Role |
|---|---|
| `bot__1_.py` | The full JLL production bot — extract system prompt, tool logic, gather flow |
| `integration__1_.js` | The JLL proxy server — shows all API routes and field names |
| `config.py` | Your existing settings (keep as-is, add new keys) |
| `logger.py` | Your existing logging (keep as-is) |
| `main.py` | Your existing entry point (keep as-is) |
| `pipeline/agent.py` | **Replace this entirely** with the new JLL agent below |

Keep `bot__1_.py` and `integration__1_.js` open in your IDE as reference — you will copy exact field names, route paths, and response structures from them.

---

## Folder Structure After This Build

```
pipecat-voice-agent/
├── .env                        ← add JLL_PROXY_URL here
├── config.py                   ← add 4 new settings
├── logger.py                   ← no changes
├── main.py                     ← no changes
├── requirements.txt            ← add httpx
├── pipeline/
│   ├── __init__.py             ← create (empty)
│   ├── agent.py                ← replace entirely
│   ├── jll_client.py           ← create new
│   ├── tools.py                ← create new
│   └── prompts.py              ← create new
└── logs/
    └── agent.log
```

---

## Step 1 — Update `.env`

Add these lines to your existing `.env` file:

```env
# JLL Integration
JLL_PROXY_URL=http://localhost:3000/api/integration

# JLL agent identity (spoken in greeting)
JLL_ASSISTANT_NAME=Priya

# LLM tuning for consistent tone
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=250

# Startup greeting (override default)
STARTUP_GREETING=Hello! I'm Priya from JLL Homes. I can help you find your perfect property. Which city are you looking in?
```

> **Ask yourself before proceeding:** Is your JLL integration proxy (`integration__1_.js`) running locally, or are you hitting `https://jll-backend.ibism.com` directly? Set `JLL_PROXY_URL` accordingly. The proxy file shows all routes — for example `/api/integration/proxy/search` maps to `search_properties`.

---

## Step 2 — Update `config.py`

Add these 4 new settings inside the `Settings` class, after the existing LLM block:

```python
# ── JLL Integration ───────────────────────────────────────────
JLL_PROXY_URL: str = "http://localhost:3000/api/integration"
JLL_ASSISTANT_NAME: str = "Priya"

# ── LLM fine-tuning ───────────────────────────────────────────
LLM_TEMPERATURE: float = 0.4
```

The full updated `Settings` class should look like this:

```python
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
    LLM_MAX_TOKENS: int = 250
    LLM_TEMPERATURE: float = 0.4
    MAX_HISTORY_TURNS: int = 10

    # ── JLL Integration ───────────────────────────────────────
    JLL_PROXY_URL: str = "http://localhost:3000/api/integration"
    JLL_ASSISTANT_NAME: str = "Priya"

    # ── Startup ───────────────────────────────────────────────
    STARTUP_GREETING: str = (
        "Hello! I'm Priya from JLL Homes. I can help you find your perfect property. "
        "Which city are you looking in — Chennai, Bengaluru, or Hyderabad?"
    )

    # ── Logging ────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "logs/agent.log"

    model_config = SettingsConfigDict(env_file=".env")
```

---

## Step 3 — Update `requirements.txt`

Add `httpx` for async HTTP calls to the JLL proxy:

```
# Pipecat core + required service extras
pipecat-ai[deepgram,groq,cartesia,silero,local]>=0.0.50

# Config & env
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Audio I/O (local microphone + speaker)
pyaudio>=0.2.14

# Logging extras
rich>=13.0.0

# HTTP client for JLL API calls
httpx>=0.27.0
```

Run:
```powershell
.\venv\Scripts\python.exe -m pip install httpx
```

---

## Step 4 — Create `pipeline/__init__.py`

Create this file with no content (just an empty file). It makes `pipeline` a Python package.

```python
```

---

## Step 5 — Create `pipeline/prompts.py`

This file contains the JLL system prompt. It is derived from the `DEFAULT_SYSTEM_PROMPT`, `_build_integration_system_prompt`, and `_build_compact_gather_phase_prompt` functions in `bot__1_.py`.

Create `pipeline/prompts.py` with this content:

```python
"""
pipeline/prompts.py
JLL voice agent system prompt and gather-phase prompts.
Derived from bot__1_.py: DEFAULT_SYSTEM_PROMPT, _build_integration_system_prompt,
and the gather phase logic in TranscriptionLogger.
"""

JLL_SYSTEM_PROMPT = """You are {assistant_name}, a warm and professional voice sales assistant for JLL Homes — one of India's most trusted real-estate companies.

PERSONA & TONE
- Always speak in short, clear sentences (1-2 sentences max per turn).
- Warm, confident, and helpful. Never robotic.
- Never use bullet points, numbered lists, markdown, or asterisks in your spoken responses.
- Always end a sentence with a period or question mark.
- Use Indian English naturally — "lakhs", "crores", "BHK", "possession".

YOUR ROLE
You help home buyers find properties listed by JLL across Chennai, Bengaluru, and Hyderabad.
You gather their requirements and search the JLL database using tools.

GATHERING REQUIREMENTS (do this in order before calling search_properties)
You must collect ALL of the following before searching:
1. City — ask if not stated. Supported: Chennai, Bengaluru, Hyderabad.
2. Property type — ask if not stated. Example: apartment, villa, plot.
3. Budget — ask for a range (min and max in rupees). Interpret "80 lakhs" as 8000000, "1 crore" as 10000000.
4. Location / area — ask which area or locality in the city.

If the user provides all four in one sentence, proceed directly to search_properties. Do not ask again for what they already told you.

SEARCHING PROPERTIES
- Call search_properties with city, property_type, min_price, max_price, location.
- Present results as: "I found [N] options. First is [Name] in [location], priced at [price]. Want to hear more about this one or shall I list the next option?"
- Never fabricate property names, prices, or locations. Only speak what the tool returns.
- If 0 results: say "I couldn't find properties matching that exactly. Let me check what areas fit your budget." Then call areas_by_budget.

SHOWING MORE
- If user says "show more", "next", "other options" — call search_properties again with page incremented.
- If results are exhausted, say so honestly and offer to widen the search.

PROPERTY DETAILS
- If user asks about a specific property by name or number, call get_property_details with the property_id.
- Speak key details: name, location, BHK configurations, price, possession date, developer.
- Do not guess details you don't have.

CALLBACK / SITE VISIT
- If user wants a site visit, call schedule_site_visit.
- If user wants a callback, call submit_callback.
- After submitting, confirm: "Done! The JLL team will reach out to you shortly."

ENDING THE CALL
- If user says goodbye, thank them: "Thank you for speaking with JLL Homes. Have a wonderful day!"

STRICT RULES
- Never invent property data.
- Never ask for a field the user already provided.
- Keep every spoken response under 40 words.
- Do not speak tool call syntax or JSON.
"""


GATHER_PHASE_PROMPT = """CURRENT SESSION STATE
You are in the requirement-gathering phase. You have collected:
{gathered_summary}

NEXT STEP: {next_question}
Ask only this one question. Do not search yet.
"""


def build_system_prompt(assistant_name: str) -> str:
    """Return the fully formatted JLL system prompt."""
    return JLL_SYSTEM_PROMPT.format(assistant_name=assistant_name)


def build_gather_hint(gathered: dict) -> str:
    """
    Return a short hint message appended to the system prompt
    showing what has been collected so far.
    Mirrors the GatherStateHint logic from bot__1_.py.
    """
    city = gathered.get("city", "")
    prop_type = gathered.get("property_type", "")
    location = gathered.get("location", "")
    min_price = gathered.get("min_price")
    max_price = gathered.get("max_price")

    parts = []
    if city:
        parts.append(f"city={city}")
    if prop_type:
        parts.append(f"type={prop_type}")
    if location:
        parts.append(f"area={location}")
    if min_price or max_price:
        lo = f"{int(min_price):,}" if min_price else "0"
        hi = f"{int(max_price):,}" if max_price else "∞"
        parts.append(f"budget=₹{lo}–₹{hi}")

    gathered_summary = ", ".join(parts) if parts else "nothing yet"

    # Determine next question (mirrors _next_deterministic_gather_prompt in bot__1_.py)
    if not prop_type:
        next_q = "What kind of property are you looking for — apartment, villa, or plot?"
    elif not (min_price or max_price):
        next_q = "What is your budget range?"
    elif not location:
        next_q = f"Which area in {city or 'the city'} are you interested in?"
    else:
        next_q = "I have all the details. Searching now."

    return GATHER_PHASE_PROMPT.format(
        gathered_summary=gathered_summary,
        next_question=next_q,
    )
```

---

## Step 6 — Create `pipeline/jll_client.py`

This file makes HTTP calls to the JLL integration proxy. The routes come directly from `integration__1_.js`.

Open `integration__1_.js` and note these routes — they are what you will call:
- `GET /proxy/search` → search_properties
- `GET /proxy/property-details` → get_property_details
- `GET /proxy/areas-by-budget` → areas_by_budget

Create `pipeline/jll_client.py`:

```python
"""
pipeline/jll_client.py
Async HTTP client for the JLL integration proxy.

Route reference: integration__1_.js
  GET  {JLL_PROXY_URL}/proxy/search             → search_properties
  GET  {JLL_PROXY_URL}/proxy/property-details   → get_property_details
  GET  {JLL_PROXY_URL}/proxy/areas-by-budget    → areas_by_budget
  POST {JLL_PROXY_URL}/proxy/callback           → submit_callback
  POST {JLL_PROXY_URL}/proxy/site-visit         → schedule_site_visit

All responses follow the shape:
  { "success": true, "data": [...], "total": N, "has_more": bool }
"""

import logging
from typing import Any

import httpx

from config import settings

log = logging.getLogger("JLL-CLIENT")

_BASE = settings.JLL_PROXY_URL.rstrip("/")

# Shared async client — created once, reused across calls
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


async def search_properties(
    city: str,
    property_type: str = "",
    location: str = "",
    min_price: int | None = None,
    max_price: int | None = None,
    bedrooms: str = "",
    page: int = 1,
    limit: int = 3,
    exclude_slugs: list[str] | None = None,
    sort_preference: str = "",
    call_id: str = "",
) -> dict:
    """
    Search JLL properties.
    Maps to: GET /proxy/search in integration__1_.js (handleSearchProperties).
    Field names match the query params in integration__1_.js lines 728-735.
    """
    params: dict[str, Any] = {
        "city": city,
        "page": str(page),
        "limit": str(limit),
    }
    if property_type:
        params["property_type"] = property_type
    if location:
        params["location"] = location
    if min_price is not None:
        params["min_price"] = str(min_price)
    if max_price is not None:
        params["max_price"] = str(max_price)
    if bedrooms:
        params["bedrooms"] = bedrooms
    if sort_preference:
        params["sort_preference"] = sort_preference
    if exclude_slugs:
        import json
        params["exclude_slugs"] = json.dumps(exclude_slugs)
    if call_id:
        params["call_id"] = call_id

    log.info(
        f"[search_properties] city={city} type={property_type} "
        f"loc={location} budget={min_price}-{max_price} page={page}"
    )

    try:
        r = await _get_client().get(f"{_BASE}/proxy/search", params=params)
        r.raise_for_status()
        result = r.json()
        count = len(result.get("data") or [])
        log.info(f"[search_properties] returned {count} results")
        return result
    except httpx.HTTPStatusError as e:
        log.error(f"[search_properties] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return {"success": False, "error": str(e), "data": []}
    except Exception as e:
        log.error(f"[search_properties] failed: {e}")
        return {"success": False, "error": str(e), "data": []}


async def get_property_details(
    property_id: str,
    city: str = "",
    call_id: str = "",
) -> dict:
    """
    Fetch full details for a single property.
    Maps to: GET /proxy/property-details in integration__1_.js.
    property_id is the Project_Slug from search results.
    """
    params: dict[str, Any] = {"property_id": property_id}
    if city:
        params["city"] = city
    if call_id:
        params["call_id"] = call_id

    log.info(f"[get_property_details] property_id={property_id}")

    try:
        r = await _get_client().get(f"{_BASE}/proxy/property-details", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[get_property_details] failed: {e}")
        return {"success": False, "error": str(e)}


async def areas_by_budget(
    city: str,
    property_type: str = "",
    min_price: int | None = None,
    max_price: int | None = None,
    call_id: str = "",
) -> dict:
    """
    Find areas matching a budget.
    Maps to: GET /proxy/areas-by-budget in integration__1_.js (handleAreasByBudget).
    Used when search_properties returns 0 results due to budget mismatch.
    """
    params: dict[str, Any] = {"city": city}
    if property_type:
        params["property_type"] = property_type
    if min_price is not None:
        params["min_price"] = str(min_price)
    if max_price is not None:
        params["max_price"] = str(max_price)
    if call_id:
        params["call_id"] = call_id

    log.info(f"[areas_by_budget] city={city} budget={min_price}-{max_price}")

    try:
        r = await _get_client().get(f"{_BASE}/proxy/areas-by-budget", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[areas_by_budget] failed: {e}")
        return {"success": False, "error": str(e), "areas": []}


async def submit_callback(
    name: str,
    phone: str,
    city: str = "",
    property_id: str = "",
    call_id: str = "",
) -> dict:
    """
    Submit a callback request.
    Maps to: POST /proxy/callback in integration__1_.js.
    """
    payload = {
        "name": name,
        "phone": phone,
        "city": city,
        "property_id": property_id,
        "call_id": call_id,
    }
    log.info(f"[submit_callback] name={name} phone={phone}")
    try:
        r = await _get_client().post(f"{_BASE}/proxy/callback", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[submit_callback] failed: {e}")
        return {"success": False, "error": str(e)}


async def schedule_site_visit(
    name: str,
    phone: str,
    property_id: str,
    preferred_date: str = "",
    call_id: str = "",
) -> dict:
    """
    Schedule a site visit.
    Maps to: POST /proxy/site-visit in integration__1_.js.
    """
    payload = {
        "name": name,
        "phone": phone,
        "property_id": property_id,
        "preferred_date": preferred_date,
        "call_id": call_id,
    }
    log.info(f"[schedule_site_visit] property_id={property_id} name={name}")
    try:
        r = await _get_client().post(f"{_BASE}/proxy/site-visit", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[schedule_site_visit] failed: {e}")
        return {"success": False, "error": str(e)}


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None
```

---

## Step 7 — Create `pipeline/tools.py`

This file defines the LLM tool schemas and their handler functions.
The tool names and parameter shapes come from `IntegrationToolHandler.build_tool_schemas()` in `bot__1_.py` and the endpoint configs in `integration__1_.js` lines 13920–13940.

Create `pipeline/tools.py`:

```python
"""
pipeline/tools.py
JLL LLM tool definitions and handlers.

Tool names and parameter shapes are taken from:
  - bot__1_.py: IntegrationToolHandler, lines 13906–14100
  - integration__1_.js: endpoint config block, lines 13920–13940

The 5 tools the LLM can call:
  1. search_properties
  2. get_property_details
  3. areas_by_budget
  4. submit_callback
  5. schedule_site_visit
"""

import json
import logging
from typing import Any

from pipeline import jll_client

log = logging.getLogger("JLL-TOOLS")


# ── Tool schemas (passed to Groq as function definitions) ─────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": (
                "Search JLL real estate listings. "
                "Call this after collecting city, property_type, budget, and location from the user. "
                "Returns a list of matching properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name. One of: Chennai, Bengaluru, Hyderabad.",
                    },
                    "property_type": {
                        "type": "string",
                        "description": "Type of property. Examples: Apartment, Villa, Plot.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Area or locality within the city. Example: OMR, T Nagar, Whitefield.",
                    },
                    "min_price": {
                        "type": "integer",
                        "description": "Minimum budget in rupees. 80 lakhs = 8000000.",
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "Maximum budget in rupees. 1 crore = 10000000.",
                    },
                    "bedrooms": {
                        "type": "string",
                        "description": "BHK count as string. Examples: '2', '3', '4'.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for pagination. Start at 1. Increment for 'show more'.",
                        "default": 1,
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_property_details",
            "description": (
                "Get full details for a specific property. "
                "Call this when the user asks about a specific property from the search results. "
                "Use the property_id from search_properties results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {
                        "type": "string",
                        "description": "The project_slug or property_id from search results.",
                    },
                    "city": {
                        "type": "string",
                        "description": "City of the property.",
                    },
                },
                "required": ["property_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "areas_by_budget",
            "description": (
                "Find areas or localities that have properties within the user's budget. "
                "Call this when search_properties returns zero results due to budget being too low for the requested area."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to search in.",
                    },
                    "property_type": {
                        "type": "string",
                        "description": "Type of property.",
                    },
                    "min_price": {
                        "type": "integer",
                        "description": "Minimum budget in rupees.",
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "Maximum budget in rupees.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_callback",
            "description": (
                "Submit a callback request when the user wants the JLL team to call them back. "
                "Collect the user's name and phone number before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "User's full name.",
                    },
                    "phone": {
                        "type": "string",
                        "description": "User's phone number.",
                    },
                    "city": {
                        "type": "string",
                        "description": "City of interest.",
                    },
                    "property_id": {
                        "type": "string",
                        "description": "Property ID if the user is interested in a specific property.",
                    },
                },
                "required": ["name", "phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_site_visit",
            "description": (
                "Schedule a site visit for a property. "
                "Collect the user's name, phone, and preferred date before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "User's full name.",
                    },
                    "phone": {
                        "type": "string",
                        "description": "User's phone number.",
                    },
                    "property_id": {
                        "type": "string",
                        "description": "Property ID from search results.",
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Preferred date for the visit. Example: 'next Saturday', '2024-12-20'.",
                    },
                },
                "required": ["name", "phone", "property_id"],
            },
        },
    },
]


# ── Tool handler ──────────────────────────────────────────────────────────────

class JLLToolHandler:
    """
    Executes LLM tool calls and formats results for voice delivery.
    Mirrors IntegrationToolHandler in bot__1_.py, simplified for terminal use.
    """

    def __init__(self):
        # Session memory — mirrors _gathered_search in bot__1_.py
        self.gathered: dict = {}
        # Shown property slugs for pagination dedup — mirrors _shown_slugs in bot__1_.py
        self.shown_slugs: set = set()
        # Current JLL page number — mirrors _jll_page in bot__1_.py
        self.current_page: int = 1
        # Last search key for detecting "show more" vs fresh search
        self._last_search_key: str = ""
        # Property details cache — mirrors _property_details_map in bot__1_.py
        self._property_details_map: dict = {}

    def _search_key(self, args: dict) -> str:
        return "|".join([
            str(args.get("city") or "").lower(),
            str(args.get("location") or "").lower(),
            str(args.get("property_type") or "").lower(),
            str(args.get("min_price") or ""),
            str(args.get("max_price") or ""),
        ])

    async def handle(self, tool_name: str, args: dict) -> str:
        """
        Dispatch a tool call and return a voice-ready result string.
        Called by the pipeline after the LLM emits a function_call frame.
        """
        log.info(f"[Tool] {tool_name} args={json.dumps(args, ensure_ascii=False)[:200]}")

        if tool_name == "search_properties":
            return await self._handle_search(args)
        elif tool_name == "get_property_details":
            return await self._handle_details(args)
        elif tool_name == "areas_by_budget":
            return await self._handle_areas(args)
        elif tool_name == "submit_callback":
            return await self._handle_callback(args)
        elif tool_name == "schedule_site_visit":
            return await self._handle_site_visit(args)
        else:
            return f"Unknown tool: {tool_name}"

    async def _handle_search(self, args: dict) -> str:
        # Update gathered search memory (mirrors _gathered_search in bot__1_.py)
        for field in ("city", "property_type", "location", "min_price", "max_price", "bedrooms"):
            if args.get(field):
                self.gathered[field] = args[field]

        # Detect "show more" vs fresh search
        new_key = self._search_key(args)
        if new_key == self._last_search_key and self.shown_slugs:
            self.current_page += 1
        else:
            self.current_page = int(args.get("page") or 1)
            self.shown_slugs.clear()
            self._last_search_key = new_key

        result = await jll_client.search_properties(
            city=args.get("city", ""),
            property_type=args.get("property_type", ""),
            location=args.get("location", ""),
            min_price=args.get("min_price"),
            max_price=args.get("max_price"),
            bedrooms=str(args.get("bedrooms") or ""),
            page=self.current_page,
            limit=3,
            exclude_slugs=list(self.shown_slugs),
        )

        items = result.get("data") or []
        has_more = result.get("has_more", False)

        if not items:
            self.gathered["result_count"] = 0
            return (
                "SEARCH_RESULT: zero_results\n"
                f"city={args.get('city')} location={args.get('location')} "
                f"type={args.get('property_type')} budget={args.get('min_price')}-{args.get('max_price')}\n"
                "No properties found matching these criteria."
            )

        # Cache property details and track shown slugs
        lines = [f"SEARCH_RESULT: {len(items)} properties found (has_more={has_more})"]
        for i, p in enumerate(items, 1):
            slug = p.get("Project_Slug") or p.get("project_slug") or p.get("property_id") or ""
            name = p.get("Project_Name_Original") or p.get("Project_Name") or p.get("name") or "?"
            location = p.get("Location") or p.get("location") or p.get("Micro_Market") or ""
            price = p.get("starting_price") or p.get("price") or ""

            configs = p.get("configurations") or p.get("configs") or []
            bhk_types = list({
                c.get("Config_Type") or c.get("type") or ""
                for c in configs
                if c.get("Config_Type") or c.get("type")
            })
            bhk_summary = "/".join(sorted(bhk_types)[:3]) or ""

            if slug:
                self.shown_slugs.add(slug)
                self._property_details_map[slug] = p

            lines.append(
                f"{i}. name={name} | location={location} | bhk={bhk_summary} | "
                f"price={price} | property_id={slug}"
            )

        self.gathered["result_count"] = len(items)
        return "\n".join(lines)

    async def _handle_details(self, args: dict) -> str:
        property_id = str(args.get("property_id") or "").strip()
        city = str(args.get("city") or self.gathered.get("city") or "").strip()

        # Check in-memory cache first (mirrors _property_details_map in bot__1_.py)
        cached = self._property_details_map.get(property_id)

        result = await jll_client.get_property_details(property_id=property_id, city=city)
        data = result.get("data") or cached or {}

        if not data:
            return f"PROPERTY_DETAILS: not_found property_id={property_id}"

        name = data.get("name") or data.get("Project_Name") or property_id
        location = data.get("location") or data.get("Location") or ""
        price = data.get("starting_price") or data.get("price") or "Price on request"
        possession = data.get("possession") or data.get("Possession") or ""
        developer = data.get("developer") or ""
        if isinstance(developer, dict):
            developer = developer.get("name") or ""

        configs = data.get("configs") or data.get("configurations") or []
        bhk_lines = []
        for c in configs[:4]:
            cfg_type = c.get("Config_Type") or c.get("type") or ""
            cfg_price = c.get("FinalPrice") or c.get("All_Price") or c.get("price") or ""
            cfg_area = c.get("super_builtup") or c.get("carpet_area") or c.get("area") or ""
            if cfg_type:
                bhk_lines.append(f"  {cfg_type}: {cfg_area} sqft @ {cfg_price}")

        lines = [
            f"PROPERTY_DETAILS: {name}",
            f"location={location}",
            f"starting_price={price}",
            f"developer={developer}",
            f"possession={possession}",
        ]
        if bhk_lines:
            lines.append("configurations:\n" + "\n".join(bhk_lines))

        return "\n".join(lines)

    async def _handle_areas(self, args: dict) -> str:
        result = await jll_client.areas_by_budget(
            city=args.get("city", self.gathered.get("city", "")),
            property_type=args.get("property_type", self.gathered.get("property_type", "")),
            min_price=args.get("min_price", self.gathered.get("min_price")),
            max_price=args.get("max_price", self.gathered.get("max_price")),
        )

        areas = result.get("areas") or []
        if not areas:
            return "AREAS_RESULT: No areas found within this budget."

        area_names = [a.get("name") or a.get("area") or "" for a in areas[:5] if a.get("name") or a.get("area")]
        return f"AREAS_RESULT: Areas with properties in your budget: {', '.join(area_names)}"

    async def _handle_callback(self, args: dict) -> str:
        result = await jll_client.submit_callback(
            name=args.get("name", ""),
            phone=args.get("phone", ""),
            city=args.get("city", self.gathered.get("city", "")),
            property_id=args.get("property_id", ""),
        )
        if result.get("success"):
            return "CALLBACK_SUBMITTED: success The JLL team will contact you shortly."
        return f"CALLBACK_SUBMITTED: error {result.get('error', 'Unknown error')}"

    async def _handle_site_visit(self, args: dict) -> str:
        result = await jll_client.schedule_site_visit(
            name=args.get("name", ""),
            phone=args.get("phone", ""),
            property_id=args.get("property_id", ""),
            preferred_date=args.get("preferred_date", ""),
        )
        if result.get("success"):
            return "SITE_VISIT_SCHEDULED: success Your site visit has been booked."
        return f"SITE_VISIT_SCHEDULED: error {result.get('error', 'Unknown error')}"
```

---

## Step 8 — Replace `pipeline/agent.py`

This is the main pipeline. It wires Deepgram STT → Groq LLM (with JLL tools) → Cartesia TTS → speaker, exactly like your current agent, but now with:
- The JLL system prompt
- Tool call handling
- Gather-state awareness

**Replace the entire content of `pipeline/agent.py`** with:

```python
"""
pipeline/agent.py
JLL Voice Sales Agent — local terminal pipeline.

Replaces the general-purpose agent.py.
Wires: Mic → Deepgram STT → Groq LLM (JLL tools) → Cartesia TTS → Speaker

Tool call flow (derived from IntegrationToolHandler in bot__1_.py):
  LLM emits function_call → ToolCallHandler.handle() → jll_client HTTP call
  → result injected back into LLM context → LLM speaks the result

References used:
  - bot__1_.py: run_outbound_pipeline(), pipeline assembly, TranscriptionLogger
  - integration__1_.js: proxy routes, field names
"""

from __future__ import annotations

import asyncio
import json
import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMMessagesFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams

from config import settings
from logger import (
    get_logger,
    log_pipeline_event,
    log_stt_result,
    log_tts_complete,
)
from pipeline.prompts import build_system_prompt, build_gather_hint
from pipeline.tools import JLLToolHandler, TOOL_SCHEMAS
from pipeline import jll_client

log = get_logger("agent")


async def run_agent() -> None:
    log.info("JLL Voice Agent starting…")
    log_pipeline_event("INIT", "Building pipeline components")

    # ── Transport (mic + speaker) ─────────────────────────────────────────────
    log_pipeline_event("TRANSPORT", "Initialising LocalAudioTransport")
    transport = LocalAudioTransport(
        LocalAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=float(settings.SILENCE_THRESHOLD_MS) / 1000,
                    min_volume=0.6,
                )
            ),
            vad_audio_passthrough=True,
        )
    )

    # ── STT ───────────────────────────────────────────────────────────────────
    log_pipeline_event("STT", "Initialising Deepgram nova-2")
    stt = DeepgramSTTService(
        api_key=settings.DEEPGRAM_API_KEY,
        live_options={
            "model": "nova-2",
            "language": "en-IN",
            "encoding": "linear16",
            "sample_rate": settings.SAMPLE_RATE,
            "channels": settings.CHANNELS,
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        },
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    log_pipeline_event("LLM", f"Initialising Groq model={settings.LLM_MODEL}")
    llm = GroqLLMService(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        tools=TOOL_SCHEMAS,
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    log_pipeline_event("TTS", "Initialising Cartesia sonic-2")
    tts = CartesiaTTSService(
        api_key=settings.CARTESIA_API_KEY,
        voice_id=settings.CARTESIA_VOICE_ID,
        model="sonic-2",
        speed=0.0,        # neutral speed
        emotion="content", # fixed emotion — consistent tone
    )

    # ── LLM Context ───────────────────────────────────────────────────────────
    # Build system prompt (derived from build_system_prompt() in prompts.py,
    # which mirrors DEFAULT_SYSTEM_PROMPT + _build_integration_system_prompt in bot__1_.py)
    system_prompt = build_system_prompt(settings.JLL_ASSISTANT_NAME)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}]
    )
    context_aggregator = llm.create_context_aggregator(context)

    # ── Tool handler ──────────────────────────────────────────────────────────
    tool_handler = JLLToolHandler()

    # ── Pipeline assembly ─────────────────────────────────────────────────────
    log_pipeline_event("PIPELINE", "Assembling pipeline stages")
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(allow_interruptions=True),
    )

    # ── Tool call handler ─────────────────────────────────────────────────────
    # Pipecat fires this callback when the LLM emits a function_call.
    # We call our JLLToolHandler, inject the result back into context,
    # then re-run the LLM to produce a spoken response.
    # This mirrors how IntegrationToolHandler.handle() works in bot__1_.py.
    @llm.event_handler("on_tool_call")
    async def handle_tool_call(llm_service, tool_call):
        tool_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        log.info(f"[Tool] LLM called {tool_name}({args})")

        # Update gather hint in system prompt based on what we now know
        _update_gather_hint(context, tool_handler)

        # Execute the tool
        result_text = await tool_handler.handle(tool_name, args)

        log.info(f"[Tool] {tool_name} result: {result_text[:120]}")

        # Inject tool result into LLM context
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

        # Re-run LLM to speak the result
        await task.queue_frames([LLMMessagesFrame(context.messages)])

    # ── Startup greeting ──────────────────────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_connected(transport_obj, client):
        log_pipeline_event("GREET", "Playing JLL startup greeting")
        if settings.STARTUP_GREETING:
            await task.queue_frames([TTSSpeakFrame(settings.STARTUP_GREETING)])
        log.info("✅ JLL Agent ready. Speak into the microphone. Press Ctrl+C to stop.")

    # ── STT transcript logging ────────────────────────────────────────────────
    @stt.event_handler("on_transcription")
    async def on_transcript(stt_service, text, is_final):
        log_stt_result(text, is_final)
        if is_final:
            log.info(f"[USER] {text}")
            # Update gather hint after each user turn
            _update_gather_hint(context, tool_handler)

    # ── TTS logging ───────────────────────────────────────────────────────────
    @tts.event_handler("on_tts_stopped")
    async def on_tts_done(tts_service):
        log_tts_complete()

    # ── Runner ────────────────────────────────────────────────────────────────
    log_pipeline_event("READY", "Pipeline assembled — starting runner")
    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        log_pipeline_event("CLEANUP", "Pipeline task cancelled")
        await jll_client.close_client()
        log.info("Agent stopped cleanly.")


def _update_gather_hint(context: LLMContext, tool_handler: JLLToolHandler) -> None:
    """
    Inject a gather-state hint into the system message so the LLM always
    knows what has been collected and what to ask next.
    Mirrors GatherStateHint processor in bot__1_.py.
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
```

---

## Step 9 — Verify Your `.env` Has All Required Keys

Your final `.env` should look like this:

```env
# API Keys
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
CARTESIA_API_KEY=your_cartesia_key
CARTESIA_VOICE_ID=db6b0ed5-d5d3-463d-ae85-518a07d3c2b4

# Audio
SAMPLE_RATE=16000
CHANNELS=1

# VAD
VAD_AGGRESSIVENESS=2
SILENCE_THRESHOLD_MS=800

# LLM
LLM_MODEL=llama-3.3-70b-versatile
LLM_MAX_TOKENS=250
LLM_TEMPERATURE=0.4
MAX_HISTORY_TURNS=10

# JLL Integration
JLL_PROXY_URL=http://localhost:3000/api/integration
JLL_ASSISTANT_NAME=Priya

# Startup greeting
STARTUP_GREETING=Hello! I'm Priya from JLL Homes. I can help you find your perfect property. Which city are you looking in — Chennai, Bengaluru, or Hyderabad?

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=logs/agent.log
```

---

## Step 10 — Run the Agent

```powershell
.\venv\Scripts\python.exe main.py
```

Expected terminal output:

```
╔══════════════════════════════════════════════╗
║   🎙  Pipecat AI Voice Agent  🎙             ║
║   STT : Deepgram nova-2                      ║
║   LLM : Groq llama-3.3-70b-versatile         ║
║   TTS : Cartesia sonic-2                     ║
╚══════════════════════════════════════════════╝

INFO  INIT      Building pipeline components
INFO  STT       Initialising Deepgram nova-2
INFO  LLM       Initialising Groq model=llama-3.3-70b-versatile
INFO  TTS       Initialising Cartesia sonic-2
INFO  GREET     Playing JLL startup greeting
INFO  ✅ JLL Agent ready. Speak into the microphone.
```

After the greeting plays, speak naturally:

> "I'm looking for a 2BHK apartment in Chennai, around 80 lakhs, in OMR."

The agent will call `search_properties`, speak the results, and handle follow-ups.

---

## Important Notes

**About the JLL proxy URL:**
The `JLL_PROXY_URL` must point to a running instance of `integration__1_.js`. If you are not running it locally, you need to either:
- Start the Node.js integration server locally on port 3000, OR
- Point `JLL_PROXY_URL` directly at `https://jll-backend.ibism.com` and adjust the route paths in `jll_client.py` to match the JLL backend directly (instead of the proxy routes)

**About route paths:**
The routes in `jll_client.py` (`/proxy/search`, `/proxy/property-details`, `/proxy/areas-by-budget`) are the integration proxy routes from `integration__1_.js`. If the proxy is not running, you will need to replace these with the direct JLL backend routes found in `integration__1_.js` — look for `JLL_BASE + '/api/user/search/projects'` and similar.

**About `on_tool_call` event:**
The exact event name (`on_tool_call`) depends on which version of `pipecat-ai` is installed. Check the Pipecat docs or the `GroqLLMService` source for the correct event name if it throws. Common alternatives: `on_function_call`, `on_llm_function_call`.

**About Pipecat imports:**
Some imports (`LLMMessagesFrame`, `create_context_aggregator`) may vary by pipecat version. If you get import errors, check `pipecat-ai` changelog for your installed version (`pip show pipecat-ai`).

---

## Files Summary

| File | Action |
|---|---|
| `.env` | Add 3 new keys: `JLL_PROXY_URL`, `JLL_ASSISTANT_NAME`, `LLM_TEMPERATURE` |
| `config.py` | Add 2 new settings: `JLL_PROXY_URL`, `JLL_ASSISTANT_NAME`, `LLM_TEMPERATURE` |
| `requirements.txt` | Add `httpx>=0.27.0` |
| `pipeline/__init__.py` | Create empty file |
| `pipeline/prompts.py` | Create new — JLL system prompt |
| `pipeline/jll_client.py` | Create new — HTTP calls to JLL proxy |
| `pipeline/tools.py` | Create new — LLM tool schemas and handlers |
| `pipeline/agent.py` | Replace entirely — JLL-aware pipeline |
| `main.py` | No changes |
| `logger.py` | No changes |
