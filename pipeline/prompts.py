import os

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "system_prompt.txt")


def _load_system_prompt() -> str:
    """Load system prompt from system_prompt.txt next to the project root."""
    try:
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Minimal fallback if file is missing
        return (
            "You are {assistant_name}, a voice sales assistant for JLL Homes. "
            "Help users find properties in Chennai, Bengaluru, or Hyderabad."
        )


JLL_SYSTEM_PROMPT = _load_system_prompt()



GATHER_PHASE_PROMPT = """CURRENT SESSION STATE
You are in the requirement-gathering phase. You have collected:
{gathered_summary}

NEXT STEP: {next_question}
Ask only this one question. Do not search yet.
"""


def build_system_prompt(assistant_name: str) -> str:
    """Return the fully formatted JLL system prompt.

    Safe for prompts that do NOT contain {assistant_name} (e.g. the current
    system_prompt.txt which hardcodes 'Riya').
    """
    try:
        prompt = JLL_SYSTEM_PROMPT.format(assistant_name=assistant_name)
    except KeyError:
        prompt = JLL_SYSTEM_PROMPT  # no placeholder — use as-is

    # Tell the LLM how to handle the hidden startup trigger
    prompt += (
        "\n\nSTARTUP: When the user message is exactly '[BEGIN]', "
        "deliver the OPENING message defined above. Do not mention '[BEGIN]' to the caller."
    )
    return prompt


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
    if not city:
        next_q = "Which city are you looking in — Chennai, Bengaluru, or Hyderabad?"
    elif not prop_type:
        next_q = "What kind of property are you looking for — apartment, villa, or plot?"
    elif not (min_price or max_price):
        next_q = "What is your budget range?"
    elif not location:
        next_q = f"Which area in {city} are you interested in?"
    else:
        next_q = "I have all the details. Searching now."

    return GATHER_PHASE_PROMPT.format(
        gathered_summary=gathered_summary,
        next_question=next_q,
    )
