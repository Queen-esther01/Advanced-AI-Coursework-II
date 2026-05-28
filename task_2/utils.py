import json
import re

import pandas as pd

WEY2WAT_STOP_CODES = [
    "WEY", "UPW", "DCH", "MTN", "WRM", "WOO", "HAM", "HOL", "PKS", "POO",
    "BSM", "BMH", "POK", "CHR", "BCU", "NWM", "SOU", "HNA", "SOA", "SWG",
    "WIN", "FRM", "SWY", "WOK", "HAV", "SDN", "HSL", "BEU", "KWB", "GLD",
    "ANF", "BSK", "WIM", "WBY", "TTN", "CLJ", "BFN", "RDB", "WPL", "HOK",
    "WYB", "ESL", "FNB", "MBK", "WNF", "FLE", "PTC", "WAL", "SUR", "SHW",
    "HER", "CSA", "MIC", "ESH", "BKO", "SNS", "EAD", "VXH", "WAT",
]

WAT2WEY_STOP_CODES = [
    "WAT", "VXH", "CLJ", "WOK", "WIM", "EFF", "VIR", "SNS", "BSK", "GLD",
    "WIN", "FNB", "SUR", "ESH", "HSL", "HAV", "SOA", "PTR", "HER", "WYB",
    "BKO", "FRM", "SOU", "FLE", "WAL", "MIC", "SHW", "ESL", "BCU", "BFN",
    "SWG", "HOK", "WNF", "WBY", "BMH", "NWM", "SDN", "BSM", "TTN", "MBK",
    "ANF", "RDB", "PKS", "CHR", "POK", "POO", "BEU", "HAM", "SWY", "WRM",
    "HOL", "HNA", "WOO", "DCH", "MTN", "UPW", "WEY",
]

_SUPPORTED_STOP_CODES = set(WEY2WAT_STOP_CODES) | set(WAT2WEY_STOP_CODES)

# Common passenger names that do not match the CSV literally (e.g. WATERLOO LONDON).
_STATION_ALIASES = {
    "waterloo": "WAT",
    "waterloo london": "WAT",
    "london waterloo": "WAT",
    "london": "WAT",
    "weymouth": "WEY",
    "wareham": "WRM",
    "hamworthy": "HAM",
}

_STATION_LOOKUP = None
_THOUGHT_TOKEN_PATTERNS = (
    re.compile(r"<\|channel\|?[^>]*>", re.IGNORECASE),
    re.compile(r"<\|?channel\|?>", re.IGNORECASE),
    re.compile(r"<\|[^|>]*\|>", re.IGNORECASE),
)

MAX_TOOL_ROUNDS = 5
DEFAULT_MINUTES_PER_STOP = 6.8

DELAY_ASSISTANT_SYSTEM_PROMPT = """
You are a helpful railway delay assistant for journeys from Weymouth (WEY) to
London Waterloo (WAT) and vice versa.

Your job is to guide the passenger through a short chat and collect the details
needed by a predictive delay model. Ask clear follow-up questions when
information is missing, including:
- the current station/location of the train
- the passenger's destination station
- the current delay in minutes
- the planned departure or arrival time at the current stop — use 24-hour
  time or am/pm (e.g. 17:55 or 5:55pm; bare '5:55' is treated as evening)

Do not ask for stops remaining, remaining journey time, or expected arrival
time at the destination. Those are calculated automatically by the tools.

Keep replies concise and conversational. Ask only one question at a time.
Never repeat the same question twice in one message, and do not re-ask for
details the passenger already gave in this chat.

Call check_station_coverage only once when the passenger first states where
they are and where they are going, or if they change stations. Do not call it
again for follow-up questions (e.g. "how long until I arrive?") if stations are
already known.

When coverage is confirmed and you have current delay and planned time at the
current stop, call get_train_delay with train_journey, current_location,
destination, current_delay, and planned_time_at_current_stop. Do not call
get_train_delay until you have both delay and planned time.

Use get_covered_stations only if the passenger asks which stations are supported.

After tools return, you must always send a clear text reply to the passenger.
Never end with only tool calls and no message.

After a tool returns a prediction, explain the expected delay clearly to the
passenger using the predicted_delay_minutes and reason fields. Do not invent
numbers that were not returned by a tool.

Do not ask for unnecessary fields like day_of_week.
""".strip()


def has_delay_inputs(journey_context):
    """True when journey context has the fields required for get_train_delay."""
    if journey_context.get("current_delay") is None:
        return False
    return bool(journey_context.get("planned_time_at_current_stop"))


def build_system_prompt(journey_context):
    """Build system prompt, including remembered journey details when set."""
    prompt = DELAY_ASSISTANT_SYSTEM_PROMPT
    if not journey_context.get("coverage_confirmed"):
        return prompt

    lines = [
        "",
        "Confirmed journey for this chat (do not call check_station_coverage again",
        "unless the passenger changes stations):",
        f"- train_journey: {journey_context.get('train_journey', '?')}",
        f"- current_location: {journey_context.get('current_location', '?')} "
        f"({journey_context.get('current_station_code', '?')})",
        f"- destination: {journey_context.get('destination', '?')} "
        f"({journey_context.get('destination_station_code', '?')})",
    ]
    if journey_context.get("current_delay") is not None:
        lines.append(
            f"- current_delay (minutes): {journey_context['current_delay']}"
        )
    if journey_context.get("planned_time_at_current_stop"):
        lines.append(
            "- planned_time_at_current_stop: "
            f"{journey_context['planned_time_at_current_stop']}"
        )
    lines.append(
        "For arrival or delay questions, call get_train_delay directly using "
        "these details (update delay or time if the passenger gives new values)."
    )
    return prompt + "\n".join(lines)


def update_journey_context_from_tool(journey_context, tool_name, arguments_json, result):
    """Remember journey details from tool results for later turns in this chat."""
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        arguments = {}

    if tool_name == "check_station_coverage" and result.get("covered"):
        journey_context["coverage_confirmed"] = True
        journey_context["train_journey"] = result.get("journey")
        journey_context["current_location"] = arguments.get("current_location", "")
        journey_context["destination"] = arguments.get("destination", "")
        journey_context["current_station_code"] = result.get("current_station_code")
        journey_context["destination_station_code"] = result.get(
            "destination_station_code"
        )

    if tool_name == "get_train_delay" and "error" not in result:
        journey_context["coverage_confirmed"] = True
        journey_context["train_journey"] = result.get("train_journey")
        journey_context["current_location"] = result.get("current_location")
        journey_context["destination"] = result.get("destination")
        journey_context["current_station_code"] = result.get("current_station_code")
        journey_context["destination_station_code"] = result.get(
            "destination_station_code"
        )
        journey_context["current_delay"] = arguments.get("current_delay")
        journey_context["planned_time_at_current_stop"] = arguments.get(
            "planned_time_at_current_stop"
        )


def get_stops_from_journey(journey):
    """
    Looks up the ordered stop list for a WEY↔WAT journey and formats each
    station as a readable name with CRS code. Returns a list of strings such as
    ['Weymouth (WEY)', 'Upwey (UPW)', ...], or an empty list for unknown
    journeys. Required for the get_covered_stations tool and route coverage
    explanations in the delay assistant.
    """
    journey_key = (journey or "").strip().upper()
    if journey_key == "WEY2WAT":
        stop_codes = WEY2WAT_STOP_CODES
    elif journey_key == "WAT2WEY":
        stop_codes = WAT2WEY_STOP_CODES
    else:
        return []

    station_lookup = {}
    try:
        station_df = pd.read_csv("data/StationNameAndCode.csv", usecols=["NAME", "CRS"])
        station_lookup = {
            str(row["CRS"]).strip().upper(): str(row["NAME"]).strip().title()
            for _, row in station_df.iterrows()
        }
    except Exception:
        station_lookup = {}

    return [f"{station_lookup.get(code, code)} ({code})" for code in stop_codes]


def clean_model_text(text):
    """
    Removes leaked model markup such as <|channel>thought tokens from streamed
    LLM output. Returns cleaned plain text safe to show in the chat UI. Required
    for filtering Gemma/reasoning-style artefacts before they reach the user.
    """
    if not text:
        return ""
    cleaned = text
    for pattern in _THOUGHT_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _load_station_lookup():
    """
    Loads station CRS codes and names from StationNameAndCode.csv into cached
    lookup dictionaries keyed by code and normalised name. Returns
    {'by_code': {...}, 'by_name': {...}}. Required for resolving passenger
    station names to CRS codes during coverage checks.
    """
    global _STATION_LOOKUP
    if _STATION_LOOKUP is not None:
        return _STATION_LOOKUP

    lookup = {"by_code": {}, "by_name": {}}
    try:
        station_df = pd.read_csv("data/StationNameAndCode.csv", usecols=["NAME", "CRS"])
        for _, row in station_df.iterrows():
            code = str(row["CRS"]).strip().upper()
            name = str(row["NAME"]).strip()
            lookup["by_code"][code] = name
            lookup["by_name"][re.sub(r"\s+", " ", name.lower())] = code
    except Exception:
        pass

    _STATION_LOOKUP = lookup
    return lookup


def _normalise_station_text(value):
    """
    Lowercases and strips punctuation from a station name or free-text location
    so it can be matched against the station CSV. Returns a normalised string.
    Required for fuzzy station name matching in _resolve_station_code.
    """
    cleaned = re.sub(r"[^a-z0-9& ]+", " ", (value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _score_station_name_match(normalised_query, station_name):
    """
    Scores how well a normalised query matches a CSV station name. Lower is
    better. Returns None when there is no reasonable match.
    """
    if normalised_query == station_name:
        return (0, -len(station_name))

    query_words = normalised_query.split()
    name_words = station_name.split()
    if query_words and all(word in name_words for word in query_words):
        return (1, -len(station_name))

    if normalised_query in station_name:
        return (2, -len(station_name))

    if station_name in normalised_query:
        return (3, -len(station_name))

    return None


def _resolve_station_code(station_text):
    """
    Maps a station name or CRS code provided by the user to a three-letter CRS
    code using aliases, exact CSV names, and fuzzy matching. Returns the CRS
    code string, or None if no match is found. Required for check_station_coverage
    and journey direction detection.
    """
    station_text = (station_text or "").strip()
    if not station_text:
        return None

    lookup = _load_station_lookup()
    upper = station_text.upper()
    if upper in lookup["by_code"]:
        return upper

    normalised = _normalise_station_text(station_text)
    if normalised in _STATION_ALIASES:
        return _STATION_ALIASES[normalised]

    if normalised in lookup["by_name"]:
        return lookup["by_name"][normalised]

    best_match = None
    for name, code in lookup["by_name"].items():
        score = _score_station_name_match(normalised, name)
        if score is None:
            continue

        # Prefer stations on the supported WEY↔WAT route over other UK stations.
        if code in _SUPPORTED_STOP_CODES:
            score = (score[0] - 1, score[1])

        if best_match is None or score < best_match[0]:
            best_match = (score, code)

    if best_match is not None:
        return best_match[1]

    return None


def resolve_station_code(station_text):
    """
    Public wrapper for _resolve_station_code. Maps a station name or CRS code
    to a three-letter CRS code. Returns the code string or None.
    Required for delay model inference and route calculations in delay_tool.
    """
    return _resolve_station_code(station_text)


def compute_stops_remaining(journey, current_location, destination):
    """
    Returns median historical stops remaining for this journey segment, falling
    back to the geographic route list when no historical pair exists. Required
    for building stops_remaining model input in get_train_delay.
    """
    from task_2.historical_lookup import lookup_journey_metrics

    return lookup_journey_metrics(journey, current_location, destination)[
        "stops_remaining"
    ]


def compute_remaining_minutes(journey, current_location, destination):
    """
    Returns median historical remaining journey minutes for this segment, falling
    back to a stops-based estimate when no historical pair exists. Required for
    building remaining_minutes model input in get_train_delay.
    """
    from task_2.historical_lookup import lookup_journey_metrics

    return lookup_journey_metrics(journey, current_location, destination)[
        "remaining_minutes"
    ]


def _journey_for_stations(current_code, destination_code):
    """
    Determines whether two CRS codes lie on the same supported WEY↔WAT route
    with the current stop before the destination. Returns 'WEY2WAT', 'WAT2WEY',
    or None if the pair is invalid or in the wrong direction. Required for
    check_station_coverage before calling the delay prediction tool.
    """
    for journey, stop_codes in (
        ("WEY2WAT", WEY2WAT_STOP_CODES),
        ("WAT2WEY", WAT2WEY_STOP_CODES),
    ):
        if current_code not in stop_codes or destination_code not in stop_codes:
            continue
        if stop_codes.index(current_code) < stop_codes.index(destination_code):
            return journey
    return None


def check_station_coverage(current_location, destination):
    """
    Validates that the passenger's current location and destination are on the
    supported Weymouth ↔ Waterloo route and in a valid direction. Returns a
    dict with covered (bool), journey ('WEY2WAT'/'WAT2WEY'/None), station codes,
    and a human-readable message. Required for the check_station_coverage LLM
    tool before requesting a delay prediction.
    """
    current_code = _resolve_station_code(current_location)
    destination_code = _resolve_station_code(destination)

    if not current_code or not destination_code:
        return {
            "covered": False,
            "journey": None,
            "message": (
                "I could not recognise one or both stations. This assistant only "
                "covers the Weymouth ↔ Waterloo route."
            ),
        }

    if (
        current_code not in _SUPPORTED_STOP_CODES
        or destination_code not in _SUPPORTED_STOP_CODES
    ):
        return {
            "covered": False,
            "journey": None,
            "current_station_code": current_code,
            "destination_station_code": destination_code,
            "message": (
                f"{current_location} ({current_code}) and/or {destination} "
                f"({destination_code}) are outside the supported WEY↔WAT route."
            ),
        }

    journey = _journey_for_stations(current_code, destination_code)
    if not journey:
        return {
            "covered": False,
            "journey": None,
            "current_station_code": current_code,
            "destination_station_code": destination_code,
            "message": (
                "Those stations are on the route, but not in the correct direction "
                "for this journey."
            ),
        }

    lookup = _load_station_lookup()
    current_name = lookup["by_code"].get(current_code, current_code)
    destination_name = lookup["by_code"].get(destination_code, destination_code)
    return {
        "covered": True,
        "journey": journey,
        "current_station_code": current_code,
        "destination_station_code": destination_code,
        "message": (
            f"Covered journey: {journey} from {current_name} ({current_code}) "
            f"to {destination_name} ({destination_code})."
        ),
    }


def merge_stream_tool_calls(accumulator, tool_call_deltas):
    """
    Merges incremental tool-call fragments from a streaming OpenRouter response
    into a single accumulator dict keyed by tool-call index. Returns nothing;
    updates accumulator in place. Required for reconstructing complete tool
    calls before they can be executed in the LLM tool loop.
    """
    for tool_call in tool_call_deltas:
        index = tool_call.index
        if index not in accumulator:
            accumulator[index] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }

        if tool_call.id:
            accumulator[index]["id"] = tool_call.id
        if tool_call.type:
            accumulator[index]["type"] = tool_call.type
        if tool_call.function:
            if tool_call.function.name:
                accumulator[index]["function"]["name"] += tool_call.function.name
            if tool_call.function.arguments:
                accumulator[index]["function"]["arguments"] += (
                    tool_call.function.arguments
                )


def tool_calls_from_accumulator(accumulator):
    """
    Converts the merged streaming tool-call accumulator into a list of complete
    tool-call dicts ready for execution. Returns an empty list when no valid
    tool calls were received. Required for deciding when to run tools and
    append results back into the LLM conversation.
    """
    if not accumulator:
        return []

    tool_calls = []
    for index in sorted(accumulator):
        tool_call = accumulator[index]
        if not tool_call["function"]["name"]:
            continue
        tool_calls.append(tool_call)
    return tool_calls


def execute_tool_call(tool_name, arguments_json):
    """
    Dispatches an LLM tool request to the matching Python handler, parses the
    JSON arguments string, and returns the tool result as a dict. Returns an
    error dict if arguments are invalid or the tool is unknown. Required for
    the tool-calling loop in main.py to run get_train_delay, check_station_coverage,
    and get_covered_stations.
    """
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return {"error": "Invalid tool arguments"}

    try:
        if tool_name == "get_train_delay":
            from task_2.delay_tool import get_train_delay

            return get_train_delay(
                train_journey=arguments.get("train_journey", ""),
                current_location=arguments.get("current_location", ""),
                destination=arguments.get("destination", ""),
                current_delay=arguments.get("current_delay"),
                planned_time_at_current_stop=arguments.get(
                    "planned_time_at_current_stop", ""
                ),
            )
        if tool_name == "check_station_coverage":
            return check_station_coverage(
                arguments.get("current_location", ""),
                arguments.get("destination", ""),
            )
        if tool_name == "get_covered_stations":
            journey = arguments.get("journey", "")
            return {"journey": journey, "stations": get_stops_from_journey(journey)}
        return {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        return {"error": str(exc)}
