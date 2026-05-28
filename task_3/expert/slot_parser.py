import re

from difflib import SequenceMatcher, get_close_matches

from data_loader import load_stations
from expert.station_index import canonical_station_name
from nlp_utils import nlp

_VAGUE_STATION_PHRASES = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "my",
        "our",
        "there",
        "here",
        "station",
        "the station",
        "a station",
    }
)

_stations_df = None
_station_names: list[str] | None = None
_passenger_displays: list[str] | None = None
_crs_to_display: dict[str, str] | None = None

_FUZZY_MIN_RATIO = 0.72
_MIN_FUZZY_QUERY_LEN = 5


def _station_names_list() -> list[str]:
    global _stations_df, _station_names
    if _station_names is None:
        _stations_df = load_stations()
        _station_names = _stations_df["NAME"].tolist()
    return _station_names


def _sorted_tokens(phrase: str) -> str:
    return " ".join(sorted(re.split(r"[\s\-]+", phrase.lower())))


def _is_passenger_station(name: str) -> bool:
    nl = name.lower()
    skip = (
        "shop",
        "depot",
        "store",
        "siding",
        "yard",
        "works",
        "tunnel",
        "dep ",
        "ctl",
        "tq",
        "cre",
        "sigs",
        "dpl",
    )
    if any(term in nl for term in skip):
        return False
    if any(ch.isdigit() for ch in name):
        return False
    return True


def _passenger_display_list() -> list[str]:
    global _passenger_displays
    if _passenger_displays is not None:
        return _passenger_displays

    seen: set[str] = set()
    displays: list[str] = []
    for name in _station_names_list():
        if not _is_passenger_station(name):
            continue
        display = _display_station_name(name)
        key = display.lower()
        if key not in seen:
            seen.add(key)
            displays.append(display)

    _passenger_displays = displays
    return displays


def _token_similarity(query: str, token: str) -> float:
    return SequenceMatcher(None, query, token).ratio()


def _fuzzy_station_match(phrase: str) -> str | None:
    query = phrase.lower().strip()
    if len(query) < _MIN_FUZZY_QUERY_LEN:
        return None

    candidates: list[tuple[float, str, bool, int]] = []
    for display in _passenger_display_list():
        tokens = [t for t in re.split(r"[\s\-]+", display.lower()) if len(t) >= 4]
        if not tokens:
            continue
        best_token = max(tokens, key=lambda t: _token_similarity(query, t))
        score = _token_similarity(query, best_token)
        dl = display.lower()
        score = max(
            score,
            SequenceMatcher(None, query, dl).ratio(),
            SequenceMatcher(None, _sorted_tokens(query), _sorted_tokens(dl)).ratio(),
        )
        if score < _FUZZY_MIN_RATIO:
            continue
        first_token_match = tokens[0] == best_token
        candidates.append((score, display, first_token_match, len(display)))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (-c[0], not c[2], c[3]))
    top_score = candidates[0][0]
    top = [c for c in candidates if top_score - c[0] <= 0.02]
    preferred = [c for c in top if c[2]] or top
    preferred.sort(key=lambda c: (c[3], -c[0]))
    return preferred[0][1]


def _crs_lookup(phrase: str) -> str | None:
    global _crs_to_display
    if _crs_to_display is None:
        df = load_stations()
        mapping: dict[str, str] = {}
        for name, crs in zip(df["NAME"], df["CRS"]):
            code = str(crs).strip().upper()
            if len(code) == 3 and code.isalpha():
                mapping[code] = friendly_station_label(_display_station_name(str(name)))
        _crs_to_display = mapping

    code = phrase.strip().upper()
    if len(code) == 3 and code.isalpha():
        return _crs_to_display.get(code)
    return None


def _lookup_station_csv(phrase: str) -> str | None:
    crs_station = _crs_lookup(phrase)
    if crs_station:
        return crs_station

    names = _station_names_list()
    lower = phrase.lower().strip()
    if lower in ("waterloo", "waterloo lt"):
        wat = _crs_lookup("WAT")
        if wat:
            return wat
    for name in names:
        if name.lower() == lower:
            return _display_station_name(name)

    starts = [n for n in names if n.lower().startswith(lower)]
    if len(starts) == 1:
        return _display_station_name(starts[0])

    sorted_query = _sorted_tokens(lower)
    if sorted_query != lower:
        for name in names:
            if _sorted_tokens(_display_station_name(name)) == sorted_query:
                return _display_station_name(name)

    contains = [n for n in names if lower in n.lower()]
    if len(contains) == 1:
        return _display_station_name(contains[0])

    matches = get_close_matches(lower, [n.lower() for n in names], n=1, cutoff=0.86)
    if matches:
        idx = [n.lower() for n in names].index(matches[0])
        return _display_station_name(names[idx])

    return _fuzzy_station_match(phrase)


LINE_BETWEEN = re.compile(
    r"(?:block(?:age|ed)?|blocked\s+line)?\s*between\s+(.+?)\s+and\s+(.+?)(?:\.|,|$|\s+with|\s+one|\s+both)",
    re.I,
)
LINE_FROM_TO = re.compile(
    r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\.|,|$|\s+is|\s+are|\s+with|\s+one|\s+both)",
    re.I,
)
BLOCKAGE_FROM = re.compile(
    r"(?:block(?:age|ed)?|blocked|blockade)(?:\s+\w+){0,6}?\s+from\s+(.+?)(?:\s+to\s+|\s*$|\.|,)",
    re.I,
)
STATION_AT = re.compile(
    r"(?:disruption|incident|problem)\s+at\s+(.+?)(?:\s+station)?(?=\s|$|\.|,)",
    re.I,
)

BLOCKAGE_MENTION = re.compile(r"\b(blockage|blocked|blockade)\b", re.I)
EVENT_LINE = re.compile(
    r"\b(line\s+block(?:age|ed)?|blocked\s+line|blockage\s+between|blockade)\b",
    re.I,
)
EVENT_STATION = re.compile(
    r"\b(station\s+disruption|disruption\s+at|incident\s+at|problem\s+at)\b",
    re.I,
)

SEVERITY_WORDS_FULL = frozenset(
    {"full", "complete", "both", "all", "entire", "total", "closed"}
)
SEVERITY_WORDS_PARTIAL = frozenset({"partial", "one", "single"})

INFO_TOPIC_KEYWORDS = {
    "staff": ("signaller", "signaler", "station staff", "staff", "crew", "divert", "terminate"),
    "passengers": ("passenger", "bus replacement", "replacement bus", "taxi", "customer"),
    "contacts": ("contact", "phone", "number", "call"),
    "routes": ("alternative route", "diversion", "detour", "via "),
}

ROLE_KEYWORDS = {
    "signaller": ("signaller", "signaler", "signalling", "signal"),
    "station_staff": ("station staff", "platform", "dispatcher", "gateline", "staff"),
    "control": ("control", "controller", "operations control"),
}

TIME_PATTERNS = (
    re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b"),
    re.compile(r"\b([1-9]|1[0-2])\s*(am|pm)\b", re.I),
)

DURATION_PATTERN = re.compile(
    r"\b(?:(about|around|approx(?:imately)?)\s+)?(\d{1,3})\s*(hours?|hrs?|h|minutes?|mins?|m)\b",
    re.I,
)


def _clean_station_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    phrase = re.sub(r"\s+(station|junction)$", "", phrase, flags=re.I)
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.I)
    phrase = re.sub(
        r"\b(full|partial|blockage|blocked|blockade|complete|both|lines?)\b",
        "",
        phrase,
        flags=re.I,
    )
    return " ".join(phrase.split())


def _display_station_name(official: str) -> str:
    lower = official.lower()
    for suffix in (" station", " junction"):
        if lower.endswith(suffix):
            return official[: -len(suffix)].title()
    return official.title()


def resolve_station_name(phrase: str, *, min_len: int = 3) -> str | None:
    return _resolve_station_phrase(phrase, min_len=max(min_len, _MIN_FUZZY_QUERY_LEN))


def _resolve_station_phrase(phrase: str, *, min_len: int = 5) -> str | None:
    stripped = phrase.strip()
    if stripped.lower() in _VAGUE_STATION_PHRASES:
        return None
    crs_station = _crs_lookup(stripped)
    if crs_station:
        return crs_station
    cleaned = _clean_station_phrase(phrase)
    if cleaned.lower() in _VAGUE_STATION_PHRASES:
        return None
    if len(cleaned) < min_len:
        return None
    return _lookup_station_csv(cleaned)


def extract_stations_nlp(text: str) -> list[str]:
    doc = nlp(text)
    found: list[str] = []
    seen: set[str] = set()

    for ent in doc.ents:
        if ent.label_ not in ("GPE", "FAC", "LOC"):
            continue
        name = _resolve_station_phrase(ent.text, min_len=4)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            found.append(name)

    if not found:
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if len(text) < 5:
                continue
            name = _resolve_station_phrase(text)
            if name and name.lower() not in seen:
                seen.add(name.lower())
                found.append(name)

    return found


def parse_severity(text: str) -> str | None:
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", " ", t)
    words = set(t.split())

    if words & SEVERITY_WORDS_PARTIAL and not words & SEVERITY_WORDS_FULL:
        return "one_line_blocked"
    if words & SEVERITY_WORDS_FULL or t in ("full blockage", "both lines blocked"):
        return "both_lines_blocked"
    if re.search(r"\b(both\s+lines?\s+blocked|full\s+block|complete\s+block)\b", t):
        return "both_lines_blocked"
    if re.search(r"\b(one\s+line\s+blocked|partial|single\s+line)\b", t):
        return "one_line_blocked"
    if t == "full" or t.startswith("full "):
        return "both_lines_blocked"
    if t == "partial" or t.startswith("partial"):
        return "one_line_blocked"
    return None


def parse_event_type_answer(text: str) -> str | None:
    if parse_line_endpoints(text) != (None, None):
        return None
    lower = text.strip().lower()
    if re.search(
        r"\b(station\s+disruption|statuin\s+disruption|tation\s+disruption|"
        r"\w+ation\s+disruption|at\s+one\s+station|single\s+station)\b",
        lower,
    ):
        return "station_disruption"
    if re.search(r"\b(line\s+block|between\s+two|two\s+stations)\b", lower):
        return "line_blockage"
    if "station" in lower.split() and "line" not in lower:
        return "station_disruption"
    if "line" in lower or "between" in lower:
        return "line_blockage"
    return None


def friendly_station_label(name: str) -> str:
    lower = name.lower()
    if "waterloo" in lower and "merseyside" not in lower and "east" not in lower:
        return "London Waterloo"
    if lower in ("london br", "london bridge"):
        return "London"
    if lower.startswith("london ") and "waterloo" not in lower:
        return "London"
    if "weymouth" in lower:
        return "Weymouth"
    return canonical_station_name(name)


def parse_station_correction(text: str) -> str | None:
    stripped = text.strip()
    if not stripped or len(stripped) > 35:
        return None
    if parse_event_type_answer(text) or parse_severity(text):
        return None
    return parse_single_station_answer(text)


def parse_single_station_answer(text: str) -> str | None:
    if parse_severity(text) or parse_event_type_answer(text):
        return None

    raw = text.strip()
    raw = re.sub(r"^(to|from|at)\s+", "", raw, flags=re.I)

    crs_station = _crs_lookup(raw)
    if crs_station:
        return crs_station

    raw = _clean_station_phrase(raw)
    if not raw:
        return None
    if len(raw) < 5:
        return None

    station = resolve_station_name(raw, min_len=5)
    if station:
        return station

    nlp_stations = extract_stations_nlp(text)
    if len(nlp_stations) == 1:
        return nlp_stations[0]
    return None


def _resolve_route_station(phrase: str) -> str | None:
    station = _resolve_station_phrase(phrase, min_len=4)
    if station:
        return friendly_station_label(station)
    return None


def parse_line_endpoints(text: str) -> tuple[str | None, str | None]:
    for pattern in (LINE_FROM_TO, LINE_BETWEEN):
        match = pattern.search(text)
        if match:
            a = _resolve_route_station(match.group(1))
            b = _resolve_route_station(match.group(2))
            if a and b and a.lower() != b.lower():
                return a, b

    doc = nlp(text)
    from_idx = to_idx = None
    for i, token in enumerate(doc):
        low = token.text.lower()
        if low == "from" and from_idx is None:
            from_idx = i
        elif low == "to" and to_idx is None and from_idx is not None:
            to_idx = i

    if from_idx is not None and to_idx is not None and to_idx > from_idx:
        from_span = doc[from_idx + 1 : to_idx].text.strip()
        to_span = doc[to_idx + 1 :].text.strip()
        to_span = re.split(r"\s+(?:and|with|is|are|both|one|full|partial)\b", to_span, 1, flags=re.I)[0]
        a = _resolve_route_station(from_span)
        b = _resolve_route_station(to_span)
        if a and b and a.lower() != b.lower():
            return a, b

    match = re.search(r"\bto\s+([a-z][a-z\s'-]+?)(?:\s*$|\.|,)", text, re.I)
    if match and BLOCKAGE_MENTION.search(text):
        fs = parse_partial_from_station(text)
        ts = _resolve_route_station(match.group(1))
        if fs and ts:
            return fs, ts
    return None, None


def parse_partial_from_station(text: str) -> str | None:
    match = BLOCKAGE_FROM.search(text)
    if match:
        return resolve_station_name(match.group(1), min_len=4)

    if not BLOCKAGE_MENTION.search(text):
        return None

    match = re.search(
        r"\bfrom\s+([a-z][a-z\s'-]+?)(?:\s+to\s+|\s*$|\.|,)",
        text,
        re.I,
    )
    if match:
        return resolve_station_name(match.group(1), min_len=4)
    return None


def parse_station_at_disruption(text: str) -> str | None:
    if not EVENT_STATION.search(text) and "disruption" not in text.lower():
        return None
    if re.search(r"\bdisruption\s+at\s+the\s+station\b", text, re.I):
        return None
    match = STATION_AT.search(text)
    if match:
        phrase = match.group(1).strip()
        if phrase.lower() in _VAGUE_STATION_PHRASES:
            return None
        return resolve_station_name(phrase, min_len=4)
    if EVENT_STATION.search(text):
        stations = extract_stations_nlp(text)
        if len(stations) == 1:
            return stations[0]
    return None


def mentions_blockage(text: str) -> bool:
    return bool(BLOCKAGE_MENTION.search(text))


def parse_info_topics(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for topic, keywords in INFO_TOPIC_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.append(topic)
    return found


def parse_staff_role(text: str) -> str | None:
    lower = text.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return role
    return None


def parse_incident_time(text: str) -> str | None:
    lower = text.lower()
    for pattern in TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    for label in ("morning", "afternoon", "evening", "night", "tonight", "now"):
        if re.search(rf"\b{label}\b", lower):
            return label
    return None


def parse_duration_minutes(text: str) -> int | None:
    match = DURATION_PATTERN.search(text)
    if not match:
        return None
    amount = int(match.group(2))
    unit = match.group(3).lower()
    if unit.startswith(("hour", "hr", "h")):
        return amount * 60
    return amount


def parse_day_type(text: str) -> str | None:
    lower = text.lower()
    if re.search(r"\b(sunday|sundays)\b", lower):
        return "sunday"
    if re.search(r"\b(weekday|weekdays|monday|tuesday|wednesday|thursday|friday)\b", lower):
        return "weekday"
    return None


def parse_service_period(text: str) -> str | None:
    lower = text.lower()
    if "off peak" in lower or "off-peak" in lower:
        return "off_peak"
    if "peak" in lower or "rush hour" in lower:
        return "peak"
    if re.search(r"\b([7-9])\s*(am)\b", lower) or re.search(r"\b([4-6])\s*(pm)\b", lower):
        return "peak"
    return None


def _apply_route_endpoints(state, fs: str | None, ts: str | None) -> bool:
    if fs and ts:
        state.from_station = friendly_station_label(fs)
        state.to_station = friendly_station_label(ts)
        state.event_type = "line_blockage"
        state.station = None
        state.pending_slot = None
        return True
    return False


def fill_pending_slot(state, text: str) -> bool:
    slot = state.pending_slot
    if not slot:
        return False

    fs, ts = parse_line_endpoints(text)
    if _apply_route_endpoints(state, fs, ts):
        return True

    if slot == "event_type":
        value = parse_event_type_answer(text)
        if value:
            state.event_type = value
            state.pending_slot = None
            return True
        if st := parse_single_station_answer(text):
            state.from_station = friendly_station_label(st)
            state.event_type = "line_blockage"
            state.station = None
            state.pending_slot = "to_station"
            return False

    elif slot in ("from_station", "to_station", "station"):
        value = parse_single_station_answer(text)
        if value:
            if slot == "from_station":
                state.from_station = friendly_station_label(value)
                if state.event_type == "line_blockage":
                    state.station = None
            elif slot == "to_station":
                if (
                    state.from_station
                    and value.lower() == state.from_station.lower()
                ):
                    return False
                state.to_station = friendly_station_label(value)
            else:
                state.station = value
                state.event_type = "station_disruption"
            state.pending_slot = None
            return True

    elif slot == "severity":
        value = parse_severity(text)
        if value:
            state.severity = value
            state.pending_slot = None
            return True
    elif slot == "staff_role":
        value = parse_staff_role(text)
        if value:
            state.staff_role = value
            state.pending_slot = None
            return True
    elif slot == "incident_time":
        value = parse_incident_time(text)
        if value:
            state.incident_time = value
            state.pending_slot = None
            return True
    elif slot == "duration_minutes":
        value = parse_duration_minutes(text)
        if value is not None:
            state.duration_minutes = value
            state.pending_slot = None
            return True

    return False


def apply_message_slots(state, text: str) -> bool:
    if mentions_blockage(text):
        state.mentions_blockage = True

    fs, ts = parse_line_endpoints(text)
    if _apply_route_endpoints(state, fs, ts):
        return True

    if state.pending_slot and fill_pending_slot(state, text):
        return True

    partial_from = parse_partial_from_station(text)
    station_at = parse_station_at_disruption(text)
    if re.search(r"\bdisruption\s+at\s+(?:the\s+)?station\b", text, re.I):
        state.event_type = state.event_type or "station_disruption"

    if fs and not ts:
        state.from_station = friendly_station_label(fs)
        state.event_type = state.event_type or "line_blockage"
    elif ts and not fs:
        state.to_station = friendly_station_label(ts)
        state.event_type = state.event_type or "line_blockage"
    elif partial_from:
        if not state.from_station:
            state.from_station = friendly_station_label(partial_from)
        state.event_type = state.event_type or "line_blockage"

    event_answer = parse_event_type_answer(text)
    if event_answer:
        state.event_type = event_answer

    if station_at and state.event_type != "line_blockage":
        state.station = station_at
        state.event_type = "station_disruption"

    if state.event_type == "line_blockage":
        state.station = None

    severity = parse_severity(text)
    if severity:
        state.severity = severity

    role = parse_staff_role(text)
    if role:
        state.staff_role = role

    incident_time = parse_incident_time(text)
    if incident_time:
        state.incident_time = incident_time

    duration = parse_duration_minutes(text)
    if duration is not None:
        state.duration_minutes = duration

    day_type = parse_day_type(text)
    if day_type:
        state.day_type = day_type

    service_period = parse_service_period(text)
    if service_period:
        state.service_period = service_period

    if state.incident_time and not state.service_period:
        if state.incident_time in ("morning", "afternoon", "evening"):
            state.service_period = "peak" if state.incident_time in ("morning", "evening") else "off_peak"
        elif state.incident_time in ("night", "tonight"):
            state.service_period = "off_peak"

    for topic in parse_info_topics(text):
        if topic not in state.info_needed:
            state.info_needed.append(topic)

    if state.mentions_blockage and not state.event_type:
        if partial_from or fs or ts:
            state.event_type = "line_blockage"

    if state.event_type == "station_disruption":
        correction = parse_station_correction(text)
        if correction:
            state.station = friendly_station_label(correction)

    return False
