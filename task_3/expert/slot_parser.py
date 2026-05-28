import re
from difflib import SequenceMatcher, get_close_matches

from data_loader import load_stations
from nlp_utils import nlp
from task_3.expert.nlp_slots import (
    NlpExtraction,
    analyse_incident_text,
    extract_route_phrases,
    get_doc,
)
from task_3.expert.nlp_slots import extract_duration_minutes as nlp_extract_duration
from task_3.expert.nlp_slots import extract_event_type as nlp_extract_event_type
from task_3.expert.nlp_slots import extract_incident_time as nlp_extract_incident_time
from task_3.expert.nlp_slots import extract_info_topics as nlp_extract_info_topics
from task_3.expert.nlp_slots import extract_severity as nlp_extract_severity
from task_3.expert.nlp_slots import extract_staff_role as nlp_extract_staff_role
from task_3.expert.station_index import canonical_station_name

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
        "stations",
        "the station",
        "a station",
        "two stations",
        "2 stations",
        "two station",
        "2 station",
        "staff",
        "signaller",
        "signaler",
        "control",
        "station staff",
    }
)

_ROLE_WORDS = frozenset(
    {"staff", "signaller", "signaler", "control", "station staff"}
)

_stations_df = None
_station_names: list[str] | None = None
_passenger_displays: list[str] | None = None
_crs_to_display: dict[str, str] | None = None

_FUZZY_MIN_RATIO = 0.72
_MIN_FUZZY_QUERY_LEN = 5

ROLE_DEFAULT_EVENT_TYPE: dict[str, str] = {
    "station_staff": "station_disruption",
    "signaller": "line_blockage",
    "control": "line_blockage",
}


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
    if query in _VAGUE_STATION_PHRASES:
        return None
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


def _clean_station_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    phrase = re.sub(r"\s+(station|junction)$", "", phrase, flags=re.I)
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.I)
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
            chunk_text = chunk.text.strip()
            if len(chunk_text) < 5:
                continue
            name = _resolve_station_phrase(chunk_text)
            if name and name.lower() not in seen:
                seen.add(name.lower())
                found.append(name)

    return found


def _resolve_route_station(phrase: str) -> str | None:
    station = _resolve_station_phrase(phrase, min_len=4)
    if station:
        return friendly_station_label(station)
    return None


def _resolve_route_pair(
    from_phrase: str | None, to_phrase: str | None
) -> tuple[str | None, str | None]:
    if not from_phrase or not to_phrase:
        return None, None
    a = _resolve_route_station(from_phrase)
    b = _resolve_route_station(to_phrase)
    if a and b and a.lower() != b.lower():
        return a, b
    return None, None


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


def parse_line_endpoints(text: str) -> tuple[str | None, str | None]:
    from_phrase, to_phrase = extract_route_phrases(text)
    return _resolve_route_pair(from_phrase, to_phrase)


def parse_partial_from_station(text: str) -> str | None:
    doc = get_doc(text)
    for tok in doc:
        if tok.text.lower() != "from":
            continue
        for child in tok.children:
            if child.dep_ == "pobj":
                return resolve_station_name(_subtree_text(child), min_len=4)
        if tok.i + 1 < len(doc):
            phrase = doc[tok.i + 1 : tok.i + 6].text
            if " to " not in phrase.lower():
                return resolve_station_name(phrase.split(",")[0], min_len=4)
    return None


def _subtree_text(token) -> str:
    return " ".join(t.text for t in token.subtree if not t.is_punct).strip()


def parse_station_at_disruption(text: str) -> str | None:
    parsed = analyse_incident_text(text)
    if not parsed.mentions_disruption and not parsed.station_phrase:
        return None
    if parsed.station_phrase:
        return resolve_station_name(parsed.station_phrase, min_len=4)
    stations = extract_stations_nlp(text)
    if len(stations) == 1:
        return stations[0]
    return None


def mentions_blockage(text: str) -> bool:
    return analyse_incident_text(text).mentions_blockage


def parse_severity(text: str) -> str | None:
    return nlp_extract_severity(text)


def parse_event_type_answer(text: str) -> str | None:
    if parse_line_endpoints(text) != (None, None):
        return None
    parsed = analyse_incident_text(text)
    if parsed.two_station_answer:
        return "line_blockage"
    return nlp_extract_event_type(text)


def parse_station_correction(text: str) -> str | None:
    stripped = text.strip()
    if not stripped or len(stripped) > 35:
        return None
    if stripped.lower() in _ROLE_WORDS:
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


def parse_info_topics(text: str) -> list[str]:
    return nlp_extract_info_topics(text)


def parse_staff_role(text: str, *, pending_staff_slot: bool = False) -> str | None:
    return nlp_extract_staff_role(text, pending_staff_slot=pending_staff_slot)


def apply_role_event_defaults(state) -> bool:
    if not state.staff_role:
        return False
    default = ROLE_DEFAULT_EVENT_TYPE.get(state.staff_role)
    if not default:
        return False

    has_line_detail = bool(state.from_station or state.to_station)
    has_station_detail = bool(state.station)

    if state.event_type == default:
        return False

    if not state.event_type:
        state.event_type = default
        if default == "line_blockage":
            state.station = None
        return True

    if (
        default == "station_disruption"
        and not has_line_detail
        and not has_station_detail
    ):
        state.event_type = default
        state.from_station = None
        state.to_station = None
        return True

    if default == "line_blockage" and not has_station_detail and not has_line_detail:
        state.event_type = default
        state.station = None
        return True

    return False


def role_assumption_prefix(staff_role: str | None, event_type: str | None) -> str:
    if staff_role == "station_staff" and event_type == "station_disruption":
        return "As **station staff**, I'm treating this as a **station disruption**. "
    if staff_role == "signaller" and event_type == "line_blockage":
        return "As a **signaller**, I'm treating this as a **line blockage**. "
    if staff_role == "control" and event_type == "line_blockage":
        return "From **control**, I'm assuming a **line blockage** (say **station disruption** if not). "
    return ""


def parse_incident_time(text: str) -> str | None:
    return nlp_extract_incident_time(text)


def parse_duration_minutes(text: str) -> int | None:
    return nlp_extract_duration(text)


def parse_day_type(text: str) -> str | None:
    doc = get_doc(text)
    for tok in doc:
        low = tok.text.lower()
        if low in ("sunday", "sundays"):
            return "sunday"
        if low in (
            "weekday",
            "weekdays",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
        ):
            return "weekday"
    return None


def parse_service_period(text: str) -> str | None:
    lower = text.lower()
    if "off peak" in lower or "off-peak" in lower:
        return "off_peak"
    if "peak" in lower or "rush hour" in lower:
        return "peak"
    if re.search(r"\b([7-9])\s*(am)\b", lower) or re.search(
        r"\b([4-6])\s*(pm)\b", lower
    ):
        return "peak"
    return None


def _apply_slots_from_nlp(state, parsed: NlpExtraction) -> None:
    if parsed.mentions_blockage:
        state.mentions_blockage = True

    fs, ts = _resolve_route_pair(parsed.route_from, parsed.route_to)
    if fs and ts:
        state.from_station = fs
        state.to_station = ts
        state.event_type = "line_blockage"
        state.station = None
    elif fs:
        state.from_station = friendly_station_label(fs)
        state.event_type = state.event_type or "line_blockage"

    if parsed.event_type:
        state.event_type = parsed.event_type

    if parsed.station_phrase and state.event_type != "line_blockage":
        station = resolve_station_name(parsed.station_phrase, min_len=4)
        if station:
            state.station = friendly_station_label(station)
            state.event_type = "station_disruption"

    if state.event_type == "line_blockage":
        state.station = None

    if parsed.severity:
        state.severity = parsed.severity
    if parsed.staff_role:
        state.staff_role = parsed.staff_role
    if parsed.incident_time:
        state.incident_time = parsed.incident_time
    if parsed.duration_minutes is not None:
        state.duration_minutes = parsed.duration_minutes

    for topic in parsed.info_topics:
        if topic not in state.info_needed:
            state.info_needed.append(topic)


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
                if state.from_station and value.lower() == state.from_station.lower():
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
        value = parse_staff_role(text, pending_staff_slot=True)
        if value:
            state.staff_role = value
            state.pending_slot = None
            apply_role_event_defaults(state)
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
    if state.pending_slot and fill_pending_slot(state, text):
        return True

    parsed = analyse_incident_text(text)
    _apply_slots_from_nlp(state, parsed)

    apply_role_event_defaults(state)

    if (
        parsed.mentions_blockage
        and not state.event_type
        and (state.from_station or state.to_station)
    ):
        state.event_type = "line_blockage"

    if state.incident_time and not state.service_period:
        if state.incident_time in ("morning", "afternoon", "evening"):
            state.service_period = (
                "peak" if state.incident_time in ("morning", "evening") else "off_peak"
            )
        elif state.incident_time in ("night", "tonight"):
            state.service_period = "off_peak"

    day_type = parse_day_type(text)
    if day_type:
        state.day_type = day_type

    service_period = parse_service_period(text)
    if service_period:
        state.service_period = service_period

    if state.event_type == "station_disruption":
        correction = parse_station_correction(text)
        if correction:
            state.station = friendly_station_label(correction)

    return False
