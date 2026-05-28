from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llm_client import LLMClient
from task_3.expert.facts import Incident, WantsInfo
from task_3.expert.incident_engine import IncidentEngine
from task_3.expert.plan_retriever import image_paths_from_chunks, retrieve_plans
from task_3.expert.response_builder import build_disruption_reply
from task_3.expert.slot_parser import (
    apply_message_slots,
    friendly_station_label,
    mentions_blockage,
    parse_line_endpoints,
    parse_single_station_answer,
    parse_station_correction,
)
from task_3.expert.station_index import INDEXED_STATION_HINT
from task_3.indexing.vector_store import VectorStore

ProgressCallback = Callable[[str], None] | None
INCIDENT_KEYWORDS = (
    "blockage",
    "blocked",
    "blockade",
    "disruption",
    "contingency",
    "incident",
    "diversion",
    "signaller",
    "signaler",
    "contingency plan",
    "station disruption",
    "line blocked",
    "both lines",
    "replacement bus",
    "staff plan",
)

UNRECOGNISED_STATION = (
    "I didn't recognise that station name. Use the full name (e.g. **Norwich**, "
    "**London Waterloo**) or a **3-letter CRS code** (e.g. **WAT**, **WEY**)."
)

_PASSENGER_BOOKING_HINTS = (
    "ticket",
    "tickets",
    "cheapest",
    "cheap fare",
    "fare",
    "book a",
    "book ",
    "buy a",
    "buy ",
    "need a ticket",
    "want a ticket",
    "get a ticket",
    "travel from",
    "travelling from",
    "going from",
    "journey from",
)


@dataclass
class IncidentState:
    stage: str = "idle"
    event_type: str | None = None
    from_station: str | None = None
    to_station: str | None = None
    station: str | None = None
    severity: str | None = None
    staff_role: str | None = None
    incident_time: str | None = None
    day_type: str | None = None
    service_period: str | None = None
    duration_minutes: int | None = None
    derived_actions: list[str] = field(default_factory=list)
    info_needed: list[str] = field(default_factory=list)
    mentions_blockage: bool = False
    pending_slot: str | None = None


def _is_passenger_booking_intent(user_input: str) -> bool:
    lower = user_input.lower()
    return any(hint in lower for hint in _PASSENGER_BOOKING_HINTS)


def is_incident_intent(user_input: str) -> bool:
    lower = user_input.lower()
    if _is_passenger_booking_intent(user_input):
        return False
    if any(kw in lower for kw in INCIDENT_KEYWORDS):
        return True
    if mentions_blockage(user_input):
        return True
    if parse_line_endpoints(user_input) != (None, None):
        return any(kw in lower for kw in INCIDENT_KEYWORDS) or mentions_blockage(
            user_input
        )
    return False


def incident_slots_complete(state: IncidentState) -> bool:
    if not state.staff_role:
        return False
    if state.event_type == "station_disruption":
        return bool(state.station)
    if state.event_type != "line_blockage":
        return False
    if not (
        state.from_station and state.to_station and state.severity and state.staff_role
    ):
        return False
    if state.severity == "both_lines_blocked":
        return state.incident_time is not None and state.duration_minutes is not None
    return True


def reset_incident_state(state: IncidentState) -> None:
    state.stage = "idle"
    state.event_type = None
    state.from_station = None
    state.to_station = None
    state.station = None
    state.severity = None
    state.staff_role = None
    state.incident_time = None
    state.day_type = None
    state.service_period = None
    state.duration_minutes = None
    state.derived_actions = []
    state.info_needed = []
    state.mentions_blockage = False
    state.pending_slot = None


def _sync_pending_slot(state: IncidentState) -> None:
    slot = state.pending_slot
    if not slot:
        return
    filled = {
        "event_type": state.event_type,
        "from_station": state.from_station,
        "to_station": state.to_station,
        "station": state.station,
        "severity": state.severity,
        "staff_role": state.staff_role,
        "incident_time": state.incident_time,
        "duration_minutes": state.duration_minutes,
    }
    if filled.get(slot):
        state.pending_slot = None


def _declare_facts(engine: IncidentEngine, state: IncidentState) -> None:
    incident_attrs: dict = {}
    if state.event_type:
        incident_attrs["event_type"] = state.event_type
    if state.from_station:
        incident_attrs["from_station"] = state.from_station
    if state.to_station:
        incident_attrs["to_station"] = state.to_station
    if state.station:
        incident_attrs["station"] = state.station
    if state.severity:
        incident_attrs["severity"] = state.severity
    if state.staff_role:
        incident_attrs["staff_role"] = state.staff_role
    if state.incident_time:
        incident_attrs["incident_time"] = state.incident_time
    if state.day_type:
        incident_attrs["day_type"] = state.day_type
    if state.service_period:
        incident_attrs["service_period"] = state.service_period
    if state.duration_minutes is not None:
        incident_attrs["duration_minutes"] = state.duration_minutes
    if incident_attrs:
        engine.declare(Incident(**incident_attrs))
    for topic in state.info_needed:
        engine.declare(WantsInfo(topic=topic))


def run_incident_engine(state: IncidentState) -> IncidentEngine:
    engine = IncidentEngine()
    engine.reset()
    engine.info_topics = list(state.info_needed)
    _declare_facts(engine, state)
    engine.run()
    return engine


def _slot_prompt(state: IncidentState) -> str | None:
    prompts = {
        "event_type": (
            "Please say **line blockage** (between two stations) or **station disruption** (at one station)."
        ),
        "from_station": "Which station is at the **start** of the blocked section?",
        "to_station": (
            "Which station is at the **end** of the blocked section"
            + (f" (after **{state.from_station}**)?" if state.from_station else "?")
            + " You can use the full name or a CRS code (e.g. **WAT** for Waterloo)."
        ),
        "station": "Which **station** is affected?",
        "severity": (
            f"For the blockage between **{state.from_station}** and **{state.to_station}**, "
            "is it **full** (both lines) or **partial** (one line)?"
            if state.from_station and state.to_station
            else "Is it a **full** or **partial** blockage?"
        ),
        "staff_role": "Which role are you acting as: **signaller**, **station staff**, or **control**?",
        "incident_time": "What time is the incident (e.g. **08:30**, **6pm**, **morning**)?",
        "duration_minutes": "How long has this lasted or is expected to last (e.g. **30 min**, **1 hour**)?",
    }
    return prompts.get(state.pending_slot or "")


def _report_progress(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def process_incident_input(
    user_input: str,
    state: IncidentState,
    *,
    llm_client: LLMClient,
    vector_store: VectorStore | None = None,
    progress: ProgressCallback = None,
) -> tuple[str, list[Path]]:
    lower = user_input.lower().strip()

    if lower in ("reset", "bye"):
        reset_incident_state(state)
        return "Contingency assistant reset. How can I help with the disruption?", []

    if state.stage == "idle":
        if not is_incident_intent(user_input):
            return (
                "I can help with **line blockages** and **station disruptions** using SWR contingency plans. "
                "Say what happened, or start with your role: **signaller**, **station staff**, or **control**."
            ), []
        state.stage = "collecting"

    prev_pending = state.pending_slot
    slots_complete = apply_message_slots(state, user_input)
    if not slots_complete:
        _sync_pending_slot(state)

    correction = parse_station_correction(user_input)
    if (
        state.event_type == "station_disruption"
        and correction
        and not state.pending_slot
        and prev_pending not in ("staff_role", None)
    ):
        state.station = friendly_station_label(correction)

    if prev_pending and state.pending_slot == prev_pending:
        if prev_pending in ("from_station", "to_station", "station"):
            if not parse_single_station_answer(user_input):
                return UNRECOGNISED_STATION, []
        retry = _slot_prompt(state)
        if retry:
            return retry, []

    engine = run_incident_engine(state)

    if engine.action == "ask":
        state.stage = "collecting"
        state.pending_slot = engine.pending_slot
        return engine.message, []

    state.pending_slot = None
    state.derived_actions = list(engine.derived_actions)

    if engine.action == "retrieve" and engine.retrieval_query and engine.plan_source:
        _report_progress(progress, "Searching contingency plans...")
        store = vector_store or VectorStore()
        route_stations = None
        if state.from_station and state.to_station:
            route_stations = [state.from_station, state.to_station]
        chunks = retrieve_plans(
            store,
            query=engine.retrieval_query,
            plan_source=engine.plan_source,
            station_filter=engine.station_filter,
            staff_role=engine.role_focus or state.staff_role,
            service_period=engine.time_focus or state.service_period,
            derived_actions=state.derived_actions,
            route_stations=route_stations,
        )
        if not chunks and engine.station_filter:
            state.stage = "collecting"
            state.pending_slot = "station"
            label = friendly_station_label(engine.station_filter)
            return (
                f"There is no station disruption plan available for **{label}**. "
                f"Plans currently available for: {INDEXED_STATION_HINT}. "
                "Try the full name or a CRS code (**WEY**, **WAT**)."
            ), []
        _report_progress(progress, "Summarising staff and passenger advice...")
        images = image_paths_from_chunks(chunks)
        reply = build_disruption_reply(
            chunks,
            retrieval_query=engine.retrieval_query,
            info_topics=engine.info_topics,
            llm_client=llm_client,
            station=engine.station_filter or state.station,
            event_type=state.event_type,
            staff_role=engine.role_focus or state.staff_role,
            service_period=engine.time_focus or state.service_period,
            incident_time=state.incident_time,
            derived_actions=state.derived_actions,
        )
        state.stage = "done"
        return reply, images

    if state.stage == "done":
        return (
            "This incident is already summarised. Type **reset** to start a new disruption enquiry."
        ), []

    state.stage = "collecting"
    return (
        "I need a bit more detail. Tell me your role (**signaller**, **station staff**, **control**), "
        "where the disruption is, and whether it is a **line blockage** or **station disruption**."
    ), []
