import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llm_client import LLMClient
from task_3.incident_handler import (
    IncidentState,
    incident_slots_complete,
    is_incident_intent,
    process_incident_input,
    reset_incident_state,
)
from task_3.indexing.vector_store import VectorStore

# Task1 Imports
from ticket_finder import TicketState, is_ticket_intent, process_ticket_input

load_dotenv(verbose=True)

SYSTEM_PROMPT = """
You are a helpful railway delay assistant for journeys from Weymouth (WEY) to
London Waterloo (WAT) and vice versa.

Your job is to guide the passenger through a short chat and collect the details
needed by a predictive arrival-time model. Ask clear follow-up questions when
information is missing, including:
- the train/service they are on
- the current station or current location of the train
- the passenger's destination station
- the current delay in minutes
- any relevant disruption information the passenger provides

Keep replies concise and conversational. Ask only one or two questions at a
time. When you know the current location and destination, you MUST call
check_station_coverage before saying whether stations are on the route.
If coverage is confirmed, call get_train_delay with train_journey,
current_location, destination, current_delay, and planned_time_at_current_stop.
Use get_covered_stations if the passenger asks which stations are supported.

Never tell the passenger a station is outside the route without calling
check_station_coverage first. Wareham, Hamworthy, Poole, and Bournemouth are
on the Dorset section of this line.

After a tool returns a prediction, explain the expected delay clearly to the
passenger using the predicted_delay_minutes and reason fields. Do not invent
numbers that were not returned by a tool.

Do not ask for unnecessary fields like day_of_week.
""".strip()

INITIAL_GREETINGS = [
    "Hello! I can help with **Weymouth–Waterloo delay information** or **SWR disruption/contingency plans** "
    "(line blockages and station disruptions). What do you need?",
    "Hi — ask about a **train delay** on the WEY–WAT line, or describe a **line blockage** / **station disruption** "
    "for staff and passenger advice from the plans.",
    "Welcome. For passengers: share your service and where you are. "
    "For staff: say e.g. **station disruption at Weymouth** or a **blockage between two stations**.",
    "Hello! How can I help with your journey today?",
    "Welcome, let me know how I can help.",
    "Good day. Tell me about your journey and what you need help with.",
    "Hello! Stuck at a station or running late or need to book a new journey? I can help you with that.",
]

llm_client = LLMClient(system_prompt=SYSTEM_PROMPT)

DISRUPTION_LLM = LLMClient()


@dataclass
class HandlerResult:
    text: str | None = None
    use_llm: bool = False
    images: list[str] = field(default_factory=list)


def _processing_label(
    user_input: str,
    ticket_state: TicketState,
    incident_state: IncidentState,
) -> str:
    ticket_active = ticket_state.stage != "idle" or is_ticket_intent(user_input)
    incident_active = incident_state.stage != "idle" or is_incident_intent(user_input)
    if ticket_active and (not incident_active or ticket_state.stage != "idle"):
        return "Processing your journey details..."
    if incident_active:
        if not incident_state.staff_role or incident_state.pending_slot == "staff_role":
            return "Collecting your role and incident details..."
        if incident_state.pending_slot or not incident_slots_complete(incident_state):
            return "Collecting disruption details..."
        return "Looking up contingency plans..."
    return "Thinking..."


def _paths_to_strings(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths if Path(p).is_file()]


def render_assistant_content(content: str, images: list[str] | None = None) -> None:
    st.markdown(content)
    for raw in images or []:
        path = Path(raw)
        if path.is_file():
            caption = path.name
            section = path.parent.name.replace("-", " ").title()
            if section and section != "Extracted Media":
                caption = f"{section} — {path.name}"
            st.image(str(path), caption=caption, use_container_width=True)


st.set_page_config(page_title="AI Train Assistant", layout="centered")
st.title("AI Train Assistant")

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": random.choice(INITIAL_GREETINGS)}
    ]

# Initialise ticket state (holds conversation progress)
if "ticket_state" not in st.session_state:
    st.session_state.ticket_state = TicketState()

if "incident_state" not in st.session_state:
    st.session_state.incident_state = IncidentState()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_content(
                message["content"],
                message.get("images"),
            )
        else:
            st.markdown(message["content"])


def process_user_input(
    user_input,
    ticket_state,
    incident_state,
    *,
    progress=None,
    vector_store: VectorStore | None = None,
) -> HandlerResult:
    """
    Decide whether to handle with ticket state machine or LLM.
    """
    store = vector_store or st.session_state.vector_store
    lower_input = user_input.lower()
    if lower_input == "reset":
        ticket_state.__dict__.update(TicketState().__dict__)
        reset_incident_state(incident_state)
        return HandlerResult(text="Conversation reset. How can I help you?")
    if lower_input in ("yes", "bye"):
        if ticket_state.stage != "idle":
            reply = process_ticket_input(user_input, ticket_state)
            return HandlerResult(text=reply)
        if incident_state.stage != "idle":
            reply, images = process_incident_input(
                user_input,
                incident_state,
                llm_client=DISRUPTION_LLM,
                vector_store=store,
                progress=progress,
            )
            return HandlerResult(text=reply, images=_paths_to_strings(images))
    ticket_active = ticket_state.stage != "idle" or is_ticket_intent(user_input)
    incident_active = incident_state.stage != "idle" or is_incident_intent(user_input)
    if ticket_active and (not incident_active or ticket_state.stage != "idle"):
        reply = process_ticket_input(user_input, ticket_state)
        return HandlerResult(text=reply)
    if incident_active:
        reply, images = process_incident_input(
            user_input,
            incident_state,
            llm_client=DISRUPTION_LLM,
            vector_store=store,
            progress=progress,
        )
        return HandlerResult(text=reply, images=_paths_to_strings(images))
    return HandlerResult(use_llm=True)


# Accept user input
if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ticket_state = st.session_state.ticket_state
    incident_state = st.session_state.incident_state
    label = _processing_label(prompt, ticket_state, incident_state)

    with st.chat_message("assistant"):
        ticket_active = ticket_state.stage != "idle" or is_ticket_intent(prompt)
        incident_active = incident_state.stage != "idle" or is_incident_intent(prompt)
        will_use_llm = (
            not ticket_active
            and not incident_active
            and prompt.lower() not in ("reset", "yes", "bye")
        )

        if will_use_llm:
            with st.spinner("Thinking..."):
                stream = llm_client.stream_delay_assistant(st.session_state.messages)
                try:
                    first_chunk = next(stream)
                except StopIteration:
                    first_chunk = None

                def stream_from_first():
                    if first_chunk is not None:
                        yield first_chunk
                    yield from stream

                reply = st.write_stream(stream_from_first()) or ""
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            with st.status(label, expanded=False) as status:

                def on_progress(message: str) -> None:
                    status.update(label=message)

                result = process_user_input(
                    prompt,
                    ticket_state,
                    incident_state,
                    progress=on_progress,
                    vector_store=st.session_state.vector_store,
                )
            reply = result.text or ""
            render_assistant_content(reply, result.images)
            msg: dict = {"role": "assistant", "content": reply}
            if result.images:
                msg["images"] = result.images
            st.session_state.messages.append(msg)
