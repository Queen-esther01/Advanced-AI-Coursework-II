import random
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from incident_handler import (
    IncidentState,
    is_incident_intent,
    process_incident_input,
    reset_incident_state,
)
from llm_client import LLMClient

# Task1 Imports
from ticket_finder import TicketState, is_ticket_intent, process_ticket_input

load_dotenv(verbose=True)

SYSTEM_PROMPT = """
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

Keep replies concise and conversational. Ask only one or two questions at a
time. When you know the current location and destination, call
check_station_coverage before attempting a prediction. If coverage is confirmed,
call get_train_delay with train_journey, current_location, destination,
current_delay, and planned_time_at_current_stop.
Use get_covered_stations if the passenger asks which stations are supported.

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
    if incident_state.stage != "idle" or is_incident_intent(user_input):
        if incident_state.stage == "collecting" or is_incident_intent(user_input):
            return "Looking up contingency plans..."
        return "Generating contingency advice..."
    if ticket_state.stage != "idle" or is_ticket_intent(user_input):
        return "Processing your journey details..."
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


def process_user_input(user_input, ticket_state, incident_state) -> HandlerResult:
    """
    Decide whether to handle with ticket state machine or LLM.
    """
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
            )
            return HandlerResult(text=reply, images=_paths_to_strings(images))
    if incident_state.stage != "idle" or is_incident_intent(user_input):
        reply, images = process_incident_input(
            user_input,
            incident_state,
            llm_client=DISRUPTION_LLM,
        )
        return HandlerResult(text=reply, images=_paths_to_strings(images))
    if ticket_state.stage != "idle" or is_ticket_intent(user_input):
        reply = process_ticket_input(user_input, ticket_state)
        return HandlerResult(text=reply)
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
        will_use_llm = (
            ticket_state.stage == "idle"
            and incident_state.stage == "idle"
            and not is_incident_intent(prompt)
            and not is_ticket_intent(prompt)
            and prompt.lower() not in ("reset", "yes", "bye")
        )

        if will_use_llm:
            with st.spinner("Thinking..."):
                result = process_user_input(prompt, ticket_state, incident_state)
                stream = llm_client.stream_reply(st.session_state.messages)
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
            with st.spinner(label):
                result = process_user_input(prompt, ticket_state, incident_state)
            reply = result.text or ""
            render_assistant_content(reply, result.images)
            msg: dict = {"role": "assistant", "content": reply}
            if result.images:
                msg["images"] = result.images
            st.session_state.messages.append(msg)
