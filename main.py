import random
from dotenv import load_dotenv
import streamlit as st
from llm_client import LLMClient

# Task1 Imports
from ticket_finder import (
    is_ticket_intent,
    process_ticket_input,
    TicketState,
)

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
    "Hello! How can I help with your journey today?",
    "Hi there — travelling between Weymouth and London Waterloo? Tell me what you need.",
    "Good to see you. Are you on a train now, or planning a trip from Weymouth or Waterloo?",
    "Welcome, let me know how I can help.",
    "Hi there. I help passengers on the South Western route between Weymouth and Waterloo — how can I assist?",
    "Good day. Tell me about your journey and what you need help with.",
    "Hello! Stuck at a station or running late? I can help you with that.",
]

llm_client = LLMClient(system_prompt=SYSTEM_PROMPT)


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

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def process_user_input(user_input, ticket_state):
    """
    Decide whether to handle with ticket state machine or LLM.
    Returns a reply string and a boolean indicating if LLM was used.
    """
    lower_input = user_input.lower()
    # Special commands that must go to ticket handler (even if state idle)
    if lower_input in ['reset', 'yes', 'bye']:
        reply = process_ticket_input(user_input, ticket_state)
        return reply, False
    # If we are already in an active ticket conversation (stage not idle), use ticket handler.
    # Or if the user shows ticket intent (even if idle), use ticket handler.
    if ticket_state.stage != 'idle' or is_ticket_intent(user_input):
        reply = process_ticket_input(user_input, ticket_state)
        return reply, False
    else:
        # No active ticket conversation and no ticket intent → use LLM delay assistant
        return None, True

# Accept user input
if prompt := st.chat_input("Type a message..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Decide how to respond
    reply, use_llm = process_user_input(prompt, st.session_state.ticket_state)

    if use_llm:
        # Use LLM (streaming) for delay assistance
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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
        # Use ticket handler reply (non‑streaming)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})