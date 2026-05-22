import os
import random

import streamlit as st
from dotenv import load_dotenv

from llm_client import LLMClient

# Task1 Imports
from ticket_finder import (
    TicketState,
    is_ticket_intent,
    process_ticket_input,
    reset_ticket_state,
)

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
time. When enough information has been gathered, summarise the details in a
structured way so they can be passed to the predictive model.

Do not invent a predicted arrival time. If a model prediction is provided by the
application, explain it clearly in minutes and as an estimated arrival time. If
no model prediction is provided yet, say what information is still needed.
""".strip()

INITIAL_GREETINGS = [
    "Hello! How can I help with your Weymouth–Waterloo journey today?",
    "Hi there — travelling between Weymouth and London Waterloo? Tell me what you need.",
    "Welcome aboard. I can help with delays and arrival estimates on the WEY–WAT line.",
    "Good to see you. Are you on a train now, or planning a trip from Weymouth or Waterloo?",
    "Hello! Share your service and where you are, and I will help work out what is going on.",
    "Hi — I am your delay assistant for Weymouth ↔ Waterloo services. What is happening on your train?",
    "Welcome. If you are delayed or unsure when you will arrive, I can guide you through the details.",
    "Hello! Which station are you at, and where are you headed?",
    "Hi there. I help passengers on the South Western route between Weymouth and Waterloo — how can I assist?",
    "Good day. Tell me about your train and I will ask the right follow-up questions.",
    "Hello! Stuck at a station or running late? I am here to collect the details for an arrival estimate.",
    "Hi — whether you are at Weymouth, Waterloo, or somewhere in between, I can help with delay information.",
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
    if lower_input in ["reset", "yes", "bye"]:
        reply = process_ticket_input(user_input, ticket_state)
        return reply, False
    # If we are already in an active ticket conversation (stage not idle), use ticket handler.
    # Or if the user shows ticket intent (even if idle), use ticket handler.
    if ticket_state.stage != "idle" or is_ticket_intent(user_input):
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
