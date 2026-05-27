import os
import random
from dotenv import load_dotenv
import streamlit as st
from openrouter import OpenRouter
from openrouter.types import UNSET

load_dotenv(verbose=True)

MODEL_NAME = "google/gemma-4-31b-it"

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


def stream_llm_reply(messages):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        yield "OPENROUTER_API_KEY is not set, so I cannot call the LLM yet."
        return

    model_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

    try:
        with OpenRouter(api_key=api_key) as client:
            response = client.chat.send(
                model=MODEL_NAME,
                messages=model_messages,
                stream=True,
            )
            with response as event_stream:
                for chunk in event_stream:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content and content is not UNSET:
                        yield content
    except Exception as exc:
        yield f"I could not get a model response: {exc}"


st.set_page_config(page_title="AI Train Ticket Bot", layout="centered")

st.title("AI Train Ticket Bot")

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": random.choice(INITIAL_GREETINGS)}
    ]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type a message..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ask the LLM using the full conversation so far (streamed into the chat).
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = stream_llm_reply(st.session_state.messages)
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
