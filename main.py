import json
import os
import random
from dotenv import load_dotenv
import streamlit as st
from openrouter import OpenRouter
from openrouter.types import UNSET

# Task1 Imports
from ticket_finder import (
    is_ticket_intent,
    process_ticket_input,
    TicketState,
    reset_ticket_state,
)

# Task2 Imports
from task_2.delay_tool import get_tools_for_session
from task_2.utils import (
    MAX_TOOL_ROUNDS,
    build_system_prompt,
    clean_model_text,
    execute_tool_call,
    has_delay_inputs,
    merge_stream_tool_calls,
    tool_calls_from_accumulator,
    update_journey_context_from_tool,
)

load_dotenv(verbose=True)

MODEL_NAME = "openai/gpt-4o-mini"

INITIAL_GREETINGS = [
    "Hello! How can I help with your journey today?",
    "Hi there — travelling between Weymouth and London Waterloo? Tell me what you need.",
    "Good to see you. Are you on a train now, or planning a trip from Weymouth or Waterloo?",
    "Welcome, let me know how I can help.",
    "Hi there. I help passengers on the South Western route between Weymouth and Waterloo — how can I assist?",
    "Good day. Tell me about your journey and what you need help with.",
    "Hello! Stuck at a station or running late? I can help you with that.",
]


def stream_llm_reply(messages, journey_context=None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        yield "OPENROUTER_API_KEY is not set, so I cannot call the LLM yet."
        return

    journey_context = journey_context if journey_context is not None else {}
    active_tools = get_tools_for_session(journey_context.get("coverage_confirmed"))

    model_messages = [
        {"role": "system", "content": build_system_prompt(journey_context)},
        *messages,
    ]

    try:
        with OpenRouter(api_key=api_key) as client:
            tools_called = set()
            any_text_yielded = False

            for round_index in range(MAX_TOOL_ROUNDS):
                content_parts = []
                tool_call_accumulator = {}

                response = client.chat.send(
                    model=MODEL_NAME,
                    messages=model_messages,
                    stream=True,
                    tools=active_tools,
                )
                with response as event_stream:
                    for chunk in event_stream:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta
                        if delta.content and delta.content is not UNSET:
                            cleaned = clean_model_text(delta.content)
                            if cleaned:
                                content_parts.append(cleaned)

                        if delta.tool_calls:
                            merge_stream_tool_calls(
                                tool_call_accumulator, delta.tool_calls
                            )

                tool_calls = tool_calls_from_accumulator(tool_call_accumulator)
                round_text = "".join(content_parts)

                if not tool_calls:
                    if round_text:
                        any_text_yielded = True
                        yield round_text

                    if round_index + 1 < MAX_TOOL_ROUNDS:
                        if (
                            journey_context.get("coverage_confirmed")
                            and has_delay_inputs(journey_context)
                            and "get_train_delay" not in tools_called
                        ):
                            model_messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Use get_train_delay now with the journey "
                                        "details you have, then explain the "
                                        "predicted delay to me in plain language."
                                    ),
                                }
                            )
                            continue
                        if "get_train_delay" in tools_called and not any_text_yielded:
                            model_messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Explain the get_train_delay result to me in "
                                        "plain language."
                                    ),
                                }
                            )
                            continue

                    if not any_text_yielded:
                        yield (
                            "Sorry — I could not get a text reply from the assistant "
                            "after running the tools. Please try again, or re-send "
                            "your journey details (location, destination, delay, "
                            "and planned time)."
                        )
                    return

                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
                model_messages.append(assistant_message)

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tools_called.add(tool_name)
                    tool_result = execute_tool_call(
                        tool_name,
                        tool_call["function"]["arguments"],
                    )
                    update_journey_context_from_tool(
                        journey_context,
                        tool_name,
                        tool_call["function"]["arguments"],
                        tool_result,
                    )
                    if journey_context.get("coverage_confirmed"):
                        active_tools = get_tools_for_session(True)

                    model_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(tool_result),
                        }
                    )

            yield (
                "Sorry, I could not finish the tool request after several attempts."
            )
    except Exception as exc:
        yield f"I could not get a model response: {exc}"


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

if "journey_context" not in st.session_state:
    st.session_state.journey_context = {}

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
    if prompt.strip().lower() == "reset":
        st.session_state.journey_context = {}

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Decide how to respond
    reply, use_llm = process_user_input(prompt, st.session_state.ticket_state)

    if use_llm:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = st.write_stream(
                    stream_llm_reply(
                        st.session_state.messages,
                        journey_context=st.session_state.journey_context,
                    )
                ) or ""
            if not reply.strip():
                reply = (
                    "Sorry — no reply was shown. Please try again or re-send your "
                    "journey details."
                )
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        # Use ticket handler reply (non‑streaming)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})