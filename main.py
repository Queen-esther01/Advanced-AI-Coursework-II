import os
from dotenv import load_dotenv
import streamlit as st
from openrouter import OpenRouter

load_dotenv(verbose=True)

print(os.getenv("OPENROUTER_API_KEY"))
print("hello")



with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
    response = client.chat.send(
        model="google/gemma-4-31b-it",
        messages=[
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
    )
    print(response.choices[0].message.content)


st.set_page_config(page_title="AAI Chat", page_icon="💬", layout="centered")

st.title("💬 AAI Chat")

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type a message…"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a simple echo response (replace with your model later)
    reply = f"You said: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
