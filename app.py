import streamlit as st

from transcription import transcribe_video_real_time
from vector_database.qdrant_manager import QdrantManager
import time
import threading
from lecture_simulator.simulate_lecture import simulate_lecture_updates




@st.cache_resource
def initialize_threads():
    """Initialize and start the file watcher and lecture simulator threads once."""
    qdrant_mgr = QdrantManager()  # Singleton instance of QdrantManager

    return qdrant_mgr


# Initialize threads once, outside Streamlit's re-run scope
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True  # Flag to prevent re-initialization

    # Start threads
    st.session_state["qdrant_mgr"] = initialize_threads()

# Streamlit UI
st.title("In Class Chatbot")
st.write("This is a chatbot that can help you with your queries during the class.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    with st.spinner("Getting response..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Handle user input with Qdrant
        try:
            response = st.session_state["qdrant_mgr"].chat(user_input)
        except Exception as e:
            response = f"An error occurred: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
