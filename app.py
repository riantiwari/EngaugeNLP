import streamlit as st
from transcription import transcribe_video_real_time
from vector_database.qdrant_manager import QdrantManager
import time
import threading
from lecture_simulator.simulate_lecture import simulate_lecture_updates
import threading


def start_transcription(qdrant_manager: QdrantManager = None):
    print("Starting transcription...")
    video_file = "sample_video.mp4"  # Replace with your video file path
    transcribe_video_real_time(video_file, chunk_duration=20.0, overlap=2, qdrant_manager=qdrant_manager)


# Initialize threads once, outside Streamlit's re-run scope
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True  # Flag to prevent re-initialization

    # Start threads
    st.session_state["qdrant_mgr"] = QdrantManager()
    st.session_state["qdrant_mgr"].create_collection("lecture_collection")
    st.session_state["conversation_history"] = []
    transcription_thread = threading.Thread(target=start_transcription, args=(st.session_state["qdrant_mgr"],))
    transcription_thread.start()


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
            response = st.session_state["qdrant_mgr"].chat("lecture_collection", user_input, st.session_state["conversation_history"])
        except Exception as e:
            response = f"An error occurred: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
