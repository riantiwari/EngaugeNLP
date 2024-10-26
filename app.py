import streamlit as st
from vector_database.qdrant_manager import QdrantManager
import time
import threading
from lecture_simulator.simulate_lecture import simulate_lecture_updates


def process_file(file_path: str, qdrant_mgr: QdrantManager):
    """Watch file for changes and add new content to Qdrant."""
    print(f"Watching for changes in {file_path}...")

    with open(file_path, 'r') as file:
        file.seek(0, 2)  # Move to end of file
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.5)
                continue

            # Process new line
            line = line.strip()
            if line:
                print(f"New line detected: {line}")
                qdrant_mgr.add_text(line)
                print(f"Added to Qdrant database: {line}...")



@st.cache_resource
def initialize_threads(file_path, input_file):
    """Initialize and start the file watcher and lecture simulator threads once."""
    qdrant_mgr = QdrantManager()  # Singleton instance of QdrantManager

    # Start file-watcher thread
    file_watcher_thread = threading.Thread(
        target=process_file, args=(file_path, qdrant_mgr), daemon=True
    )
    file_watcher_thread.start()

    # Start lecture simulation thread
    lecture_sim_thread = threading.Thread(
        target=simulate_lecture_updates, args=(input_file, file_path), daemon=True
    )
    lecture_sim_thread.start()

    return qdrant_mgr, file_watcher_thread, lecture_sim_thread


# Initialize threads once, outside Streamlit's re-run scope
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True  # Flag to prevent re-initialization
    # Prepare files
    open("lecture_simulator/simulated_lecture_output.txt", "w").close()
    file_to_watch = "lecture_simulator/simulated_lecture_output.txt"
    input_file = "lecture_simulator/lecture_input.txt"

    # Start threads
    st.session_state["qdrant_mgr"], st.session_state["file_watcher_thread"], st.session_state["lecture_sim_thread"] = initialize_threads(
        file_to_watch, input_file
    )

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
