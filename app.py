import streamlit as st
from chatbot import LectureChatbot
from vector_database.qdrant_manager import QdrantManager

st.title("In Class Chatbot")
st.write("This is a chatbot that can help you with your queries during the class.")

chatbot = LectureChatbot(QdrantManager())

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    spinner_text = "Getting response..."

    with st.spinner(spinner_text):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            response = chatbot.chat(user_input)
        except:
            response = f"An error occurred. Try again."

    st.session_state.messages.append({"role": "assistant", "content": "HI THERE"})
    st.chat_message("assistant").write(response)
