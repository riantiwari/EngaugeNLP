from langchain.llms import LLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from typing import Optional
import requests


class LocalLlama3LLM(LLM):
    def __init__(self, url: str):
        self.url = url

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        data = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        headers = {'Content-Type': 'application/json'}

        response = requests.post(self.url, headers=headers, json=data)
        result = response.json().get('message', {}).get('content', "")

        if stop:
            for token in stop:
                result = result.split(token)[0]

        return result

    @property
    def _llm_type(self) -> str:
        return "local_llama3"


# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the custom LLM
url = "http://localhost:8000"  # Replace with your local model URL
llm = LocalLlama3LLM(url=url)

# Create a conversation chain with the custom LLM and memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)


# Function to interact with the conversation
def get_response(user_input: str):
    response = conversation.predict(input=user_input)
    return response
