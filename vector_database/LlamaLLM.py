from langchain.llms.base import LLM  # Correct import path
import requests
import ollama

url = "http://localhost:11434/api/chat"


# Custom LLM wrapper for Llama model
class LlamaLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call the parent constructor with kwargs


    def _call(self, prompt: str, stop=None):
        """Calls the local Llama model API with the prompt."""
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt },
            ],
            # stream=False
        )

        return response['message']['content']

    @property
    def _llm_type(self):
        return "local_llama3"

