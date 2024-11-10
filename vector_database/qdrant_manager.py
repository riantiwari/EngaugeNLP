from langchain.llms.base import LLM  # Correct import path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests
import os

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
url = "http://localhost:11434/api/chat"


# Custom LLM wrapper for Llama model
class LlamaLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Make sure to call the parent constructor with kwargs

    def _call(self, prompt: str, stop=None):
        """Calls the local Llama model API with the prompt."""
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

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        return response_data['message']['content']

    @property
    def _llm_type(self):
        return "local_llama3"


class QdrantManager:
    def __init__(self, collection_name="lecture_collection"):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.current_id = 0
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.setup_collection()

        # Initialize LLM and LangChain components
        self.llm = LlamaLLM()
        self.memory = ConversationBufferMemory()

        # Updated PromptTemplate
        self.prompt_template = PromptTemplate(
            input_variables=["input"],  # Ensure variables align
            template="{input}"  # Template now includes all variables
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template, memory=self.memory)
        # print(self.prompt_template.input_variables)

    def setup_collection(self, vector_size=384):  # 384 is the size for all-MiniLM-L6-v2
        """Initialize the Qdrant collection"""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{self.collection_name}' created successfully")

    def add_text(self, text: str, time_stamp):
        embedding = self.model.encode(text)

        metadata = {
            "text": text,
            "time_stamp": time_stamp
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self.current_id,
                    vector=embedding.tolist(),
                    payload=metadata
                )
            ]
        )
        self.current_id += 1

    def search_similar(self, prompt, limit: int = 30):
        embedding = self.model.encode(prompt)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit
        )
        return results

    def chat(self, prompt):
        # Search for similar context in the database
        results = self.search_similar(prompt)
        combined_text = " ".join([result.payload['text'] for result in results])

        input = f"Context: {combined_text}\nQuery: {prompt}"

        # Use LangChain's LLMChain with history, context, and query
        response = self.llm_chain.predict(input=input)
        self.memory.chat_memory.add_user_message(prompt)
        self.memory.chat_memory.add_ai_message(response)

        return response
