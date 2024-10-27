from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import requests
import os
url = "http://localhost:11434/api/chat"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QdrantManager:
    def __init__(self, collection_name="lecture_collection"):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.current_id = 0
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.setup_collection()

    def setup_collection(self, vector_size=384):  # 384 is the size for all-MiniLM-L6-v2
        """Initialize the Qdrant collection"""
        try:
            # Delete collection if it exists
            self.client.delete_collection(self.collection_name)
        except:
            pass

        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{self.collection_name}' created successfully")

    def add_text(self, text: str):
        embedding = self.model.encode(text)

        """Add text and its embedding to the collection"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self.current_id,
                    vector=embedding.tolist(),
                    payload={"text": text}
                )
            ]
        )
        self.current_id += 1

    def search_similar(self, prompt, limit: int = 30):
        embedding = self.model.encode(prompt)

        """Search for similar texts in the collection"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit
        )

        return results

    def chat(self, prompt):
        results = self.search_similar(prompt)

        combined_text = ""

        for result in results:
            print(f"Text: {result.payload['text']}")
            combined_text += result.payload['text'] + " "

        return llama3(prompt + "Context: " + combined_text)


def llama3(prompt):
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

    return response.json()['message']['content']