from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class QdrantManager:
    def __init__(self, collection_name="lecture_collection"):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.current_id = 0
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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

    def search_similar(self, prompt, limit: int = 5):
        embedding = self.model.encode(prompt)

        """Search for similar texts in the collection"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit
        )

        return results
