# Example search usage
from qdrant_client import QdrantClient

from chatbot import llama3
from vector_database.qdrant_manager import QdrantManager
from vector_database.tokenizer_and_embedding import tokenize_and_embed

query_text = "Who is my teacher?"
query_embedding = tokenize_and_embed(query_text)
client = QdrantClient("localhost", port=6333)
manager = QdrantManager()

results = manager.search_similar(query_embedding)

combined_text = ""

for result in results:
    # print(f"Text: {result.payload['text']}")
    combined_text += result.payload['text'] + " "

print(llama3("Who is teaching this class?" + "Context:" + combined_text))