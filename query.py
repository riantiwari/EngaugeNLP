# Example search usage
from qdrant_client import QdrantClient

from chatbot import LectureChatbot
from vector_database.qdrant_manager import QdrantManager

query_text = "Who is my teacher?"
manager = QdrantManager()

chatbot = LectureChatbot(manager)

results = manager.search_similar(query_text)

combined_text = ""

for result in results:
    # print(f"Text: {result.payload['text']}")
    combined_text += result.payload['text'] + " "

print(chatbot.chat("Who is teaching this class?" + "Context:" + combined_text))