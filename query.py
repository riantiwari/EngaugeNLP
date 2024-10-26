from vector_database.qdrant_manager import QdrantManager

query_text = "Who is my teacher?"
manager = QdrantManager()

results = manager.search_similar(query_text)

combined_text = ""

for result in results:
    # print(f"Text: {result.payload['text']}")
    combined_text += result.payload['text'] + " "

print(manager.chat("Who is teaching this class?" + "Context:" + combined_text))