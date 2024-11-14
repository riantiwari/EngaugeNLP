from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from vector_database.LlamaLLM import LlamaLLM

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QdrantManager:
    def __init__(self, collection_name="lecture_collection"):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.current_id = 0
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.setup_collection()
        self.drawing_transcription = {}

        # Initialize LLM and LangChain components
        self.llm = LlamaLLM()
        self.memory = ConversationBufferMemory()

        # PromptTemplate to handle the combination of query and context
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input"],  # Ensure history is an input variable
            template="{history}\n{input}"  # Include history in template
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template, memory=self.memory)

    def setup_collection(self, vector_size=384):
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

    def search_similar(self, prompt, limit: int = 30, similarity_threshold: float = 0.2):
        embedding = self.model.encode(prompt)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit,
            with_payload=True
        )

        filtered_results = []

        # print("Top 3 most similar documents:")
        for result in results:
            if result.score >= similarity_threshold:
                filtered_results.append(result)
                print(f"Document: {result.payload['text']}")
                print(f"Score: {result.score}")  # This is the similarity score

        return filtered_results

    def chat(self, prompt):
        # Search for similar context in the database
        results = self.search_similar(prompt)

        if not results:
            return "No relevant context found. How can I help you?"

        drawing_context = []

        for result in results:
            timestamp = int(result.payload['time_stamp'])

            for timer in range(timestamp - 5, timestamp + 6):
                if timer in self.drawing_transcription:
                    text = self.drawing_transcription[timer]
                    drawing_context.append(text)

        combined_text = " ".join([result.payload['text'] for result in results])
        drawing_text = " ".join(drawing_context)

        input = f"Context: {combined_text}\nTeacher's Drawing: {drawing_text}\nUser: {prompt}\n"

        # Use LangChain's LLMChain to handle the prompt with history and context
        response = self.llm_chain.predict(input=input)

        # Print memory for debugging
        print("Conversation History in Memory:", self.memory.load_memory_variables({})['history'])

        return response

    def add_drawing_text(self, text, time_stamp):
        self.drawing_transcription[time_stamp] = text