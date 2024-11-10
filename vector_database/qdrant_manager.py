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


        combined_text = " ".join([result.payload['text'] for result in results])

        input = f"Context: {combined_text}\nUser: {prompt}\n"

        # Use LangChain's LLMChain to handle the prompt with history and context
        response = self.llm_chain.predict(input=input)

        # Print memory for debugging
        print("Conversation History in Memory:", self.memory.load_memory_variables({})['history'])

        return response
