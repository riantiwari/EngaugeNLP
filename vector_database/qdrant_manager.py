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

        self.drawing_start_times = []
        self.drawing_text = []

        # Initialize LLM and LangChain components
        self.llm = LlamaLLM()
        self.memory = ConversationBufferMemory()


        # Kshitij Prompt
        rules = """
                You are an AI assistant specifically trained to answer questions based ONLY on the provided lecture content. 
                Your knowledge is limited to the information given in the context. Follow these rules strictly:

                - Only use information explicitly stated in the provided context.
                - If the context doesn't contain relevant information to answer the question, say 
                  "I don't have enough information to answer that question based on the provided lecture content."
                - Do not use any external knowledge or make assumptions beyond what's in the context.
                - If asked about topics not covered in the context, state that the lecture content doesn't cover that topic.
                - Be precise and concise in your answers, citing specific parts of the context when possible.
                - If the question is ambiguous or unclear based on the context, ask for clarification.
                - Never claim to know more than what's provided in the context.
                - If the context contains conflicting information, point out the inconsistency without resolving it.
                - Remember, your role is to interpret and relay the information from the lecture content, not to provide additional knowledge or opinions.
                """

        # PromptTemplate to handle the combination of query and context
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input"],  # Ensure history is an input variable
            template=f"{rules}\n\n{{history}}\n{{input}}"  # Include history in template
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

    def add_text(self, text: str, start_time, end_time):
        embedding = self.model.encode(text)

        metadata = {
            "text": text,
            "start_time": start_time,
            "end_time": end_time
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

        drawing_context = set()


        for result in results:
            start_time = result.payload['start_time']
            end_time = result.payload['end_time']

            index = self.special_binary_search(start_time)

            if index != -1:
                while index < len(self.drawing_start_times) and self.drawing_start_times[index] <= end_time:
                    drawing_context.add(self.drawing_text[index])
                    index += 1

        combined_text = " ".join([ f" {result.payload['start_time']}:{result.payload['end_time']} {result.payload['text']}" for result in results])
        drawing_text = " ".join(drawing_context)

        input = f"Context: {combined_text}\nTeacher's Drawing: {drawing_text}\nUser: {prompt}\n"

        # Use LangChain's LLMChain to handle the prompt with history and context
        response = self.llm_chain.predict(input=input)

        print(f"Human input: {input}")

        # Print memory for debugging
        # print("Conversation History in Memory:", self.memory.load_memory_variables({})['history'])

        return response

    def special_binary_search(self, start_time):
        start = 0
        end = len(self.drawing_start_times) - 1

        if start_time < self.drawing_start_times[0]:
            return 0
        elif start_time > self.drawing_start_times[len(self.drawing_start_times) - 1]:
            return len(self.drawing_start_times) - 1

        while start <= end:
            mid = (start + end) // 2

            if self.drawing_start_times[mid] == start_time:
                return mid
            elif self.drawing_start_times[mid] < start_time:
                if mid + 1 < len(self.drawing_start_times) and self.drawing_start_times[mid + 1] > start_time: # Found the closest drawing
                    return mid
                else:
                    start = mid + 1
            else: # self.drawing_start_times[mid] > start_time
                if mid - 1 >= 0 and self.drawing_start_times[mid - 1] < start_time:
                    return mid - 1
                else:
                    end = mid - 1

        return -1


    def add_drawing_text(self, text, time_stamp):
        self.drawing_text.append(text)
        self.drawing_start_times.append(time_stamp)