from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Distance
from sentence_transformers import SentenceTransformer
import os
import ollama

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def llama(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response['message']['content']


def special_binary_search(drawing_start_times, start_time):
    """
    Special binary search method that can work with any list of start times

    :param drawing_start_times: List of drawing start times
    :param start_time: Time to search for
    :return: Index of closest drawing start time
    """
    start = 0
    end = len(drawing_start_times) - 1

    if not drawing_start_times:
        return -1

    if start_time < drawing_start_times[0]:
        return 0
    elif start_time > drawing_start_times[-1]:
        return len(drawing_start_times) - 1

    while start <= end:
        mid = (start + end) // 2

        if drawing_start_times[mid] == start_time:
            return mid
        elif drawing_start_times[mid] < start_time:
            if mid + 1 < len(drawing_start_times) and drawing_start_times[mid + 1] > start_time:
                return mid
            else:
                start = mid + 1
        else:
            if mid - 1 >= 0 and drawing_start_times[mid - 1] < start_time:
                return mid - 1
            else:
                end = mid - 1

    return -1


class QdrantManager:
    def __init__(self, host="localhost", port=6333):
        """
        Initialize Qdrant client with support for multiple collections

        :param host: Qdrant server host
        :param port: Qdrant server port
        """
        self.client = QdrantClient(host, port=port)

        # Store collections with their metadata
        self.collections = {}

        # Shared embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_collection(self, collection_name, vector_size=384):
        """
        Create a new collection with specific parameters

        :param collection_name: Unique name for the collection
        :param vector_size: Dimension of embedding vectors
        :return: Collection metadata dictionary
        """
        try:
            # Delete existing collection if it exists
            self.client.delete_collection(collection_name)
        except:
            pass

        # Create new collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        # Initialize collection metadata
        collection_metadata = {
            'current_id': 0,
            'drawing_start_times': [],
            'drawing_text': [],
        }

        self.collections[collection_name] = collection_metadata

        print(f"Collection '{collection_name}' created successfully")
        return collection_metadata

    def delete_collection(self, collection_name):
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        self.client.delete_collection(collection_name)
        self.collections.pop(collection_name)

    def add_text(self, collection_name, text: str, start_time: int, end_time: int):
        """
        Add text to a specific collection

        :param collection_name: Name of the collection
        :param text: Text to be added
        :param start_time: Start timestamp
        :param end_time: End timestamp
        """
        # Ensure collection exists
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        # Get collection metadata
        collection_metadata = self.collections[collection_name]

        # Encode text
        embedding = self.model.encode(text)

        metadata = {
            "text": text,
            "start_time": start_time,
            "end_time": end_time
        }

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=collection_metadata['current_id'],
                    vector=embedding.tolist(),
                    payload=metadata
                )
            ]
        )

        # Increment ID
        collection_metadata['current_id'] += 1

    def _search_similar(self, collection_name, prompt, limit: int = 30, similarity_threshold: float = 0.2):
        """
        Search for similar texts in a specific collection

        :param collection_name: Name of the collection to search
        :param prompt: Search query
        :param limit: Maximum number of results
        :param similarity_threshold: Minimum similarity score
        :return: Filtered search results
        """
        # Ensure collection exists
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        embedding = self.model.encode(prompt)
        results = self.client.search(
            collection_name=collection_name,
            query_vector=embedding.tolist(),
            limit=limit,
            with_payload=True
        )

        filtered_results = []

        for result in results:
            if result.score >= similarity_threshold:
                filtered_results.append(result)
                print(f"Document: {result.payload['text']}")
                print(f"Score: {result.score}")

        return filtered_results

    def add_drawing_text(self, collection_name, text, time_stamp):
        """
        Add drawing text to a specific collection

        :param collection_name: Name of the collection
        :param text: Drawing text
        :param time_stamp: Timestamp of the drawing
        """
        # Ensure collection exists
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection_metadata = self.collections[collection_name]
        collection_metadata['drawing_text'].append(text)
        collection_metadata['drawing_start_times'].append(time_stamp)

    def chat(self, collection_name, prompt, conversation_history: list):
        """
        Chat using a specific collection

        :param collection_name: Name of the collection to use
        :param prompt: User prompt
        :return: AI response
        """
        # Ensure collection exists
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection_metadata = self.collections[collection_name]

        # Search for similar context in the database
        results = self._search_similar(collection_name, prompt)

        if not results:
            return "No relevant context found. How can I help you?"

        drawing_context = set()
        drawing_start_times = collection_metadata['drawing_start_times']
        drawing_text = collection_metadata['drawing_text']

        for result in results:
            start_time = result.payload['start_time']
            end_time = result.payload['end_time']

            index = special_binary_search(drawing_start_times, start_time)

            if index != -1:
                while index < len(drawing_start_times) and drawing_start_times[index] <= end_time:
                    drawing_context.add(f"{drawing_start_times[index] // 60}:{(drawing_start_times[index] % 60):02} - {drawing_text[index]}")
                    index += 1

        # Construct context with previous conversation history
        history_context = "\n".join(conversation_history[-6:]) if conversation_history else ""

        combined_text = " ".join(
            [f" {result.payload['start_time']//60}:{(result.payload['start_time']%60):02}-{result.payload['end_time']//60}:{(result.payload['end_time'] % 60):02} {result.payload['text']}" for result in
             results])
        drawing_text_str = " ".join(drawing_context)

        # Construct rules and input
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

        # Prepare input for LLM
        input_text = f"{rules}\n\nPrevious Conversation:\n{history_context}\n\nContext: {combined_text}\nTeacher's Drawing: {drawing_text_str}\n\nUser: {prompt}\n"
        print(f"Input text: {input_text}")

        # Generate response
        response = llama(prompt=input_text)

        return response

