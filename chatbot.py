import requests


url = "http://localhost:11434/api/chat"


class LectureChatbot:
    def __init__(self, qdrant_manager):
        self.qdrant_manager = qdrant_manager

    def chat(self, prompt):
        results = self.qdrant_manager.search_similar(prompt)

        combined_text = ""

        for result in results:
            print(f"Text: {result.payload['text']}")
            combined_text += result.payload['text'] + " "

        return self.llama3(prompt + "Context: " + combined_text)

    def llama3(self, prompt):
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
