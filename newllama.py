import ollama

# Check if Ollama supports streaming by looking for a "stream" or "progress" argument
response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "user", "content": "Who is this"},
    ],
    stream=True  # This is hypothetical; check Ollama's documentation for this feature
)

# If the stream is supported, you would handle partial updates here
for partial_message in response:
    print(partial_message["message"]["content"], end="")
