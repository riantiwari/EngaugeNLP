
def tokenize_and_embed(text):
    # Generate the embedding for the provided text
    embedding = model.encode(text)
    return embedding