from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Define a GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Run this application using: uvicorn main:app --reload

