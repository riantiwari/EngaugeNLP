import asyncio

from fastapi import FastAPI, HTTPException, Body, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from starlette.websockets import WebSocketDisconnect

from vector_database.qdrant_manager import QdrantManager  # Assuming you have this class imported
from whisper import load_model
import tempfile
import os


# Global QdrantManager instance
qdrant_manager: QdrantManager = None

# Initialize QdrantManager using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_manager
    global model
    print("Initializing QdrantManager...")
    # Initialize QdrantManager instance
    qdrant_manager = QdrantManager()
    # Load the Whisper model
    model = load_model("small", device='cpu')

    # Perform any setup with QdrantClient, like creating collections, etc.
    yield

    print("Cleaning up QdrantManager...")
    qdrant_manager = None

# Create FastAPI app with the lifespan event
app = FastAPI(lifespan=lifespan)

# Root endpoint to check server status
@app.get("/")
async def read_root():
    if qdrant_manager:
        return {"message": "Qdrant client is ready!"}
    else:
        return {"message": "Qdrant client is not initialized."}



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")

    # Temporary storage for audio chunks
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    temp_audio_path = temp_audio.name
    temp_audio.close()

    try:
        while True:  # Keep connection open until client disconnects
            try:
                # Wait for an audio chunk
                audio_chunk = await websocket.receive_bytes()

                # Append received chunk to the temporary audio file
                with open(temp_audio_path, "ab") as f:
                    f.write(audio_chunk)

                # Process the audio chunk
                result = model.transcribe(temp_audio_path)
                transcription = result.get("text", "")
                print(f"Transcription: {transcription}")

                # Send transcription back to the frontend
                await websocket.send_text(transcription)

            except KeyboardInterrupt:
                break

            except asyncio.TimeoutError:
                print("Timeout waiting for audio chunk.")
                break

            except WebSocketDisconnect:
                print("Client disconnected.")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")

    finally:
        # Cleanup temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        print("WebSocket disconnected.")


# Endpoint to create a collection
@app.post("/create_collection/{collection_name}")
async def create_collection(collection_name: str):
    if qdrant_manager:
        try:
            result = qdrant_manager.create_collection(collection_name=collection_name)
            if result:
                return JSONResponse(
                    status_code=201,
                    content={"message": f"Collection '{collection_name}' created successfully."}
                )
            else:
                raise HTTPException(status_code=400, detail="Failed to create collection")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized.")

# Endpoint to delete a collection
@app.delete("/delete_collection/{collection_name}")
async def delete_collection(collection_name: str):
    if qdrant_manager:
        try:
            result = qdrant_manager.delete_collection(collection_name=collection_name)
            if result:
                return {"message": f"Collection '{collection_name}' deleted successfully."}
            else:
                raise HTTPException(status_code=400, detail="Failed to delete collection")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized.")

# Endpoint to add text to a collection
@app.post("/add_text/{collection_name}")
async def add_text(collection_name: str, text: str = Body(...), start_time: int = Body(...), end_time: int = Body(...)):
    if qdrant_manager:
        try:
            qdrant_manager.add_text(collection_name=collection_name, text=text, start_time=start_time, end_time=end_time)
            return {"message": "Text added successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding text: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized.")

# Endpoint to add drawing text to a collection
@app.post("/add_drawing_text/{collection_name}")
async def add_drawing_text(collection_name: str, text: str = Body(...), timestamp: int = Body(...)):
    if qdrant_manager:
        try:
            qdrant_manager.add_drawing_text(collection_name=collection_name, text=text, time_stamp=timestamp)
            return {"message": "Drawing text added successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding drawing text: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized.")

# Endpoint for /chat
@app.get("/chat/{collection_name}")
async def chat(collection_name: str, body: dict = Body(...)):
    if qdrant_manager:
        try:
            # Extract `prompt` and `conversation_history` from the body
            prompt = body.get("prompt")
            conversation_history = body.get("conversation_history", [])

            # Validate input
            if not isinstance(prompt, str) or not isinstance(conversation_history, list):
                raise HTTPException(status_code=400,
                                    detail="Invalid input format. Ensure 'prompt' is a string and 'conversation_history' is a list.")

            # Use QdrantManager to interact with the collection and generate a response
            response = qdrant_manager.chat(collection_name, prompt, conversation_history)
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized.")

