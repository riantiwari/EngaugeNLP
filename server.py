import asyncio

from fastapi import FastAPI, HTTPException, Body, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from starlette.websockets import WebSocketDisconnect

from vector_database.qdrant_manager import QdrantManager  # Assuming you have this class imported
from whisper import load_model
import tempfile
import os


import numpy as np
import whisper
from threading import Thread, Event
import time
import subprocess

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn



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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint to check server status
@app.get("/")
async def read_root():
    if qdrant_manager:
        return {"message": "Qdrant client is ready!"}
    else:
        return {"message": "Qdrant client is not initialized."}



def create_collection(collection_name: str):
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


def add_text(collection_name: str, text: str, start_time: int, end_time: int):
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
@app.post("/chat/{collection_name}")
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



class LiveTranscriber:
    def __init__(self, model_name="small", chunk_time=10, overlap_seconds=2):
        self.model = whisper.load_model(model_name, device="cpu")
        self.is_running = False
        self.stop_event = Event()
        self.chunk_time = chunk_time
        self.overlap_seconds = overlap_seconds

        # Calculation for overlap bytes:
        # 16000 samples/sec * 4 bytes/sample * overlap_seconds
        self.sample_rate = 16000
        self.bytes_per_sample = 4
        self.overlap_bytes = self.overlap_seconds * self.sample_rate * self.bytes_per_sample

        # ffmpeg command to decode webm/opus to raw PCM float32 mono 16kHz
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error',
            '-f', 'webm', '-codec:a', 'opus', '-i', 'pipe:0',
            '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1'
        ]
        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def process_audio(self):
        pcm_buffer = bytearray()
        last_time = time.time()

        while self.is_running and not self.stop_event.is_set():
            chunk = self.ffmpeg_proc.stdout.read(1024)
            if not chunk:
                time.sleep(0.01)
                continue
            pcm_buffer.extend(chunk)

            if time.time() - last_time >= self.chunk_time:

                # Time to transcribe
                if len(pcm_buffer) > 0:
                    data_copy = bytes(pcm_buffer)
                    audio_data = np.frombuffer(data_copy, dtype=np.float32)
                    result = self.model.transcribe(audio_data)
                    text = result["text"].strip()
                    if text:
                        print("Transcription:", text)
                        add_text(collection_name="lecture", text=text, start_time=0, end_time=0)

                    # Keep the last overlap_bytes of audio as context for next chunk
                    if len(pcm_buffer) > self.overlap_bytes:
                        pcm_buffer = pcm_buffer[-self.overlap_bytes:]
                    else:
                        # If buffer is smaller than overlap_bytes, just keep it as is
                        pcm_buffer = bytearray(pcm_buffer)

                last_time = time.time()

        # After stopping, transcribe what remains
        if len(pcm_buffer) > 0:
            data_copy = bytes(pcm_buffer)
            audio_data = np.frombuffer(data_copy, dtype=np.float32)
            result = self.model.transcribe(audio_data)
            text = result["text"].strip()
            if text:
                print("Final Transcription:", text)

    def start(self):
        self.is_running = True
        self.process_thread = Thread(target=self.process_audio, daemon=True)
        self.process_thread.start()

    def stop(self):
        self.is_running = False
        self.stop_event.set()
        if self.ffmpeg_proc:
            self.ffmpeg_proc.stdin.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        if self.ffmpeg_proc:
            self.ffmpeg_proc.terminate()
            self.ffmpeg_proc.wait()

    def write_audio(self, data):
        if self.ffmpeg_proc and self.ffmpeg_proc.poll() is None:
            self.ffmpeg_proc.stdin.write(data)
            self.ffmpeg_proc.stdin.flush()

transcriber = LiveTranscriber(model_name="small", chunk_time=10, overlap_seconds=2)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    transcriber.start()

    create_collection("lecture")

    try:
        while True:
            message = await websocket.receive_bytes()
            transcriber.write_audio(message)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print("Error:", e)
    finally:
        transcriber.stop()

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="localhost", port=8000)