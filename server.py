from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from vector_database.qdrant_manager import QdrantManager  # Assuming you have this class imported

# Global QdrantManager instance
qdrant_manager: QdrantManager = None

# Initialize QdrantManager using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_manager
    print("Initializing QdrantManager...")
    # Initialize QdrantManager instance
    qdrant_manager = QdrantManager()

    # Perform any setup with QdrantClient, like creating collections, etc.
    # For example: qdrant_manager.create_collection("example_collection")

    # Yield control to FastAPI app, allowing it to start accepting requests
    yield

    print("Cleaning up QdrantManager...")
    # Perform cleanup tasks if necessary (like closing connections)
    # No explicit cleanup is necessary for QdrantManager by default, but it's a good practice
    qdrant_manager = None

# Create FastAPI app with the lifespan event
app = FastAPI(lifespan=lifespan)

# Example endpoint using QdrantClient instance
@app.get("/")
async def read_root():
    if qdrant_manager:
        # Example: Interact with Qdrant using QdrantManager (if you have methods in QdrantManager)
        # For instance, retrieving all collections or fetching data from a collection
        # collections = qdrant_manager.list_collections()  # Replace with an actual method
        return {"message": "Qdrant client is ready!"}
    else:
        return {"message": "Qdrant client is not initialized."}

# Example endpoint to create a collection
@app.post("/create_collection/{collection_name}")
async def create_collection(collection_name: str):
    if qdrant_manager:
        try:
            # Assuming QdrantManager has a method `create_collection` to create a collection
            result = qdrant_manager.setup_collection(vector_size=384)
            if result:
                # Return success response with 201 Created status
                return JSONResponse(
                    status_code=201,
                    content={"message": f"Collection '{collection_name}' created successfully."}
                )
            else:
                # Return failure response if the collection couldn't be created
                raise HTTPException(status_code=400, detail="Failed to create collection")
        except Exception as e:
            # Return an error response if something goes wrong
            raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")
    else:
        # Return error response if qdrant_manager is not initialized
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized.")
