import time
from vector_database.qdrant_manager import QdrantManager


def process_file(file_path: str, qdrant_mgr: QdrantManager):
    """Watch file for changes and add new content to Qdrant"""
    print(f"Watching for changes in {file_path}...")

    with open(file_path, 'r') as file:
        # Move to end of file
        file.seek(0, 2)

        while True:
            line = file.readline()
            if not line:
                time.sleep(0.5)
                continue

            # Process new line
            line = line.strip()
            if line:
                print(f"New line detected: {line}")

                # Add to Qdrant
                qdrant_mgr.add_text(line)
                print(f"Added to Qdrant database: {line}...")


def main():
    # Initialize Qdrant manager and setup collection
    qdrant_mgr = QdrantManager()
    qdrant_mgr.setup_collection()

    # File to watch
    file_to_watch = "lecture_simulator/simulated_lecture_output.txt"

    try:
        # Start processing
        process_file(file_to_watch, qdrant_mgr)
    except KeyboardInterrupt:
        print("\nStopping the process...")


if __name__ == "__main__":
    main()