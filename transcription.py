import os
import tempfile
import moviepy.editor as mp
import whisper
import spacy
import logging
import sys

from vector_database.qdrant_manager import QdrantManager



def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def transcribe_video_real_time(video_file, chunk_duration=20.0, overlap=2.0, qdrant_manager: QdrantManager = None, collection_name="lecture_collection"):
    """
    Transcribe a video file in real-time by sentences with accurate timestamps.

    Parameters:
    - video_file (str): Path to the video file.
    - chunk_duration (float): Duration of each audio chunk in seconds.
    - overlap (float): Overlap duration between consecutive chunks in seconds.
    """
    line_number = 0

    # Check if the video file exists
    if not os.path.isfile(video_file):
        print(f"Error: The video file '{video_file}' does not exist.")
        return

    # Initialize the Whisper model
    model = whisper.load_model("small")


    # Load SpaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Failed to download SpaCy model: {e}")
            return
    except Exception as e:
        print(f"Error loading SpaCy model: {e}")
        return

    # Open the video file
    try:
        with mp.VideoFileClip(video_file) as video:
            if video.audio is None:
                print("Error: The video file has no audio track.")
                return

            audio = video.audio
            total_duration = audio.duration
            print(f"Total duration: {total_duration:.2f} seconds")

            current_time = 0.0
            buffer_text = ""  # Buffer to hold partial sentences

            # Store already processed segments to avoid duplicates
            processed_segments = []

            while current_time < total_duration:
                end_time = min(current_time + chunk_duration, total_duration)
                try:
                    # Extract the audio chunk with overlap
                    start_time = max(current_time - overlap, 0.0)
                    audio_chunk = audio.subclip(start_time, end_time)

                    # Create a temporary file for the audio chunk
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        temp_audio_path = temp_audio_file.name

                    # Write the audio chunk to the temporary file
                    audio_chunk.write_audiofile(
                        temp_audio_path, codec="pcm_s16le", fps=16000, verbose=False, logger=None
                    )

                    # Transcribe the audio chunk
                    result = model.transcribe(temp_audio_path, language='en', verbose=False)

                    # Remove the temporary audio chunk file
                    os.remove(temp_audio_path)

                    # Process each segment
                    for segment in result['segments']:
                        seg_start = segment['start'] + start_time  # Adjust start time based on chunk
                        seg_end = segment['end'] + start_time
                        seg_text = segment['text'].strip()

                        if not seg_text:
                            continue

                        # Check if the segment has already been processed
                        if any(abs(seg_start - ps['start']) < 0.1 for ps in processed_segments):
                            continue  # Skip duplicate segments

                        # Append to buffer
                        buffer_text += " " + seg_text if buffer_text else seg_text

                        # Use SpaCy to detect sentences
                        doc = nlp(buffer_text)
                        sentences = list(doc.sents)

                        # Print all complete sentences except the last one (which might be incomplete)
                        for sentence in sentences[:-1]:
                            sentence_text = sentence.text.strip()
                            if sentence_text:
                                # Find the timestamp for the sentence
                                # Approximate by taking the start time of the first segment contributing to the sentence
                                # This requires mapping sentences to segments
                                # For simplicity, assign the current chunk's start_time
                                timestamp = format_timestamp(seg_start)
                                # print(seg_start)
                                # print(f"{timestamp} - \"{sentence_text}\"")
                                qdrant_manager.add_text(collection_name, sentence_text, int(seg_start), int(seg_end))

                                with open('sample_visual_notes.txt') as file:
                                    lines = file.readlines()  # Reads all lines into a list
                                    if line_number < len(lines):
                                        timestamp, text = lines[line_number].split(' - ')

                                        minutes, seconds = map(int, timestamp.split(":"))
                                        total_seconds = minutes * 60 + seconds

                                        if seg_start > total_seconds:
                                            line_number += 1
                                            qdrant_manager.add_drawing_text(collection_name, text, total_seconds)

                                            # print(f"Drawing:{timestamp} {text}")
                                    else:
                                        print("Line number is out of range.")

                        # Update buffer with the last (possibly incomplete) sentence
                        if sentences:
                            buffer_text = sentences[-1].text.strip()
                        else:
                            buffer_text = ""

                        # Mark this segment as processed
                        processed_segments.append({'start': seg_start, 'end': seg_end})

                except Exception as e:
                    print(f"Error processing chunk {current_time:.2f}-{end_time:.2f}: {e}")

                # Update the current time
                current_time += chunk_duration

                # Optional: Delay can be introduced here if needed
                # time.sleep(chunk_duration)

            # After processing all chunks, print any remaining text in the buffer with the end timestamp
            if buffer_text:
                timestamp = format_timestamp(total_duration)
                # print(f"{timestamp} - \"{buffer_text}\"")
                print(f"{timestamp}")

    except Exception as e:
        print(f"Error processing video file: {e}")

    print("Transcription completed.")

# Example usage
# if __name__ == "__main__":
#     print("Starting transcription...")
#     video_file = "sample_video.mp4"  # Replace with your video file path
#     transcribe_video_real_time(video_file, chunk_duration=20.0, overlap=2, qdrant_manager=QdrantManager())