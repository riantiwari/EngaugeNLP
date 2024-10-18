# src/main.py

import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VIDEO_PATH
from src.audio_processing.audio_extractor import AudioExtractor
from src.transcription.whisper_handler import WhisperTranscriber
from src.post_processing.cleaner import TextCleaner
from src.utils.utils import setup_logging

def main():
    # Set up logging
    logger = setup_logging()

    # Initialize components
    audio_extractor = AudioExtractor(video_path=VIDEO_PATH)
    transcriber = WhisperTranscriber()
    cleaner = TextCleaner()

    # Process video
    logger.info("Starting video processing...")
    for chunk_info in audio_extractor.extract_audio_chunks():
        start_time = chunk_info['start_time']
        audio_data = chunk_info['audio_data']

        # Transcribe audio chunk
        transcription_result = transcriber.transcribe(audio_data, start_time)

        # Clean and output transcription
        for segment in transcription_result['segments']:
            cleaned_text = cleaner.clean_text(segment['text'])
            if cleaned_text.strip():
                timestamped_text = f"[{segment['start']:.2f} - {segment['end']:.2f}] {cleaned_text}"
                print(timestamped_text)
                # Optionally, write to a file or database

    logger.info("Video processing completed.")

if __name__ == '__main__':
    main()