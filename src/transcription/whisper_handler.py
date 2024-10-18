# src/transcription/whisper_handler.py

import whisper
from config import WHISPER_MODEL
import torch

class WhisperTranscriber:
    def __init__(self, model_name=WHISPER_MODEL):
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the model
        self.model = whisper.load_model(model_name).to(self.device)

    def transcribe(self, audio_data, start_time):
        # Transcribe the audio
        result = self.model.transcribe(audio_data, language='en', word_timestamps=False)
        # Adjust timestamps
        for segment in result['segments']:
            segment['start'] += start_time
            segment['end'] += start_time
        return result