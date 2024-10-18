# src/audio_processing/audio_extractor.py

import ffmpeg
import numpy as np
import os
from config import CHUNK_DURATION, VIDEO_PATH
from src.utils.utils import get_video_duration

class AudioExtractor:
    def __init__(self, video_path=VIDEO_PATH, chunk_duration=CHUNK_DURATION):
        self.video_path = video_path
        self.chunk_duration = chunk_duration
        self.total_duration = get_video_duration(self.video_path)
        self.num_chunks = int(np.ceil(self.total_duration / self.chunk_duration))

    def extract_audio_chunks(self):
        for i in range(self.num_chunks):
            start_time = i * self.chunk_duration
            try:
                out, _ = (
                    ffmpeg
                    .input(self.video_path, ss=start_time, t=self.chunk_duration)
                    .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                    .run(capture_stdout=True, capture_stderr=True)
                )
                audio_data = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
                yield {
                    'start_time': start_time,
                    'audio_data': audio_data
                }
            except ffmpeg.Error as e:
                print(f"An error occurred during audio extraction: {e.stderr.decode()}")
                continue