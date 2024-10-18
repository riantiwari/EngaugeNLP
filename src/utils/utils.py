# src/utils/utils.py

import ffmpeg
import logging
import os
from config import LOG_LEVEL

def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"An error occurred while getting video duration: {e.stderr.decode()}")
        return 0.0

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    return logger