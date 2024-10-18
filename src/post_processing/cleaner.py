# src/post_processing/cleaner.py

import re
from config import FILLER_WORDS

class TextCleaner:
    def __init__(self, filler_words=FILLER_WORDS):
        self.filler_words = filler_words
        # Create regex pattern for filler words
        self.pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.filler_words)) + r')\b', flags=re.IGNORECASE)

    def clean_text(self, text):
        # Remove filler words
        cleaned_text = self.pattern.sub('', text)
        # Remove extra spaces
        cleaned_text = re.sub(' +', ' ', cleaned_text)
        # Fix casing and punctuation if needed
        cleaned_text = cleaned_text.strip().capitalize()
        return cleaned_text