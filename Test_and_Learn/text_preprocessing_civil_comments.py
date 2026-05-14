import torch
import json
import re
import html
from datasets import load_dataset
import string

# NOTE**   This code extracts 100 samples of cleaned text

# =========================
# 1. Define Stop Words
# =========================
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "will", "with", "i", "me", "my", "we", "you",
    "your", "this", "but", "not", "can", "do", "have", "should", "would"
}

# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# STOP_WORDS = set(stopwords.words('english'))

# =========================
# 2. Preprocessing
# =========================
def preprocess_text(text: str) -> str | None:
    """
    Clean and validate a civil_comments text entry.
    Returns None if the text should be skipped.
    """
    if not isinstance(text, str):
        return None

    # Convert to lowercase
    text = text.lower()

    # Decode HTML entities (e.g &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # Strip HTML tags (e.g. <br>, <b>, <i>)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize unicode punctuation to ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove all punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    text = " ".join(filtered_words)

    # Return None if text is empty after filtering
    if not text:
        return None

    return text


# =========================
# 3. Load + Filter Dataset
# =========================
raw_data = load_dataset("google/civil_comments")["test"].select(range(100))

processed_data = []
for item in raw_data:
    clean_text = preprocess_text(item["text"])
    if clean_text is None:
        continue

    processed_data.append({
        "text": clean_text,
        "raw_toxicity": item["toxicity"]   # Keep raw float score as-is
    })

# Deduplicate by exact text match
seen = set()
deduplicated_data = []
for item in processed_data:
    if item["text"] not in seen:
        seen.add(item["text"])
        deduplicated_data.append(item)

print(f"Samples after preprocessing + dedup: {len(deduplicated_data)}")
data = deduplicated_data[:100]

print(data)
