from transformers import pipeline
from collections import Counter
import re

# Read the input text
with open("article.txt", "r", encoding="utf-8") as f:
    long_text = f.read().strip()  

max_chunk = 512

def chunk_text(text, max_tokens=max_chunk):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = chunk_text(long_text)

# Abstractive summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
abstractive_summaries = []
for chunk in chunks:
    if chunk.strip():  # Avoid empty chunks
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        abstractive_summaries.append(summary)
abstractive_final = " ".join(abstractive_summaries)

# Extractive summarization
def extractive_summary(text, ratio=0.3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    words = re.findall(r'\w+', text.lower())
    if not words or not sentences:
        raise ValueError("Text too short for extractive summary.")
    freq = Counter(words)
    scores = {sent: sum(freq[w] for w in re.findall(r'\w+', sent.lower())) for sent in sentences}
    top_n = max(1, int(len(sentences) * ratio))  # At least 1 sentence
    top_sentences = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return " ".join(top_sentences)

try:
    extractive_final = extractive_summary(long_text, ratio=0.3)
except ValueError as e:
    extractive_final = str(e)

print("\nExtractive Summary\n")
print(extractive_final)
print("\nAbstractive Summary\n")
print(abstractive_final)
