import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
with open("Data/structure_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
chunks = [
    chunk["text"]
    for chunk in data
]

print("Total chunks:", len(chunks))

if chunks:
    print("First chunk:")
    print(chunks[0])