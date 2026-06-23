import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


with open("Data/section_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)


texts = [
    f"""
    Disease: {item["disease"]}
    Section: {item["section"]}
    Content: {item["content"]}
    """
    for item in data
]


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


embeddings = model.encode(texts)

embeddings = np.array(
    embeddings,
    dtype="float32"
)
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]

index_cosine = faiss.IndexFlatIP(dimension)

index_cosine.add(embeddings)
os.makedirs("faiss", exist_ok=True)

faiss.write_index(
    index_cosine,
    "faiss/cosine_index.faiss"
)
print("Cosine Index saved successfully!")
print("Total vectors:", index_cosine.ntotal)