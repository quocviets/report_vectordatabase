import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
os.makedirs("faiss", exist_ok=True)
with open("Data/section_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
texts = [
    f"""
    Disease: {item["disease"]}
    Section: {item["section"]}
    Content: {item["content"]}
    """
    for item in data
]
embeddings = model.encode(texts)
embeddings = np.array(
    embeddings,
    dtype="float32"
)


dimension = embeddings.shape[1]


index_l2 = faiss.IndexFlatL2(dimension)


index_l2.add(embeddings)


faiss.write_index(
    index_l2,
    "faiss/l2_index.faiss"
)


print("L2 Index saved successfully!")