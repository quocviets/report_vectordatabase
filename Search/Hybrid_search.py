import json 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import json
from rank_bm25 import BM25Okapi
import numpy as np

with open("Data\section_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    f"""
    Disease: {item["disease"]}
    Section: {item["section"]}
    Content: {item["content"]}
"""
    for item in data
]
tokenized_docs = [
    doc.lower().split()
    for doc in documents
]
bm25= BM25Okapi(tokenized_docs)
print("BM25 is ready")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
index_cosine = faiss.read_index(
    "faiss/cosine_index.faiss"
)

print("Cosine FAISS ready")
Query = "Cách điều trị bệnh tuyến trùng"
Tokenized_query = Query.lower().split()
bm25_scores = bm25.get_scores(Tokenized_query)
query_embedding = model.encode([Query])
query_embedding = np.array(
    query_embedding,
    dtype="float32"
)
faiss.normalize_L2(query_embedding)
k = len(data)
cosine_scores, cosine_indices = index_cosine.search(
    query_embedding,
    k
)
full_cosine_scores = np.zeros(len(data))
for scores, idx in zip(
    cosine_scores[0],
    cosine_indices[0]
):
    full_cosine_scores[idx] = scores

bm25_norm = (
    bm25_scores - bm25_scores.min()
) / (
    bm25_scores.max() - bm25_scores.min()
)

cosine_norm = (
    full_cosine_scores - full_cosine_scores.min()
) / (
    full_cosine_scores.max() - full_cosine_scores.min()
)

alpha = 0.5

hybrid_scores = (
    alpha * bm25_norm +
    (1 - alpha) * cosine_norm
)

k = 3

top_indices = np.argsort(
    hybrid_scores
)[::-1][:k]

print("\n===== Hybrid Search =====")

for rank, idx in enumerate(top_indices):
    print(f"\nTop {rank + 1}")

    print("Hybrid:", hybrid_scores[idx])
    print("BM25:", bm25_scores[idx])
    print("Cosine:", full_cosine_scores[idx])

    print(data[idx])