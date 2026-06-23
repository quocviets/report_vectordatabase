import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
with open("Data/section_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
index_l2 = faiss.read_index(
    "l2_index.faiss"
)

index_cosine = faiss.read_index(
    "cosine_index.faiss"
)
query = "Thuốc trị khô vằn"
query_vector = model.encode([query])

query_vector = np.array(
    query_vector,
    dtype="float32"
)
k = 3

distance_l2, index_l2_result = index_l2.search(
    query_vector,
    k
)

query_cosine = query_vector.copy()

faiss.normalize_L2(query_cosine)

score_cosine, index_cosine_result = index_cosine.search(
    query_cosine,
    k
)
print("\n===== L2 Distance =====")

for i, idx in enumerate(index_l2_result[0]):
    print(f"\nTop {i+1}")
    print("Distance:", distance_l2[0][i])
    print(data[idx])
print("\n===== Cosine Similarity =====")

for i, idx in enumerate(index_cosine_result[0]):
    print(f"\nTop {i+1}")
    print("Score:", score_cosine[0][i])
    print(data[idx])