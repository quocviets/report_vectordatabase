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
query = "Cách điều trị bệnh đạo ôn cô bổng"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)
k = 3
top_indices = np.argsort(scores)[::-1][:k]

print("\n===== BM25 Search =====")

for rank, idx in enumerate(top_indices):
    print(f"\nTop {rank + 1}")
    print("Score:", scores[idx])
    print(data[idx])