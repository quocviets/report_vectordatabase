import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
with open(
    "Data/test_case.json",
    "r",
    encoding="utf-8"
) as f:
    test_cases = json.load(f)
with open(
    "Data/section_chunks.json",
    "r",
    encoding="utf-8"
) as f:
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
bm25 = BM25Okapi(tokenized_docs)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
index_cosine = faiss.read_index(
    "faiss/cosine_index.faiss"
)
index_l2 = faiss.read_index(
    "faiss/l2_index.faiss"
)
def bm25_search(query, k=3):
    # Tokenize query
    tokenized_query = query.lower().split()

    # Tính BM25 score
    scores = bm25.get_scores(tokenized_query)

    # Lấy top k score cao nhất
    top_indices = scores.argsort()[::-1][:k]

    # Trả về kết quả
    results = [
        data[idx]
        for idx in top_indices
    ]

    return results
def normalize(scores):
    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (
        scores - scores.min()
    ) / (
        scores.max() - scores.min()
    )
def cosine_search(query, k=3):
    # Query -> Embedding
    query_embedding = model.encode([query])

    # Chuyển về float32
    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Normalize để tính Cosine Similarity
    faiss.normalize_L2(query_embedding)

    # Search Top-k trên FAISS
    scores, indices = index_cosine.search(
        query_embedding,
        k
    )

    # Lấy dữ liệu tương ứng
    results = [
        data[idx]
        for idx in indices[0]
    ]

    return results
def l2_search(query, k=3):
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index_l2.search(
        query_embedding,
        k
    )

    results = [
        data[idx]
        for idx in indices[0]
    ]

    return results
def hybrid_search(query, k=3, alpha=0.7):

    # ===== BM25 Score =====
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )


    # ===== Cosine Score =====
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    cosine_scores, cosine_indices = (
        index_cosine.search(
            query_embedding,
            len(data)
        )
    )


    # ===== Mapping lại Cosine Score =====
    full_cosine_scores = np.zeros(
        len(data)
    )


    for score, idx in zip(
        cosine_scores[0],
        cosine_indices[0]
    ):
        full_cosine_scores[idx] = score


    # ===== Normalize BM25 =====
    bm25_norm = normalize(bm25_scores)


    # ===== Normalize Cosine =====
    cosine_norm = normalize(full_cosine_scores)


    # ===== Combine Score =====
    hybrid_scores = (
        alpha * bm25_norm
        +
        (1 - alpha) * cosine_norm
    )


    # ===== Ranking =====
    top_indices = (
        hybrid_scores.argsort()
        [::-1][:k]
    )


    # ===== Return Result =====
    results = [
        data[idx]
        for idx in top_indices
    ]


    return results
def evaluate_method(search_function, name):
    top1_correct = 0
    top3_correct = 0

    total = len(test_cases)

    errors = []

    print(f"\n===== Evaluate {name} =====")

    for test in test_cases:
        query = test["query"]
        expected = test["expected"]

        results = search_function(query, k=3)

        # Check Top-1
        top1_match = False

        if expected.lower() in results[0]["disease"].lower():
            top1_correct += 1
            top1_match = True


        # Check Top-3
        for result in results:
            if expected.lower() in result["disease"].lower():
                top3_correct += 1
                break


        # Save Error
        if not top1_match:
            errors.append({
                "query": query,
                "expected": expected,
                "predict": results[0]["disease"],
                "top3": [
                    result["disease"]
                    for result in results
                ]
            })


    print(
        f"Top-1 Accuracy: {top1_correct}/{total} = {top1_correct / total:.2%}"
    )

    print(
        f"Top-3 Accuracy: {top3_correct}/{total} = {top3_correct / total:.2%}"
    )


    print("\n===== Error Analysis =====")

    for error in errors:
        print("-" * 50)
        print("Query:", error["query"])
        print("Expected:", error["expected"])
        print("Predict:", error["predict"])
        print("Top-3:", error["top3"])
evaluate_method(
    bm25_search,
    "BM25"
)

evaluate_method(
    cosine_search,
    "Cosine"
)

evaluate_method(
    hybrid_search,
    "Hybrid"
)
evaluate_method(
    l2_search,
    "L2"
)