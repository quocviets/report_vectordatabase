import faiss
import json

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_chunks():

    with open(
        "C:\TMA_Intern\Data\structure_chunks.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def semantic_search(
    query,
    top_k=5
):

    index = faiss.read_index(
        "Output/faiss.index"
    )

    chunks = load_chunks()

    query_vector = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")


    distances, indices = index.search(
        query_vector,
        top_k
    )


    results = []

    for i, idx in enumerate(indices[0]):

        results.append(
            {
                "rank": i + 1,
                "distance": float(
                    distances[0][i]
                ),
                "text": chunks[idx]["text"]
            }
        )

    return results


if __name__ == "__main__":

    query = input(
        "Enter your query: "
    )

    results = semantic_search(
        query
    )


    for result in results:

        print(
            f"\nTop {result['rank']}"
        )

        print(
            f"Distance: {result['distance']}"
        )

        print(
            result["text"]
        )