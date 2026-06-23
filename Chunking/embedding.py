import json
import numpy as np

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_chunks(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings():

    data = load_chunks(
        "Data\section_chunks.json"
    )

    texts = [
        f"""
        Disease: {item["disease"]}
        Section: {item["section"]}
        Content: {item["content"]}
        """
        for item in data
]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    np.save(
        "Output/Section-base/embeddings.npy",
        embeddings
    )

    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    create_embeddings()