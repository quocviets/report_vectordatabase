import faiss
import numpy as np


def create_index():

    embeddings = np.load(
        "Output/embeddings.npy"
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        "Output/faiss.index"
    )

    print(
        "Total vectors:",
        index.ntotal
    )


if __name__ == "__main__":
    create_index()