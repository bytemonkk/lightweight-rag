
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

DATA_PATH = Path("data/countries.txt")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 2


def load_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Document is empty: {path}")

    return text


def split_text(text: str) -> list[str]:
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(text), step):
        chunk = text[start : start + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)

    return chunks


def main() -> None:
    text = load_text(DATA_PATH)
    chunks = split_text(text)

    print(f"Document: {DATA_PATH}")
    print(f"Chunks indexed: {len(chunks)}")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    retriever = NearestNeighbors(
        n_neighbors=min(TOP_K, len(chunks)),
        metric="cosine",
        algorithm="brute",
    )
    retriever.fit(embeddings)

    question = "Which places are famous for islands and beaches?"
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    distances, indices = retriever.kneighbors(query_embedding)

    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks:")

    for rank, (index, distance) in enumerate(
        zip(indices[0], distances[0]), start=1
    ):
        print(f"\n{rank}. Cosine similarity: {1 - distance:.4f}")
        print(chunks[index])


if __name__ == "__main__":
    main()
