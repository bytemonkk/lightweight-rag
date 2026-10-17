
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from test_chunking import DATA_DIR, load_documents, chunk_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_NEIGHBORS = 3


class SklearnRetriever:
    def __init__(self, n_neighbors=3):
        self.retriever = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.chunks = []

    def fit(self, chunks, embeddings):
        """Build an index from the document chunk embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have equal lengths.")

        self.chunks = chunks
        self.retriever.fit(embeddings)
        print("Vector index built successfully!")

    def retrieve(self, query_embedding):
        """Return the most relevant chunks for a query embedding."""
        distances, indices = self.retriever.kneighbors(
            [query_embedding]
        )

        results = [
            (self.chunks[idx], float(distance))
            for idx, distance in zip(indices[0], distances[0])
        ]

        return results


def main() -> None:
    # Load documents and create the article's 25-word chunks
    documents = load_documents(DATA_DIR)
    all_chunks = []

    for document in documents:
        all_chunks.extend(chunk_text(document["text"]))

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")

    # Generate embeddings for all chunks
    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
    )

    # Fit the retriever
    vector_db = SklearnRetriever(n_neighbors=N_NEIGHBORS)
    vector_db.fit(all_chunks, chunk_embeddings)

    # Embed a question and retrieve relevant chunks
    question = "Which Asian countries are best known for islands and beaches?"
    query_embedding = model.encode(question)

    results = vector_db.retrieve(query_embedding)

    print(f"\nQuestion: {question}")
    print(f"\nTop {N_NEIGHBORS} retrieved chunks:")

    for rank, (chunk, distance) in enumerate(results, start=1):
        print(f"\n{rank}. Cosine distance: {distance:.4f}")
        print(chunk)


if __name__ == "__main__":
    main()
