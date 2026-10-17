
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

documents = [
    "Paris is the capital of France and a major European city.",
    "Tokyo is the capital of Japan, known for its technology and culture.",
    "The Maldives is famous for tropical islands, beaches, and coral reefs.",
    "Cricket is a popular sport played with a bat and ball.",
    "Bali is an Indonesian island known for beaches and tourism.",
]

query = "Which places are famous for islands and beaches?"

def main() -> None:
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    retriever = NearestNeighbors(
        n_neighbors=3,
        metric="cosine",
        algorithm="brute",
    )
    retriever.fit(document_embeddings)

    distances, indices = retriever.kneighbors(query_embedding)

    print(f"\nQuery: {query}")
    print("\nTop 3 retrieved documents:")

    for rank, (index, distance) in enumerate(
        zip(indices[0], distances[0]), start=1
    ):
        similarity = 1.0 - distance
        print(f"\n{rank}. Similarity: {similarity:.4f}")
        print(f"   {documents[index]}")

if __name__ == "__main__":
    main()