
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

sentences = [
    "The capital of France is Paris.",
    "Paris is the capital city of France.",
    "I enjoy playing cricket on weekends.",
]

def main() -> None:
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    similarities = cosine_similarity(embeddings)

    print(f"\nNumber of sentences: {len(sentences)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print("\nPairwise cosine similarities:")

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(
                f"\nSentence {i + 1} vs Sentence {j + 1}: "
                f"{similarities[i, j]:.4f}"
            )
            print(f"  A: {sentences[i]}")
            print(f"  B: {sentences[j]}")

if __name__ == "__main__":
    main()
