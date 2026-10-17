
from sentence_transformers import SentenceTransformer

from test_chunking import DATA_DIR, load_documents, chunk_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    # 1. Load the original dataset
    documents = load_documents(DATA_DIR)

    # 2. Split each document into chunks
    all_chunks = []

    for document in documents:
        all_chunks.extend(chunk_text(document["text"]))

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")

    # 3. Load the embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}")
    embedding_model = SentenceTransformer(MODEL_NAME)

    # 4. Generate embeddings for all chunks
    print("\nGenerating embeddings...")
    chunk_embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True,
    )

    # 5. Verify the results
    print("\nEmbedding generation complete!")
    print(f"Number of chunks: {len(all_chunks)}")
    print(f"Embedding matrix shape: {chunk_embeddings.shape}")
    print(f"First chunk: {all_chunks[0][:150]}...")
    print(f"First embedding (first 5 values): {chunk_embeddings[0][:5]}")


if __name__ == "__main__":
    main()
